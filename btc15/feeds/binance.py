"""
BTC price feed — Kraken WebSocket as primary (Binance blocked on US IPs).
Kraken is one of the CF Benchmarks BRTI constituent exchanges, making it
a better proxy for Kalshi's settlement price than Binance anyway.

Keeps the same public API (BinanceFeed class name preserved for compatibility).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Coroutine, Optional

import websockets

log = logging.getLogger(__name__)

TickHandler = Callable[[float, float, int], Coroutine]


@dataclass
class Tick:
    price: float
    qty: float
    ts_ms: int

    @property
    def ts(self) -> float:
        return self.ts_ms / 1000


@dataclass
class OHLCBar:
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    ts: float
    interval_sec: int
    trade_count: int = 0

    @property
    def pct_change(self) -> float:
        return (self.close - self.open) / self.open if self.open else 0.0

    @property
    def body_pct(self) -> float:
        return (self.close - self.open) / self.open if self.open else 0.0


class BinanceFeed:
    """
    Real-time BTC/USD price feed via Kraken WebSocket v2.
    Named BinanceFeed for API compatibility with the rest of the codebase.
    """

    STREAM_URL = "wss://ws.kraken.com/v2"

    def __init__(self, bar_interval_sec: int = 60, lookback_bars: int = 200):
        self.bar_interval_sec = bar_interval_sec
        self.lookback_bars = lookback_bars

        self._tick_handlers: list[TickHandler] = []
        self._bar_handlers: list[Callable] = []

        self._ticks: deque[Tick] = deque(maxlen=5000)
        self._bars: deque[OHLCBar] = deque(maxlen=lookback_bars)
        self._current_bar: Optional[_BarAccumulator] = None

        self._last_price: float = 0.0
        self._running = False
        self._reconnect_delay = 2.0

    @property
    def latest_price(self) -> float:
        return self._last_price

    @property
    def bars(self) -> list[OHLCBar]:
        return list(self._bars)

    @property
    def close_prices(self) -> list[float]:
        return [b.close for b in self._bars]

    @property
    def volumes(self) -> list[float]:
        return [b.volume for b in self._bars]

    def recent_ticks(self, seconds: float = 60) -> list[Tick]:
        cutoff = time.time() - seconds
        return [t for t in self._ticks if t.ts >= cutoff]

    def realized_vol_annualized(self, lookback_bars: int = 20) -> float:
        import numpy as np
        closes = self.close_prices[-lookback_bars - 1:]
        if len(closes) < 5:
            return 0.80
        returns = np.diff(np.log(closes))
        bars_per_year = 365 * 24 * 3600 / self.bar_interval_sec
        return float(np.std(returns) * np.sqrt(bars_per_year))

    def on_tick(self, handler: TickHandler):
        self._tick_handlers.append(handler)

    def on_bar(self, handler: Callable):
        self._bar_handlers.append(handler)

    async def run(self):
        self._running = True
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"Kraken feed disconnected: {e}. Reconnecting in {self._reconnect_delay}s")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 1.5, 30.0)
            else:
                self._reconnect_delay = 2.0

    async def stop(self):
        self._running = False

    async def _connect_and_listen(self):
        log.info(f"Connecting to Kraken WebSocket: {self.STREAM_URL}")
        async with websockets.connect(
            self.STREAM_URL,
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            # Subscribe to ticker (trade-level updates)
            await ws.send(json.dumps({
                "method": "subscribe",
                "params": {
                    "channel": "ticker",
                    "symbol": ["BTC/USD"],
                    "event_trigger": "trades",
                }
            }))
            self._reconnect_delay = 2.0
            log.info("Kraken WebSocket connected — receiving BTC/USD ticker")

            async for raw in ws:
                try:
                    data = json.loads(raw)
                    await self._handle_message(data)
                except Exception as e:
                    log.error(f"Kraken handler error: {e}", exc_info=True)

    async def _handle_message(self, data: dict):
        channel = data.get("channel")
        if channel != "ticker":
            return

        events = data.get("data", [])
        if not events:
            return

        ticker = events[0]
        price = float(ticker.get("last", 0))
        if not price:
            return

        qty = float(ticker.get("volume", 0) or 0)
        ts_ms = int(time.time() * 1000)

        tick = Tick(price=price, qty=qty, ts_ms=ts_ms)
        self._ticks.append(tick)
        self._last_price = price

        await self._update_bar(tick)

        for handler in self._tick_handlers:
            try:
                await handler(price, qty, ts_ms)
            except Exception as e:
                log.error(f"Tick handler error: {e}")

    async def _update_bar(self, tick: Tick):
        bar_ts_ms = (tick.ts_ms // (self.bar_interval_sec * 1000)) * (self.bar_interval_sec * 1000)

        if self._current_bar is None:
            self._current_bar = _BarAccumulator(bar_ts_ms, self.bar_interval_sec)

        if tick.ts_ms >= self._current_bar.ts_ms + self.bar_interval_sec * 1000:
            finished = self._current_bar.to_ohlc()
            self._bars.append(finished)
            log.debug(f"Bar: O={finished.open:.0f} H={finished.high:.0f} "
                      f"L={finished.low:.0f} C={finished.close:.0f}")
            for handler in self._bar_handlers:
                try:
                    await handler(finished)
                except Exception as e:
                    log.error(f"Bar handler error: {e}")
            self._current_bar = _BarAccumulator(bar_ts_ms, self.bar_interval_sec)

        self._current_bar.add(tick)

    async def seed_from_rest(self, limit: int = 100):
        """Seed initial bars from Kraken REST OHLC endpoint."""
        import aiohttp
        interval_map = {60: 1, 300: 5, 900: 15}
        interval = interval_map.get(self.bar_interval_sec, 1)
        url = f"https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval={interval}&count={limit}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    body = await resp.json()
            result = body.get("result", {})
            ohlc_data = result.get("XXBTZUSD", result.get("XBTUSD", []))
            for k in ohlc_data[-limit:]:
                # Kraken OHLC: [time, open, high, low, close, vwap, volume, count]
                bar = OHLCBar(
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[6]),
                    vwap=float(k[5]),
                    ts=int(k[0]),
                    interval_sec=self.bar_interval_sec,
                    trade_count=int(k[7]),
                )
                self._bars.append(bar)
            if self._bars:
                self._last_price = self._bars[-1].close
            log.info(f"Seeded {len(self._bars)} bars from Kraken REST (last close: ${self._last_price:,.2f})")
        except Exception as e:
            log.warning(f"Failed to seed bars from Kraken REST: {e}")


class _BarAccumulator:
    def __init__(self, ts_ms: int, interval_sec: int):
        self.ts_ms = ts_ms
        self.interval_sec = interval_sec
        self._open: Optional[float] = None
        self._high: float = float("-inf")
        self._low: float = float("inf")
        self._close: float = 0.0
        self._vol: float = 0.0
        self._vol_price: float = 0.0
        self._count: int = 0

    def add(self, tick: Tick):
        if self._open is None:
            self._open = tick.price
        self._high = max(self._high, tick.price)
        self._low = min(self._low, tick.price)
        self._close = tick.price
        self._vol += tick.qty
        self._vol_price += tick.qty * tick.price
        self._count += 1

    def to_ohlc(self) -> OHLCBar:
        vwap = self._vol_price / self._vol if self._vol > 0 else self._close
        return OHLCBar(
            open=self._open or self._close,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._vol,
            vwap=vwap,
            ts=self.ts_ms / 1000,
            interval_sec=self.interval_sec,
            trade_count=self._count,
        )
