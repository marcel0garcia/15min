"""
BTC price feed — Coinbase Exchange WebSocket.

Coinbase is a CF Benchmarks BRTI constituent exchange (same index Kalshi
uses for KXBTC settlement), requires no auth, and is reliable on US IPs.

This is the single-venue price source. The consolidated BRTI feed
(btc15/feeds/brti_feed.py) wraps the venue connectors used for recording
into the same interface so the engine can swap between them via
feeds.price_source config.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Callable, Optional

import websockets

from btc15.feeds.types import BarAccumulator, OHLCBar, Tick, TickHandler

log = logging.getLogger(__name__)


class CoinbaseFeed:
    """Real-time BTC/USD price feed via Coinbase Exchange WebSocket."""

    STREAM_URL = "wss://ws-feed.exchange.coinbase.com"

    def __init__(self, bar_interval_sec: int = 60, lookback_bars: int = 200):
        self.bar_interval_sec = bar_interval_sec
        self.lookback_bars = lookback_bars

        self._tick_handlers: list[TickHandler] = []
        self._bar_handlers: list[Callable] = []

        self._ticks: deque[Tick] = deque(maxlen=5000)
        self._bars: deque[OHLCBar] = deque(maxlen=lookback_bars)
        self._current_bar: Optional[BarAccumulator] = None

        self._last_price: float = 0.0
        self._running = False
        self._reconnect_delay = 2.0

    @property
    def latest_price(self) -> float:
        return self._last_price

    @property
    def bars(self) -> list[OHLCBar]:
        return list(self._bars)

    def partial_bar(
        self,
        min_age_sec: float = 5.0,
        min_ticks: int = 5,
    ) -> Optional[OHLCBar]:
        """Snapshot of the in-progress (unclosed) bar from the accumulator.

        Returns the partial bar as a regular OHLCBar (open/high/low/close
        reflect what's happened so far in the current bar interval), or None
        when the partial is too young to be informative (< min_age_sec or
        < min_ticks).

        Used by the ensemble's technical_momentum component so RSI/MACD/BB
        update once per scan instead of once per bar interval — the slow
        minute-scale signal becomes responsive at the second timescale.
        """
        acc = self._current_bar
        if acc is None or acc._open is None:
            return None
        age = time.time() - acc.ts_ms / 1000
        if age < min_age_sec or acc._count < min_ticks:
            return None
        return acc.to_ohlc()

    @property
    def bars_with_partial(self) -> list[OHLCBar]:
        """Closed bars plus the live partial bar (when sufficient).

        Drop-in replacement for `bars` when callers want per-scan freshness
        on indicators that read bar series (RSI, MACD, BB, trend regression).
        """
        partial = self.partial_bar()
        if partial is None:
            return list(self._bars)
        return list(self._bars) + [partial]

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
        log.info(f"Connecting to Coinbase WebSocket: {self.STREAM_URL}")
        async with websockets.connect(
            self.STREAM_URL,
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            # Subscribe to matches channel — per-trade price + actual size.
            await ws.send(json.dumps({
                "type": "subscribe",
                "product_ids": ["BTC-USD"],
                "channels": ["matches"],
            }))
            self._reconnect_delay = 2.0
            log.info("Coinbase WebSocket connected — receiving BTC-USD trade feed")

            async for raw in ws:
                try:
                    data = json.loads(raw)
                    await self._handle_message(data)
                except Exception as e:
                    log.error(f"Coinbase handler error: {e}", exc_info=True)

    async def _handle_message(self, data: dict):
        msg_type = data.get("type")
        # "match" = live trade; "last_match" = snapshot of most recent trade on subscribe
        if msg_type not in ("match", "last_match"):
            return
        if data.get("product_id") != "BTC-USD":
            return

        price = float(data.get("price", 0))
        if not price:
            return

        qty = float(data.get("size", 0) or 0)
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
            self._current_bar = BarAccumulator(bar_ts_ms, self.bar_interval_sec)

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
            self._current_bar = BarAccumulator(bar_ts_ms, self.bar_interval_sec)

        self._current_bar.add(tick)

    async def seed_from_rest(self, limit: int = 100):
        """Seed initial bars from Coinbase Exchange REST candles endpoint."""
        import aiohttp
        from datetime import datetime, timezone, timedelta
        # Coinbase granularity: 60=1m, 300=5m, 900=15m
        granularity = self.bar_interval_sec
        end = datetime.now(timezone.utc)
        start = end - timedelta(seconds=granularity * limit)
        url = (
            f"https://api.exchange.coinbase.com/products/BTC-USD/candles"
            f"?granularity={granularity}"
            f"&start={start.isoformat()}"
            f"&end={end.isoformat()}"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    candles = await resp.json()
            # Coinbase candles: [[time, low, high, open, close, volume], ...] newest first
            for k in reversed(candles[-limit:]):
                bar = OHLCBar(
                    open=float(k[3]),
                    high=float(k[2]),
                    low=float(k[1]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    vwap=float(k[4]),   # Coinbase candles have no VWAP; use close
                    ts=int(k[0]),
                    interval_sec=self.bar_interval_sec,
                )
                self._bars.append(bar)
            if self._bars:
                self._last_price = self._bars[-1].close
            log.info(f"Seeded {len(self._bars)} bars from Coinbase REST (last close: ${self._last_price:,.2f})")
        except Exception as e:
            log.warning(f"Failed to seed bars from Coinbase REST: {e}")


# BarAccumulator moved to btc15.feeds.types
