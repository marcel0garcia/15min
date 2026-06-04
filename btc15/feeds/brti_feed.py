"""BRTI consolidated price feed.

Drop-in replacement for CoinbaseFeed that consumes the live consolidated
BRTI mid (median of Coinbase / Kraken / Bitstamp top-of-book, with MAD
outlier rejection — same algorithm as btc15.recording.brti.reconstruct,
running live in the engine's _live_brti_loop).

The feed itself owns no WebSocket — prices are pushed in via push_brti()
from the engine when each fresh consolidated mid is computed. Historical
bars are seeded from Coinbase Exchange REST candles at startup so the
technical-indicator stack (RSI / MACD / BB / trend regression) has
context before live BRTI ticks accumulate.

Why Coinbase REST for the seed when the live source is the consolidated
BRTI: Coinbase IS a BRTI constituent (~33% of the consolidated mid in
calm regimes), so the seeded historical context is close enough to BRTI
for indicator calibration. Once live BRTI begins ticking, all new bars
are derived from the consolidated mid.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Callable, Optional

from btc15.feeds.types import BarAccumulator, OHLCBar, Tick, TickHandler

log = logging.getLogger(__name__)


class BRTIPriceFeed:
    """Consolidated BRTI feed exposing the CoinbaseFeed interface."""

    def __init__(self, bar_interval_sec: int = 60, lookback_bars: int = 200):
        self.bar_interval_sec = bar_interval_sec
        self.lookback_bars = lookback_bars

        self._tick_handlers: list[TickHandler] = []
        self._bar_handlers: list[Callable] = []

        self._ticks: deque[Tick] = deque(maxlen=5000)
        self._bars: deque[OHLCBar] = deque(maxlen=lookback_bars)
        self._current_bar: Optional[BarAccumulator] = None

        self._last_price: float = 0.0
        self._last_tick_ts: float = 0.0
        self._running = False
        self._run_task: Optional[asyncio.Task] = None

    # ── PriceAggregator-compatible interface ──────────────────────────────
    # The engine consumes self.price_feed via this surface and stays agnostic
    # about whether the underlying source is single-venue Coinbase or the
    # consolidated BRTI mid.

    @property
    def latest_price(self) -> float:
        return self._last_price

    @property
    def current_price(self) -> float:
        """PriceAggregator-style alias for latest_price."""
        return self._last_price

    @property
    def feed(self):
        """Self-reference so engine code that historically went through
        the aggregator's .feed attribute (e.g. price_feed.feed.on_tick)
        keeps working when price_feed is the BRTI feed directly."""
        return self

    def realized_vol(self, lookback: int = 20) -> float:
        """Alias matching PriceAggregator.realized_vol(lookback)."""
        return self.realized_vol_annualized(lookback)

    def feed_age_sec(self) -> float:
        if self._last_tick_ts == 0:
            return float("inf")
        return time.time() - self._last_tick_ts

    @property
    def bars(self) -> list[OHLCBar]:
        return list(self._bars)

    def partial_bar(
        self,
        min_age_sec: float = 5.0,
        min_ticks: int = 5,
    ) -> Optional[OHLCBar]:
        """Snapshot of the in-progress (unclosed) bar from the accumulator,
        or None if too young / too sparse for the technical indicators to
        consume safely."""
        acc = self._current_bar
        if acc is None or acc._open is None:
            return None
        age = time.time() - acc.ts_ms / 1000
        if age < min_age_sec or acc._count < min_ticks:
            return None
        return acc.to_ohlc()

    @property
    def bars_with_partial(self) -> list[OHLCBar]:
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

    def on_tick(self, handler: TickHandler) -> None:
        self._tick_handlers.append(handler)

    def on_bar(self, handler: Callable) -> None:
        self._bar_handlers.append(handler)

    # ── Push API — engine's _live_brti_loop calls this on each fresh BRTI ──

    async def push_brti(self, price: float, ts_sec: float) -> None:
        """Ingest a fresh consolidated BRTI mid. Updates last_price, the
        tick buffer, the current bar accumulator (rolling over to a new
        bar on interval boundaries), and fires registered handlers."""
        if not price or price <= 0:
            return
        ts_ms = int(ts_sec * 1000)
        tick = Tick(price=price, qty=0.0, ts_ms=ts_ms)
        self._ticks.append(tick)
        self._last_price = price
        self._last_tick_ts = ts_sec
        await self._update_bar(tick)

        for handler in self._tick_handlers:
            try:
                await handler(price, 0.0, ts_ms)
            except Exception as e:
                log.error(f"BRTI tick handler error: {e}")

    async def _update_bar(self, tick: Tick) -> None:
        bar_ts_ms = (tick.ts_ms // (self.bar_interval_sec * 1000)) * (self.bar_interval_sec * 1000)
        if self._current_bar is None:
            self._current_bar = BarAccumulator(bar_ts_ms, self.bar_interval_sec)
        if tick.ts_ms >= self._current_bar.ts_ms + self.bar_interval_sec * 1000:
            finished = self._current_bar.to_ohlc()
            self._bars.append(finished)
            log.debug(
                f"BRTI bar: O={finished.open:.0f} H={finished.high:.0f} "
                f"L={finished.low:.0f} C={finished.close:.0f}"
            )
            for handler in self._bar_handlers:
                try:
                    await handler(finished)
                except Exception as e:
                    log.error(f"BRTI bar handler error: {e}")
            self._current_bar = BarAccumulator(bar_ts_ms, self.bar_interval_sec)
        self._current_bar.add(tick)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Seed historical bars from Coinbase REST then spawn the (no-op
        for now) run-loop. Live BRTI ticks arrive via push_brti() from
        the engine's _live_brti_loop, so no WebSocket of our own."""
        self._running = True
        await self.seed_from_rest(limit=150)
        self._run_task = asyncio.create_task(self._idle_loop(), name="brti-feed-idle")
        log.info("[BRTI] BRTIPriceFeed started — awaiting live BRTI pushes")

    async def stop(self) -> None:
        self._running = False
        if self._run_task is not None:
            self._run_task.cancel()
            self._run_task = None

    async def _idle_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            pass

    async def seed_from_rest(self, limit: int = 100) -> None:
        """Seed historical bars from Coinbase REST candles. Coinbase is a
        BRTI constituent so this is a reasonable approximation for the
        technical-indicator warm-up period."""
        import aiohttp
        from datetime import datetime, timezone, timedelta
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
            for k in reversed(candles[-limit:]):
                bar = OHLCBar(
                    open=float(k[3]),
                    high=float(k[2]),
                    low=float(k[1]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    vwap=float(k[4]),
                    ts=int(k[0]),
                    interval_sec=self.bar_interval_sec,
                )
                self._bars.append(bar)
            if self._bars:
                self._last_price = self._bars[-1].close
            log.info(
                f"[BRTI] Seeded {len(self._bars)} historical bars from Coinbase "
                f"REST (last close: ${self._last_price:,.2f}) — live BRTI ticks "
                f"will replace this once the engine's _live_brti_loop starts pushing"
            )
        except Exception as e:
            log.warning(f"[BRTI] Failed to seed historical bars: {e}")
