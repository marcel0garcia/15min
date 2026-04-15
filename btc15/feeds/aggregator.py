"""Multi-feed price aggregator — primary Binance WS + REST fallback."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import aiohttp

from btc15.feeds.binance import BinanceFeed, OHLCBar

log = logging.getLogger(__name__)


class PriceAggregator:
    """
    Manages the Binance WebSocket feed and provides a unified interface
    for price + bar data to the rest of the system.

    Falls back to Coinbase REST if primary WS is stale.
    """

    STALE_THRESHOLD_SEC = 30  # consider feed stale if no tick in this many seconds

    def __init__(
        self,
        bar_interval_sec: int = 60,
        lookback_bars: int = 200,
        coinbase_rest_url: str = "https://api.coinbase.com/v2/prices/BTC-USD/spot",
    ):
        self.feed = BinanceFeed(
            bar_interval_sec=bar_interval_sec,
            lookback_bars=lookback_bars,
        )
        self._coinbase_url = coinbase_rest_url
        self._last_tick_ts: float = 0.0
        self._fallback_price: Optional[float] = None
        self._fallback_price_ts: float = 0.0   # when the fallback was last fetched
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._last_stale_log: float = 0.0
        self._FALLBACK_STALE_SEC = 8  # warn if fallback itself is stale

    # ── Startup / Shutdown ───────────────────────────────────────────────

    async def start(self):
        self._running = True
        # Seed historical bars first
        await self.feed.seed_from_rest(limit=150)

        # Register internal tick tracker
        self.feed.on_tick(self._on_tick)

        # Start WebSocket in background
        self._tasks.append(asyncio.create_task(self.feed.run(), name="binance-ws"))
        # Start fallback poller
        self._tasks.append(asyncio.create_task(self._fallback_loop(), name="price-fallback"))
        # Start bar heartbeat — emits synthetic flat bars if no ticks arrive
        self._tasks.append(asyncio.create_task(self._bar_heartbeat(), name="bar-heartbeat"))
        log.info("PriceAggregator started")

    async def stop(self):
        self._running = False
        await self.feed.stop()
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()

    async def _on_tick(self, __price: float, __qty: float, __ts_ms: int):
        self._last_tick_ts = time.time()

    async def _bar_heartbeat(self):
        """
        Force-emit synthetic flat bars when no ticks arrive.
        Prevents technical indicators from freezing on a stale last bar.
        Checks every bar_interval_sec; if no real bar was emitted in that window
        and we have a current price, synthesise a flat bar at that price.
        """
        interval = self.feed.bar_interval_sec
        while self._running:
            await asyncio.sleep(interval)
            try:
                price = self.feed.latest_price
                if not price:
                    continue
                bars = self.feed.bars
                now = time.time()
                # Only emit a heartbeat bar if no real bar arrived in the last interval
                if bars and (now - bars[-1].ts) < interval * 1.5:
                    continue
                from btc15.feeds.binance import OHLCBar
                synthetic = OHLCBar(
                    open=price, high=price, low=price, close=price,
                    volume=0.0, vwap=price,
                    ts=now - (now % interval),
                    interval_sec=interval,
                    trade_count=0,
                )
                self.feed._bars.append(synthetic)
                log.debug(f"[HEARTBEAT] Synthetic flat bar emitted @ ${price:,.2f} (no ticks in {interval}s)")
            except Exception as e:
                log.debug(f"Bar heartbeat error: {e}")

    async def _fallback_loop(self):
        """Poll Coinbase REST every 5 seconds as a sanity check / fallback.
        Fetches immediately on start so there's always a price before the first
        Kraken tick arrives (prevents RuntimeError during early startup)."""
        while self._running:
            try:
                price = await self._fetch_coinbase_price()
                if price is not None:
                    self._fallback_price = price
                    self._fallback_price_ts = time.time()
            except Exception:
                pass
            await asyncio.sleep(5)

    async def _fetch_coinbase_price(self) -> Optional[float]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._coinbase_url,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data = await resp.json()
                    # Coinbase Exchange ticker: {"price": "66807.00", ...}
                    # Coinbase consumer API:    {"data": {"amount": "66807.00"}}
                    price = data.get("price") or data.get("data", {}).get("amount")
                    return float(price) if price else None
        except Exception as e:
            log.debug(f"Coinbase REST fallback failed: {e}")
            return None

    # ── Data access ──────────────────────────────────────────────────────

    @property
    def is_live(self) -> bool:
        return time.time() - self._last_tick_ts < self.STALE_THRESHOLD_SEC

    @property
    def current_price(self) -> float:
        """Best available BTC price."""
        ws_price = self.feed.latest_price
        if ws_price and self.is_live:
            return ws_price
        if self._fallback_price:
            now = time.time()
            if now - self._last_stale_log > 30:
                fallback_age = now - self._fallback_price_ts if self._fallback_price_ts else float("inf")
                if fallback_age > self._FALLBACK_STALE_SEC:
                    log.warning(
                        f"Primary WS stale — Coinbase fallback is also old ({fallback_age:.1f}s)"
                    )
                else:
                    log.warning(
                        f"Primary WS stale — using Coinbase REST price (age {fallback_age:.1f}s)"
                    )
                self._last_stale_log = now
            return self._fallback_price
        # Both feeds unavailable — return 0 and let callers skip this cycle
        log.warning("No price data available from any feed — skipping cycle")
        return 0.0

    @property
    def bars(self) -> list[OHLCBar]:
        return self.feed.bars

    @property
    def close_prices(self) -> list[float]:
        return self.feed.close_prices

    def realized_vol(self, lookback: int = 20) -> float:
        return self.feed.realized_vol_annualized(lookback)

    def feed_age_sec(self) -> float:
        """Seconds since last tick received on primary WS."""
        if self._last_tick_ts == 0:
            return float("inf")
        return time.time() - self._last_tick_ts

    def fallback_age_sec(self) -> float:
        """Seconds since last successful Coinbase fallback fetch."""
        if self._fallback_price_ts == 0:
            return float("inf")
        return time.time() - self._fallback_price_ts

    def recent_ticks(self, seconds: float = 60):
        return self.feed.recent_ticks(seconds)

    def on_bar(self, handler):
        self.feed.on_bar(handler)

    def on_tick(self, handler):
        self.feed.on_tick(handler)
