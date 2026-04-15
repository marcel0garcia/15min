"""Kalshi WebSocket client for real-time market data."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Coroutine, Optional

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

from btc15.config import KalshiConfig

log = logging.getLogger(__name__)

MessageHandler = Callable[[dict], Coroutine]


class KalshiWebSocket:
    """
    Connects to Kalshi's WebSocket API and dispatches messages to registered handlers.

    Subscriptions:
      - orderbook_delta: real-time orderbook updates
      - ticker: price/volume updates
      - trade: public trade feed
      - fill: our fills (requires auth)
    """

    CHANNELS = ("orderbook_delta", "ticker", "trade", "fill")

    def __init__(self, config: KalshiConfig, token: Optional[str] = None,
                 auth_header_factory=None):
        self.cfg = config
        self._token = token  # kept for reference; not used for WS auth (header-only)
        self._auth_header_factory = auth_header_factory  # callable → dict of RSA headers
        self._ws = None
        self._cmd_id = 0
        self._running = False
        self._handlers: dict[str, list[MessageHandler]] = {ch: [] for ch in self.CHANNELS}
        self._subscribed_tickers: set[str] = set()
        self._reconnect_delay = 2.0
        self._connection_count = 0
        # Optional async callable invoked after every reconnect (not the first connect).
        # Used by the engine to reconcile resting orders that filled during the WS gap.
        self.on_reconnect: Optional[MessageHandler] = None

    def on(self, channel: str, handler: MessageHandler):
        """Register a coroutine handler for a channel."""
        if channel not in self._handlers:
            self._handlers[channel] = []
        self._handlers[channel].append(handler)

    async def subscribe(self, tickers: list[str], channels: Optional[list[str]] = None):
        """Subscribe to channels for given market tickers."""
        channels = channels or ["orderbook_delta", "ticker"]
        self._subscribed_tickers.update(tickers)
        if self._ws:
            await self._send_subscribe(tickers, channels)

    async def _send_subscribe(self, tickers: list[str], channels: list[str]):
        self._cmd_id += 1
        msg = {
            "id": self._cmd_id,
            "cmd": "subscribe",
            "params": {"channels": channels, "market_tickers": tickers},
        }
        await self._ws.send(json.dumps(msg))
        log.debug(f"Subscribed to {channels} for {tickers}")

    async def run(self):
        """Connect and run forever with auto-reconnect."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except InvalidStatus as e:
                body = ""
                try:
                    body = e.response.body.decode("utf-8", errors="replace")
                except Exception:
                    pass
                log.error(
                    f"WebSocket rejected by Kalshi (HTTP {e.response.status_code}): {body!r}. "
                    f"Reconnecting in {self._reconnect_delay}s..."
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 1.5, 30.0)
            except Exception as e:
                log.warning(f"WebSocket disconnected: {e}. Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 1.5, 30.0)
            else:
                self._reconnect_delay = 2.0

    async def stop(self):
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _connect_and_listen(self):
        url = self.cfg.ws_url
        log.info(f"Connecting to Kalshi WebSocket: {url}")

        # Kalshi WS auth is ALWAYS header-based during the HTTP upgrade handshake.
        # There is no post-connection login command. RSA key required.
        if not self._auth_header_factory:
            raise RuntimeError(
                "Kalshi WebSocket requires RSA key authentication. "
                "Set KALSHI_API_KEY and KALSHI_RSA_KEY_PATH in your .env file. "
                "Email/password auth is not supported for WebSocket connections."
            )
        extra_headers = self._auth_header_factory()
        if not extra_headers.get("KALSHI-ACCESS-KEY"):
            raise RuntimeError(
                "RSA auth factory returned empty headers — KALSHI_API_KEY or "
                "KALSHI_RSA_KEY_PATH is missing or the key failed to load. "
                "Check your .env file."
            )

        async with websockets.connect(
            url,
            additional_headers=extra_headers,
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            self._ws = ws
            self._reconnect_delay = 2.0
            self._connection_count += 1
            log.info(f"Kalshi WebSocket connected (connection #{self._connection_count})")

            # On reconnects, notify the engine so it can reconcile resting orders
            # that may have filled during the disconnect window.
            if self._connection_count > 1 and self.on_reconnect:
                asyncio.create_task(self.on_reconnect())

            # Re-subscribe to any previously subscribed tickers
            if self._subscribed_tickers:
                await self._send_subscribe(
                    list(self._subscribed_tickers),
                    ["orderbook_delta", "ticker", "fill"],
                )

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    await self._dispatch(msg)
                except json.JSONDecodeError as e:
                    log.warning(f"WS JSON error: {e}")
                except Exception as e:
                    log.error(f"WS dispatch error: {e}", exc_info=True)

    async def _dispatch(self, msg: dict):
        msg_type = msg.get("type", "")
        channel = msg.get("channel", "")

        # Subscription confirmations
        if msg_type in ("subscribed", "error"):
            log.debug(f"WS: {msg}")
            return

        # Route to handlers
        target = channel or msg_type
        handlers = self._handlers.get(target, [])
        for handler in handlers:
            try:
                await handler(msg)
            except Exception as e:
                log.error(f"WS handler error in '{target}': {e}", exc_info=True)


class MarketDataCache:
    """
    Thread-safe cache of latest orderbook/ticker data from WebSocket.
    Handlers write here; strategy reads from here.
    """

    def __init__(self):
        self._tickers: dict[str, dict] = {}      # ticker → {yes_bid, yes_ask, volume, ...}
        self._orderbooks: dict[str, dict] = {}   # ticker → {yes_bids: [...], yes_asks: [...]}
        self._lock = asyncio.Lock()

    async def handle_ticker(self, msg: dict):
        data = msg.get("msg", {})
        ticker = data.get("market_ticker")
        if not ticker:
            return
        async with self._lock:
            self._tickers[ticker] = {
                "yes_bid": data.get("yes_bid"),
                "yes_ask": data.get("yes_ask"),
                "no_bid": data.get("no_bid"),
                "no_ask": data.get("no_ask"),
                "last_price": data.get("last_price"),
                "volume": data.get("volume"),
                "ts": time.time(),
            }

    async def handle_orderbook_delta(self, msg: dict):
        data = msg.get("msg", {})
        ticker = data.get("market_ticker")
        if not ticker:
            return
        async with self._lock:
            if ticker not in self._orderbooks:
                self._orderbooks[ticker] = {"yes_bids": {}, "yes_asks": {}}
            ob = self._orderbooks[ticker]
            # Apply delta updates: price → size (0 = remove)
            for price, size in data.get("yes", []):
                if size == 0:
                    ob["yes_bids"].pop(price, None)
                else:
                    ob["yes_bids"][price] = size
            # NO bid levels: YES ask = 100 - no_bid_price
            for price, size in data.get("no", []):
                yes_ask_price = 100 - price
                if size == 0:
                    ob["yes_asks"].pop(yes_ask_price, None)
                else:
                    ob["yes_asks"][yes_ask_price] = size
            ob["ts"] = time.time()

    async def get_ticker(self, ticker: str) -> Optional[dict]:
        async with self._lock:
            return self._tickers.get(ticker)

    async def get_best_prices(self, ticker: str) -> tuple[Optional[float], Optional[float]]:
        """Returns (best_yes_bid, best_yes_ask) in cents from the WS orderbook cache.
        Falls back to ticker-channel snapshot when orderbook has no levels.
        Returns (None, None) when no data is cached yet."""
        async with self._lock:
            ob = self._orderbooks.get(ticker, {})
            bids = ob.get("yes_bids", {})
            asks = ob.get("yes_asks", {})
            best_bid = max(bids.keys(), default=None)
            best_ask = min(asks.keys(), default=None)
            t = self._tickers.get(ticker, {})
            # Only fall back to ticker snapshot if the orderbook has no levels.
            # Never coerce None → 0 here; callers decide how to handle missing data.
            return (
                float(best_bid) if best_bid is not None else t.get("yes_bid"),
                float(best_ask) if best_ask is not None else t.get("yes_ask"),
            )

    async def get_cache_age(self, ticker: str) -> Optional[float]:
        """Returns seconds since the last WS update for ticker, or None if never seen."""
        async with self._lock:
            ob_ts = self._orderbooks.get(ticker, {}).get("ts")
            tk_ts = self._tickers.get(ticker, {}).get("ts")
            latest = max(t for t in (ob_ts, tk_ts) if t is not None) if (ob_ts or tk_ts) else None
            return (time.time() - latest) if latest is not None else None

    async def apply_snapshot(self, ticker: str, orderbook) -> None:
        """
        Overwrite the cached orderbook with a full REST snapshot (Orderbook object).
        Preserves all depth levels — avoids the previous bug where the snapshot
        replaced a multi-level book with a single price at size=1.
        Called by the periodic refresh loop to repair any WS delta drift.
        """
        async with self._lock:
            self._orderbooks[ticker] = {
                "yes_bids": {p: s for p, s in orderbook.yes_bids},
                "yes_asks": {p: s for p, s in orderbook.yes_asks},
                "ts": time.time(),
            }
            entry = self._tickers.setdefault(ticker, {})
            best_bid = orderbook.best_yes_bid
            best_ask = orderbook.best_yes_ask
            if best_bid is not None:
                entry["yes_bid"] = best_bid
            if best_ask is not None:
                entry["yes_ask"] = best_ask
            entry["ts"] = time.time()

    async def get_orderbook_depth(self, ticker: str) -> tuple[float, float]:
        """
        Returns (total_yes_bid_volume, total_yes_ask_volume) in contracts.
        Used by the ensemble model for orderbook imbalance signal.
        Returns (0, 0) when no data is cached yet.
        """
        async with self._lock:
            ob = self._orderbooks.get(ticker, {})
            bid_vol = float(sum(ob.get("yes_bids", {}).values()))
            ask_vol = float(sum(ob.get("yes_asks", {}).values()))
            return bid_vol, ask_vol
