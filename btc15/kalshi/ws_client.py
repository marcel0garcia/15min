"""Kalshi WebSocket client for real-time market data."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Coroutine, Optional

import websockets
from websockets.exceptions import ConnectionClosed

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
        self._token = token
        self._auth_header_factory = auth_header_factory  # callable → dict, for RSA users
        self._ws = None
        self._cmd_id = 0
        self._running = False
        self._handlers: dict[str, list[MessageHandler]] = {ch: [] for ch in self.CHANNELS}
        self._subscribed_tickers: set[str] = set()
        self._reconnect_delay = 2.0

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

        # RSA users auth via HTTP upgrade headers; token users auth via login cmd
        extra_headers = self._auth_header_factory() if self._auth_header_factory else {}

        async with websockets.connect(
            url,
            additional_headers=extra_headers,
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            self._ws = ws
            self._reconnect_delay = 2.0
            log.info("Kalshi WebSocket connected")

            # Token-based login (email/password users only)
            if self._token and not self._auth_header_factory:
                self._cmd_id += 1
                await ws.send(json.dumps({
                    "id": self._cmd_id,
                    "cmd": "login",
                    "params": {"token": self._token},
                }))

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
            for price, size in data.get("no", []):
                if size == 0:
                    ob["yes_asks"].pop(price, None)
                else:
                    ob["yes_asks"][price] = size
            ob["ts"] = time.time()

    async def get_ticker(self, ticker: str) -> Optional[dict]:
        async with self._lock:
            return self._tickers.get(ticker)

    async def get_best_prices(self, ticker: str) -> tuple[Optional[float], Optional[float]]:
        """Returns (best_yes_bid, best_yes_ask) in cents."""
        async with self._lock:
            ob = self._orderbooks.get(ticker, {})
            bids = ob.get("yes_bids", {})
            asks = ob.get("yes_asks", {})
            best_bid = max(bids.keys(), default=None)
            best_ask = min(asks.keys(), default=None)
            t = self._tickers.get(ticker, {})
            return (
                float(best_bid) if best_bid else t.get("yes_bid"),
                float(best_ask) if best_ask else t.get("yes_ask"),
            )

    async def apply_snapshot(self, ticker: str, yes_bid: Optional[float], yes_ask: Optional[float]):
        """
        Overwrite the cached orderbook and ticker with a fresh REST snapshot.
        Called by the periodic refresh loop to repair any WS delta drift.
        """
        async with self._lock:
            self._orderbooks[ticker] = {
                "yes_bids": {yes_bid: 1} if yes_bid is not None else {},
                "yes_asks": {yes_ask: 1} if yes_ask is not None else {},
                "ts": time.time(),
            }
            entry = self._tickers.setdefault(ticker, {})
            if yes_bid is not None:
                entry["yes_bid"] = yes_bid
            if yes_ask is not None:
                entry["yes_ask"] = yes_ask
            entry["ts"] = time.time()
