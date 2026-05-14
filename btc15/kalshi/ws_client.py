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

    CHANNELS = ("orderbook_delta", "orderbook_snapshot", "ticker", "trade", "fill")

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
        try:
            await self._ws.send(json.dumps(msg))
        except ConnectionClosed:
            # Connection dropped mid-send; the reconnect path in _connect_and_listen
            # will resubscribe to self._subscribed_tickers when it reconnects.
            log.debug(f"Skipped subscribe for {tickers} — WS reconnecting")
            return
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

        try:
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
        finally:
            # Clear the stale reference so subscribe() doesn't try to send on a
            # closed socket between disconnect and the next reconnect.
            self._ws = None

    async def _dispatch(self, msg: dict):
        msg_type = msg.get("type", "")
        channel = msg.get("channel", "")

        # Subscription confirmations
        if msg_type in ("subscribed", "error"):
            log.debug(f"WS: {msg}")
            return

        # Route to handlers. Prefer msg_type when it's a known channel
        # (Kalshi sends type="orderbook_snapshot" with channel="orderbook_delta"
        # for the initial snapshot — we need the snapshot handler, not the delta one).
        if msg_type in self._handlers:
            target = msg_type
        else:
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
        # Injected by the engine: fire-and-forget REST refresh when cache looks stale.
        # Signature: async def refresh(ticker: str) -> None
        self.rest_refresh: Optional[Callable[[str], Coroutine]] = None
        # Staleness threshold in seconds — reads older than this trigger an
        # out-of-band REST refresh on the next call.
        self.staleness_sec: float = 3.0
        # Per-ticker cooldown so we don't spam REST when a ticker has gone idle.
        self._refresh_inflight: dict[str, float] = {}
        # Per-sid expected next sequence number. Populated by snapshots, verified
        # on every delta. A mismatch means we dropped a message → trigger resync.
        self._seq_expected: dict[int, int] = {}

    async def handle_ticker(self, msg: dict):
        data = msg.get("msg", {})
        ticker = data.get("market_ticker")
        if not ticker:
            return

        def _to_cents(dollars_val, cents_val):
            """Prefer *_dollars string; fall back to legacy integer cents field."""
            if dollars_val is not None:
                try:
                    return float(dollars_val) * 100
                except (TypeError, ValueError):
                    pass
            if cents_val is not None:
                try:
                    v = float(cents_val)
                    # Normalize: API may return 0.47 (dollars) or 47 (cents)
                    return v * 100 if v <= 1.0 else v
                except (TypeError, ValueError):
                    pass
            return None

        async with self._lock:
            self._tickers[ticker] = {
                "yes_bid": _to_cents(data.get("yes_bid_dollars"), data.get("yes_bid")),
                "yes_ask": _to_cents(data.get("yes_ask_dollars"), data.get("yes_ask")),
                "no_bid":  _to_cents(data.get("no_bid_dollars"),  data.get("no_bid")),
                "no_ask":  _to_cents(data.get("no_ask_dollars"),  data.get("no_ask")),
                "last_price": _to_cents(data.get("price_dollars"), data.get("last_price")),
                "volume": data.get("volume_fp") or data.get("volume"),
                "ts": time.time(),
            }

    async def handle_orderbook_snapshot(self, msg: dict):
        """
        Full orderbook snapshot from WS (sent on subscribe and occasionally after
        gaps). Unlike a delta, this message is authoritative — clear any prior
        state for the ticker so stale levels don't linger.

        Schema (Kalshi v2, current as of 2026):
          msg.yes_dollars_fp: list[[price_dollars_str, size_fp_str]]
          msg.no_dollars_fp:  list[[price_dollars_str, size_fp_str]]
        """
        sid = msg.get("sid")
        seq = msg.get("seq")
        data = msg.get("msg", {})
        ticker = data.get("market_ticker")
        if not ticker:
            return
        async with self._lock:
            self._orderbooks[ticker] = {"yes_bids": {}, "yes_asks": {}}
            ob = self._orderbooks[ticker]
            for price_dollars, size_fp in data.get("yes_dollars_fp", []):
                try:
                    price_cents = int(round(float(price_dollars) * 100))
                    size = float(size_fp)
                except (TypeError, ValueError):
                    continue
                if size > 0:
                    ob["yes_bids"][price_cents] = size
            for price_dollars, size_fp in data.get("no_dollars_fp", []):
                try:
                    price_cents = int(round(float(price_dollars) * 100))
                    size = float(size_fp)
                except (TypeError, ValueError):
                    continue
                if size > 0:
                    ob["yes_asks"][100 - price_cents] = size
            ob["ts"] = time.time()
        if sid is not None and seq is not None:
            # Snapshot is authoritative — reset seq tracking.
            try:
                self._seq_expected[int(sid)] = int(seq) + 1
            except (TypeError, ValueError):
                pass

    async def handle_orderbook_delta(self, msg: dict):
        """
        Single-level signed delta (Kalshi v2 current schema):
          msg.price_dollars: "0.960"   (string)
          msg.delta_fp:      "-54.00"  (signed fixed-point string)
          msg.side:          "yes" | "no"
        A gap in `seq` per `sid` means we dropped a message — trigger a REST
        refresh so the cache re-converges on truth.
        """
        sid = msg.get("sid")
        seq = msg.get("seq")
        data = msg.get("msg", {})
        ticker = data.get("market_ticker")
        if not ticker:
            return

        gap_detected = False
        if sid is not None and seq is not None:
            try:
                sid_i, seq_i = int(sid), int(seq)
                expected = self._seq_expected.get(sid_i)
                if expected is not None and seq_i != expected:
                    gap_detected = True
                    log.warning(
                        f"[SEQ GAP] {ticker} sid={sid_i} expected={expected} got={seq_i}"
                    )
                self._seq_expected[sid_i] = seq_i + 1
            except (TypeError, ValueError):
                pass

        price_dollars = data.get("price_dollars")
        delta_fp = data.get("delta_fp")
        side = data.get("side")
        if price_dollars is None or delta_fp is None or side not in ("yes", "no"):
            return
        try:
            price_cents = int(round(float(price_dollars) * 100))
            delta = float(delta_fp)
        except (TypeError, ValueError):
            return

        async with self._lock:
            if ticker not in self._orderbooks:
                self._orderbooks[ticker] = {"yes_bids": {}, "yes_asks": {}}
            ob = self._orderbooks[ticker]
            if side == "yes":
                book = ob["yes_bids"]
                key = price_cents
            else:
                book = ob["yes_asks"]
                key = 100 - price_cents
            new_size = book.get(key, 0.0) + delta
            if new_size <= 0:
                book.pop(key, None)
            else:
                book[key] = new_size
            ob["ts"] = time.time()

        if gap_detected and self.rest_refresh is not None:
            asyncio.create_task(self.rest_refresh(ticker))

    async def get_ticker(self, ticker: str) -> Optional[dict]:
        async with self._lock:
            return self._tickers.get(ticker)

    async def get_best_prices(self, ticker: str) -> tuple[Optional[float], Optional[float]]:
        """Returns (best_yes_bid, best_yes_ask) in cents.
        Prefers whichever source (orderbook vs ticker snapshot) was updated most
        recently. When both caches are older than `staleness_sec`, fires an
        out-of-band REST refresh (non-blocking) so the next call is fresh.
        Returns (None, None) when no data is cached yet."""
        async with self._lock:
            ob = self._orderbooks.get(ticker, {})
            bids = ob.get("yes_bids", {})
            asks = ob.get("yes_asks", {})
            ob_ts = ob.get("ts")
            best_bid = max(bids.keys(), default=None)
            best_ask = min(asks.keys(), default=None)
            t = self._tickers.get(ticker, {})
            t_ts = t.get("ts")

            # Pick the fresher source when both have data.
            ob_fresh = (ob_ts is not None) and (best_bid is not None or best_ask is not None)
            tk_fresh = (t_ts is not None) and (t.get("yes_bid") is not None or t.get("yes_ask") is not None)
            if ob_fresh and tk_fresh:
                if (t_ts or 0) > (ob_ts or 0):
                    bid = t.get("yes_bid") if t.get("yes_bid") is not None else (float(best_bid) if best_bid is not None else None)
                    ask = t.get("yes_ask") if t.get("yes_ask") is not None else (float(best_ask) if best_ask is not None else None)
                else:
                    bid = float(best_bid) if best_bid is not None else t.get("yes_bid")
                    ask = float(best_ask) if best_ask is not None else t.get("yes_ask")
            elif ob_fresh:
                bid = float(best_bid) if best_bid is not None else None
                ask = float(best_ask) if best_ask is not None else None
            elif tk_fresh:
                bid = t.get("yes_bid")
                ask = t.get("yes_ask")
            else:
                bid = None
                ask = None

            latest_ts = max((ts for ts in (ob_ts, t_ts) if ts is not None), default=None)

        # Out-of-band staleness check (outside the lock to avoid holding it across log/IO).
        if latest_ts is not None and self.rest_refresh is not None:
            age = time.time() - latest_ts
            if age > self.staleness_sec:
                last = self._refresh_inflight.get(ticker, 0.0)
                # Debounce: one refresh per ticker per staleness window
                if time.time() - last > self.staleness_sec:
                    self._refresh_inflight[ticker] = time.time()
                    log.warning(f"[STALE CACHE] {ticker} age={age:.1f}s — triggering REST refresh")
                    asyncio.create_task(self.rest_refresh(ticker))

        return bid, ask

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
