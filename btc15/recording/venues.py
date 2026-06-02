"""BTC venue top-of-book WS connectors for BRTI reconstruction.

Three read-only connectors (Coinbase, Kraken, Bitstamp) — BRTI's most-liquid
constituents. Each is a standalone asyncio task with its own reconnect loop;
they write `{venue, bid, ask, bid_sz, ask_sz, recv_ts, exch_ts}` rows into the
session's venue_ticks.jsonl.

Phase 1 records top-of-book only. Per-venue full L2 firehose is out of scope
unless a measured reconstruction-error blow-out in Phase 2 demands it.

No engine integration — the bot does not consume venue feeds yet. Phase 3
swaps the ensemble brain to consume the consolidated mid built from these.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed

log = logging.getLogger(__name__)


class _BaseVenueWS:
    name: str = ""
    url: str = ""

    def __init__(self, recorder, max_msg_per_sec: int = 0):
        self.recorder = recorder
        self.max_msg_per_sec = max_msg_per_sec
        self._running = False
        self._last_emit_ts = 0.0
        self._reconnect_delay = 2.0
        self._msg_count = 0

    async def _subscribe(self, ws):
        raise NotImplementedError

    async def _handle_message(self, msg: dict) -> None:
        raise NotImplementedError

    async def run(self) -> None:
        self._running = True
        log.info(f"[REC-{self.name.upper()}] starting venue tap")
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(
                    f"[REC-{self.name.upper()}] disconnected: {e!r} — "
                    f"reconnect in {self._reconnect_delay:.0f}s"
                )
                try:
                    await asyncio.sleep(self._reconnect_delay)
                except asyncio.CancelledError:
                    break
                self._reconnect_delay = min(self._reconnect_delay * 1.5, 30.0)
            else:
                self._reconnect_delay = 2.0

    async def stop(self) -> None:
        self._running = False

    async def _connect_and_listen(self) -> None:
        async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:
            self._reconnect_delay = 2.0
            await self._subscribe(ws)
            log.info(f"[REC-{self.name.upper()}] connected")
            async for raw in ws:
                if not self._running:
                    break
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                try:
                    await self._handle_message(msg)
                except Exception as e:
                    log.debug(f"[REC-{self.name.upper()}] handler error: {e!r}")

    def _emit(
        self,
        bid: Optional[float],
        ask: Optional[float],
        bid_sz: Optional[float] = None,
        ask_sz: Optional[float] = None,
        exch_ts: Optional[str] = None,
    ) -> None:
        if bid is None or ask is None:
            return
        if self.max_msg_per_sec > 0:
            now = time.time()
            min_interval = 1.0 / self.max_msg_per_sec
            if now - self._last_emit_ts < min_interval:
                return
            self._last_emit_ts = now
        self._msg_count += 1
        self.recorder.write_venue({
            "recv_ts": time.time(),
            "venue": self.name,
            "bid": bid,
            "ask": ask,
            "bid_sz": bid_sz,
            "ask_sz": ask_sz,
            "exch_ts": exch_ts,
        })


# ── Coinbase Advanced Trade ──────────────────────────────────────────────────
# wss://ws-feed.exchange.coinbase.com — `ticker` channel emits best_bid/ask
# on every match. No auth required for public market data.
class CoinbaseWS(_BaseVenueWS):
    name = "coinbase"
    url = "wss://ws-feed.exchange.coinbase.com"

    async def _subscribe(self, ws):
        await ws.send(json.dumps({
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channels": ["ticker"],
        }))

    async def _handle_message(self, msg: dict) -> None:
        if msg.get("type") != "ticker":
            return
        try:
            bid = float(msg["best_bid"])
            ask = float(msg["best_ask"])
        except (KeyError, ValueError, TypeError):
            return
        bid_sz = _safe_float(msg.get("best_bid_size"))
        ask_sz = _safe_float(msg.get("best_ask_size"))
        self._emit(bid, ask, bid_sz, ask_sz, exch_ts=msg.get("time"))


# ── Kraken v2 ────────────────────────────────────────────────────────────────
# wss://ws.kraken.com/v2 — `book` channel emits a snapshot then delta updates.
# An update's qty=0 removes that price level; otherwise the level is set to qty.
# We maintain a small per-side dict so we can emit the post-update top-of-book.
# Chosen over `ticker` (which only fires on quote change) so BRTI reconstruction
# stays responsive when BTC is calm.
class KrakenWS(_BaseVenueWS):
    name = "kraken"
    url = "wss://ws.kraken.com/v2"

    def __init__(self, recorder, max_msg_per_sec: int = 0, depth: int = 10):
        super().__init__(recorder, max_msg_per_sec)
        self._depth = depth
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}

    async def _subscribe(self, ws):
        await ws.send(json.dumps({
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": ["BTC/USD"],
                "depth": self._depth,
            },
        }))

    async def _handle_message(self, msg: dict) -> None:
        if msg.get("channel") != "book":
            return
        msg_type = msg.get("type")
        data = msg.get("data")
        if not data or not isinstance(data, list):
            return
        row = data[0]
        if msg_type == "snapshot":
            self._bids = {}
            self._asks = {}
            for lvl in row.get("bids") or []:
                p = _safe_float(lvl.get("price"))
                q = _safe_float(lvl.get("qty"))
                if p is not None and q is not None and q > 0:
                    self._bids[p] = q
            for lvl in row.get("asks") or []:
                p = _safe_float(lvl.get("price"))
                q = _safe_float(lvl.get("qty"))
                if p is not None and q is not None and q > 0:
                    self._asks[p] = q
        elif msg_type == "update":
            for lvl in row.get("bids") or []:
                p = _safe_float(lvl.get("price"))
                q = _safe_float(lvl.get("qty"))
                if p is None or q is None:
                    continue
                if q <= 0:
                    self._bids.pop(p, None)
                else:
                    self._bids[p] = q
            for lvl in row.get("asks") or []:
                p = _safe_float(lvl.get("price"))
                q = _safe_float(lvl.get("qty"))
                if p is None or q is None:
                    continue
                if q <= 0:
                    self._asks.pop(p, None)
                else:
                    self._asks[p] = q
        else:
            return

        if not self._bids or not self._asks:
            return
        best_bid = max(self._bids.keys())
        best_ask = min(self._asks.keys())
        self._emit(
            best_bid, best_ask,
            self._bids[best_bid], self._asks[best_ask],
            exch_ts=row.get("timestamp"),
        )


# ── Bitstamp v2 ──────────────────────────────────────────────────────────────
# wss://ws.bitstamp.net — `order_book_btcusd` channel emits full 100-deep
# book on every change. We only read top-of-book.
class BitstampWS(_BaseVenueWS):
    name = "bitstamp"
    url = "wss://ws.bitstamp.net"

    async def _subscribe(self, ws):
        await ws.send(json.dumps({
            "event": "bts:subscribe",
            "data": {"channel": "order_book_btcusd"},
        }))

    async def _handle_message(self, msg: dict) -> None:
        if msg.get("event") != "data":
            return
        data = msg.get("data") or {}
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        if not bids or not asks:
            return
        try:
            bid = float(bids[0][0])
            ask = float(asks[0][0])
            bid_sz = float(bids[0][1])
            ask_sz = float(asks[0][1])
        except (IndexError, ValueError, TypeError):
            return
        self._emit(bid, ask, bid_sz, ask_sz, exch_ts=data.get("microtimestamp") or data.get("timestamp"))


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── Driver: build the enabled set per config and run them ────────────────────

def build_venue_tasks(cfg, recorder) -> list:
    """Return a list of `(coroutine, ws_instance)` tuples for each enabled
    venue. Caller is responsible for asyncio.create_task() and lifecycle."""
    out = []
    rate = cfg.recording.venue_max_msg_per_sec
    if cfg.recording.venue_coinbase:
        ws = CoinbaseWS(recorder, max_msg_per_sec=rate)
        out.append(("venue-coinbase", ws))
    if cfg.recording.venue_kraken:
        ws = KrakenWS(recorder, max_msg_per_sec=rate)
        out.append(("venue-kraken", ws))
    if cfg.recording.venue_bitstamp:
        ws = BitstampWS(recorder, max_msg_per_sec=rate)
        out.append(("venue-bitstamp", ws))
    return out
