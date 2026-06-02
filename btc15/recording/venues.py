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
import binascii
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
# wss://ws.kraken.com/v2 — `book` channel with CRC32-checksum-validated state.
#
# The naive approach (track levels in a dict, apply update deltas) drifts
# when levels fall out of the top-N depth window without an explicit
# qty=0 removal message, producing crossed-book reads. Kraken's protocol
# is built around a CRC32 checksum sent on every update — when our local
# book matches Kraken's, the checksum agrees; on mismatch we drop the
# connection and resubscribe to get a fresh snapshot.
#
# JSON is parsed with parse_float=str so the byte-exact original numeric
# strings survive into the checksum input (the algorithm is sensitive to
# decimal precision — "0.05415" and "0.054150" produce different CRCs).


def _kraken_strip(value_str: str) -> str:
    """Strip decimal point and leading zeros from a numeric string.

    Used to build the input to Kraken v2's CRC32 book checksum.
    '0.05435'        -> '5435'
    '1234.56789012'  -> '123456789012'
    '0.0' or '0'     -> '0'
    """
    if value_str is None:
        return ""
    s = str(value_str).replace(".", "").lstrip("0")
    return s if s else "0"


class KrakenWS(_BaseVenueWS):
    name = "kraken"
    url = "wss://ws.kraken.com/v2"

    def __init__(self, recorder, max_msg_per_sec: int = 0, depth: int = 10):
        super().__init__(recorder, max_msg_per_sec)
        self._depth = depth
        # price_str -> qty_str. Storing original strings is required so the
        # CRC32 verifier sees byte-exact what Kraken hashed on the server.
        self._bids: dict[str, str] = {}
        self._asks: dict[str, str] = {}
        self._needs_resync = False
        self._checksum_mismatches = 0
        self._checksum_passes = 0

    async def _subscribe(self, ws):
        self._bids = {}
        self._asks = {}
        self._needs_resync = False
        await ws.send(json.dumps({
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": ["BTC/USD"],
                "depth": self._depth,
            },
        }))

    async def _connect_and_listen(self) -> None:
        """Override base: parse JSON with parse_float=str (checksum needs
        exact original numeric strings), and drop the connection on checksum
        mismatch so the outer loop reconnects and gets a fresh snapshot."""
        async with websockets.connect(
            self.url, ping_interval=20, ping_timeout=20
        ) as ws:
            self._reconnect_delay = 2.0
            await self._subscribe(ws)
            log.info(f"[REC-{self.name.upper()}] connected (book channel, depth={self._depth})")
            async for raw in ws:
                if not self._running:
                    break
                try:
                    msg = json.loads(raw, parse_float=str)
                except (json.JSONDecodeError, TypeError):
                    continue
                try:
                    await self._handle_message(msg)
                except Exception as e:
                    log.debug(f"[REC-{self.name.upper()}] handler error: {e!r}")
                if self._needs_resync:
                    log.warning(
                        f"[REC-{self.name.upper()}] resyncing — closing connection "
                        f"(mismatches={self._checksum_mismatches}, "
                        f"passes={self._checksum_passes})"
                    )
                    await ws.close()
                    return

    async def _handle_message(self, msg: dict) -> None:
        if msg.get("channel") != "book":
            return
        msg_type = msg.get("type")
        data = msg.get("data")
        if not data or not isinstance(data, list):
            return
        row = data[0]

        if msg_type == "snapshot":
            self._bids.clear()
            self._asks.clear()
            self._apply_levels(row.get("bids") or [], self._bids)
            self._apply_levels(row.get("asks") or [], self._asks)
        elif msg_type == "update":
            self._apply_levels(row.get("bids") or [], self._bids)
            self._apply_levels(row.get("asks") or [], self._asks)
        else:
            return

        # Trim to depth — Kraken's checksum is computed over the top-N
        # levels only, so any deeper levels we accidentally retained are
        # outside the validated scope and should be discarded.
        if len(self._bids) > self._depth:
            self._bids = dict(sorted(self._bids.items(),
                                     key=lambda x: float(x[0]),
                                     reverse=True)[:self._depth])
        if len(self._asks) > self._depth:
            self._asks = dict(sorted(self._asks.items(),
                                     key=lambda x: float(x[0]))[:self._depth])

        # Validate checksum if present in this message.
        expected = row.get("checksum")
        if expected is not None:
            computed = self._compute_checksum()
            try:
                if int(expected) != computed:
                    self._checksum_mismatches += 1
                    self._needs_resync = True
                    return
            except (TypeError, ValueError):
                pass
            self._checksum_passes += 1

        # Emit top-of-book.
        if not self._bids or not self._asks:
            return
        bid_items = sorted(self._bids.items(),
                           key=lambda x: float(x[0]), reverse=True)
        ask_items = sorted(self._asks.items(), key=lambda x: float(x[0]))
        best_bid_str, best_bid_qty = bid_items[0]
        best_ask_str, best_ask_qty = ask_items[0]
        best_bid = float(best_bid_str)
        best_ask = float(best_ask_str)
        if best_bid >= best_ask:
            # Defensive — checksum validation should prevent this. Skip
            # rather than emit a crossed book.
            return
        self._emit(
            best_bid, best_ask,
            _safe_float(best_bid_qty), _safe_float(best_ask_qty),
            exch_ts=row.get("timestamp"),
        )

    def _apply_levels(self, levels: list, target: dict) -> None:
        for lvl in levels:
            p = lvl.get("price")
            q = lvl.get("qty")
            if p is None or q is None:
                continue
            try:
                qf = float(q)
            except (TypeError, ValueError):
                continue
            ps = str(p)
            if qf <= 0:
                target.pop(ps, None)
            else:
                target[ps] = str(q)

    def _compute_checksum(self) -> int:
        """Kraken v2 book CRC32: concat of top-N asks (price asc) then
        top-N bids (price desc), each level contributing
        strip(price) + strip(qty). Then CRC32 of the UTF-8 bytes."""
        ask_items = sorted(self._asks.items(),
                           key=lambda x: float(x[0]))[:self._depth]
        bid_items = sorted(self._bids.items(),
                           key=lambda x: float(x[0]), reverse=True)[:self._depth]
        parts: list[str] = []
        for price_str, qty_str in ask_items:
            parts.append(_kraken_strip(price_str))
            parts.append(_kraken_strip(qty_str))
        for price_str, qty_str in bid_items:
            parts.append(_kraken_strip(price_str))
            parts.append(_kraken_strip(qty_str))
        return binascii.crc32("".join(parts).encode("utf-8"))


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
