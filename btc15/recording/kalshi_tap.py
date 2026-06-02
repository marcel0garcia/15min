"""Kalshi raw-frame tap.

Attaches as an additional handler on the engine's KalshiWebSocket and writes
every orderbook_snapshot, orderbook_delta, and trade message verbatim to the
session's kalshi_frames.jsonl with a recv_ts stamp. Read-only — never sends
messages, never affects other handlers' dispatch.

We deliberately do NOT maintain a live normalized book here; raw frames are
the source of truth and replay reconstructs the book offline. The `ticker`
channel is skipped (derived from deltas) and `fill` is skipped (already
captured by the engine via _handle_fill → trades.csv).
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


class KalshiRawTap:
    def __init__(self, recorder):
        self.recorder = recorder

    def attach(self, ws) -> None:
        if not self.recorder.enabled:
            return
        ws.on("orderbook_snapshot", self._on_snapshot)
        ws.on("orderbook_delta", self._on_delta)
        ws.on("trade", self._on_trade)
        log.info("[REC-KAL] Attached to KalshiWebSocket (snapshot/delta/trade)")

    async def _on_snapshot(self, msg: dict) -> None:
        self.recorder.write_kalshi({
            "recv_ts": time.time(),
            "kind": "snapshot",
            "raw": msg,
        })

    async def _on_delta(self, msg: dict) -> None:
        self.recorder.write_kalshi({
            "recv_ts": time.time(),
            "kind": "delta",
            "raw": msg,
        })

    async def _on_trade(self, msg: dict) -> None:
        self.recorder.write_kalshi({
            "recv_ts": time.time(),
            "kind": "trade",
            "raw": msg,
        })
