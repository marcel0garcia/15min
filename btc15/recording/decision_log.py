"""Unified per-scan decision log.

Emits one row per (ticker, scan, action) — including no-action scans. Captures
the same per-component context as personas._log_fire_instrumentation plus
fields needed for the negative-space replay (joining 'what would the trades I
didn't make have realized?').

Engine-side emitter — personas is not touched. Legacy logs/fires.jsonl still
gets the positive-fire rows from personas; this is the additive unified stream.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

log = logging.getLogger(__name__)


# Reason codes — extend as needed. The engine can infer most of these without
# touching personas internals; finer-grained "which gate inside personas
# killed it" is recoverable post-hoc by parsing bot.log SIGNAL/ENTRY/STOP lines
# at the same recv_ts.
REASON_CODES = {
    # No-action negatives (engine can infer all of these)
    "OUTSIDE_ENTRY_WINDOW",     # secs > max_entry_seconds or < min
    "OUTSIDE_PRICE_BAND",       # market price outside entry_price_by_phase
    "ALREADY_HOLDING",          # ticker already in autotrader.positions
    "STOP_COOLDOWN",            # under stop_cooldown_seconds
    "REVERSAL_COOLDOWN",        # under reversal_cooldown_seconds
    "EVALUATED_NO_ACTION",      # personas evaluated but returned [] — finer reason in bot.log
    "AUTO_TRADE_OFF",           # cfg.strategy.auto_trade is false (signal-only mode)
    # Action positives
    "ENTRY_FIRED",              # directional entry posted
    "ENTRY_PYRAMID",            # add-to-winner
    "REVERSAL_EXIT",            # exit on signal flip
    "PROFIT_TAKE",
    "LOSS_CUT",
    "EMERGENCY_STOP",
    "MM_QUOTE_POSTED",
    "MM_CANCEL",
    "ARB_PAIR_FOUND",
    "SETTLEMENT_LOCK_ENTRY",
    "GTC_ESCALATION",
    "MANUAL",
}


def _phase_of(secs: float) -> str:
    """Map secs_remaining to the phase label used by min_confidence_by_phase
    and the audit tooling. Boundaries align with config.entry_price_by_phase."""
    if secs > 540:
        return "early"
    if secs > 300:
        return "mid"
    if secs > 180:
        return "prime"
    return "late"


def _classify_action(action) -> str:
    """Map an Action returned from AutoTrader.evaluate to a reason_code."""
    reason = (getattr(action, "reason", "") or "").lower()
    atype = getattr(action, "action_type", "") or ""

    if "emergency_stop" in reason:
        return "EMERGENCY_STOP"
    if "loss_cut" in reason:
        return "LOSS_CUT"
    if "profit_take" in reason:
        return "PROFIT_TAKE"
    if "reversal" in reason:
        return "REVERSAL_EXIT"
    if "pyramid" in reason:
        return "ENTRY_PYRAMID"
    if "settlement_lock" in reason:
        return "SETTLEMENT_LOCK_ENTRY"
    if "mm_" in reason or "market_make" in reason or "mm post" in reason:
        return "MM_QUOTE_POSTED"
    if "mm cancel" in reason or "mm_cancel" in reason:
        return "MM_CANCEL"
    if "arb" in reason:
        return "ARB_PAIR_FOUND"
    if "escalat" in reason:
        return "GTC_ESCALATION"
    if atype in ("post_only", "ioc", "buy"):
        return "ENTRY_FIRED"
    return "ENTRY_FIRED"


class DecisionLog:
    def __init__(self, recorder, session_label: str, config_hash: str, brain_version: str):
        self.recorder = recorder
        self.session_label = session_label
        self.config_hash = config_hash
        self.brain_version = brain_version

    def emit(
        self,
        *,
        ticker: str,
        secs: float,
        output: Any,                       # ModelOutput or None
        orderbook: dict,
        flow_info: Optional[dict],
        action: Any = None,                # Action or None
        reason_code: str,
        extra: Optional[dict] = None,
    ) -> str:
        if not self.recorder.enabled:
            return ""

        yes_bid = orderbook.get("yes_bid") if orderbook else None
        yes_ask = orderbook.get("yes_ask") if orderbook else None
        kalshi_mid = None
        if yes_bid is not None and yes_ask is not None:
            try:
                yb, ya = float(yes_bid), float(yes_ask)
                if yb > 0 and ya > 0:
                    kalshi_mid = round((yb + ya) / 2, 2)
            except (TypeError, ValueError):
                pass

        decision_id = f"D{uuid.uuid4().hex[:10]}"
        record: dict = {
            "ts": time.time(),
            "decision_id": decision_id,
            "session_label": self.session_label,
            "config_hash": self.config_hash,
            "brain_version": self.brain_version,
            "ticker": ticker,
            "secs_remaining": round(float(secs), 1),
            "phase": _phase_of(secs),
            "reason_code": reason_code,
            # Market context
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "kalshi_mid": kalshi_mid,
        }

        if output is not None:
            record.update({
                "prob_yes": _r(getattr(output, "prob_yes", None), 4),
                "prob_no": _r(getattr(output, "prob_no", None), 4),
                "confidence": _r(getattr(output, "confidence", None), 4),
                "edge_yes": _r(getattr(output, "edge_yes", None), 4),
                "edge_no": _r(getattr(output, "edge_no", None), 4),
                "raw_confidence": _r(getattr(output, "raw_confidence", None), 4),
                "raw_edge_yes": _r(getattr(output, "raw_edge_yes", None), 4),
                "raw_edge_no": _r(getattr(output, "raw_edge_no", None), 4),
                "prob_orderbook": _r(getattr(output, "prob_orderbook", None), 4),
                "prob_technical": _r(getattr(output, "prob_technical", None), 4),
                "prob_trend": _r(getattr(output, "prob_trend", None), 4),
                "prob_binary_options": _r(getattr(output, "prob_binary_options", None), 4),
                "prob_ml": _r(getattr(output, "prob_ml", None), 4),
                "recommended_side": getattr(output, "recommended_side", None),
            })

        if flow_info:
            record["flow_yes_volume"] = flow_info.get("yes_volume")
            record["flow_no_volume"] = flow_info.get("no_volume")

        if action is not None:
            record["action"] = getattr(action, "action_type", None)
            record["side"] = getattr(action, "side", None)
            record["contracts"] = getattr(action, "contracts", None)
            record["price_cents"] = getattr(action, "price_cents", None)
            record["action_reason"] = getattr(action, "reason", None)
        else:
            record["action"] = "none"

        if extra:
            record.update(extra)

        self.recorder.write_decision(record)
        return decision_id


def _r(val, ndigits: int):
    if val is None:
        return None
    try:
        return round(float(val), ndigits)
    except (TypeError, ValueError):
        return None
