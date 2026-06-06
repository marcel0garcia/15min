"""Phase 3 step 5 diagnostic: trace why each brain's signals get filtered
by the personas entry gates.

For every decision row in a session, replays the production entry-gate
chain against each brain's prob/conf/edge and records WHICH gate (if any)
killed it. Aggregates per-brain so you can see:

  - How many opportunities did each brain see?
  - How many would have fired?
  - For the rejections, which gate caught them?

Gates replicated (from personas._check_directional_entry):
  1. Time window:     min_secs <= secs <= max_secs
  2. Price band:      kalshi_mid in entry_price_by_phase[phase]
  3. Confidence:      conf >= min_confidence_by_phase[phase]
  4. Recommended:     prob_yes != 0.5 (has a directional opinion)
  5. Edge floor:      edge_side >= min_edge
  6. Tier-1 suppress: edge > 0.25 AND conf < 0.52
  7. Tier-2 suppress: edge >= 0.35 AND conf < 0.65

A row is counted against the FIRST gate it fails. Rows that pass all
gates are "would-fire" candidates.

Doesn't replicate every personas knob (K-tick confirmation, conf-fade,
EWMA raw-floor, MM-resting suppression, cooldowns) — those depend on
state across scans and would require full replay. This focuses on the
per-row gates, which is where DIR vs FV calibration most matters.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


GATE_ORDER = [
    "outside_time_window",
    "outside_price_band",
    "below_min_confidence",
    "no_recommended_side",
    "below_min_edge",
    "entry_suppressed_tier1",
    "entry_suppressed_tier2",
    "WOULD_FIRE",
]


DEFAULT_MIN_SECS = 60
DEFAULT_MAX_SECS = 870
DEFAULT_MIN_EDGE = 0.10
DEFAULT_MIN_CONFIDENCE_BY_PHASE = {
    "early": 0.55,
    "mid": 0.48,
    "prime": 0.50,
    "late": 0.55,
}
DEFAULT_ENTRY_PRICE_BY_PHASE = {
    "early": (10, 60),
    "mid": (35, 80),
    "prime": (20, 85),
    "late": (10, 95),
}


def _iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _phase_of(secs: float) -> str:
    if secs > 540:
        return "early"
    if secs > 300:
        return "mid"
    if secs > 180:
        return "prime"
    return "late"


@dataclass
class GateTrace:
    brain: str
    n_rows_evaluated: int = 0
    n_skipped_no_prob: int = 0
    by_gate: dict = field(default_factory=lambda: defaultdict(int))
    would_fire_examples: list = field(default_factory=list)  # first 10 rows that pass

    @property
    def n_would_fire(self) -> int:
        return self.by_gate.get("WOULD_FIRE", 0)


def _trace_row(
    row: dict, brain: str,
    min_secs: int, max_secs: int, min_edge: float,
    min_conf_by_phase: dict, entry_price_by_phase: dict,
) -> Optional[str]:
    """Return the name of the first gate this row fails for `brain`, or
    "WOULD_FIRE" if it clears them all. Returns None if the row lacks the
    fields needed to evaluate this brain (e.g. FV degenerate)."""
    if brain == "fv":
        if row.get("fv_degenerate"):
            return None
        prob_yes = row.get("fv_prob_yes")
        conf = row.get("fv_confidence")
    else:
        prob_yes = row.get("prob_yes")
        conf = row.get("confidence")
    if prob_yes is None or conf is None:
        return None

    secs = row.get("secs_remaining")
    if secs is None:
        return None

    if secs < min_secs or secs > max_secs:
        return "outside_time_window"

    yes_bid = row.get("yes_bid")
    yes_ask = row.get("yes_ask")
    if yes_bid is None or yes_ask is None:
        return "outside_price_band"

    # The personas gate checks the SIDE-SPECIFIC entry price (raw_price),
    # not the YES mid. For YES side we'd pay yes_ask; for NO we'd pay
    # (100 - yes_bid). This is the gate that was silently killing every
    # FV NO-side entry on cheap-YES markets (cheap YES = expensive NO).
    if prob_yes >= 0.5:
        entry_price = float(yes_ask)
    else:
        entry_price = 100.0 - float(yes_bid)

    phase = _phase_of(secs)
    ph_min, ph_max = entry_price_by_phase.get(phase, (10, 95))
    if not (ph_min <= entry_price <= ph_max):
        return "outside_price_band"

    min_conf = min_conf_by_phase.get(phase, 0.5)
    if conf < min_conf:
        return "below_min_confidence"

    if prob_yes == 0.5:
        return "no_recommended_side"

    # Edge for the recommended side, computed against the Kalshi mid the
    # same way the ensemble's edge_yes/no is computed.
    if yes_bid is None or yes_ask is None:
        return "below_min_edge"  # can't compute edge -> treat as failing
    try:
        mid_frac = (float(yes_bid) + float(yes_ask)) / 200.0
    except (TypeError, ValueError):
        return "below_min_edge"

    if prob_yes > 0.5:
        edge = prob_yes - mid_frac
    else:
        edge = (1.0 - prob_yes) - (1.0 - mid_frac)  # = mid_frac - prob_yes

    if edge < min_edge:
        return "below_min_edge"

    # ENTRY SUPPRESSED tiers (the calibration-mismatch traps for FV).
    if edge > 0.25 and conf < 0.52:
        return "entry_suppressed_tier1"
    if edge >= 0.35 and conf < 0.65:
        return "entry_suppressed_tier2"

    return "WOULD_FIRE"


def trace_session(
    session_dir: Path,
    *,
    min_secs: int = DEFAULT_MIN_SECS,
    max_secs: int = DEFAULT_MAX_SECS,
    min_edge: float = DEFAULT_MIN_EDGE,
    min_conf_by_phase: dict = None,
    entry_price_by_phase: dict = None,
) -> tuple[GateTrace, GateTrace, dict]:
    """Walk decisions.jsonl; trace each row against the gates for both
    brains. Returns (DIR trace, FV trace, action_counts)."""
    if min_conf_by_phase is None:
        min_conf_by_phase = DEFAULT_MIN_CONFIDENCE_BY_PHASE
    if entry_price_by_phase is None:
        entry_price_by_phase = DEFAULT_ENTRY_PRICE_BY_PHASE

    dir_trace = GateTrace(brain="DIR")
    fv_trace = GateTrace(brain="FV")
    action_counts: dict[str, int] = defaultdict(int)

    for row in _iter_jsonl(session_dir / "decisions.jsonl"):
        action_counts[row.get("action") or "none"] += 1

        for brain, trace in (("dir", dir_trace), ("fv", fv_trace)):
            gate = _trace_row(
                row, brain,
                min_secs=min_secs, max_secs=max_secs, min_edge=min_edge,
                min_conf_by_phase=min_conf_by_phase,
                entry_price_by_phase=entry_price_by_phase,
            )
            if gate is None:
                trace.n_skipped_no_prob += 1
                continue
            trace.n_rows_evaluated += 1
            trace.by_gate[gate] += 1
            if gate == "WOULD_FIRE" and len(trace.would_fire_examples) < 10:
                trace.would_fire_examples.append({
                    "ticker": row.get("ticker"),
                    "secs": row.get("secs_remaining"),
                    "prob_yes": row.get("fv_prob_yes" if brain == "fv" else "prob_yes"),
                    "conf": row.get("fv_confidence" if brain == "fv" else "confidence"),
                    "kalshi_mid": row.get("kalshi_mid"),
                })

    return dir_trace, fv_trace, dict(action_counts)
