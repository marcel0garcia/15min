"""Phase 3 step 4.5: counterfactual P&L analyzer.

Brier validates calibration. P&L validates execution outcome. Both should
agree, but P&L is the number that funds the account.

This module simulates what each brain would have made on a recorded
session, assuming **hold-to-settlement** for every fired entry. That's an
upper bound — it ignores DIR's actual stop-loss + profit-take exit logic
— but it lets us compare brains apples-to-apples: just signals, no exit
policy noise.

Three numbers come out:
  - DIR actual:           realized P&L from logs/trades.csv (real fills + exits)
  - DIR hold-to-settle:   simulated P&L assuming DIR's entries held to close
  - FV  hold-to-settle:   simulated P&L assuming FV's entries held to close

Comparing DIR-actual to DIR-htps reveals what DIR's exit logic cost or
saved. Comparing FV-htps to DIR-htps reveals which brain's *signals* are
more profitable, independent of exit policy.

Simulated entry gates (intentionally minimal — matches the high-frequency
production gates without trying to replicate every personas.py
suppression tier):
  - secs_remaining in [min_secs, max_secs]
  - confidence ≥ min_confidence (per-phase)
  - edge (prob_yes - market_implied_yes) on the directional side ≥ min_edge
  - Kalshi mid price in entry_price_by_phase band
  - One entry per (ticker, side); no pyramiding sim

Kalshi mechanics (cents):
  Buy YES at yes_ask → settles YES: P&L = +(100 - yes_ask) per contract
                       settles NO:  P&L = -yes_ask per contract
  Buy NO  at (100 - yes_bid) → settles NO:  P&L = +yes_bid per contract
                               settles YES: P&L = -(100 - yes_bid) per contract
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


# Default gates roughly matching the current production config.yaml. The
# simulator caller can override.
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
DEFAULT_CONTRACTS_PER_TRADE = 1


def _phase_of(secs: float) -> str:
    if secs > 540:
        return "early"
    if secs > 300:
        return "mid"
    if secs > 180:
        return "prime"
    return "late"


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


def _load_results_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    out = {}
    for ticker, rec in raw.items():
        if isinstance(rec, dict) and rec.get("status") == "finalized":
            r = rec.get("result")
            if r in ("yes", "no"):
                out[ticker] = r
    return out


# ── Single-row simulator ─────────────────────────────────────────────────────

@dataclass
class SimulatedTrade:
    ticker: str
    side: str               # "yes" or "no"
    entry_price_cents: int  # what we paid to enter
    contracts: int
    secs_at_entry: float
    phase: str
    prob_yes: float
    confidence: float
    edge: float
    pnl_cents: int          # realized assuming hold-to-settlement
    outcome: str            # "yes"/"no" from settlement


def _simulate_entry_from_row(
    row: dict,
    prob_yes_field: str,
    confidence_field: str,
    *,
    results: dict[str, str],
    min_secs: int,
    max_secs: int,
    min_edge: float,
    min_confidence_by_phase: dict,
    entry_price_by_phase: dict,
    contracts: int,
) -> Optional[SimulatedTrade]:
    """Decide whether the brain would have entered on this row; if yes,
    compute the hold-to-settlement P&L."""
    prob_yes = row.get(prob_yes_field)
    conf = row.get(confidence_field)
    if prob_yes is None or conf is None:
        return None
    if row.get("fv_degenerate") and prob_yes_field == "fv_prob_yes":
        return None

    secs = row.get("secs_remaining")
    if secs is None or secs < min_secs or secs > max_secs:
        return None

    phase = _phase_of(secs)
    if conf < min_confidence_by_phase.get(phase, 0.5):
        return None

    yes_bid = row.get("yes_bid")
    yes_ask = row.get("yes_ask")
    kalshi_mid = row.get("kalshi_mid")
    if yes_bid is None or yes_ask is None or kalshi_mid is None:
        return None
    try:
        yes_bid = float(yes_bid)
        yes_ask = float(yes_ask)
        kalshi_mid = float(kalshi_mid)
    except (TypeError, ValueError):
        return None
    if yes_bid <= 0 or yes_ask <= 0:
        return None

    ph_min, ph_max = entry_price_by_phase.get(phase, (10, 95))
    if not (ph_min <= kalshi_mid <= ph_max):
        return None

    # Pick directional side and compute edge against the market-implied prob.
    if prob_yes >= 0.5:
        side = "yes"
        market_implied = yes_ask / 100.0
        edge = prob_yes - market_implied
        entry_price = int(round(yes_ask))
    else:
        side = "no"
        market_implied = (100.0 - yes_bid) / 100.0
        edge = (1.0 - prob_yes) - market_implied
        entry_price = int(round(100.0 - yes_bid))

    if edge < min_edge:
        return None

    ticker = row.get("ticker")
    outcome = results.get(ticker)
    if outcome is None:
        return None  # market hasn't settled; can't score

    if side == "yes":
        per_contract = (100 - entry_price) if outcome == "yes" else (-entry_price)
    else:  # "no"
        per_contract = (yes_bid) if outcome == "no" else (-(100 - yes_bid))
    per_contract = int(round(per_contract))

    return SimulatedTrade(
        ticker=ticker, side=side, entry_price_cents=entry_price, contracts=contracts,
        secs_at_entry=float(secs), phase=phase,
        prob_yes=float(prob_yes), confidence=float(conf), edge=float(edge),
        pnl_cents=per_contract * contracts, outcome=outcome,
    )


# ── Per-session simulator ────────────────────────────────────────────────────

@dataclass
class BrainPnL:
    name: str
    trades: list = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_pnl_cents(self) -> int:
        return sum(t.pnl_cents for t in self.trades)

    @property
    def total_pnl_dollars(self) -> float:
        return self.total_pnl_cents / 100.0

    @property
    def n_wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl_cents > 0)

    @property
    def win_rate(self) -> Optional[float]:
        return self.n_wins / self.n_trades if self.n_trades else None

    def per_phase(self) -> dict:
        out = defaultdict(lambda: {"n": 0, "pnl_cents": 0, "wins": 0})
        for t in self.trades:
            out[t.phase]["n"] += 1
            out[t.phase]["pnl_cents"] += t.pnl_cents
            if t.pnl_cents > 0:
                out[t.phase]["wins"] += 1
        return dict(out)


@dataclass
class PnLAnalysisResult:
    session_id: str
    n_decision_rows: int
    n_settled_rows: int
    settled_tickers: int
    dir_simulated: BrainPnL
    fv_simulated: BrainPnL
    dir_actual_pnl_dollars: Optional[float]   # from trades.csv if available
    dir_actual_n_fills: int
    disagreement_pnl_cents: dict  # {"dir_only": cents, "fv_only": cents, "both": cents}


def simulate_brain_trades(
    decisions: list[dict],
    prob_yes_field: str,
    confidence_field: str,
    results: dict[str, str],
    *,
    min_secs: int = DEFAULT_MIN_SECS,
    max_secs: int = DEFAULT_MAX_SECS,
    min_edge: float = DEFAULT_MIN_EDGE,
    min_confidence_by_phase: dict = None,
    entry_price_by_phase: dict = None,
    contracts: int = DEFAULT_CONTRACTS_PER_TRADE,
) -> list[SimulatedTrade]:
    """Walk decision rows; emit one simulated trade per (ticker, side) the
    first time the gates clear. Hold-to-settlement P&L."""
    if min_confidence_by_phase is None:
        min_confidence_by_phase = DEFAULT_MIN_CONFIDENCE_BY_PHASE
    if entry_price_by_phase is None:
        entry_price_by_phase = DEFAULT_ENTRY_PRICE_BY_PHASE

    seen: set[tuple[str, str]] = set()
    trades = []
    for row in decisions:
        trade = _simulate_entry_from_row(
            row, prob_yes_field, confidence_field,
            results=results, min_secs=min_secs, max_secs=max_secs,
            min_edge=min_edge,
            min_confidence_by_phase=min_confidence_by_phase,
            entry_price_by_phase=entry_price_by_phase,
            contracts=contracts,
        )
        if trade is None:
            continue
        key = (trade.ticker, trade.side)
        if key in seen:
            continue
        seen.add(key)
        trades.append(trade)
    return trades


def _load_actual_pnl_from_trades_csv(
    trades_csv: Path,
    session_label: str,
    results: dict[str, str],
) -> tuple[float, int]:
    """Compute DIR's actual P&L from the trade log, restricted to one session.

    trades.csv has three schema versions per memory; we branch on column
    count. We look at entry rows ('source' containing 'auto' or similar)
    and settle each at market_results_cache outcome (hold-to-settlement
    on ACTUAL fills — different from simulated entries since these are
    the prices the bot actually paid).
    """
    if not trades_csv.exists():
        return 0.0, 0
    total_pnl_cents = 0
    n_fills = 0
    try:
        with open(trades_csv) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                # Schema check: current header has 10 cols
                # trade_id,timestamp,ticker,side,contracts,price_cents,
                # cost_usd,source,mode,session
                try:
                    if len(row) >= 10:
                        ticker = row[2]
                        side = row[3].lower()
                        contracts = int(float(row[4]))
                        price_cents = int(float(row[5]))
                        sess = row[9]
                    else:
                        continue
                except (ValueError, IndexError):
                    continue
                if sess != session_label:
                    continue
                # Skip exits / partial / settlements — we want entry rows
                # to hold-to-settle from. Heuristic: side is plain "yes"/"no".
                if side not in ("yes", "no"):
                    continue
                outcome = results.get(ticker)
                if outcome is None:
                    continue
                if side == "yes":
                    pc = (100 - price_cents) if outcome == "yes" else (-price_cents)
                else:
                    # 'no' side: bought NO at price_cents
                    pc = (100 - price_cents) if outcome == "no" else (-price_cents)
                total_pnl_cents += pc * contracts
                n_fills += 1
    except Exception:
        pass
    return total_pnl_cents / 100.0, n_fills


def analyze_pnl(
    session_dir: Path,
    results_cache_path: Path,
    trades_csv: Path,
    *,
    min_secs: int = DEFAULT_MIN_SECS,
    max_secs: int = DEFAULT_MAX_SECS,
    min_edge: float = DEFAULT_MIN_EDGE,
    min_confidence_by_phase: dict = None,
    entry_price_by_phase: dict = None,
    contracts: int = DEFAULT_CONTRACTS_PER_TRADE,
) -> PnLAnalysisResult:
    """Full DIR-vs-FV P&L counterfactual for one recorded session."""
    decisions = list(_iter_jsonl(session_dir / "decisions.jsonl"))
    results = _load_results_cache(results_cache_path)

    meta_path = session_dir / "meta.json"
    session_label = session_dir.name
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            session_label = meta.get("session_label", session_label)
        except Exception:
            pass

    n_settled = sum(1 for d in decisions if results.get(d.get("ticker")) is not None)
    settled_tickers = {d.get("ticker") for d in decisions if results.get(d.get("ticker")) is not None}

    dir_trades = simulate_brain_trades(
        decisions, "prob_yes", "confidence", results,
        min_secs=min_secs, max_secs=max_secs, min_edge=min_edge,
        min_confidence_by_phase=min_confidence_by_phase,
        entry_price_by_phase=entry_price_by_phase, contracts=contracts,
    )
    fv_trades = simulate_brain_trades(
        decisions, "fv_prob_yes", "fv_confidence", results,
        min_secs=min_secs, max_secs=max_secs, min_edge=min_edge,
        min_confidence_by_phase=min_confidence_by_phase,
        entry_price_by_phase=entry_price_by_phase, contracts=contracts,
    )

    # Disagreement: trades only one brain would have taken
    dir_keys = {(t.ticker, t.side) for t in dir_trades}
    fv_keys = {(t.ticker, t.side) for t in fv_trades}
    both_keys = dir_keys & fv_keys
    dir_only_keys = dir_keys - fv_keys
    fv_only_keys = fv_keys - dir_keys
    disagreement = {
        "both_n": len(both_keys),
        "dir_only_n": len(dir_only_keys),
        "fv_only_n": len(fv_only_keys),
        "dir_only_pnl_cents": sum(t.pnl_cents for t in dir_trades if (t.ticker, t.side) in dir_only_keys),
        "fv_only_pnl_cents": sum(t.pnl_cents for t in fv_trades if (t.ticker, t.side) in fv_only_keys),
        "both_dir_pnl_cents": sum(t.pnl_cents for t in dir_trades if (t.ticker, t.side) in both_keys),
        "both_fv_pnl_cents": sum(t.pnl_cents for t in fv_trades if (t.ticker, t.side) in both_keys),
    }

    actual_pnl, actual_n = _load_actual_pnl_from_trades_csv(
        trades_csv, session_label, results,
    )

    return PnLAnalysisResult(
        session_id=session_dir.name,
        n_decision_rows=len(decisions),
        n_settled_rows=n_settled,
        settled_tickers=len(settled_tickers),
        dir_simulated=BrainPnL("DIR (hold-to-settle)", dir_trades),
        fv_simulated=BrainPnL("FV (hold-to-settle)", fv_trades),
        dir_actual_pnl_dollars=actual_pnl,
        dir_actual_n_fills=actual_n,
        disagreement_pnl_cents=disagreement,
    )
