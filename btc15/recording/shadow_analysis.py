"""Phase 3 shadow comparison analyzer.

Pure functions over a recorded session's decision_log + market_results_cache.
Computes Brier scores for DIR (ensemble) vs FV (fair-value) on settled
markets, plus an agreement matrix showing who wins on disagreement.

Brier convention:
  Brier(p, y) = (p - y)^2
    where p ∈ [0,1] is the predicted P(YES), y ∈ {0,1} the realized outcome.
  Lower is better. 0 = perfect. 0.25 = constant-50% baseline.
  An overconfident brain that's also miscalibrated gets > 0.25 (worse than
  predicting nothing). The legacy ensemble's audit found 0.283 — that's our
  bar to beat.

Aggregations:
  - Overall Brier per brain
  - Per-phase (early/mid/prime/late) Brier
  - Per-confidence-band Brier (does HIGH confidence actually predict better?)
  - Agreement matrix on the four possible (DIR_side, FV_side) × (outcome)
    combinations, with sample sizes
  - Calibration buckets (decile of predicted prob → realized win rate)
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


# ── Loaders ──────────────────────────────────────────────────────────────────

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


def load_results_cache(path: Path) -> dict[str, str]:
    """ticker → 'yes'|'no' for finalized markets only."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    out = {}
    for ticker, rec in raw.items():
        if isinstance(rec, dict) and rec.get("status") == "finalized":
            result = rec.get("result")
            if result in ("yes", "no"):
                out[ticker] = result
    return out


# ── Per-row scoring ──────────────────────────────────────────────────────────

def brier(prob: float, outcome: int) -> float:
    """Brier score for a single prediction. outcome ∈ {0, 1}."""
    return (prob - outcome) ** 2


def side_of(prob: float) -> Optional[str]:
    """Classify a P(YES) into a directional bet, or None for fence-sitting."""
    if prob > 0.5:
        return "yes"
    if prob < 0.5:
        return "no"
    return None


# ── Main analysis ────────────────────────────────────────────────────────────

@dataclass
class BrainScores:
    n_rows: int = 0
    brier_sum: float = 0.0
    correct_side: int = 0     # how often the directional bet matched outcome
    decisive: int = 0         # rows where prob != 0.5 (had an opinion)
    per_phase: dict = field(default_factory=lambda: defaultdict(lambda: {"n": 0, "brier": 0.0}))
    per_conf_band: dict = field(default_factory=lambda: defaultdict(lambda: {"n": 0, "brier": 0.0, "won": 0}))

    @property
    def mean_brier(self) -> Optional[float]:
        return self.brier_sum / self.n_rows if self.n_rows else None

    @property
    def directional_accuracy(self) -> Optional[float]:
        return self.correct_side / self.decisive if self.decisive else None


@dataclass
class AnalysisResult:
    session_id: str
    n_total_rows: int
    n_settled_rows: int
    n_with_fv: int                    # rows where fv_prob_yes was populated
    dir_scores: BrainScores
    fv_scores: BrainScores
    baseline_brier: float             # constant-50% predictor (always 0.25)
    agreement: dict                   # ("agree_yes", etc.) → {n, won_dir, won_fv}
    calibration_dir: list             # [(pred_bucket, realized_wr, n)]
    calibration_fv: list
    settled_tickers: int              # unique markets we scored on


def _conf_band(conf: float) -> str:
    """Bucket confidence into bands for breakdown."""
    if conf < 0.20:
        return "0.0–0.2"
    if conf < 0.40:
        return "0.2–0.4"
    if conf < 0.60:
        return "0.4–0.6"
    if conf < 0.80:
        return "0.6–0.8"
    return "0.8–1.0"


def _phase_of(secs: float) -> str:
    if secs > 540:
        return "early"
    if secs > 300:
        return "mid"
    if secs > 180:
        return "prime"
    return "late"


def _update_brain_score(
    scores: BrainScores, prob: float, outcome: int, phase: str, conf: float
) -> None:
    scores.n_rows += 1
    b = brier(prob, outcome)
    scores.brier_sum += b
    s = side_of(prob)
    if s is not None:
        scores.decisive += 1
        if (s == "yes" and outcome == 1) or (s == "no" and outcome == 0):
            scores.correct_side += 1
    scores.per_phase[phase]["n"] += 1
    scores.per_phase[phase]["brier"] += b
    band = _conf_band(conf)
    scores.per_conf_band[band]["n"] += 1
    scores.per_conf_band[band]["brier"] += b
    won = (s == "yes" and outcome == 1) or (s == "no" and outcome == 0)
    if s is not None and won:
        scores.per_conf_band[band]["won"] += 1


def _calibration_buckets(rows: list[tuple[float, int]], n_buckets: int = 10) -> list:
    """Group (prob, outcome) by predicted-prob decile and compute realized win rate."""
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r[0])
    buckets = []
    bucket_size = max(1, len(rows) // n_buckets)
    for i in range(0, len(rows), bucket_size):
        chunk = rows[i:i + bucket_size]
        mean_pred = sum(p for p, _ in chunk) / len(chunk)
        realized = sum(y for _, y in chunk) / len(chunk)
        buckets.append((round(mean_pred, 3), round(realized, 3), len(chunk)))
    return buckets


def analyze_session(
    session_dir: Path,
    results_cache_path: Path,
) -> AnalysisResult:
    """Run the full DIR-vs-FV comparison on a recorded session."""
    decisions = list(_iter_jsonl(session_dir / "decisions.jsonl"))
    results = load_results_cache(results_cache_path)

    dir_scores = BrainScores()
    fv_scores = BrainScores()
    agreement: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "dir_correct": 0, "fv_correct": 0}
    )
    calib_dir_rows: list[tuple[float, int]] = []
    calib_fv_rows: list[tuple[float, int]] = []
    settled_tickers: set = set()
    n_with_fv = 0
    n_settled = 0

    for d in decisions:
        ticker = d.get("ticker")
        outcome_str = results.get(ticker)
        if outcome_str is None:
            continue
        outcome = 1 if outcome_str == "yes" else 0
        settled_tickers.add(ticker)
        n_settled += 1

        secs = float(d.get("secs_remaining", 0))
        phase = _phase_of(secs)

        dir_prob = d.get("prob_yes")
        dir_conf = d.get("confidence", 0.0)
        if dir_prob is not None:
            _update_brain_score(dir_scores, dir_prob, outcome, phase, dir_conf)
            calib_dir_rows.append((dir_prob, outcome))

        fv_prob = d.get("fv_prob_yes")
        fv_conf = d.get("fv_confidence", 0.0)
        fv_degenerate = d.get("fv_degenerate", False)
        if fv_prob is not None and not fv_degenerate:
            n_with_fv += 1
            _update_brain_score(fv_scores, fv_prob, outcome, phase, fv_conf)
            calib_fv_rows.append((fv_prob, outcome))

        # Agreement matrix — only when both brains had a decisive opinion
        if dir_prob is not None and fv_prob is not None and not fv_degenerate:
            dir_side = side_of(dir_prob)
            fv_side = side_of(fv_prob)
            if dir_side is not None and fv_side is not None:
                key = (
                    f"agree_{dir_side}" if dir_side == fv_side
                    else f"dir_{dir_side}_fv_{fv_side}"
                )
                cell = agreement[key]
                cell["n"] += 1
                dir_won = (dir_side == "yes" and outcome == 1) or (dir_side == "no" and outcome == 0)
                fv_won = (fv_side == "yes" and outcome == 1) or (fv_side == "no" and outcome == 0)
                if dir_won:
                    cell["dir_correct"] += 1
                if fv_won:
                    cell["fv_correct"] += 1

    return AnalysisResult(
        session_id=session_dir.name,
        n_total_rows=len(decisions),
        n_settled_rows=n_settled,
        n_with_fv=n_with_fv,
        dir_scores=dir_scores,
        fv_scores=fv_scores,
        baseline_brier=0.25,
        agreement=dict(agreement),
        calibration_dir=_calibration_buckets(calib_dir_rows),
        calibration_fv=_calibration_buckets(calib_fv_rows),
        settled_tickers=len(settled_tickers),
    )


def analyze_all_sessions(
    recordings_root: Path,
    results_cache_path: Path,
) -> list[AnalysisResult]:
    """Run the analysis across every session under recordings_root."""
    results = []
    for session_dir in sorted(recordings_root.iterdir()):
        if not session_dir.is_dir():
            continue
        if not (session_dir / "decisions.jsonl").exists():
            continue
        results.append(analyze_session(session_dir, results_cache_path))
    return results


def merge_results(rs: list[AnalysisResult]) -> AnalysisResult:
    """Aggregate per-session results into a single cross-session view.
    Useful when a single session is too small for stable Brier estimates."""
    if not rs:
        return AnalysisResult(
            session_id="(empty)", n_total_rows=0, n_settled_rows=0,
            n_with_fv=0, dir_scores=BrainScores(), fv_scores=BrainScores(),
            baseline_brier=0.25, agreement={}, calibration_dir=[],
            calibration_fv=[], settled_tickers=0,
        )

    merged_dir = BrainScores()
    merged_fv = BrainScores()
    merged_agreement: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "dir_correct": 0, "fv_correct": 0}
    )
    total_rows = 0
    settled_rows = 0
    with_fv = 0
    settled_tickers_set: set = set()

    for r in rs:
        total_rows += r.n_total_rows
        settled_rows += r.n_settled_rows
        with_fv += r.n_with_fv
        settled_tickers_set.add(r.session_id)  # sessions, not tickers — semantic shift

        for brain_src, brain_dst in [(r.dir_scores, merged_dir), (r.fv_scores, merged_fv)]:
            brain_dst.n_rows += brain_src.n_rows
            brain_dst.brier_sum += brain_src.brier_sum
            brain_dst.correct_side += brain_src.correct_side
            brain_dst.decisive += brain_src.decisive
            for phase, cell in brain_src.per_phase.items():
                brain_dst.per_phase[phase]["n"] += cell["n"]
                brain_dst.per_phase[phase]["brier"] += cell["brier"]
            for band, cell in brain_src.per_conf_band.items():
                brain_dst.per_conf_band[band]["n"] += cell["n"]
                brain_dst.per_conf_band[band]["brier"] += cell["brier"]
                brain_dst.per_conf_band[band]["won"] += cell.get("won", 0)
        for key, cell in r.agreement.items():
            merged_agreement[key]["n"] += cell["n"]
            merged_agreement[key]["dir_correct"] += cell["dir_correct"]
            merged_agreement[key]["fv_correct"] += cell["fv_correct"]

    return AnalysisResult(
        session_id=f"merged({len(rs)} sessions)",
        n_total_rows=total_rows,
        n_settled_rows=settled_rows,
        n_with_fv=with_fv,
        dir_scores=merged_dir,
        fv_scores=merged_fv,
        baseline_brier=0.25,
        agreement=dict(merged_agreement),
        # Calibration buckets don't merge cleanly without re-walking; skip
        # for the merged view. Per-session calibration is the right tool.
        calibration_dir=[],
        calibration_fv=[],
        settled_tickers=0,
    )
