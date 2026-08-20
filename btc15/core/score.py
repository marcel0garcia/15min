"""Offline scoring: does the model beat THE MARKET, per slice?

The legacy analysis scored the model's Brier against 0.25 (the
always-say-50% baseline). That is the wrong bar. A Brier of 0.12 is
worthless if the Kalshi mid scores 0.10 — we would be paying fees to
underperform the price we could have just accepted.

This module scores three things on every observation the engine logged:

    brier_model   (p_model - outcome)^2
    brier_market  (p_market - outcome)^2
    delta         brier_market - brier_model   (positive == we are better)

aggregated overall and by slice (phase x price band), with a paired
bootstrap confidence interval on the delta so a slice with 12
observations doesn't get promoted on noise.

Input: the decisions JSONL written by core/engine.py, plus a mapping of
ticker -> settled outcome ("yes"/"no"). Output: SliceReport rows, ready
to print or to paste into config as enabled_slices.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional


@dataclass
class Observation:
    ticker: str
    phase: str
    band: str
    p_model: float
    p_market: Optional[float]
    outcome: int          # 1 == YES settled
    secs: float


# A slice must contain at least this many DISTINCT SETTLED MARKETS before it
# can be promoted. Not observations — markets. One 15-minute window is one
# coin flip no matter how many times we scanned it, and KXBTC15M lists only
# one market at a time, so 30 markets is ~7.5 hours of runtime.
MIN_MARKETS_FOR_VERDICT = 30


@dataclass
class SliceReport:
    key: str
    n: int                              # observations (scans), autocorrelated
    n_markets: int                      # distinct settled markets — the real n
    brier_model: float
    brier_market: Optional[float]
    delta: Optional[float]              # market - model; > 0 means we win
    delta_ci_low: Optional[float]
    delta_ci_high: Optional[float]
    mean_p_model: float
    realized_yes_rate: float

    @property
    def verdict(self) -> str:
        if self.delta is None or self.delta_ci_low is None:
            return "no_market_data"
        if self.n_markets < MIN_MARKETS_FOR_VERDICT:
            return "insufficient"
        if self.delta_ci_low > 0:
            return "BEATS_MARKET"
        if self.delta_ci_high < 0:
            return "loses_to_market"
        return "inconclusive"


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


def load_observations(
    decisions_path: Path,
    outcomes: dict[str, str],
    *,
    dedup_seconds: float = 15.0,
) -> list[Observation]:
    """Read decision rows and pair them with settled outcomes.

    The engine logs one row per market per scan (1 Hz), so raw rows are
    massively autocorrelated — 900 rows from one 15-minute market is one
    independent event, not 900. `dedup_seconds` thins each ticker's rows
    to at most one per bucket, which keeps the sample honest for the
    bootstrap.
    """
    seen: set[tuple[str, int]] = set()
    obs: list[Observation] = []
    for row in _iter_jsonl(decisions_path):
        ticker = row.get("ticker")
        if not ticker or ticker not in outcomes:
            continue
        p_model = row.get("p_model")
        if p_model is None:
            continue
        secs = float(row.get("secs") or 0.0)
        bucket = int(secs // dedup_seconds)
        key = (ticker, bucket)
        if key in seen:
            continue
        seen.add(key)
        obs.append(Observation(
            ticker=ticker,
            phase=row.get("phase") or "?",
            band=row.get("band") or "?",
            p_model=float(p_model),
            p_market=(float(row["p_market"]) if row.get("p_market") is not None else None),
            outcome=1 if outcomes[ticker] == "yes" else 0,
            secs=secs,
        ))
    return obs


def _cluster_bootstrap_ci(
    clusters: list[list[float]],
    *,
    iters: int = 2000,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float]:
    """Paired bootstrap that resamples MARKETS, not observations.

    This is the difference between a real R1 gate and a rubber stamp. The
    engine logs one row per market per second; even thinned to one row per
    15 seconds, a single 15-minute window contributes ~36 rows that share
    one outcome and one price path. Resampling those rows independently
    treats 36 views of one coin flip as 36 coin flips, and the interval
    comes out roughly sqrt(36) = 6x too narrow. A slice could then be
    promoted to `enabled_slices` — and traded with real money — on the
    strength of five markets.

    Resampling whole tickers with replacement keeps the correlation inside
    the cluster where it belongs. The interval gets much wider. That is not
    the method being pessimistic; it is the earlier interval having been
    wrong.
    """
    clusters = [c for c in clusters if c]
    if not clusters:
        return (0.0, 0.0)
    rng = random.Random(seed)
    k = len(clusters)
    # Pre-reduce each cluster to (sum, count). A resample only ever needs
    # those two numbers, so re-summing the members on every iteration is
    # pure waste: at 78 clusters over 40k observations it turned each of
    # 2000 iterations into a 40k-element sum, and scoring a sweep took
    # longer than the replays it was scoring. Same arithmetic, same seed,
    # same intervals — O(k) per iteration instead of O(n).
    sums = [sum(c) for c in clusters]
    lens = [len(c) for c in clusters]
    means = []
    for _ in range(iters):
        total = 0.0
        count = 0
        for _ in range(k):
            j = rng.randrange(k)
            total += sums[j]
            count += lens[j]
        means.append(total / count if count else 0.0)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (lo, hi)


def _score(obs: list[Observation], key: str) -> SliceReport:
    n = len(obs)
    bm = sum((o.p_model - o.outcome) ** 2 for o in obs) / n
    paired = [o for o in obs if o.p_market is not None]
    n_markets = len({o.ticker for o in obs})
    if paired:
        bk = sum((o.p_market - o.outcome) ** 2 for o in paired) / len(paired)
        by_ticker: dict[str, list[float]] = defaultdict(list)
        for o in paired:
            by_ticker[o.ticker].append(
                (o.p_market - o.outcome) ** 2 - (o.p_model - o.outcome) ** 2
            )
        diffs = [d for c in by_ticker.values() for d in c]
        delta = sum(diffs) / len(diffs)
        lo, hi = _cluster_bootstrap_ci(list(by_ticker.values()))
    else:
        bk = delta = lo = hi = None
    return SliceReport(
        key=key, n=n, n_markets=n_markets,
        brier_model=bm,
        brier_market=bk,
        delta=delta, delta_ci_low=lo, delta_ci_high=hi,
        mean_p_model=sum(o.p_model for o in obs) / n,
        realized_yes_rate=sum(o.outcome for o in obs) / n,
    )


def score_slices(obs: Iterable[Observation]) -> list[SliceReport]:
    """Overall row first, then one row per phase:band slice, ordered by
    sample size."""
    obs = list(obs)
    if not obs:
        return []
    reports = [_score(obs, "ALL")]
    groups: dict[str, list[Observation]] = defaultdict(list)
    for o in obs:
        groups[f"{o.phase}:{o.band}"].append(o)
    for key, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        reports.append(_score(rows, key))
    return reports


def calibration_buckets(obs: Iterable[Observation], n_buckets: int = 10) -> list[dict]:
    """Predicted-probability decile -> realized YES rate. A well-calibrated
    model tracks the diagonal; systematic deviation means sigma is wrong."""
    buckets: dict[int, list[Observation]] = defaultdict(list)
    for o in obs:
        idx = min(n_buckets - 1, int(o.p_model * n_buckets))
        buckets[idx].append(o)
    out = []
    for idx in sorted(buckets):
        rows = buckets[idx]
        out.append({
            "bucket": f"{idx / n_buckets:.1f}-{(idx + 1) / n_buckets:.1f}",
            "n": len(rows),
            "mean_predicted": sum(r.p_model for r in rows) / len(rows),
            "realized": sum(r.outcome for r in rows) / len(rows),
        })
    return out


def enabled_slice_suggestion(reports: list[SliceReport]) -> list[str]:
    """The slices that earned a place in the policy: those whose paired
    bootstrap CI on the Brier delta is entirely above zero."""
    return [r.key for r in reports if r.key != "ALL" and r.verdict == "BEATS_MARKET"]
