"""Evaluate many configurations against the same recorded corpus.

## What a sweep result is, and is not

A sweep ranks configurations on data that has already happened. Run
enough configurations against a small corpus and the winner is whichever
one best fits that corpus's noise — which is not a property of the
market. Three defences are built in and none of them is optional:

  1. **Markets, not scans.** Every result reports `n_markets`, the count
     of distinct settled 15-minute windows behind it. That is the real
     sample size. Kalshi lists one market at a time, so 30 markets is
     ~7.5 hours of runtime and 300 is ~75 hours.
  2. **A holdout.** `--holdout` splits the corpus by time: configs are
     ranked on the earlier sessions and re-scored on the later ones. A
     config that wins in-sample and dies out-of-sample was fitted to
     noise, and the report says so per row.
  3. **Multiplicity.** Testing 200 configs at 95% confidence gets you
     ~10 false winners by construction. `n_configs` is printed with the
     results as a standing reminder, and the promotion rule stays what it
     always was: a slice earns `enabled_slices` from `score`, on data the
     sweep did not choose it with.

The honest use of this tool is to *narrow* a large knob space to a few
candidates, then let live paper sessions arbitrate. It is a hypothesis
generator, not a verdict.
"""
from __future__ import annotations

import itertools
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from btc15.config import CoreConfig
from btc15.core.score import SliceReport, enabled_slice_suggestion, score_slices
from btc15.research.corpus import (
    LoadedSession, load_results_cache, load_session, resolve_outcomes,
)
from btc15.research.replay import (
    ReplayResult, merge_results, replay_session, thin_observations, with_overrides,
)


@dataclass
class SweepRow:
    """One configuration, scored over the corpus."""
    overrides: dict
    n_markets: int
    n_settled: int
    n_obs: int
    n_trades: int
    trades_per_day: float
    pnl_usd: float
    fees_usd: float
    win_rate: Optional[float]
    brier_model: Optional[float]
    brier_market: Optional[float]
    delta: Optional[float]
    delta_ci_low: Optional[float]
    delta_ci_high: Optional[float]
    verdict: str
    winning_slices: list = field(default_factory=list)
    holdout_delta: Optional[float] = None
    holdout_pnl_usd: Optional[float] = None
    holdout_trades: Optional[int] = None
    reject_top: dict = field(default_factory=dict)

    # Unit suffixes carry no information in a comparison table and cost the
    # column width that makes it readable.
    _DROP_SUFFIXES = ("_cents", "_usd", "_sec", "_seconds")

    @property
    def label(self) -> str:
        if not self.overrides:
            return "baseline"
        parts = []
        for k, v in sorted(self.overrides.items()):
            name = k
            for suf in self._DROP_SUFFIXES:
                if name.endswith(suf) and len(name) > len(suf) + 2:
                    name = name[: -len(suf)]
                    break
            parts.append(f"{name}={v}")
        return " ".join(parts)


def parse_knob(spec: str) -> tuple[str, list]:
    """`core.sigma_floor=0.2,0.3,0.4` -> ('sigma_floor', [0.2, 0.3, 0.4]).

    The `core.` prefix is optional and stripped: every sweepable knob lives
    on CoreConfig, and accepting the prefix keeps the syntax identical to
    `run --set`.
    """
    if "=" not in spec:
        raise ValueError(f"--knob expects field=v1,v2,...  got {spec!r}")
    field_name, raw = spec.split("=", 1)
    field_name = field_name.strip()
    if field_name.startswith("core."):
        field_name = field_name[len("core."):]
    if not hasattr(CoreConfig(), field_name):
        raise ValueError(f"unknown CoreConfig field {field_name!r}")
    current = getattr(CoreConfig(), field_name)
    values = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if isinstance(current, bool):
            values.append(tok.lower() in ("1", "true", "yes", "on"))
        elif isinstance(current, int) and not isinstance(current, bool):
            values.append(int(float(tok)))
        elif isinstance(current, float):
            values.append(float(tok))
        else:
            values.append(tok)
    if not values:
        raise ValueError(f"--knob {spec!r} lists no values")
    return field_name, values


def expand_grid(knobs: dict[str, list]) -> list[dict]:
    """Full cartesian product. Empty knobs -> a single baseline config."""
    if not knobs:
        return [{}]
    names = sorted(knobs)
    return [
        dict(zip(names, combo))
        for combo in itertools.product(*(knobs[n] for n in names))
    ]


# ── Worker ───────────────────────────────────────────────────────────────────
# Module-level so ProcessPoolExecutor can pickle it. Sessions are loaded once
# per worker and cached: parsing JSONL dominates replay cost otherwise.

_CACHE: dict[str, tuple[LoadedSession, dict, dict]] = {}


def _load_cached(session_dir: str, results_cache: str, allow_twap: bool):
    key = f"{session_dir}|{allow_twap}"
    hit = _CACHE.get(key)
    if hit is None:
        sess = load_session(Path(session_dir))
        official = load_results_cache(Path(results_cache))
        outcomes, source = resolve_outcomes(
            sess, official, allow_twap_fallback=allow_twap,
        )
        hit = (sess, outcomes, source)
        _CACHE[key] = hit
    return hit


def _eval_config(args) -> dict:
    (base_core, overrides, session_dirs, results_cache, allow_twap,
     holdout_dirs, dedup_seconds) = args
    core = with_overrides(base_core, overrides)

    def run_over(dirs) -> ReplayResult:
        results = []
        for sd in dirs:
            sess, outcomes, source = _load_cached(sd, results_cache, allow_twap)
            results.append(replay_session(sess, core, outcomes, source))
        return merge_results(results)

    merged = run_over(session_dirs)
    # Thin before scoring, exactly as core/score.py does — see
    # replay.thin_observations for why the un-thinned average is not a
    # property of the model.
    scored_obs = thin_observations(merged.observations, dedup_seconds)
    reports = score_slices(scored_obs) if scored_obs else []
    overall: Optional[SliceReport] = reports[0] if reports else None

    row = SweepRow(
        overrides=overrides,
        n_markets=(overall.n_markets if overall else 0),
        n_settled=merged.n_settled_markets,
        n_obs=len(scored_obs),
        n_trades=merged.n_trades,
        trades_per_day=merged.trades_per_day,
        pnl_usd=merged.realized_pnl_usd,
        fees_usd=merged.fees_usd,
        win_rate=merged.win_rate,
        brier_model=(overall.brier_model if overall else None),
        brier_market=(overall.brier_market if overall else None),
        delta=(overall.delta if overall else None),
        delta_ci_low=(overall.delta_ci_low if overall else None),
        delta_ci_high=(overall.delta_ci_high if overall else None),
        verdict=(overall.verdict if overall else "no_data"),
        winning_slices=enabled_slice_suggestion(reports) if reports else [],
        reject_top=dict(sorted(
            merged.reject_counts.items(), key=lambda kv: -kv[1],
        )[:6]),
    )

    if holdout_dirs:
        ho = run_over(holdout_dirs)
        ho_obs = thin_observations(ho.observations, dedup_seconds)
        ho_reports = score_slices(ho_obs) if ho_obs else []
        row.holdout_delta = ho_reports[0].delta if ho_reports else None
        row.holdout_pnl_usd = ho.realized_pnl_usd
        row.holdout_trades = ho.n_trades

    return asdict(row)


def run_sweep(
    base_core: CoreConfig,
    session_dirs: Sequence[Path],
    configs: Sequence[dict],
    *,
    results_cache: Path,
    allow_twap_fallback: bool = True,
    holdout_frac: float = 0.0,
    workers: Optional[int] = None,
    dedup_seconds: float = 15.0,
    progress=None,
) -> list[SweepRow]:
    """Replay every config over the corpus. Returns rows, best delta first.

    The holdout split is by TIME, never at random: sessions are ordered
    oldest-first by `discover_sessions`, and the last `holdout_frac` are
    withheld. A random split would leak, because scans from the same
    15-minute market would land on both sides of it.
    """
    dirs = [str(d) for d in session_dirs]
    train, holdout = dirs, []
    if holdout_frac > 0 and len(dirs) >= 4:
        cut = max(1, int(len(dirs) * (1.0 - holdout_frac)))
        train, holdout = dirs[:cut], dirs[cut:]

    tasks = [
        (base_core, cfg, train, str(results_cache), allow_twap_fallback,
         holdout, dedup_seconds)
        for cfg in configs
    ]
    workers = workers or max(1, min(len(tasks), (os.cpu_count() or 2)))

    rows: list[dict] = []
    if workers == 1 or len(tasks) == 1:
        for i, t in enumerate(tasks):
            rows.append(_eval_config(t))
            if progress:
                progress(i + 1, len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(_eval_config, tasks)):
                rows.append(r)
                if progress:
                    progress(i + 1, len(tasks))

    out = [SweepRow(**r) for r in rows]
    out.sort(key=lambda r: (
        -(r.delta if r.delta is not None else -math.inf),
        -r.pnl_usd,
    ))
    return out


def write_report(rows: list[SweepRow], path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "meta": meta,
        "rows": [asdict(r) for r in rows],
    }, indent=2, default=str))


def pareto_frontier(rows: Iterable[SweepRow]) -> list[SweepRow]:
    """Configs not dominated on BOTH edge and frequency.

    The user asked to optimize measured edge but also said a bot that
    barely trades is useless to them. Those are two axes, not one, so the
    honest output is the frontier: every config where buying more trades
    costs edge, and picking the operating point stays a human decision.
    """
    scored = [r for r in rows if r.delta is not None]
    frontier: list[SweepRow] = []
    for r in scored:
        dominated = any(
            (o.delta >= r.delta and o.trades_per_day >= r.trades_per_day)
            and (o.delta > r.delta or o.trades_per_day > r.trades_per_day)
            for o in scored
        )
        if not dominated:
            frontier.append(r)
    frontier.sort(key=lambda r: -r.trades_per_day)
    return frontier
