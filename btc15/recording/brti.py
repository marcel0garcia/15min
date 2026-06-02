"""BRTI reconstruction — Phase 2.

Builds a consolidated BTC mid from per-venue top-of-book ticks using CF
Benchmarks' published methodology applied to our captured constituents
(Coinbase, Kraken, Bitstamp). No I/O — pure functions over iterables.

Algorithm per grid tick T:
  1. For each venue, take last known mid where recv_ts is within staleness_sec
     of T. Older readings → venue marked stale, excluded.
  2. If fewer than n_min venues have fresh data → recon_healthy=False, null mid.
  3. Compute median across fresh venue mids. Compute deviation from median
     per venue; flag as outlier if |dev| > k_mad × MAD.
  4. Recompute median excluding outliers. If excluding outliers drops below
     n_min healthy venues, keep original median but mark recon_healthy=False.
  5. Emit row: recon_brti, recon_healthy, n_venues, outlier_venues, spread.

Defaults (BTC venue dispersion is typically < $50; staleness > 5s on a
healthy WS feed is a real degradation, not microstructure noise):
  staleness_sec  = 5.0
  n_min          = 2
  k_mad          = 3.0
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict
from typing import Iterable, Optional


# ── Tunables ─────────────────────────────────────────────────────────────────

DEFAULT_STALENESS_SEC = 5.0
DEFAULT_N_MIN = 2
DEFAULT_K_MAD = 3.0
DEFAULT_GRID_INTERVAL_SEC = 1.0


# ── Result schema ────────────────────────────────────────────────────────────

@dataclass
class GridRow:
    ts: float
    recon_brti: Optional[float]
    recon_healthy: bool
    n_venues: int                       # venues with fresh data at this tick
    venues: list                        # sorted venue names contributing
    outlier_venues: list                # venues flagged + excluded by MAD
    spread: float                       # max - min of fresh venue mids
    reason: Optional[str] = None        # filled when not healthy

    def to_dict(self) -> dict:
        return asdict(self)


# ── Per-tick reconstruction ──────────────────────────────────────────────────

def reconstruct(
    venue_mids: dict,                   # venue -> mid (already filtered for freshness)
    *,
    k_mad: float = DEFAULT_K_MAD,
    n_min: int = DEFAULT_N_MIN,
) -> tuple[Optional[float], list, bool, str]:
    """Given fresh per-venue mids, return (recon_mid, outliers, healthy, reason).

    Pure — no time math here, just the algorithm. Caller is responsible for
    pre-filtering venue_mids for staleness."""
    if len(venue_mids) < n_min:
        return None, [], False, "insufficient_venues"

    mids = list(venue_mids.values())
    median = statistics.median(mids)

    # MAD outlier reject. Need ≥3 venues for MAD to be meaningful.
    outlier_venues: list = []
    if len(mids) >= 3:
        deviations = [abs(m - median) for m in mids]
        mad = statistics.median(deviations)
        if mad > 0:
            outlier_venues = sorted(
                v for v, m in venue_mids.items()
                if abs(m - median) > k_mad * mad
            )

    # Recompute median excluding outliers, but only if enough venues remain.
    if outlier_venues:
        clean = {v: m for v, m in venue_mids.items() if v not in outlier_venues}
        if len(clean) >= n_min:
            median = statistics.median(clean.values())
            return round(median, 2), outlier_venues, True, "ok_with_outliers_removed"
        # Outliers detected but excluding them would leave us under n_min →
        # keep median but flag unhealthy. This is the "venue divergence,
        # majority unclear" case.
        return round(median, 2), outlier_venues, False, "outliers_unrecoverable"

    return round(median, 2), [], True, "ok"


# ── Grid builder (walks venue ticks once, in time order) ─────────────────────

def build_grid(
    venue_ticks: Iterable[dict],
    *,
    grid_interval_sec: float = DEFAULT_GRID_INTERVAL_SEC,
    staleness_sec: float = DEFAULT_STALENESS_SEC,
    n_min: int = DEFAULT_N_MIN,
    k_mad: float = DEFAULT_K_MAD,
) -> list[GridRow]:
    """Walk venue_ticks in recv_ts order, maintain per-venue last mid, emit
    one GridRow per grid_interval_sec boundary.

    Expects each tick to be: {recv_ts, venue, bid, ask, ...}.
    """
    last_state: dict[str, tuple[float, float]] = {}  # venue -> (ts, mid)
    grid: list[GridRow] = []
    next_emit_ts: Optional[float] = None

    for tick in venue_ticks:
        try:
            ts = float(tick["recv_ts"])
            venue = tick["venue"]
            bid = float(tick["bid"])
            ask = float(tick["ask"])
        except (KeyError, TypeError, ValueError):
            continue

        if bid <= 0 or ask <= 0 or ask < bid:
            continue

        last_state[venue] = (ts, (bid + ask) / 2.0)

        if next_emit_ts is None:
            next_emit_ts = (int(ts / grid_interval_sec) * grid_interval_sec) + grid_interval_sec

        # Emit rows for every grid boundary crossed by this tick's ts.
        while ts >= next_emit_ts:
            grid.append(
                _emit_at(next_emit_ts, last_state, staleness_sec, n_min, k_mad)
            )
            next_emit_ts += grid_interval_sec

    return grid


def _emit_at(
    t: float,
    last_state: dict[str, tuple[float, float]],
    staleness_sec: float,
    n_min: int,
    k_mad: float,
) -> GridRow:
    fresh = {
        v: mid for v, (ts, mid) in last_state.items()
        if t - ts <= staleness_sec
    }
    spread = (max(fresh.values()) - min(fresh.values())) if len(fresh) > 1 else 0.0
    mid, outliers, healthy, reason = reconstruct(fresh, k_mad=k_mad, n_min=n_min)
    return GridRow(
        ts=t,
        recon_brti=mid,
        recon_healthy=healthy,
        n_venues=len(fresh),
        venues=sorted(fresh.keys()),
        outlier_venues=outliers,
        spread=round(spread, 2),
        reason=reason,
    )


# ── Stability metrics over a built grid ──────────────────────────────────────

@dataclass
class StabilityReport:
    n_rows: int
    n_healthy: int
    health_pct: float
    reason_breakdown: dict
    spread_p50: float
    spread_p95: float
    spread_max: float
    venue_uptime_pct: dict                 # venue → % of rows it was fresh
    jackknife: dict                        # venue → mean |median - median_without_v|
    n_outlier_events: int


def stability_report(grid: list[GridRow]) -> StabilityReport:
    """Summarize a built grid. Pure read-only over the grid output."""
    if not grid:
        return StabilityReport(
            n_rows=0, n_healthy=0, health_pct=0.0,
            reason_breakdown={}, spread_p50=0.0, spread_p95=0.0, spread_max=0.0,
            venue_uptime_pct={}, jackknife={}, n_outlier_events=0,
        )

    n_rows = len(grid)
    n_healthy = sum(1 for r in grid if r.recon_healthy)

    reason_breakdown: dict[str, int] = {}
    for r in grid:
        reason_breakdown[r.reason or "ok"] = reason_breakdown.get(r.reason or "ok", 0) + 1

    spreads = sorted(r.spread for r in grid if r.n_venues > 1)
    spread_p50 = spreads[len(spreads) // 2] if spreads else 0.0
    spread_p95 = spreads[int(len(spreads) * 0.95)] if spreads else 0.0
    spread_max = max(spreads) if spreads else 0.0

    # Venue uptime: % of grid rows where the venue contributed.
    all_venues: set = set()
    for r in grid:
        all_venues.update(r.venues)
    venue_uptime = {
        v: round(sum(1 for r in grid if v in r.venues) / n_rows * 100.0, 1)
        for v in sorted(all_venues)
    }

    # True jackknife requires per-row per-venue mids; the GridRow schema
    # only carries the consolidated mid + spread. Spread already bounds the
    # worst-case impact of dropping any single venue (in the n=2 case it
    # IS the jackknife distance). If Phase 3 needs finer-grained per-venue
    # robustness we'll extend GridRow to carry venue mids.
    n_outlier_events = sum(1 for r in grid if r.outlier_venues)

    return StabilityReport(
        n_rows=n_rows,
        n_healthy=n_healthy,
        health_pct=round(n_healthy / n_rows * 100.0, 2),
        reason_breakdown=reason_breakdown,
        spread_p50=spread_p50,
        spread_p95=spread_p95,
        spread_max=spread_max,
        venue_uptime_pct=venue_uptime,
        jackknife={},  # see comment above
        n_outlier_events=n_outlier_events,
    )
