"""Short-horizon realized vol estimator for the fair-value engine.

Pure functions over (timestamp, price) sequences. No I/O, no state — the
caller (engine or backtest) maintains the price buffer and passes the
recent window in.

The fair-value formula P(YES) = N(ln(S/K) / (σ √τ)) is extremely sensitive
to σ at our 15-min horizon: σ enters as `σ × √(τ/year)` which compresses
a wide range of annualized vols into a narrow effective range. So instead
of picking one estimator, this module exposes a small toolkit:

  - close_to_close: classic log-return stdev, annualized
  - rogers_satchell: uses high/low/open/close from bars; handles drift
  - blended:        weighted blend of N estimators (placeholder for the
                    time-of-day baseline once we accumulate enough sessions)

Defaults targeted at the Phase 3 spec:
  - 60-second trailing window
  - Floor σ at 0.20 (annualized) so a quiet minute can't push fair value
    to coin-flip territory artificially
  - Ceiling σ at 5.0 to bound the impact of a bad reconstruction tick
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional


SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
DEFAULT_FLOOR_SIGMA = 0.20
DEFAULT_CEILING_SIGMA = 5.0
DEFAULT_LOOKBACK_SEC = 60.0
DEFAULT_MIN_SAMPLES = 10


@dataclass
class VolEstimate:
    sigma: float                 # annualized realized vol, clamped to [floor, ceiling]
    sigma_raw: float             # pre-clamp value (None if insufficient data)
    n_samples: int               # number of return observations used
    window_sec: float            # actual time span of the sample
    method: str
    clamped: bool                # True iff floor/ceiling kicked in


def close_to_close(
    prices: Iterable[tuple[float, float]],
    *,
    lookback_sec: float = DEFAULT_LOOKBACK_SEC,
    floor: float = DEFAULT_FLOOR_SIGMA,
    ceiling: float = DEFAULT_CEILING_SIGMA,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    now_ts: Optional[float] = None,
) -> VolEstimate:
    """Annualized realized vol from log returns of sequential prices.

    Args:
      prices: iterable of (timestamp_seconds, price) pairs, newest last.
      lookback_sec: window size — only the last `lookback_sec` of data is used.
      floor / ceiling: bounds applied to the returned sigma.
      min_samples: minimum return-pair count to compute a non-fallback sigma.
      now_ts: anchor timestamp for the lookback window. If None, uses the
        last price's timestamp.

    Returns a VolEstimate; sigma falls back to `floor` when the window
    has fewer than `min_samples` returns.
    """
    sample = list(prices)
    if not sample:
        return VolEstimate(
            sigma=floor, sigma_raw=0.0, n_samples=0, window_sec=0.0,
            method="close_to_close", clamped=True,
        )

    anchor = now_ts if now_ts is not None else sample[-1][0]
    cutoff = anchor - lookback_sec
    window = [(t, p) for t, p in sample if t >= cutoff and p > 0]

    if len(window) < 2:
        return VolEstimate(
            sigma=floor, sigma_raw=0.0, n_samples=0, window_sec=0.0,
            method="close_to_close", clamped=True,
        )

    # Log returns + the inter-arrival time per return (in seconds).
    log_returns: list[float] = []
    inter_secs: list[float] = []
    for (t_a, p_a), (t_b, p_b) in zip(window[:-1], window[1:]):
        dt = t_b - t_a
        if dt <= 0:
            continue
        log_returns.append(math.log(p_b / p_a))
        inter_secs.append(dt)

    n = len(log_returns)
    if n < min_samples:
        return VolEstimate(
            sigma=floor, sigma_raw=0.0, n_samples=n,
            window_sec=window[-1][0] - window[0][0],
            method="close_to_close", clamped=True,
        )

    # Sample variance of log returns. Use n-1 denominator (unbiased).
    mean_r = sum(log_returns) / n
    var_r = sum((r - mean_r) ** 2 for r in log_returns) / max(n - 1, 1)
    mean_dt = sum(inter_secs) / n  # typical seconds per sample
    if mean_dt <= 0:
        return VolEstimate(
            sigma=floor, sigma_raw=0.0, n_samples=n,
            window_sec=window[-1][0] - window[0][0],
            method="close_to_close", clamped=True,
        )

    # Vol per second, then annualize by sqrt(seconds_per_year / mean_dt).
    sigma_per_sample = math.sqrt(var_r)
    sigma_annual = sigma_per_sample * math.sqrt(SECONDS_PER_YEAR / mean_dt)

    clamped = sigma_annual < floor or sigma_annual > ceiling
    return VolEstimate(
        sigma=max(floor, min(ceiling, sigma_annual)),
        sigma_raw=sigma_annual,
        n_samples=n,
        window_sec=window[-1][0] - window[0][0],
        method="close_to_close",
        clamped=clamped,
    )
