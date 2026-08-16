"""Blended short-horizon vol nowcast.

The TWAP fair-value formula is extremely sensitive to sigma, and a single
60-second window has two problems:

  1. Sampling noise — ~60 one-second returns give sigma a relative
     standard error near 1/sqrt(2*60) ~ 9%, and one quiet minute can
     halve the estimate.
  2. Microstructure bias — 1s returns of a reconstructed median carry
     bid-ask bounce and venue staleness, biasing sigma UP.

The blend pairs a fast window (reactive to regime breaks) with a slow
window (stable anchor) in variance space:

    sigma^2 = w_fast * sigma_fast^2 + (1 - w_fast) * sigma_slow^2

Both legs come from vol_nowcast.close_to_close with its floor/ceiling
clamps. When the slow leg has insufficient data (session warm-up), the
fast leg stands alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from btc15.models.vol_nowcast import close_to_close, VolEstimate


@dataclass
class SigmaNowcast:
    sigma: float          # annualized, blended, clamped
    sigma_fast: float
    sigma_slow: float
    n_fast: int
    n_slow: int
    blended: bool         # False when only the fast leg had data


def blended_sigma(
    ticks: Sequence[tuple[float, float]],
    *,
    now_ts: float,
    fast_sec: float = 60.0,
    slow_sec: float = 300.0,
    fast_weight: float = 0.6,
) -> SigmaNowcast:
    fast: VolEstimate = close_to_close(ticks, lookback_sec=fast_sec, now_ts=now_ts)
    slow: VolEstimate = close_to_close(ticks, lookback_sec=slow_sec, now_ts=now_ts)

    # A clamped slow estimate with few samples means warm-up — don't let a
    # floor-value slow leg drag a real fast reading around.
    if slow.n_samples < 2 * fast.n_samples and slow.n_samples < 60:
        return SigmaNowcast(
            sigma=fast.sigma, sigma_fast=fast.sigma, sigma_slow=slow.sigma,
            n_fast=fast.n_samples, n_slow=slow.n_samples, blended=False,
        )

    var = fast_weight * fast.sigma ** 2 + (1.0 - fast_weight) * slow.sigma ** 2
    return SigmaNowcast(
        sigma=math.sqrt(var), sigma_fast=fast.sigma, sigma_slow=slow.sigma,
        n_fast=fast.n_samples, n_slow=slow.n_samples, blended=True,
    )
