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

THE FLOOR IS THE MOST DANGEROUS KNOB IN THE REPO. It exists so a quiet
minute cannot drive fair value to a false certainty, but it does the
opposite when set too low: a floored sigma UNDERSTATES settlement
variance, which pushes P(YES) toward 0 or 1 exactly at the extreme
strikes where we intend to trade. In the 19AUG v3 session, 27% of scans
ran at the 0.20 floor and 27.5% of rows priced beyond 0.999 while the
market quoted 0.99. Every field below is now a config knob precisely so
`./run.sh sweep` can measure the right value instead of inheriting a
guess.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from btc15.models.vol_nowcast import (
    DEFAULT_CEILING_SIGMA, DEFAULT_FLOOR_SIGMA, DEFAULT_MIN_SAMPLES,
    close_to_close, VolEstimate,
)


@dataclass(frozen=True)
class SigmaConfig:
    """Every input to the vol nowcast, in one sweepable object."""
    fast_sec: float = 60.0
    slow_sec: float = 300.0
    fast_weight: float = 0.6
    floor: float = DEFAULT_FLOOR_SIGMA
    ceiling: float = DEFAULT_CEILING_SIGMA
    min_samples: int = DEFAULT_MIN_SAMPLES
    # Multiplicative correction applied AFTER blending and BEFORE the
    # clamp. 1.0 = trust the realized estimate. Values > 1 widen the
    # distribution (less confident probabilities); values < 1 sharpen it.
    # This is the single knob that trades model conviction against
    # calibration, and `score --calibration` is how you set it.
    scale: float = 1.0

    # Spacing, in seconds, to resample the tick series to before estimating
    # realized vol. 0 = use ticks exactly as delivered.
    #
    # Motivation, stated at the strength it was actually measured: on one
    # live session (2026-08-19, five evaluation points), sigma computed from
    # the engine's 4 Hz BRTI stream came out around 0.84x the value from the
    # same data resampled to 1 Hz. If that holds up, the estimate depends on
    # our polling rate rather than on the price process, and reading vol off
    # a faster feed makes the model MORE confident — the wrong direction
    # given the overconfidence already visible in `score --calibration`.
    #
    # The obvious explanation is wrong: a synthetic where price steps once a
    # second and is sampled at 4 Hz (three zero returns in four) reproduces
    # a ratio of 0.99, because close_to_close annualizes by the observed
    # mean inter-arrival time and so self-corrects for zero-padding. So the
    # mechanism behind 0.84 is NOT understood, and five points from one
    # session is not a result.
    #
    # This is a knob rather than a change in default behaviour for exactly
    # that reason: it costs nothing at 0.0, and it lets `sweep` settle the
    # question on real data instead of on either of our stories.
    sample_sec: float = 0.0


def resample(
    ticks: Sequence[tuple[float, float]], spacing_sec: float,
) -> list[tuple[float, float]]:
    """Keep the last tick in each `spacing_sec` bucket.

    Last, not first or mean: the estimator wants the price prevailing at the
    end of each interval, which is what a close-to-close return is defined
    on. Averaging inside the bucket would smooth the very moves being
    measured and bias vol down again.
    """
    if spacing_sec <= 0:
        return list(ticks)
    out: dict[int, tuple[float, float]] = {}
    for ts, price in ticks:
        out[int(ts / spacing_sec)] = (ts, price)
    return [out[k] for k in sorted(out)]


@dataclass
class SigmaNowcast:
    sigma: float          # annualized, blended, scaled, clamped — what the pricer uses
    sigma_fast: float
    sigma_slow: float
    n_fast: int
    n_slow: int
    blended: bool         # False when only the fast leg had data
    sigma_raw: float      # pre-clamp, pre-scale blend — diagnoses floor binding
    clamped: bool         # True iff the floor or ceiling actually bound


def blended_sigma(
    ticks: Sequence[tuple[float, float]],
    *,
    now_ts: float,
    cfg: SigmaConfig | None = None,
    # Legacy keyword form, kept so existing callers and tests still work.
    fast_sec: float | None = None,
    slow_sec: float | None = None,
    fast_weight: float | None = None,
) -> SigmaNowcast:
    if cfg is None:
        cfg = SigmaConfig()
    if fast_sec is not None or slow_sec is not None or fast_weight is not None:
        cfg = SigmaConfig(
            fast_sec=cfg.fast_sec if fast_sec is None else fast_sec,
            slow_sec=cfg.slow_sec if slow_sec is None else slow_sec,
            fast_weight=cfg.fast_weight if fast_weight is None else fast_weight,
            floor=cfg.floor, ceiling=cfg.ceiling,
            min_samples=cfg.min_samples, scale=cfg.scale,
            sample_sec=cfg.sample_sec,
        )

    if cfg.sample_sec > 0:
        ticks = resample(ticks, cfg.sample_sec)

    fast: VolEstimate = close_to_close(
        ticks, lookback_sec=cfg.fast_sec, now_ts=now_ts,
        floor=cfg.floor, ceiling=cfg.ceiling, min_samples=cfg.min_samples,
    )
    slow: VolEstimate = close_to_close(
        ticks, lookback_sec=cfg.slow_sec, now_ts=now_ts,
        floor=cfg.floor, ceiling=cfg.ceiling, min_samples=cfg.min_samples,
    )

    # Blend the UNCLAMPED legs and clamp exactly once, at the end.
    #
    # Clamping per-leg and again after blending looks harmless but destroys
    # the diagnostic: a leg floored at 0.20 blends to 0.20, the final clamp
    # sees a value already at the floor, and `clamped` comes back False. The
    # nowcast then reports a healthy sigma that is entirely manufactured by
    # the floor — and the entry guard that depends on this flag never fires.
    # `sigma_raw` below is the honest pre-clamp estimate; `sigma` is what the
    # pricer gets; `clamped` is True only when the bounds actually did work.
    fast_ok = fast.n_samples >= cfg.min_samples
    slow_ok = slow.n_samples >= cfg.min_samples
    fast_raw = fast.sigma_raw if fast_ok else None
    slow_raw = slow.sigma_raw if slow_ok else None

    if fast_raw is None and slow_raw is None:
        # No usable data at all — session warm-up. The floor stands in, and
        # we say so, which is what the policy's warm-up guard reads.
        return SigmaNowcast(
            sigma=cfg.floor, sigma_fast=fast.sigma, sigma_slow=slow.sigma,
            n_fast=fast.n_samples, n_slow=slow.n_samples, blended=False,
            sigma_raw=0.0, clamped=True,
        )

    if slow_raw is None or (slow.n_samples < 2 * fast.n_samples and slow.n_samples < 60):
        # Slow leg is still warming up — don't let it drag a real fast reading.
        raw = fast_raw if fast_raw is not None else slow_raw
        blended = False
    else:
        if fast_raw is None:
            raw, blended = slow_raw, False
        else:
            raw = math.sqrt(
                cfg.fast_weight * fast_raw ** 2
                + (1.0 - cfg.fast_weight) * slow_raw ** 2
            )
            blended = True

    scaled = raw * cfg.scale
    sigma = max(cfg.floor, min(cfg.ceiling, scaled))
    return SigmaNowcast(
        sigma=sigma, sigma_fast=fast.sigma, sigma_slow=slow.sigma,
        n_fast=fast.n_samples, n_slow=slow.n_samples, blended=blended,
        sigma_raw=raw, clamped=(scaled < cfg.floor or scaled > cfg.ceiling),
    )
