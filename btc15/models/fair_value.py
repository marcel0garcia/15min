"""Fair-value engine for KXBTC15M binary contracts.

Prices the contract directly from the no-drift digital option formula:

  P(YES) = N( ln(S/K) / (σ √τ) )

  S   spot price (consolidated BRTI in production)
  K   contract strike
  σ   annualized realized vol (from vol_nowcast)
  τ   time to settlement, in years
  N   standard normal CDF

For a true martingale underlying (which BRTI is by construction at our
15-min horizon), this returns a calibrated probability — no isotonic
recalibration layer needed if σ is right.

Confidence is derived from |z| — the distance from the coin-flip line in
vol-time units — so the existing personas gates (min_confidence_by_phase
etc.) carry over unchanged. |z| = 0 -> conf = 0, |z| = ~2 -> conf ~ 0.95.

Pure functions; no I/O. Engine integration is a separate step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


@dataclass
class FairValueOutput:
    prob_yes: float        # in [0, 1]
    prob_no: float         # = 1 - prob_yes
    z_score: float         # ln(S/K) / (σ √τ)
    confidence: float      # |prob_yes - 0.5| * 2, in [0, 1]
    sigma_used: float      # σ that fed the formula
    tau_seconds: float     # raw seconds to settlement
    inputs: dict           # {S, K, sigma, tau_seconds} for logging
    degenerate: bool       # True iff inputs were missing/invalid -> 50/50 fallback
    reason: Optional[str] = None  # diagnostic when degenerate=True


def _norm_cdf(z: float) -> float:
    """Standard normal CDF via the error function. Pure stdlib —
    avoids the scipy dependency on the engine's hot path."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def fair_value(
    *,
    spot: float,
    strike: float,
    sigma: float,
    tau_seconds: float,
) -> FairValueOutput:
    """Return P(YES) and derived diagnostics.

    The four args are all named-only to prevent accidental positional mix-ups
    (spot/strike are easy to swap and the result silently inverts).
    """
    inputs = {"spot": spot, "strike": strike, "sigma": sigma, "tau_seconds": tau_seconds}

    if spot <= 0 or strike <= 0:
        return FairValueOutput(
            prob_yes=0.5, prob_no=0.5, z_score=0.0, confidence=0.0,
            sigma_used=sigma, tau_seconds=tau_seconds, inputs=inputs,
            degenerate=True, reason="non_positive_price",
        )
    if sigma <= 0:
        return FairValueOutput(
            prob_yes=0.5, prob_no=0.5, z_score=0.0, confidence=0.0,
            sigma_used=sigma, tau_seconds=tau_seconds, inputs=inputs,
            degenerate=True, reason="non_positive_sigma",
        )
    if tau_seconds <= 0:
        # Contract has settled. The market has decided — but without an oracle
        # of the settled outcome we can only report which side BRTI is on.
        prob = 1.0 if spot > strike else 0.0 if spot < strike else 0.5
        return FairValueOutput(
            prob_yes=prob, prob_no=1.0 - prob,
            z_score=float("inf") if prob == 1.0 else float("-inf") if prob == 0.0 else 0.0,
            confidence=abs(prob - 0.5) * 2,
            sigma_used=sigma, tau_seconds=tau_seconds, inputs=inputs,
            degenerate=True, reason="tau_zero_at_or_past_settlement",
        )

    tau_years = tau_seconds / SECONDS_PER_YEAR
    sigma_sqrt_tau = sigma * math.sqrt(tau_years)
    if sigma_sqrt_tau <= 0:
        return FairValueOutput(
            prob_yes=0.5, prob_no=0.5, z_score=0.0, confidence=0.0,
            sigma_used=sigma, tau_seconds=tau_seconds, inputs=inputs,
            degenerate=True, reason="vol_time_collapsed",
        )

    z = math.log(spot / strike) / sigma_sqrt_tau
    prob_yes = _norm_cdf(z)
    prob_no = 1.0 - prob_yes
    confidence = abs(prob_yes - 0.5) * 2.0

    return FairValueOutput(
        prob_yes=prob_yes,
        prob_no=prob_no,
        z_score=z,
        confidence=confidence,
        sigma_used=sigma,
        tau_seconds=tau_seconds,
        inputs=inputs,
        degenerate=False,
        reason=None,
    )
