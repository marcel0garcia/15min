"""TWAP-aware fair value for KXBTC15M binary contracts.

The endpoint pricer in fair_value.py answers the wrong question. KXBTC15M
does NOT settle on the BRTI *at* close time — it settles on the arithmetic
average of the 60 one-second BRTI readings in the final minute before
close. Two consequences the endpoint formula misses:

  1. Before the final minute (tau > 60s), the effective horizon is shorter
     than tau: the settlement value is the average of a Brownian path over
     [T-60, T], whose variance viewed from now is

         Var = sigma_s^2 * ((tau - W) + W/3)        (W = 60s window)

     i.e. an effective tau of tau - 2W/3 = tau - 40s. The endpoint formula
     overstates settlement variance by up to 3x in the final minutes,
     which pulls its probabilities toward 0.5 and understates conviction
     exactly where the market trades hardest.

  2. Inside the final minute (tau <= 60s), part of the settlement value is
     already *known*: (W - u)/W of the average is locked in by observed
     BRTI prints (u = seconds remaining). The correct fair value conditions
     on the accrued running average A:

         settle = ((W-u)/W) * A  +  (u/W) * M
         M      = average of the remaining u seconds of the path
                  ~ Normal(S, S^2 * sigma_s^2 * u/3)   (driftless approx)

         P(YES) = P(settle >= K) = N( (S - K_eff) / (S * sigma_s * sqrt(u/3)) )
         K_eff  = (W*K - (W-u)*A) / u

     As u -> 0 this collapses to a step function of the locked-in average —
     the "settlement lock" the personas code approximates heuristically via
     a BSM >= 88% gate is just the exact limit of this formula.

Approximations (all documented, all negligible at this horizon):
  - Driftless arithmetic Brownian approximation on price (not log-price)
    inside the final minute. 15-min BTC moves are <1% in the typical case;
    the lognormal/normal discrepancy is O(move^2), well under a tenth of a
    cent of contract price.
  - The averaging is over 60 discrete 1-second samples; we price the
    continuous-time average. The discretization correction to the variance
    is O(1/60) and ignored.
  - Regime continuity: at tau = W the two regimes agree (tau_eff = W/3 on
    the log side vs u/3 = W/3 on the price side).

Pure functions; no I/O. sigma is the ANNUALIZED vol from vol_nowcast, same
convention as fair_value.fair_value().
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from btc15.models.fair_value import FairValueOutput, _norm_cdf, SECONDS_PER_YEAR


AVG_WINDOW_SEC = 60.0


def accrued_average(
    ticks: Sequence[tuple[float, float]],
    *,
    now_ts: float,
    tau_seconds: float,
    window_sec: float = AVG_WINDOW_SEC,
) -> tuple[Optional[float], int]:
    """Average price of the ticks that fall inside the settlement averaging
    window observed so far, i.e. ts >= (close_time - window_sec).

    Args:
      ticks: (timestamp_seconds, price) pairs, any order.
      now_ts: current wall clock (seconds).
      tau_seconds: seconds remaining to close.
      window_sec: settlement averaging window length.

    Returns (average or None, sample_count). None when the window hasn't
    started (tau >= window_sec) or no ticks fall inside it yet.
    """
    elapsed_in_window = window_sec - tau_seconds
    if elapsed_in_window <= 0:
        return None, 0
    cutoff = now_ts - elapsed_in_window
    inside = [p for t, p in ticks if t >= cutoff and p > 0]
    if not inside:
        return None, 0
    return sum(inside) / len(inside), len(inside)


def twap_fair_value(
    *,
    spot: float,
    strike: float,
    sigma: float,
    tau_seconds: float,
    accrued_avg: Optional[float] = None,
    accrued_count: int = 0,
    window_sec: float = AVG_WINDOW_SEC,
) -> FairValueOutput:
    """P(YES) for a contract settling on the mean BRTI over the final
    `window_sec` seconds. Drop-in replacement for fair_value.fair_value();
    returns the same FairValueOutput schema (z_score is the standardized
    distance in whichever regime applied).

    accrued_avg / accrued_count: running average of observed prices inside
    the settlement window (from accrued_average()). Only consulted when
    tau_seconds < window_sec; when missing, falls back to accrued_avg = spot
    (assumes the observed portion averaged to the current price) and flags
    the fallback in `inputs`.
    """
    inputs = {
        "spot": spot, "strike": strike, "sigma": sigma,
        "tau_seconds": tau_seconds, "accrued_avg": accrued_avg,
        "accrued_count": accrued_count, "window_sec": window_sec,
        "pricing": "twap",
    }

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
        # Window complete. The settlement value IS the accrued average.
        ref = accrued_avg if accrued_avg is not None else spot
        prob = 1.0 if ref > strike else 0.0 if ref < strike else 0.5
        return FairValueOutput(
            prob_yes=prob, prob_no=1.0 - prob,
            z_score=float("inf") if prob == 1.0 else float("-inf") if prob == 0.0 else 0.0,
            confidence=abs(prob - 0.5) * 2,
            sigma_used=sigma, tau_seconds=tau_seconds, inputs=inputs,
            degenerate=True, reason="tau_zero_at_or_past_settlement",
        )

    if tau_seconds >= window_sec:
        # Regime A — window not started. Effective variance horizon:
        # (tau - W) + W/3  =  tau - 2W/3.
        tau_eff = tau_seconds - 2.0 * window_sec / 3.0
        tau_years = tau_eff / SECONDS_PER_YEAR
        sigma_sqrt_tau = sigma * math.sqrt(tau_years)
        if sigma_sqrt_tau <= 0:
            return FairValueOutput(
                prob_yes=0.5, prob_no=0.5, z_score=0.0, confidence=0.0,
                sigma_used=sigma, tau_seconds=tau_seconds, inputs=inputs,
                degenerate=True, reason="vol_time_collapsed",
            )
        z = math.log(spot / strike) / sigma_sqrt_tau
        prob_yes = _norm_cdf(z)
        inputs["tau_eff_seconds"] = tau_eff
        return FairValueOutput(
            prob_yes=prob_yes, prob_no=1.0 - prob_yes, z_score=z,
            confidence=abs(prob_yes - 0.5) * 2.0,
            sigma_used=sigma, tau_seconds=tau_seconds, inputs=inputs,
            degenerate=False, reason=None,
        )

    # Regime B — inside the averaging window. u seconds of the average are
    # still stochastic; the rest is locked in by observed prints.
    u = tau_seconds
    locked_frac = (window_sec - u) / window_sec
    if accrued_avg is None or accrued_count <= 0:
        accrued_avg = spot
        inputs["accrued_fallback"] = "spot"

    # Threshold the remaining-path average must clear for YES:
    # settle = ((W-u)/W)*A + (u/W)*M >= K  ⟺  M >= (W*K - (W-u)*A) / u
    k_eff = (window_sec * strike - (window_sec - u) * accrued_avg) / u

    # Std dev (price scale) of the time-average of the remaining path:
    # S * sigma_annual * sqrt((u/3) / seconds_per_year).
    sd = spot * sigma * math.sqrt((u / 3.0) / SECONDS_PER_YEAR)
    if sd <= 0:
        prob_yes = 1.0 if spot > k_eff else 0.0 if spot < k_eff else 0.5
        z = float("inf") if prob_yes == 1.0 else float("-inf") if prob_yes == 0.0 else 0.0
    else:
        z = (spot - k_eff) / sd
        prob_yes = _norm_cdf(z)

    inputs["k_eff"] = k_eff
    inputs["locked_frac"] = locked_frac
    return FairValueOutput(
        prob_yes=prob_yes, prob_no=1.0 - prob_yes, z_score=z,
        confidence=abs(prob_yes - 0.5) * 2.0,
        sigma_used=sigma, tau_seconds=tau_seconds, inputs=inputs,
        degenerate=False, reason=None,
    )
