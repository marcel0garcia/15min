"""Sanity tests for the TWAP-settlement fair-value pricer.

Runnable standalone (`python tests/test_settlement_twap.py`) or via pytest.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from btc15.models.fair_value import fair_value
from btc15.models.settlement_twap import twap_fair_value, accrued_average


S = 100_000.0
SIGMA = 0.50  # annualized


def approx(a, b, tol=1e-9):
    assert abs(a - b) <= tol, f"{a} != {b} (tol {tol})"


def test_atm_is_coinflip():
    for tau in (840, 300, 90, 59, 30, 5):
        out = twap_fair_value(spot=S, strike=S, sigma=SIGMA, tau_seconds=tau)
        approx(out.prob_yes, 0.5, 1e-6)


def test_regime_continuity_at_window_boundary():
    """Regime A at tau=60+eps must match regime B at tau=60-eps (no accrued)."""
    strike = S * 1.0002
    a = twap_fair_value(spot=S, strike=strike, sigma=SIGMA, tau_seconds=60.0)
    b = twap_fair_value(spot=S, strike=strike, sigma=SIGMA, tau_seconds=59.999)
    # log vs price-scale approximation: allow a tiny numerical gap.
    assert abs(a.prob_yes - b.prob_yes) < 1e-3, (a.prob_yes, b.prob_yes)


def test_tighter_than_endpoint_before_window():
    """With tau > 60 the TWAP pricer sees LESS variance than the endpoint
    pricer (tau_eff = tau - 40), so an ITM prob must be strictly higher."""
    strike = S * 0.999  # spot above strike → ITM YES
    for tau in (90, 180, 420, 840):
        ep = fair_value(spot=S, strike=strike, sigma=SIGMA, tau_seconds=tau)
        tw = twap_fair_value(spot=S, strike=strike, sigma=SIGMA, tau_seconds=tau)
        assert tw.prob_yes > ep.prob_yes, (tau, tw.prob_yes, ep.prob_yes)


def test_lock_in_dominates_as_u_shrinks():
    """Accrued average above strike + shrinking remaining time → prob → 1
    even when spot has dipped slightly below strike."""
    strike = S
    accrued = S * 1.0005          # locked-in portion averaged above strike
    spot_dip = S * 0.9999         # spot now marginally below strike
    prev = 0.0
    for u in (50, 30, 10, 3, 1):
        out = twap_fair_value(
            spot=spot_dip, strike=strike, sigma=SIGMA, tau_seconds=u,
            accrued_avg=accrued, accrued_count=60 - int(u),
        )
        assert out.prob_yes >= prev - 1e-12, (u, out.prob_yes, prev)
        prev = out.prob_yes
    assert prev > 0.99, prev  # with 1s left the lock-in is decisive


def test_lock_in_step_function_at_settlement():
    out = twap_fair_value(
        spot=S * 0.99, strike=S, sigma=SIGMA, tau_seconds=0.0,
        accrued_avg=S * 1.001, accrued_count=60,
    )
    approx(out.prob_yes, 1.0)
    assert out.degenerate


def test_monotone_in_spot():
    strike = S
    probs = []
    for spot in (S * 0.999, S * 0.9995, S, S * 1.0005, S * 1.001):
        out = twap_fair_value(spot=spot, strike=strike, sigma=SIGMA,
                              tau_seconds=30.0, accrued_avg=S, accrued_count=30)
        probs.append(out.prob_yes)
    assert probs == sorted(probs), probs


def test_degenerate_inputs():
    assert twap_fair_value(spot=0, strike=S, sigma=SIGMA, tau_seconds=100).degenerate
    assert twap_fair_value(spot=S, strike=0, sigma=SIGMA, tau_seconds=100).degenerate
    assert twap_fair_value(spot=S, strike=S, sigma=0, tau_seconds=100).degenerate


def test_accrued_average_windowing():
    now = 1_000_000.0
    # tau=45 → window started 15s ago → only ticks with ts >= now-15 count.
    ticks = [(now - 30, 99_000.0), (now - 10, 100_000.0), (now - 2, 101_000.0)]
    avg, n = accrued_average(ticks, now_ts=now, tau_seconds=45.0)
    assert n == 2
    approx(avg, 100_500.0)
    # Window not started yet → None.
    avg, n = accrued_average(ticks, now_ts=now, tau_seconds=61.0)
    assert avg is None and n == 0


def test_fallback_without_accrued():
    """Inside the window with no accrued data: falls back to A=spot and
    stays sane (ATM → 0.5)."""
    out = twap_fair_value(spot=S, strike=S, sigma=SIGMA, tau_seconds=20.0)
    approx(out.prob_yes, 0.5, 1e-6)
    assert out.inputs.get("accrued_fallback") == "spot"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
