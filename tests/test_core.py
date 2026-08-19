"""Unit tests for the v3 decision core.

Runnable standalone (`python3 tests/test_core.py`) or via pytest.
Covers the three places a trading bot silently loses money: fee math,
fill simulation, and the gates that decide whether to fire at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from btc15.core.fees import (
    FeeCalibrator, implied_rate, taker_fee_usd, taker_fee_cents_per_contract,
)
from btc15.core.paper import PaperBroker, PaperBrokerConfig
from btc15.core.policy import (
    Policy, PolicyConfig, Position, ev_cents_per_contract, kelly_fraction,
)
from btc15.core.pricer import MarketQuote, phase_of, price_band, quote_market
from btc15.core.score import Observation, score_slices
from btc15.core.sigma import blended_sigma


def approx(a, b, tol=1e-6):
    assert abs(a - b) <= tol, f"{a} != {b} (tol {tol})"


# ── Fees ─────────────────────────────────────────────────────────────────────

def test_fee_peaks_at_fifty_cents():
    """1.75c per contract at 50c is the published anchor."""
    approx(taker_fee_usd(50, 100), 1.75, 1e-9)
    # Fee curve is symmetric and collapses at the extremes.
    assert taker_fee_cents_per_contract(50) > taker_fee_cents_per_contract(85)
    assert taker_fee_cents_per_contract(95) < 0.5
    approx(taker_fee_cents_per_contract(30), taker_fee_cents_per_contract(70), 1e-9)


def test_fee_rounds_up_to_the_cent():
    # 0.07 * 1 * 0.5 * 0.5 = 0.0175 -> rounds up to 0.02
    approx(taker_fee_usd(50, 1), 0.02, 1e-9)
    assert taker_fee_usd(50, 0) == 0.0


# ── EV / Kelly ───────────────────────────────────────────────────────────────

def test_ev_is_negative_on_a_fair_coin_at_fair_price():
    """A 50/50 bet at 50c must be strictly negative EV — that is the fee."""
    assert ev_cents_per_contract(0.50, 50) < 0


def test_ev_gate_is_stricter_mid_range_than_at_extremes():
    """Same 3-point probability advantage: worth more at 95c than at 50c,
    because the fee at 95c is ~5x cheaper. This is the whole thesis for
    trading the extremes."""
    ev_mid = ev_cents_per_contract(0.53, 50)
    ev_ext = ev_cents_per_contract(0.98, 95)
    assert ev_ext > ev_mid


def test_kelly_zero_when_not_profitable():
    assert kelly_fraction(0.50, 50) == 0.0
    assert kelly_fraction(0.30, 50) == 0.0
    assert kelly_fraction(0.0, 50) == 0.0
    assert kelly_fraction(1.0, 50) == 0.0


def test_kelly_grows_with_edge():
    a = kelly_fraction(0.60, 50)
    b = kelly_fraction(0.70, 50)
    c = kelly_fraction(0.80, 50)
    assert 0 < a < b < c < 1.0


# ── Pricer helpers ───────────────────────────────────────────────────────────

def test_phase_and_band_boundaries():
    assert phase_of(600) == "early"
    assert phase_of(400) == "mid"
    assert phase_of(200) == "prime"
    assert phase_of(50) == "late"
    assert price_band(5) == "extreme"
    assert price_band(95) == "extreme"    # symmetric by distance from 0/100
    assert price_band(25) == "outer"
    assert price_band(50) == "middle"


def test_quote_carries_market_and_model():
    ticks = [(1000.0 + i, 100_000.0) for i in range(120)]
    q = quote_market(
        ticker="T", strike=99_000.0, spot=100_000.0, secs=300.0,
        sigma=0.5, ticks=ticks, now_ts=1120.0, yes_bid=90.0, yes_ask=92.0,
    )
    assert q.prob_yes > 0.9              # deep ITM
    assert q.recommended_side == "yes"
    approx(q.market_prob_yes, 0.91)
    approx(q.exec_ask_cents("yes"), 92.0)
    approx(q.exec_ask_cents("no"), 10.0)  # 100 - yes_bid
    approx(q.exec_bid_cents("no"), 8.0)   # 100 - yes_ask


# ── Paper broker ─────────────────────────────────────────────────────────────

def _broker(cash=100.0, adverse=0.0):
    return PaperBroker(PaperBrokerConfig(starting_cash_usd=cash, adverse_cents=adverse))


BOOK = {
    # YES asks: 3 contracts at 60c, 5 at 61c, 20 at 63c
    "yes_asks": {60: 3, 61: 5, 63: 20},
    # YES bids: 4 at 58c, 10 at 57c
    "yes_bids": {58: 4, 57: 10},
}


def test_ioc_buy_walks_the_book():
    b = _broker()
    f = b.buy_ioc(ticker="T", side="yes", contracts=8, limit_cents=63, book=BOOK)
    assert f.contracts == 8
    # 3@60 + 5@61 = 485 / 8 = 60.625
    approx(f.avg_price_cents, 60.625)


def test_ioc_buy_respects_limit_and_partials():
    b = _broker()
    f = b.buy_ioc(ticker="T", side="yes", contracts=20, limit_cents=61, book=BOOK)
    assert f.contracts == 8      # only 60c and 61c levels are within the limit
    assert f.note == "partial"
    assert f.requested_contracts == 20


def test_no_fill_when_limit_below_book():
    b = _broker()
    f = b.buy_ioc(ticker="T", side="yes", contracts=5, limit_cents=55, book=BOOK)
    assert not f.filled
    assert b.cash_usd == 100.0   # no cash moved


def test_no_side_buy_mirrors_the_yes_bid_ladder():
    """Buying NO at X consumes the YES bid at (100 - X). The best NO ask
    is therefore 100 - 58 = 42c."""
    b = _broker()
    f = b.buy_ioc(ticker="T", side="no", contracts=4, limit_cents=42, book=BOOK)
    assert f.contracts == 4
    approx(f.avg_price_cents, 42.0)


def test_buy_never_spends_more_cash_than_it_has():
    b = _broker(cash=1.00)
    f = b.buy_ioc(ticker="T", side="yes", contracts=8, limit_cents=63, book=BOOK)
    assert b.cash_usd >= -1e-9
    if f.filled:
        assert f.contracts < 8


def test_empty_book_does_not_invent_a_fill():
    b = _broker()
    f = b.buy_ioc(ticker="T", side="yes", contracts=5, limit_cents=99,
                  book={"yes_asks": {}, "yes_bids": {}})
    assert not f.filled
    assert f.note == "no_fill_depth_or_limit"


def test_adverse_cents_worsens_the_fill():
    clean = _broker()
    worse = _broker(adverse=1.0)
    a = clean.buy_ioc(ticker="T", side="yes", contracts=3, limit_cents=63, book=BOOK)
    c = worse.buy_ioc(ticker="T", side="yes", contracts=3, limit_cents=63, book=BOOK)
    assert c.avg_price_cents > a.avg_price_cents


def test_cash_conservation_round_trip():
    """Buy then settle as a win: cash back = payout, and P&L equals
    (100 - entry) * n / 100 minus the entry fee."""
    b = _broker()
    start = b.cash_usd
    f = b.buy_ioc(ticker="T", side="yes", contracts=5, limit_cents=63, book=BOOK)
    assert f.filled
    spent = start - b.cash_usd
    approx(spent, f.gross_usd + f.fee_usd, 1e-9)
    pnl = b.settle(ticker="T", side="yes", contracts=f.contracts,
                   entry_cents=f.avg_price_cents, result="yes")
    approx(b.cash_usd, start - f.fee_usd + (100 - f.avg_price_cents) * f.contracts / 100, 1e-9)
    approx(pnl, (100 - f.avg_price_cents) * f.contracts / 100, 1e-9)


def test_settlement_loss_zeroes_the_position():
    b = _broker()
    f = b.buy_ioc(ticker="T", side="yes", contracts=5, limit_cents=63, book=BOOK)
    before = b.cash_usd
    pnl = b.settle(ticker="T", side="yes", contracts=f.contracts,
                   entry_cents=f.avg_price_cents, result="no")
    approx(b.cash_usd, before)                     # nothing credited
    assert pnl < 0
    assert b.losses == 1


# ── Policy ───────────────────────────────────────────────────────────────────

def _q(**kw) -> MarketQuote:
    base = dict(
        ticker="T", strike=100_000.0, spot=100_000.0, secs=300.0, phase="mid",
        prob_yes=0.75, confidence=0.5, z_score=1.0, sigma=0.5, degenerate=False,
        yes_bid=60.0, yes_ask=62.0,
        # A healthy vol nowcast by default: plenty of BRTI history behind it
        # and no clamp binding. Tests that care about the warm-up guards
        # override these explicitly.
        sigma_clamped=False, tick_span_sec=600.0,
    )
    base.update(kw)
    return MarketQuote(**base)


def _policy(**kw) -> Policy:
    return Policy(PolicyConfig(**kw))


def test_entry_fires_on_real_edge():
    p = _policy()
    d = p.evaluate_entry(_q(), bankroll_usd=100.0)
    assert d.kind == "enter", d.reject_gate
    assert d.side == "yes"
    assert d.contracts > 0
    assert d.ev_cents > 0


def test_entry_rejected_when_market_agrees_with_model():
    """No edge: model says 61%, market prices 61%. The fee makes it -EV."""
    p = _policy()
    d = p.evaluate_entry(_q(prob_yes=0.61), bankroll_usd=100.0)
    assert d.kind == "none"
    assert d.reject_gate == "ev_margin"


def test_window_gate():
    p = _policy(min_seconds=30, max_seconds=840)
    assert p.evaluate_entry(_q(secs=10), 100.0).reject_gate == "window"
    assert p.evaluate_entry(_q(secs=900), 100.0).reject_gate == "window"


def test_slice_gate_blocks_disabled_slices():
    # yes_ask 62 -> band "middle"; enable only extremes
    p = _policy(enabled_slices={"mid:extreme"})
    d = p.evaluate_entry(_q(), 100.0)
    assert d.kind == "none"
    assert d.reject_gate.startswith("slice_disabled")


def test_slice_gate_allows_enabled_slice():
    p = _policy(enabled_slices={"mid:middle"})
    assert p.evaluate_entry(_q(), 100.0).kind == "enter"


def test_spread_gate():
    p = _policy(max_spread_cents=1.0)
    assert p.evaluate_entry(_q(), 100.0).reject_gate == "spread_too_wide"


def test_no_duplicate_position():
    p = _policy()
    p.record_open(Position(ticker="T", side="yes", contracts=5, entry_cents=60,
                           cost_usd=3.0, fees_usd=0.05, opened_ts=0.0, trade_id="x"))
    assert p.evaluate_entry(_q(), 100.0).reject_gate == "already_positioned"


def test_entry_blocked_during_warmup():
    """The 2026-08-19 startup bug: entering before enough BRTI history exists
    to trust sigma. The engine fired 0.2s after start on a floored sigma,
    then flipped out four seconds later at a loss."""
    p = _policy(warmup_sec=120.0)
    d = p.evaluate_entry(_q(tick_span_sec=5.0), bankroll_usd=100.0)
    assert d.kind == "none"
    assert d.reject_gate == "warmup"
    # ...and clears once history accumulates.
    assert p.evaluate_entry(_q(tick_span_sec=300.0), bankroll_usd=100.0).kind == "enter"


def test_entry_blocked_while_sigma_is_clamped():
    """A floored sigma understates settlement variance, which drives P(YES)
    to false certainty at exactly the extreme strikes we want to trade."""
    p = _policy()
    d = p.evaluate_entry(_q(sigma_clamped=True), bankroll_usd=100.0)
    assert d.kind == "none"
    assert d.reject_gate == "sigma_clamped"
    # The guard is a knob, so the sweep can measure whether it costs us.
    p2 = _policy(reject_clamped_sigma=False)
    assert p2.evaluate_entry(_q(sigma_clamped=True), bankroll_usd=100.0).kind == "enter"


def test_crossed_book_is_refused():
    """bid STRICTLY above ask means our cache is stale, not that the market
    is offering riskless arbitrage. Observed live 2026-08-19 (bid 85/ask 84)."""
    p = _policy()
    d = p.evaluate_entry(_q(yes_bid=85.0, yes_ask=84.0), bankroll_usd=100.0)
    assert d.kind == "none"
    assert d.reject_gate == "crossed_book"


def test_locked_book_is_tradeable():
    """bid == ask means the best YES bid and best NO bid sum to exactly 100
    — a zero spread, not corruption. Rejecting these killed 60-95% of scans
    in the 19AUG sessions when the guard used >= instead of >."""
    q = _q(yes_bid=62.0, yes_ask=62.0)
    assert q.locked is True
    assert q.crossed is False
    assert _policy().evaluate_entry(q, bankroll_usd=100.0).kind == "enter"


def test_entry_budget_allows_reentry_after_close():
    """KXBTC15M lists one market at a time, so max_entries_per_market is the
    only lever that can lift trade count above one per window."""
    p = _policy(max_entries_per_market=1)
    d = p.evaluate_entry(_q(), bankroll_usd=100.0)
    assert d.kind == "enter"
    p.record_open(Position(
        ticker="T", side="yes", contracts=10, entry_cents=62.0, cost_usd=6.2,
        fees_usd=0.02, opened_ts=0.0, trade_id="x",
    ))
    p.record_close("T", 0.0, now_ts=1000.0)
    assert p.evaluate_entry(_q(), bankroll_usd=100.0).reject_gate == "max_entries_per_market"

    p2 = _policy(max_entries_per_market=2, entry_cooldown_sec=30.0)
    p2.record_open(Position(
        ticker="T", side="yes", contracts=10, entry_cents=62.0, cost_usd=6.2,
        fees_usd=0.02, opened_ts=0.0, trade_id="x",
    ))
    p2.record_close("T", 0.0, now_ts=1000.0)
    # Inside the cooldown -> blocked; past it -> allowed.
    assert p2.evaluate_entry(_q(), 100.0, now_ts=1010.0).reject_gate == "entry_cooldown"
    assert p2.evaluate_entry(_q(), 100.0, now_ts=1040.0).kind == "enter"


def test_sigma_scale_widens_the_distribution():
    """sigma_scale is the conviction-vs-calibration knob: >1 must produce
    strictly humbler probabilities on the same ticks."""
    from btc15.core.sigma import SigmaConfig
    ticks = [(1000.0 + i, 100_000.0 * (1 + 0.00002 * ((i % 7) - 3))) for i in range(400)]
    base = blended_sigma(ticks, now_ts=1400.0, cfg=SigmaConfig(scale=1.0))
    wide = blended_sigma(ticks, now_ts=1400.0, cfg=SigmaConfig(scale=2.0))
    assert wide.sigma > base.sigma
    assert wide.sigma_raw == base.sigma_raw     # scale applies after blending


def test_sigma_floor_binding_is_reported():
    """The floor is the most dangerous knob in the repo; when it binds we
    must be able to see it in the recordings."""
    from btc15.core.sigma import SigmaConfig
    flat = [(1000.0 + i, 100_000.0) for i in range(400)]   # zero realized vol
    n = blended_sigma(flat, now_ts=1400.0, cfg=SigmaConfig(floor=0.35))
    assert n.sigma == 0.35
    assert n.clamped is True
    assert n.sigma_raw < 0.35


def test_bootstrap_clusters_by_market_not_by_scan():
    """The R1 gate must not promote a slice on five coin flips.

    Before this, the CI resampled individual scan rows. A 15-minute window
    contributes dozens of rows sharing one outcome, so the interval came out
    roughly sqrt(rows-per-market) times too narrow and a slice could reach
    `enabled_slices` — and real money — on a handful of markets.
    """
    from btc15.core.score import Observation, score_slices

    def build(n_markets, per_market):
        out = []
        for m in range(n_markets):
            outcome = m % 2
            for i in range(per_market):
                out.append(Observation(
                    ticker=f"T{m}", phase="late", band="extreme",
                    p_model=0.70 + 0.001 * i, p_market=0.68,
                    outcome=outcome, secs=60.0,
                ))
        return out

    few = score_slices(build(5, 40))[0]
    assert few.n == 200 and few.n_markets == 5
    assert few.verdict == "insufficient", "5 markets must never be promotable"

    # Same rows-per-market, more markets => a tighter interval. The width
    # must track the market count, not the scan count.
    many = score_slices(build(60, 40))[0]
    assert many.n_markets == 60
    wide = few.delta_ci_high - few.delta_ci_low
    narrow = many.delta_ci_high - many.delta_ci_low
    assert narrow < wide

    # And scanning the SAME markets more often must not narrow the interval
    # much — that is the autocorrelation the cluster bootstrap absorbs.
    dense = score_slices(build(60, 160))[0]
    dense_width = dense.delta_ci_high - dense.delta_ci_low
    assert dense_width > narrow * 0.5, (
        "4x the scans on the same markets should not meaningfully shrink the CI"
    )


def test_max_open_positions():
    p = _policy(max_open_positions=1)
    p.record_open(Position(ticker="OTHER", side="yes", contracts=5, entry_cents=60,
                           cost_usd=3.0, fees_usd=0.05, opened_ts=0.0, trade_id="x"))
    assert p.evaluate_entry(_q(), 100.0).reject_gate == "max_open_positions"


def test_size_respects_caps():
    p = _policy(max_single_trade_usd=2.0)
    d = p.evaluate_entry(_q(), bankroll_usd=100_000.0)
    assert d.kind == "enter"
    assert d.contracts * d.limit_cents / 100 <= 2.0 + 1e-9


def test_halt_blocks_entries():
    p = _policy(daily_loss_limit_usd=5.0)
    p.record_close("X", -6.0)
    assert p.halted
    assert p.evaluate_entry(_q(), 100.0).reject_gate == "halted"


def test_exit_only_on_genuine_flip():
    p = _policy()
    p.record_open(Position(ticker="T", side="yes", contracts=5, entry_cents=60,
                           cost_usd=3.0, fees_usd=0.05, opened_ts=0.0, trade_id="x"))
    # Model still likes YES → hold.
    assert p.evaluate_exit(_q(prob_yes=0.75)).kind == "none"
    # Model now strongly likes NO (NO ask = 100-60 = 40c) → flip.
    d = p.evaluate_exit(_q(prob_yes=0.10))
    assert d.kind == "exit"
    assert d.side == "yes"


def test_no_exit_late_in_the_window():
    """The single exit rule must not fire near settlement — that is the
    drained-book trap the June 6 post-mortem identified."""
    p = _policy(exit_min_seconds=120)
    p.record_open(Position(ticker="T", side="yes", contracts=5, entry_cents=60,
                           cost_usd=3.0, fees_usd=0.05, opened_ts=0.0, trade_id="x"))
    d = p.evaluate_exit(_q(prob_yes=0.02, secs=45))
    assert d.kind == "none"
    assert d.reject_gate == "too_late_to_flip"


def test_no_exit_into_an_empty_book():
    p = _policy()
    p.record_open(Position(ticker="T", side="yes", contracts=5, entry_cents=60,
                           cost_usd=3.0, fees_usd=0.05, opened_ts=0.0, trade_id="x"))
    d = p.evaluate_exit(_q(prob_yes=0.02, yes_bid=None))
    assert d.kind == "none"


# ── Sigma ────────────────────────────────────────────────────────────────────

def test_blended_sigma_falls_back_during_warmup():
    ticks = [(1000.0 + i, 100_000.0 + (i % 3)) for i in range(40)]
    s = blended_sigma(ticks, now_ts=1040.0)
    assert not s.blended
    assert s.sigma > 0


def test_blended_sigma_blends_with_enough_history():
    import math
    ticks = []
    price = 100_000.0
    for i in range(400):
        price *= math.exp(0.00002 * ((i % 7) - 3))
        ticks.append((1000.0 + i, price))
    s = blended_sigma(ticks, now_ts=1400.0)
    assert s.blended
    assert min(s.sigma_fast, s.sigma_slow) - 1e-9 <= s.sigma <= max(s.sigma_fast, s.sigma_slow) + 1e-9


# ── Scorer ───────────────────────────────────────────────────────────────────

def test_scorer_detects_a_model_that_beats_the_market():
    obs = []
    for i in range(200):
        outcome = i % 2
        # Model leans the right way; market stays at the coin flip.
        obs.append(Observation(
            ticker=f"T{i}", phase="prime", band="extreme",
            p_model=0.8 if outcome else 0.2, p_market=0.5,
            outcome=outcome, secs=200.0,
        ))
    rows = score_slices(obs)
    overall = rows[0]
    assert overall.key == "ALL"
    assert overall.brier_model < overall.brier_market
    assert overall.delta > 0
    assert overall.delta_ci_low > 0
    assert overall.verdict == "BEATS_MARKET"


def test_scorer_detects_a_model_that_loses_to_the_market():
    obs = []
    for i in range(200):
        outcome = i % 2
        obs.append(Observation(
            ticker=f"T{i}", phase="mid", band="middle",
            p_model=0.5, p_market=0.9 if outcome else 0.1,
            outcome=outcome, secs=400.0,
        ))
    overall = score_slices(obs)[0]
    assert overall.verdict == "loses_to_market"


def test_scorer_marks_small_samples_insufficient():
    obs = [
        Observation(ticker=f"T{i}", phase="late", band="outer",
                    p_model=0.9, p_market=0.5, outcome=1, secs=60.0)
        for i in range(10)
    ]
    assert score_slices(obs)[0].verdict == "insufficient"


# ── Fee model uncertainty ────────────────────────────────────────────────────

def test_configurable_rate_changes_the_fee():
    """The crypto rate could not be confirmed from the build environment,
    so it must be a knob — and the knob must actually do something."""
    assert taker_fee_usd(50, 100, rate=0.14) > taker_fee_usd(50, 100, rate=0.07)
    approx(taker_fee_usd(50, 100, rate=0.14), 3.50, 1e-9)
    approx(taker_fee_usd(50, 100, rate=0.07, multiplier=2.0), 3.50, 1e-9)


def test_higher_rate_tightens_the_ev_gate():
    """If the true crypto rate is double, trades that looked +EV must
    stop qualifying. This is the failure mode the calibrator guards."""
    p = _policy(fee_rate=0.07)
    d_cheap = p.evaluate_entry(_q(prob_yes=0.68), 100.0)
    p2 = _policy(fee_rate=0.40)
    d_dear = p2.evaluate_entry(_q(prob_yes=0.68), 100.0)
    assert d_cheap.kind == "enter"
    assert d_dear.kind == "none", d_dear.ev_cents
    assert d_dear.reject_gate == "ev_margin"


def test_implied_rate_inverts_the_fee_formula():
    fee = taker_fee_usd(60, 100, rate=0.07)
    r = implied_rate(fee, 60, 100)
    assert abs(r - 0.07) < 0.005


def test_calibrator_detects_a_wrong_fee_model():
    """Feed it fills priced at double our modeled rate; it must notice."""
    cal = FeeCalibrator(rate=0.07)
    for _ in range(6):
        cal.observe(fee_usd=taker_fee_usd(60, 50, rate=0.14),
                    price_cents=60, contracts=50)
    assert cal.observed_rate is not None
    assert abs(cal.observed_rate - 0.14) < 0.01
    assert cal._warned, "a 2x fee error must raise the operator warning"


def test_calibrator_stays_quiet_when_the_model_is_right():
    cal = FeeCalibrator(rate=0.07)
    for _ in range(10):
        cal.observe(fee_usd=taker_fee_usd(60, 50, rate=0.07),
                    price_cents=60, contracts=50)
    assert not cal._warned
    assert abs(cal.observed_rate - 0.07) < 0.01


def test_calibrator_ignores_tiny_fills():
    """Ceil-rounding dominates small fills and would produce false alarms."""
    cal = FeeCalibrator(rate=0.07, min_contracts=5)
    assert cal.observe(fee_usd=0.02, price_cents=50, contracts=1) is None
    assert not cal.samples


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok    {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
