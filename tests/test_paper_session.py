"""End-to-end synthetic paper session.

Drives the full vertical — tick ingestion -> sigma -> TWAP pricer ->
policy -> paper broker -> settlement -> ledger — over scripted 15-minute
markets, with no network and no Kalshi account. This is the test that
answers "would a paper session actually be accurate?", because it checks
the invariant that matters: **cash is conserved**. Final cash must equal
starting cash plus every realized P&L minus every fee, and the ledger
must reconstruct it independently.

Two scenarios:
  1. An informed model (spot genuinely predicts settlement) must finish
     profitable after fees.
  2. A blind model (spot is pure noise vs. an adversarial settlement)
     must NOT show a fantasy profit — this is the test that would have
     caught the old "assume fill at ask, no fees" paper path.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from btc15.core.paper import PaperBroker, PaperBrokerConfig
from btc15.core.policy import Policy, PolicyConfig, Position
from btc15.core.pricer import price_band, quote_market
from btc15.core.sigma import blended_sigma

WINDOW_SEC = 900.0
SCAN_SEC = 5.0          # coarser than production's 1 Hz; keeps the test fast
START_PRICE = 100_000.0


def _make_book(yes_mid_cents: float, depth: int = 40, spread: int = 2) -> dict:
    """Two-sided book around a mid, with real depth at three levels."""
    bid = max(1, min(98, int(yes_mid_cents - spread / 2)))
    ask = max(2, min(99, int(yes_mid_cents + spread / 2)))
    return {
        "yes_bids": {bid: depth, max(1, bid - 1): depth * 2, max(1, bid - 2): depth * 3},
        "yes_asks": {ask: depth, min(99, ask + 1): depth * 2, min(99, ask + 2): depth * 3},
    }


def _simulate_window(
    *,
    rng: random.Random,
    broker: PaperBroker,
    policy: Policy,
    ticker: str,
    start_price: float,
    drift_per_sec: float,
    vol_per_sec: float,
    market_prices_fairly: bool,
    t0: float,
) -> tuple[float, str]:
    """Run one 15-minute market end to end. Returns (final_price, result)."""
    ticks: list[tuple[float, float]] = []
    price = start_price
    # Seed 5 minutes of prior history so sigma has something to chew on.
    for i in range(300, 0, -1):
        p = price * math.exp(rng.gauss(0, vol_per_sec) * math.sqrt(1.0))
        ticks.append((t0 - i, p))
    strike = round(start_price / 100) * 100  # a round strike near spot

    settle_prices: list[float] = []
    elapsed = 0.0
    while elapsed < WINDOW_SEC:
        # Advance the underlying one scan interval.
        for _ in range(int(SCAN_SEC)):
            price *= math.exp(drift_per_sec + rng.gauss(0, vol_per_sec))
            elapsed += 1.0
            now = t0 + elapsed
            ticks.append((now, price))
            if elapsed > WINDOW_SEC - 60.0:
                settle_prices.append(price)   # the settlement TWAP window

        now = t0 + elapsed
        secs_left = max(0.0, WINDOW_SEC - elapsed)
        nowcast = blended_sigma(ticks, now_ts=now)

        q = quote_market(
            ticker=ticker, strike=strike, spot=price, secs=secs_left,
            sigma=nowcast.sigma, ticks=ticks, now_ts=now,
            yes_bid=None, yes_ask=None,
        )
        # The market's own price: either fair (quoting the model's own
        # probability) or a stale coin-flip the model can exploit.
        if market_prices_fairly:
            # A 1-99 integer grid cannot represent a near-certain
            # probability, and clamping would manufacture fake edge. Where
            # the market can't quote us, skip the scan — the claim under
            # test is "where the market quotes our probability, we don't
            # trade", not "we never trade".
            if not (0.06 <= q.prob_yes <= 0.94):
                continue
            mid = q.prob_yes * 100.0
        else:
            mid = 50.0
        book = _make_book(mid)
        best_ask = min(book["yes_asks"])
        best_bid = max(book["yes_bids"])
        q.yes_bid, q.yes_ask = float(best_bid), float(best_ask)

        if ticker in policy.positions:
            ex = policy.evaluate_exit(q)
            if ex.kind == "exit":
                pos = policy.positions[ticker]
                fill = broker.sell_ioc(
                    ticker=ticker, side=pos.side, contracts=pos.contracts,
                    limit_cents=ex.limit_cents, book=book,
                    entry_cents=pos.entry_cents, entry_fee_usd=pos.fees_usd,
                    trade_id=pos.trade_id, ts=now,
                )
                if fill.filled:
                    pnl = ((fill.avg_price_cents - pos.entry_cents)
                           * fill.contracts / 100.0 - fill.fee_usd
                           - pos.fees_usd * (fill.contracts / pos.contracts))
                    policy.record_close(ticker, pnl)
        else:
            d = policy.evaluate_entry(q, bankroll_usd=broker.cash_usd)
            if d.kind == "enter":
                fill = broker.buy_ioc(
                    ticker=ticker, side=d.side, contracts=d.contracts,
                    limit_cents=d.limit_cents, book=book, ts=now,
                )
                if fill.filled:
                    policy.record_open(Position(
                        ticker=ticker, side=d.side, contracts=fill.contracts,
                        entry_cents=fill.avg_price_cents,
                        cost_usd=abs(fill.cash_delta_usd), fees_usd=fill.fee_usd,
                        opened_ts=now, trade_id=fill.trade_id, strike=strike,
                    ))

    # Settlement: the mean of the final 60s — the real instrument.
    twap = sum(settle_prices) / len(settle_prices)
    result = "yes" if twap >= strike else "no"
    pos = policy.positions.get(ticker)
    if pos is not None:
        pnl = broker.settle(
            ticker=ticker, side=pos.side, contracts=pos.contracts,
            entry_cents=pos.entry_cents, result=result,
            entry_fee_usd=pos.fees_usd, trade_id=pos.trade_id,
            ts=t0 + WINDOW_SEC,
        )
        policy.record_close(ticker, pnl)
    return price, result


def _run_session(*, seed: int, n_markets: int, market_prices_fairly: bool,
                 drift_per_sec: float) -> tuple[PaperBroker, Policy]:
    rng = random.Random(seed)
    broker = PaperBroker(PaperBrokerConfig(starting_cash_usd=100.0))
    policy = Policy(PolicyConfig(
        min_seconds=30.0, max_seconds=840.0,
        ev_margin_cents=0.75, max_single_trade_usd=10.0,
        max_per_market_usd=10.0, max_open_positions=3,
        daily_loss_limit_usd=1_000.0,   # don't halt mid-experiment
    ))
    price = START_PRICE
    t0 = 1_000_000.0
    for i in range(n_markets):
        price, _ = _simulate_window(
            rng=rng, broker=broker, policy=policy, ticker=f"KXBTC15M-T{i}",
            start_price=price, drift_per_sec=drift_per_sec,
            vol_per_sec=0.00012, market_prices_fairly=market_prices_fairly,
            t0=t0 + i * WINDOW_SEC,
        )
    return broker, policy


# ── Tests ────────────────────────────────────────────────────────────────────

def test_cash_is_conserved_across_a_full_session():
    """The invariant that makes paper trading meaningful: final cash must
    equal start + realized P&L - fees, and the ledger must agree."""
    broker, policy = _run_session(seed=11, n_markets=12,
                                  market_prices_fairly=False,
                                  drift_per_sec=0.0000015)
    assert not policy.positions, "all positions should be settled or closed"

    # realized P&L is net of every fee, so this identity is exact.
    expected = 100.0 + broker.realized_pnl_usd
    assert abs(broker.cash_usd - expected) < 1e-6, (
        f"cash={broker.cash_usd} expected={expected}"
    )
    ledger_cash = 100.0 + sum(e.cash_delta_usd for e in broker.ledger)
    assert abs(ledger_cash - broker.cash_usd) < 1e-6, (
        f"ledger={ledger_cash} cash={broker.cash_usd}"
    )
    assert broker.ledger, "a session must produce ledger entries"


def test_every_entry_has_a_matching_close():
    broker, _ = _run_session(seed=5, n_markets=10, market_prices_fairly=False,
                             drift_per_sec=0.000002)
    entries = [e for e in broker.ledger if e.kind == "entry"]
    closes = [e for e in broker.ledger if e.kind in ("exit", "settlement")]
    assert entries, "the informed scenario must actually trade"
    assert len(closes) == len(entries)
    assert broker.wins + broker.losses == len(closes)


def test_fees_are_actually_charged():
    broker, _ = _run_session(seed=5, n_markets=10, market_prices_fairly=False,
                             drift_per_sec=0.000002)
    assert broker.fees_paid_usd > 0, "a taker session must pay fees"
    for e in broker.ledger:
        if e.kind == "entry":
            assert e.fee_usd > 0
        if e.kind == "settlement":
            assert e.fee_usd == 0.0     # Kalshi charges no settlement fee


def test_informed_model_profits_against_a_stale_market():
    """When the market sits at 50c and the model actually knows something,
    the session must make money after fees. If this fails, the EV gate or
    the sizing is broken."""
    broker, _ = _run_session(seed=5, n_markets=24, market_prices_fairly=False,
                             drift_per_sec=0.000004)
    assert broker.realized_pnl_usd > 0, broker.summary()


def test_no_trades_when_the_market_prices_the_model_exactly():
    """The honesty check. If the market quotes exactly our probability,
    the fee makes every trade -EV and we must sit out completely. The old
    paper path (fill at ask, no fees) would happily 'profit' here."""
    broker, _ = _run_session(seed=3, n_markets=12, market_prices_fairly=True,
                             drift_per_sec=0.0)
    entries = [e for e in broker.ledger if e.kind == "entry"]
    assert not entries, f"fired {len(entries)} trades with zero real edge"
    assert broker.cash_usd == 100.0
    assert broker.realized_pnl_usd == 0.0


def test_position_sizing_stays_within_caps():
    broker, _ = _run_session(seed=9, n_markets=12, market_prices_fairly=False,
                             drift_per_sec=0.000003)
    for e in broker.ledger:
        if e.kind == "entry":
            notional = e.contracts * e.price_cents / 100.0
            assert notional <= 10.0 + 1e-6, f"oversized entry: ${notional:.2f}"
    assert broker.cash_usd >= -1e-9, "cash must never go negative"


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
    b, _ = _run_session(seed=5, n_markets=24, market_prices_fairly=False,
                        drift_per_sec=0.000004)
    print(f"\nsample session: {b.summary()}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
