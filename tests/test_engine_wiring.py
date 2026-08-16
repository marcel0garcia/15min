"""Engine wiring smoke test — no network, no Kalshi account.

Substitutes a fake Kalshi client and drives CoreEngine._scan() +
_check_settlements() directly, proving the real production code path
(not a reimplementation) does the full loop: quote -> decide -> paper
fill -> dashboard state -> decisions.jsonl -> settlement.

This is the test that catches wiring rot — a renamed state key, a
changed client signature, a decision row that stops carrying p_market.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from btc15.config import AppConfig
from btc15.core.engine import CoreEngine
from btc15.kalshi.models import Market, MarketStatus, Orderbook


CLOSE_IN = 300.0          # seconds until settlement
STRIKE = 100_000.0
SPOT = 100_600.0          # comfortably above the strike -> model likes YES


def _market(ticker="KXBTC15M-TEST", secs=CLOSE_IN, status=MarketStatus.OPEN,
            result=None) -> Market:
    return Market(
        ticker=ticker, series_ticker="KXBTC15M", title="test",
        status=status, yes_bid=50.0, yes_ask=52.0, no_bid=48.0, no_ask=50.0,
        last_price=51.0, volume=1000, open_interest=500, strike_price=STRIKE,
        close_time=datetime.now(timezone.utc) + timedelta(seconds=secs),
        result=result,
    )


class FakeKalshi:
    """Minimal stand-in exposing only what CoreEngine calls."""

    def __init__(self):
        self.market = _market()
        self.orders: list = []

    async def get_markets(self, **kw):
        return [self.market]

    async def get_market(self, ticker):
        return self.market

    async def get_orderbook(self, ticker, depth=10):
        return Orderbook(
            ticker=ticker,
            yes_bids=[(50.0, 200), (49.0, 300)],
            yes_asks=[(52.0, 200), (53.0, 300)],
        )


def _config(tmp: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.strategy.paper_trade = True
    cfg.strategy.auto_trade = True
    cfg.recording.enabled = True
    cfg.recording.path = str(tmp / "recordings")
    cfg.logging.log_file = str(tmp / "bot.log")
    cfg.logging.trade_log_file = str(tmp / "trades.csv")
    cfg.core.paper_starting_cash_usd = 100.0
    cfg.core.max_seconds = 900.0
    Path(cfg.recording.path).mkdir(parents=True, exist_ok=True)
    return cfg


async def _seed_engine(engine: CoreEngine, fake: FakeKalshi) -> None:
    """Feed the engine a price history and a live orderbook the way the
    real feeds would."""
    now = asyncio.get_event_loop().time()
    del now
    import time as _t
    t = _t.time()
    for i in range(400, 0, -1):
        await engine.price_feed.push_brti(SPOT + ((i % 5) - 2) * 3.0, t - i)
    await engine._market_cache.apply_snapshot(
        fake.market.ticker, await fake.get_orderbook(fake.market.ticker)
    )


async def _run() -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="btc15-wiring-"))
    try:
        cfg = _config(tmp)
        engine = CoreEngine(cfg)
        fake = FakeKalshi()
        engine._kalshi = fake
        engine._ws = type("WS", (), {"subscribe": lambda *a, **k: _noop()})()
        await _seed_engine(engine, fake)

        # ── One scan: quote -> decide -> paper fill ──────────────────────
        await engine._scan()

        out: dict = {
            "signals": dict(engine.state["signals"]),
            "markets": list(engine.state["open_markets"]),
            "sigma": engine.state["sigma_nowcast"],
            "fair_value": engine.state["fair_value"],
            "positions_after_scan": len(engine.policy.positions),
            "cash_after_entry": engine.broker.cash_usd,
        }

        # ── Settlement: market closes YES, we should be paid ─────────────
        fake.market = _market(secs=-1, status=MarketStatus.SETTLED, result="yes")
        await engine._check_settlements()
        out["positions_after_settle"] = len(engine.policy.positions)
        out["cash_after_settle"] = engine.broker.cash_usd
        out["realized"] = engine.broker.realized_pnl_usd
        out["wins"] = engine.broker.wins

        # ── Dashboard state loop populates panels ────────────────────────
        engine.running = True
        task = asyncio.create_task(engine._state_loop())
        await asyncio.sleep(1.2)
        engine.running = False
        task.cancel()
        out["risk_panel"] = dict(engine.state["risk"])
        out["balance_panel"] = dict(engine.state["balance"] or {})

        # ── Decision log ─────────────────────────────────────────────────
        engine._recorder.close()
        decisions = list(
            (Path(cfg.recording.path)).glob("*/decisions.jsonl")
        )
        rows = []
        if decisions:
            with open(decisions[0]) as f:
                rows = [json.loads(l) for l in f if l.strip()]
        out["decision_rows"] = rows
        out["trade_log"] = Path(cfg.logging.trade_log_file).read_text().splitlines()
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def _noop():
    return None


RESULT = asyncio.run(_run())


# ── Tests ────────────────────────────────────────────────────────────────────

def test_scan_produces_signals_for_the_dashboard():
    sigs = RESULT["signals"]
    assert sigs, "Signals panel would be empty"
    s = next(iter(sigs.values()))
    # Keys the existing Rich dashboard reads.
    for key in ("strike", "seconds_left", "prob_yes", "fv_prob_yes",
                "fv_confidence", "confidence", "signal", "fv_signal"):
        assert key in s, f"dashboard key missing: {key}"
    assert RESULT["markets"], "Open Markets panel would be empty"
    assert RESULT["sigma"] and RESULT["sigma"] > 0


def test_model_leans_yes_when_spot_is_above_strike():
    s = next(iter(RESULT["signals"].values()))
    assert s["prob_yes"] > 0.5, s["prob_yes"]
    assert "YES" in s["signal"]


def test_entry_fired_and_cash_left_the_account():
    assert RESULT["positions_after_scan"] == 1, "policy did not open a position"
    assert RESULT["cash_after_entry"] < 100.0, "cash did not move on entry"


def test_settlement_closes_the_position_and_pays_out():
    assert RESULT["positions_after_settle"] == 0, "position survived settlement"
    assert RESULT["wins"] == 1
    assert RESULT["realized"] > 0, "a winning YES settle must be profitable"
    # cash == start + realized (net-of-fees convention)
    assert abs(RESULT["cash_after_settle"] - (100.0 + RESULT["realized"])) < 1e-6


def test_dashboard_panels_populate():
    risk = RESULT["risk_panel"]
    for key in ("session_pnl", "session_trades", "open_positions",
                "win_rate", "halted", "fees_paid"):
        assert key in risk, f"risk panel key missing: {key}"
    assert "available" in RESULT["balance_panel"]


def test_decision_log_carries_model_and_market_probabilities():
    """Without p_market on every row, core/score.py cannot measure us
    against the market — which is the entire R1 experiment."""
    rows = RESULT["decision_rows"]
    assert rows, "no decisions.jsonl rows written"
    r = rows[0]
    for key in ("ticker", "secs", "phase", "band", "p_model", "p_market",
                "sigma", "action", "reject_gate", "ev_cents"):
        assert key in r, f"decision row missing: {key}"
    assert r["p_model"] is not None
    assert r["p_market"] is not None


def test_trade_log_csv_written_for_history_command():
    lines = RESULT["trade_log"]
    assert len(lines) >= 3, lines        # header + entry + settlement
    assert lines[0].startswith("trade_id,")
    assert any("_settled" in l for l in lines)


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
