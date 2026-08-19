#!/usr/bin/env python3
"""Tests for the offline research harness.

The harness only earns trust if replaying a recording reproduces what the
live engine did with it. `test_replay_agrees_with_the_live_run` measures
that directly against real recordings when a corpus exists, and reports
the agreement rate rather than asserting a number nobody validated.
"""
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from btc15.config import CoreConfig
from btc15.research.corpus import (
    discover_sessions, load_results_cache, load_session, resolve_outcomes,
    settlement_twap,
)
from btc15.research.replay import merge_results, replay_session, with_overrides
from btc15.research.sweep import expand_grid, pareto_frontier, parse_knob, SweepRow

PASS, FAIL = [], []


def test(fn):
    try:
        fn()
        PASS.append(fn.__name__)
        print(f"ok    {fn.__name__}")
    except AssertionError as e:
        FAIL.append((fn.__name__, str(e)))
        print(f"FAIL  {fn.__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        FAIL.append((fn.__name__, repr(e)))
        print(f"ERROR {fn.__name__}: {e!r}")


# ── Fixtures ─────────────────────────────────────────────────────────────────

def write_session(tmp: Path, *, n_scans=900, drift=1.0, strike=100_000.0,
                  bid=60.0, ask=61.0, with_book=True, t0=1_700_000_000.0,
                  wiggle=25.0):
    """A synthetic recording: one market, one scan per second, drift + noise.

    The noise is not decoration. A perfectly linear path has ~zero realized
    variance, sigma pins to its floor, and `reject_clamped_sigma` correctly
    refuses every entry — so a noiseless fixture silently tests nothing.
    Deterministic (fixed seed) so replays stay reproducible.
    """
    import random as _random
    rng = _random.Random(20260819)
    d = tmp / "SYNTH_1"
    d.mkdir(parents=True, exist_ok=True)
    close_ts = t0 + n_scans
    with open(d / "decisions.jsonl", "w") as f:
        for i in range(n_scans):
            ts = t0 + i
            spot = strike + drift * i + rng.gauss(0.0, wiggle)
            row = {
                "ts": ts, "ticker": "SYNTH-1", "strike": strike,
                "spot": round(spot, 2), "secs": round(close_ts - ts, 1),
                "phase": "mid", "band": "middle",
                "p_model": 0.5, "p_market": (bid + ask) / 200.0,
                "yes_bid": bid, "yes_ask": ask, "sigma": 0.4, "z": 0.0,
                "degenerate": False, "action": "none", "reject_gate": None,
            }
            if with_book:
                row["book"] = {
                    "yes_bids": [[bid, 500.0], [bid - 1, 500.0]],
                    "yes_asks": [[ask, 500.0], [ask + 1, 500.0]],
                }
            f.write(json.dumps(row) + "\n")
    (d / "meta.json").write_text(json.dumps({"session_id": "SYNTH_1", "mode": "paper"}))
    return d


# ── Corpus loading ───────────────────────────────────────────────────────────

def test_loads_frames_ticks_and_close_time():
    with tempfile.TemporaryDirectory() as td:
        sd = write_session(Path(td), n_scans=300)
        s = load_session(sd)
        assert len(s.frames) == 300, len(s.frames)
        assert s.n_skipped == 0
        assert s.has_depth is True
        assert len(s.close_ts) == 1
        # close_ts is ts + secs, constant across every row of the market.
        close = s.close_ts["SYNTH-1"]
        assert abs(close - (s.frames[0].ts + s.frames[0].secs)) < 1e-6
        # Ticks are the BRTI series, ascending, deduped.
        assert s.ticks == sorted(s.ticks)
        assert len(s.ticks) == 300


def test_pre_v3_rows_are_skipped_not_guessed():
    """Rows without spot/strike cannot be re-priced. They must be dropped and
    counted, never defaulted — a guessed strike is a fabricated backtest."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "OLD"
        d.mkdir()
        with open(d / "decisions.jsonl", "w") as f:
            for i in range(50):
                f.write(json.dumps({
                    "ts": 1_700_000_000.0 + i, "ticker": "OLD-1",
                    "secs_remaining": 500 - i, "yes_bid": 40, "yes_ask": 41,
                    "prob_yes": 0.5,
                }) + "\n")
        s = load_session(d)
        assert s.n_rows == 50
        assert s.n_skipped == 50
        assert s.frames == []
        assert discover_sessions(Path(td)) == []


def test_twap_refuses_to_call_a_near_strike_settlement():
    """Measured 2026-08-19: our 3-venue BRTI reconstruction matched Kalshi on
    15 of 16 markets, and the one miss settled $1.94 from the strike. Inside
    that band a synthesized outcome is a coin flip, and a coin flip used as
    a label teaches the sweep noise with the authority of data."""
    with tempfile.TemporaryDirectory() as td:
        # No drift and tiny noise -> the final TWAP sits right on the strike.
        sd = write_session(Path(td), n_scans=300, drift=0.0, wiggle=1.0)
        s = load_session(sd)
        near, _ = resolve_outcomes(s, {})
        assert near == {}, "a settlement inside the margin must stay unresolved"
        # Kalshi's own answer is still accepted there — it is authoritative.
        off, src = resolve_outcomes(s, {"SYNTH-1": "yes"})
        assert off["SYNTH-1"] == "yes" and src["SYNTH-1"] == "official"
    with tempfile.TemporaryDirectory() as td:
        # A clearly-resolved market is still called.
        sd2 = write_session(Path(td), n_scans=300, drift=2.0, wiggle=1.0)
        far, src2 = resolve_outcomes(load_session(sd2), {})
        assert far.get("SYNTH-1") == "yes"
        assert src2["SYNTH-1"] == "twap"


def test_settlement_twap_refuses_a_thin_window():
    ticks = [(1000.0 + i, 100.0 + i) for i in range(200)]
    close = 1150.0
    assert settlement_twap(ticks, close) is not None
    # Only a couple of samples in the final minute -> no synthesized outcome.
    sparse = [(1089.0, 100.0), (1090.0, 101.0)]
    assert settlement_twap(sparse, close) is None


def test_official_outcome_beats_twap_fallback():
    with tempfile.TemporaryDirectory() as td:
        sd = write_session(Path(td), n_scans=300, drift=1.0)   # ends far above strike
        s = load_session(sd)
        twap_only, src = resolve_outcomes(s, {})
        assert src.get("SYNTH-1") == "twap"
        assert twap_only["SYNTH-1"] == "yes"
        # Kalshi disagreeing is authoritative; we do not second-guess it.
        official, src2 = resolve_outcomes(s, {"SYNTH-1": "no"})
        assert official["SYNTH-1"] == "no"
        assert src2["SYNTH-1"] == "official"
        # And with the fallback disabled, an unknown market resolves to nothing.
        none_, _ = resolve_outcomes(s, {}, allow_twap_fallback=False)
        assert none_ == {}


# ── Replay behaviour ─────────────────────────────────────────────────────────

def test_replay_never_invents_a_settlement():
    """The 2026-06-06 post-mortem class of bug: a held position resolved from
    something other than a real settlement. With no outcome, the position is
    abandoned and no P&L is booked."""
    with tempfile.TemporaryDirectory() as td:
        sd = write_session(Path(td), n_scans=900, drift=2.0, bid=40, ask=41)
        s = load_session(sd)
        core = replace(CoreConfig(), warmup_sec=60.0)
        r = replay_session(s, core, outcomes={}, outcome_source={})
        assert r.n_settled_markets == 0
        assert r.realized_pnl_usd == 0.0
        assert r.unresolved_tickers == ["SYNTH-1"]


def test_replay_is_deterministic():
    with tempfile.TemporaryDirectory() as td:
        sd = write_session(Path(td), n_scans=900, drift=2.0, bid=40, ask=41)
        s = load_session(sd)
        core = replace(CoreConfig(), warmup_sec=60.0)
        outc, src = resolve_outcomes(s, {})
        a = replay_session(s, core, outc, src)
        b = replay_session(s, core, outc, src)
        assert a.n_trades == b.n_trades
        assert abs(a.realized_pnl_usd - b.realized_pnl_usd) < 1e-9
        assert a.reject_counts == b.reject_counts


def test_config_changes_actually_change_the_replay():
    """A sweep is worthless if the knobs do not reach the decision. A high EV
    margin must trade strictly less than a low one, all else equal."""
    with tempfile.TemporaryDirectory() as td:
        sd = write_session(Path(td), n_scans=900, drift=2.0, bid=40, ask=41)
        s = load_session(sd)
        outc, src = resolve_outcomes(s, {})
        base = replace(CoreConfig(), warmup_sec=60.0, max_entries_per_market=50)

        def run(margin):
            return replay_session(
                s, with_overrides(base, {"ev_margin_cents": margin}), outc, src)

        loose, strict, absurd = run(0.1), run(20.0), run(500.0)

        # A margin nothing can clear must produce no trade at all...
        assert absurd.n_trades == 0, absurd.n_trades
        assert absurd.reject_counts.get("ev_margin", 0) > 0
        # ...a permissive one must trade...
        assert loose.n_trades >= 1, loose.n_trades
        # ...and a stricter margin must wait longer for the edge to appear.
        # Trade COUNT alone cannot show this: once a position is open the
        # `already_positioned` gate masks the EV gate, so both configs enter
        # exactly once. Entry timing is what actually moves.
        assert strict.n_trades >= 1
        assert strict.trades[0].secs_at_entry < loose.trades[0].secs_at_entry, (
            strict.trades[0].secs_at_entry, loose.trades[0].secs_at_entry
        )
        assert strict.reject_counts.get("ev_margin", 0) > loose.reject_counts.get("ev_margin", 0)


def test_entry_budget_controls_trade_count():
    """max_entries_per_market is the only lever on frequency, because Kalshi
    lists exactly one market at a time."""
    with tempfile.TemporaryDirectory() as td:
        sd = write_session(Path(td), n_scans=900, drift=2.0, bid=40, ask=41)
        s = load_session(sd)
        outc, src = resolve_outcomes(s, {})
        base = replace(CoreConfig(), warmup_sec=60.0, ev_margin_cents=0.1,
                       exit_flip_min_ev_cents=1.0)
        one = replay_session(s, with_overrides(base, {"max_entries_per_market": 1}), outc, src)
        many = replay_session(s, with_overrides(base, {"max_entries_per_market": 20}), outc, src)
        assert one.n_trades <= 1
        assert many.n_trades >= one.n_trades


def test_missing_depth_is_flagged():
    """Depth-less recordings cannot show slippage, so results must say so."""
    with tempfile.TemporaryDirectory() as td:
        sd = write_session(Path(td), n_scans=200, with_book=False)
        s = load_session(sd)
        assert s.has_depth is False
        r = replay_session(s, CoreConfig(), {}, {})
        assert r.has_depth is False


def test_with_overrides_rejects_unknown_fields():
    try:
        with_overrides(CoreConfig(), {"not_a_real_knob": 1})
    except ValueError as e:
        assert "not_a_real_knob" in str(e)
    else:
        raise AssertionError("unknown knob must raise")


def test_merge_pools_spans_so_frequency_stays_honest():
    with tempfile.TemporaryDirectory() as td:
        sd = write_session(Path(td), n_scans=900, drift=2.0, bid=40, ask=41)
        s = load_session(sd)
        outc, src = resolve_outcomes(s, {})
        core = replace(CoreConfig(), warmup_sec=60.0)
        r = replay_session(s, core, outc, src)
        m = merge_results([r, r])
        assert m.n_trades == 2 * r.n_trades
        assert abs(m.span_sec - 2 * r.span_sec) < 1e-6
        # Doubling markets and trades together must leave the rate unchanged.
        if r.trades_per_day > 0:
            assert abs(m.trades_per_day - r.trades_per_day) < 1e-6
        # And the projection can never exceed one entry per market per day,
        # times the exchange's hard ceiling of 96 markets.
        assert m.trades_per_day <= 96 * m.n_markets


# ── Sweep plumbing ───────────────────────────────────────────────────────────

def test_parse_knob_types_follow_the_field():
    name, vals = parse_knob("core.sigma_floor=0.2,0.35")
    assert name == "sigma_floor" and vals == [0.2, 0.35]
    name, vals = parse_knob("max_entries_per_market=1,3")
    assert name == "max_entries_per_market" and vals == [1, 3]
    assert all(isinstance(v, int) for v in vals)
    name, vals = parse_knob("reject_clamped_sigma=true,false")
    assert vals == [True, False]
    try:
        parse_knob("nonexistent_knob=1")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown knob must raise")


def test_expand_grid_is_the_full_product():
    g = expand_grid({"a": [1, 2], "b": [3, 4, 5]})
    assert len(g) == 6
    assert {tuple(sorted(d.items())) for d in g} == {
        (("a", a), ("b", b)) for a in (1, 2) for b in (3, 4, 5)
    }
    assert expand_grid({}) == [{}]


def _row(delta, tpd, label_key="k", label_val=0):
    return SweepRow(
        overrides={label_key: label_val}, n_markets=50, n_settled=50, n_obs=500,
        n_trades=10, trades_per_day=tpd, pnl_usd=0.0, fees_usd=0.0,
        win_rate=None, brier_model=0.1, brier_market=0.11, delta=delta,
        delta_ci_low=delta - 0.01, delta_ci_high=delta + 0.01,
        verdict="inconclusive",
    )


def test_pareto_frontier_keeps_the_frequency_tradeoff_visible():
    """Optimizing edge alone hides the operating point. A config with less
    edge but far more trades is not dominated and must survive."""
    rows = [
        _row(0.05, 2.0, "a", 1),     # best edge, barely trades
        _row(0.03, 40.0, "b", 2),    # less edge, trades a lot -> on frontier
        _row(0.02, 1.0, "c", 3),     # dominated by both
    ]
    front = pareto_frontier(rows)
    kept = [r.overrides for r in front]
    assert len(front) == 2, kept
    assert {"a": 1} in kept
    assert {"b": 2} in kept
    assert {"c": 3} not in kept


# ── Fidelity against real recordings ─────────────────────────────────────────

def test_replay_agrees_with_the_live_run():
    """Replay must reproduce the live engine's own decisions on its own data.

    Perfect agreement is not expected and would be suspicious: the live
    engine feeds sigma a 4 Hz BRTI stream while a recording samples spot
    once per scan, so marginal scans can land on either side of a gate.
    What must hold is agreement on the overwhelming majority of scans —
    systematic divergence means this harness is tuning a different bot than
    the one that will trade.

    Only sessions recorded at the CURRENT git commit are asserted on.
    Comparing against a recording made by an older policy measures the
    diff between two versions of the bot, not the fidelity of replay.
    """
    from btc15.research.corpus import current_core_fingerprint
    root = Path(__file__).parent.parent / "data" / "recordings"
    head = current_core_fingerprint()

    from btc15.config import load_config
    core = load_config().core

    same_commit, checked = [], []
    for sd in discover_sessions(root):
        meta_path = sd / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        recorded = {}
        for line in open(sd / "decisions.jsonl"):
            r = json.loads(line)
            g = r.get("reject_gate")
            if g is None and r.get("action") != "enter":
                continue
            recorded[(r["ticker"], round(float(r["ts"]), 1))] = g or "enter"
        if not recorded:
            continue
        checked.append(sd)
        if meta.get("core_fingerprint") == head:
            same_commit.append((sd, recorded))

    if not same_commit:
        print(f"      (skipped — no session recorded by core {head}; "
              f"{len(checked)} session(s) from other brains present)")
        return

    total = agree = 0
    disagreements = {}
    for sd, recorded in same_commit:
        s_ = load_session(sd)
        outc, src = resolve_outcomes(
            s_, load_results_cache(root.parent / "market_results_cache.json"))
        r = replay_session(s_, core, outc, src)
        for key, live_gate in recorded.items():
            rep_gate = r.gate_by_scan.get(key)
            if rep_gate is None:
                continue
            total += 1
            if rep_gate == live_gate:
                agree += 1
            else:
                disagreements[(live_gate, rep_gate)] = (
                    disagreements.get((live_gate, rep_gate), 0) + 1)

    if not total:
        print("      (skipped — no overlapping scans)")
        return
    rate = agree / total
    print(f"      row-level gate agreement: {agree}/{total} ({rate:.1%}) "
          f"over {len(same_commit)} session(s)")
    if disagreements:
        top = sorted(disagreements.items(), key=lambda kv: -kv[1])[:4]
        for (live_g, rep_g), n in top:
            print(f"        live={live_g} replay={rep_g}  x{n}")
    assert rate >= 0.90, (
        f"replay reproduced only {rate:.1%} of the live engine's gate decisions; "
        "the harness is not modelling the same bot"
    )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            test(fn)
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    if FAIL:
        for n, e in FAIL:
            print(f"  {n}: {e}")
        sys.exit(1)
