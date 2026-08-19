"""Re-run the decision core over a recorded session under any config.

The point of this module is that it contains **no strategy of its own**.
It imports `blended_sigma`, `quote_market`, `Policy` and `PaperBroker` —
the exact objects the live engine runs — and feeds them recorded inputs.
If replay and live disagree, that is a bug in one of them, not a
difference of models. Anything reimplemented here for speed is verified
against the original in `tests/test_replay.py`.

What replay does NOT model, and why it matters when reading results:

  * **Latency.** Decisions are evaluated against the book as recorded at
    that scan. Live, a few hundred milliseconds pass between seeing a
    price and reaching the matching engine. `paper_adverse_cents` is the
    knob that buys pessimism here; it is not free realism.
  * **Market impact.** Our fills consume recorded depth but the rest of
    the book never reacts. At $10 per trade against books quoting
    thousands of contracts this is negligible; it would not be at size.
  * **Queue position.** Every fill is a taker fill. Maker strategies
    cannot be evaluated by this harness at all.

Those three are why a replay result is a hypothesis, not a track record.
The live paper session is what promotes it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

from btc15.config import CoreConfig
from btc15.core.paper import PaperBroker, PaperBrokerConfig
from btc15.core.policy import Policy, PolicyConfig, Position
from btc15.core.pricer import SliceGrid, price_band, quote_market
from btc15.core.score import Observation
from btc15.core.sigma import SigmaConfig, blended_sigma
from btc15.research.corpus import LoadedSession

# Imported, not redefined: replay must hand the vol nowcast the same span
# of history the live engine does. A different window feeds the same
# estimator different data and silently makes this a different bot.
from btc15.core.engine_constants import TICK_HISTORY_SEC  # noqa: E402


@dataclass
class ReplayTrade:
    ticker: str
    side: str
    contracts: int
    entry_cents: float
    entry_ts: float
    secs_at_entry: float
    phase: str
    band: str
    p_model_at_entry: float
    p_market_at_entry: Optional[float]
    ev_at_entry: Optional[float]
    exit_kind: str = "settled"          # 'settled' | 'flip'
    exit_cents: Optional[float] = None
    pnl_usd: float = 0.0
    fees_usd: float = 0.0
    outcome: Optional[str] = None
    outcome_source: Optional[str] = None


@dataclass
class ReplayResult:
    session_id: str
    observations: list[Observation] = field(default_factory=list)
    trades: list[ReplayTrade] = field(default_factory=list)
    n_frames: int = 0
    n_markets: int = 0
    n_settled_markets: int = 0
    realized_pnl_usd: float = 0.0
    fees_usd: float = 0.0
    starting_cash_usd: float = 0.0
    ending_cash_usd: float = 0.0
    reject_counts: dict[str, int] = field(default_factory=dict)
    unresolved_tickers: list[str] = field(default_factory=list)
    has_depth: bool = True
    span_sec: float = 0.0
    # (ticker, ts) -> gate the policy returned on that scan. Lets a test
    # compare replay against what the live engine actually recorded, which
    # is the only real check that this harness models the same bot.
    gate_by_scan: dict = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    # KXBTC15M lists exactly one market at a time, 15 minutes each.
    MARKETS_PER_DAY = 96

    @property
    def trades_per_market(self) -> float:
        if self.n_markets <= 0:
            return 0.0
        return self.n_trades / self.n_markets

    @property
    def trades_per_day(self) -> float:
        """Projected entries per day, via entries-per-market x 96.

        Deliberately NOT trades / wall-clock-span. Spans are summed when
        sessions are merged, and a corpus assembled from overlapping or very
        short segments then double-counts time: a 44-second session with one
        trade projects to ~2000 trades/day, and the merged figure exceeded
        the hard ceiling of 96 markets/day. Counting markets is immune to
        both, because the market count is the thing that is actually
        rate-limited by the exchange.
        """
        return self.trades_per_market * self.MARKETS_PER_DAY

    @property
    def win_rate(self) -> Optional[float]:
        settled = [t for t in self.trades if t.exit_kind == "settled"]
        if not settled:
            return None
        return sum(1 for t in settled if t.pnl_usd > 0) / len(settled)


def policy_config_from(core: CoreConfig, grid: SliceGrid) -> PolicyConfig:
    return PolicyConfig(
        min_seconds=core.min_seconds,
        max_seconds=core.max_seconds,
        enabled_slices=set(core.enabled_slices or []),
        ev_margin_cents=core.ev_margin_cents,
        min_confidence=core.min_confidence,
        max_spread_cents=core.max_spread_cents,
        kelly_fraction=core.kelly_fraction,
        max_single_trade_usd=core.max_single_trade_usd,
        min_single_trade_usd=core.min_single_trade_usd,
        max_open_positions=core.max_open_positions,
        max_per_market_usd=core.max_per_market_usd,
        daily_loss_limit_usd=core.daily_loss_limit_usd,
        exit_flip_min_ev_cents=core.exit_flip_min_ev_cents,
        exit_min_seconds=core.exit_min_seconds,
        slippage_cents=core.slippage_cents,
        fee_rate=core.fee_rate,
        fee_multiplier=core.fee_multiplier,
        warmup_sec=core.warmup_sec,
        reject_clamped_sigma=core.reject_clamped_sigma,
        max_entries_per_market=core.max_entries_per_market,
        entry_cooldown_sec=core.entry_cooldown_sec,
        grid=grid,
    )


def grid_from(core: CoreConfig) -> SliceGrid:
    return SliceGrid(
        phase_early_sec=core.phase_early_sec,
        phase_mid_sec=core.phase_mid_sec,
        phase_prime_sec=core.phase_prime_sec,
        band_extreme_cents=core.band_extreme_cents,
        band_outer_cents=core.band_outer_cents,
    )


def sigma_config_from(core: CoreConfig) -> SigmaConfig:
    return SigmaConfig(
        fast_sec=core.sigma_fast_sec,
        slow_sec=core.sigma_slow_sec,
        fast_weight=core.sigma_fast_weight,
        floor=core.sigma_floor,
        ceiling=core.sigma_ceiling,
        min_samples=core.sigma_min_samples,
        scale=core.sigma_scale,
        sample_sec=core.sigma_sample_sec,
    )


def replay_session(
    sess: LoadedSession,
    core: CoreConfig,
    outcomes: dict[str, str],
    outcome_source: Optional[dict[str, str]] = None,
    *,
    trade: bool = True,
) -> ReplayResult:
    """Replay one session. `outcomes` decides settlement; markets missing
    from it are never settled and never scored."""
    outcome_source = outcome_source or {}
    grid = grid_from(core)
    scfg = sigma_config_from(core)
    policy = Policy(policy_config_from(core, grid))
    broker = PaperBroker(PaperBrokerConfig(
        starting_cash_usd=core.paper_starting_cash_usd,
        adverse_cents=core.paper_adverse_cents,
        fee_rate=core.fee_rate,
        fee_multiplier=core.fee_multiplier,
    ))

    res = ReplayResult(
        session_id=sess.session_id,
        n_frames=len(sess.frames),
        n_markets=len(sess.close_ts),
        starting_cash_usd=broker.cash_usd,
        has_depth=sess.has_depth,
        span_sec=sess.span_sec,
    )

    ticks = sess.ticks
    tick_i = 0                      # left edge of the rolling history window
    tick_j = 0                      # one past the newest tick at or before now
    open_trade: dict[str, ReplayTrade] = {}

    def settle_ticker(ticker: str, now_ts: float) -> None:
        pos = policy.positions.get(ticker)
        if pos is None:
            return
        result = outcomes.get(ticker)
        if result is None:
            # No trustworthy settlement. Abandon the position rather than
            # mark it — inventing an outcome here is exactly the bug that
            # booked settled winners as -100% losses in the old logs.
            policy.positions.pop(ticker, None)
            open_trade.pop(ticker, None)
            return
        pnl = broker.settle(
            ticker=ticker, side=pos.side, contracts=pos.contracts,
            entry_cents=pos.entry_cents, result=result,
            entry_fee_usd=pos.fees_usd, trade_id=pos.trade_id,
        )
        policy.record_close(ticker, pnl, now_ts=now_ts)
        t = open_trade.pop(ticker, None)
        if t is not None:
            t.exit_kind = "settled"
            t.exit_cents = 100.0 if result == t.side else 0.0
            t.pnl_usd = pnl
            t.outcome = result
            t.outcome_source = outcome_source.get(ticker)
        res.n_settled_markets += 1

    for frame in sess.frames:
        now_ts = frame.ts

        # Advance the rolling BRTI history to `now_ts`.
        while tick_j < len(ticks) and ticks[tick_j][0] <= now_ts:
            tick_j += 1
        cutoff = now_ts - TICK_HISTORY_SEC
        while tick_i < tick_j and ticks[tick_i][0] < cutoff:
            tick_i += 1
        window = ticks[tick_i:tick_j]

        # Settle any market whose window has closed. Only one market trades
        # at a time, but rollover means two can be in flight for a scan.
        for held in list(policy.positions):
            if now_ts >= sess.close_ts.get(held, math.inf):
                settle_ticker(held, now_ts)

        nowcast = blended_sigma(window, now_ts=now_ts, cfg=scfg)
        q = quote_market(
            ticker=frame.ticker, strike=frame.strike, spot=frame.spot,
            secs=frame.secs, sigma=nowcast.sigma, ticks=window, now_ts=now_ts,
            yes_bid=frame.yes_bid, yes_ask=frame.yes_ask, grid=grid,
            sigma_clamped=nowcast.clamped,
            tick_span_sec=(now_ts - window[0][0]) if window else 0.0,
        )

        mid = q.mid_cents
        band = price_band(mid, grid) if mid is not None else "?"
        if frame.ticker in outcomes:
            res.observations.append(Observation(
                ticker=frame.ticker, phase=q.phase, band=band,
                p_model=q.prob_yes, p_market=q.market_prob_yes,
                outcome=1 if outcomes[frame.ticker] == "yes" else 0,
                secs=frame.secs,
            ))

        if not trade:
            continue

        acted = False
        if frame.ticker in policy.positions:
            ex = policy.evaluate_exit(q)
            if ex.kind == "exit":
                pos = policy.positions[frame.ticker]
                fill = broker.sell_ioc(
                    ticker=frame.ticker, side=pos.side, contracts=pos.contracts,
                    limit_cents=ex.limit_cents, book=frame.book_dicts(),
                    entry_cents=pos.entry_cents, entry_fee_usd=pos.fees_usd,
                    trade_id=pos.trade_id,
                )
                if fill.filled:
                    pnl = (
                        (fill.avg_price_cents - pos.entry_cents) * fill.contracts / 100.0
                        - fill.fee_usd
                        - pos.fees_usd * (fill.contracts / pos.contracts)
                    )
                    policy.record_close(frame.ticker, pnl, now_ts=now_ts)
                    t = open_trade.pop(frame.ticker, None)
                    if t is not None:
                        t.exit_kind = "flip"
                        t.exit_cents = fill.avg_price_cents
                        t.pnl_usd = pnl
                        t.fees_usd += fill.fee_usd
                    acted = True

        d = policy.evaluate_entry(q, broker.cash_usd, now_ts=now_ts)
        gate = d.reject_gate or ("enter" if d.kind == "enter" else "none")
        res.reject_counts[gate] = res.reject_counts.get(gate, 0) + 1
        res.gate_by_scan[(frame.ticker, round(now_ts, 1))] = gate
        if d.kind == "enter" and not acted:
            trade_id = f"{frame.ticker}-{int(now_ts)}"
            fill = broker.buy_ioc(
                ticker=frame.ticker, side=d.side, contracts=d.contracts,
                limit_cents=d.limit_cents, book=frame.book_dicts(),
                trade_id=trade_id,
            )
            if fill.filled:
                pos = Position(
                    ticker=frame.ticker, side=d.side, contracts=fill.contracts,
                    entry_cents=fill.avg_price_cents,
                    cost_usd=abs(fill.cash_delta_usd), fees_usd=fill.fee_usd,
                    opened_ts=now_ts, trade_id=trade_id, strike=frame.strike,
                )
                policy.record_open(pos)
                t = ReplayTrade(
                    ticker=frame.ticker, side=d.side, contracts=fill.contracts,
                    entry_cents=fill.avg_price_cents, entry_ts=now_ts,
                    secs_at_entry=frame.secs, phase=q.phase, band=band,
                    p_model_at_entry=q.prob_win(d.side),
                    p_market_at_entry=q.market_prob_yes,
                    ev_at_entry=d.ev_cents, fees_usd=fill.fee_usd,
                )
                open_trade[frame.ticker] = t
                res.trades.append(t)

    # Settle whatever is still open at the end of the recording.
    last_ts = sess.frames[-1].ts if sess.frames else 0.0
    for held in list(policy.positions):
        settle_ticker(held, last_ts)

    res.realized_pnl_usd = broker.realized_pnl_usd
    res.fees_usd = broker.fees_paid_usd
    res.ending_cash_usd = broker.cash_usd
    res.unresolved_tickers = sorted(set(sess.close_ts) - set(outcomes))
    return res


def merge_results(results: list[ReplayResult], label: str = "merged") -> ReplayResult:
    """Pool several sessions into one result. Spans add, so trades_per_day
    stays meaningful across a corpus assembled from many short runs."""
    out = ReplayResult(session_id=label)
    for r in results:
        out.observations.extend(r.observations)
        out.trades.extend(r.trades)
        out.n_frames += r.n_frames
        out.n_markets += r.n_markets
        out.n_settled_markets += r.n_settled_markets
        out.realized_pnl_usd += r.realized_pnl_usd
        out.fees_usd += r.fees_usd
        out.starting_cash_usd += r.starting_cash_usd
        out.ending_cash_usd += r.ending_cash_usd
        out.span_sec += r.span_sec
        out.unresolved_tickers.extend(r.unresolved_tickers)
        out.has_depth = out.has_depth and r.has_depth
        for k, v in r.reject_counts.items():
            out.reject_counts[k] = out.reject_counts.get(k, 0) + v
        out.gate_by_scan.update(r.gate_by_scan)
    return out


def with_overrides(core: CoreConfig, overrides: dict) -> CoreConfig:
    """A copy of `core` with `overrides` applied. Used by the sweep, which
    must never mutate a shared config object across worker processes."""
    unknown = [k for k in overrides if not hasattr(core, k)]
    if unknown:
        raise ValueError(f"unknown CoreConfig fields: {unknown}")
    return replace(core, **overrides)
