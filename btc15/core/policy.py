"""The entire decision policy. Four ideas, no archaeology.

The legacy stack accumulated ~15 interacting gates, most of them
calibrated to a brain that no longer runs. This module is the deliberate
replacement, and the rule for extending it is: **a gate earns its place
only by measurably beating the market on recorded data** (see
btc15/core/score.py). Anecdotes from one session are not evidence.

The four ideas:

  1. WINDOW   — trade only inside [min_secs, max_secs]. Structural: a
                market that just opened has no accrued information, and
                inside the last few seconds we cannot get filled.

  2. SLICE    — trade only (phase x price-band) slices that are enabled.
                This is the R2 knob: R1 measures which slices beat the
                market mid, and only those get switched on. Default is
                all-enabled in paper so R1 can observe everything.

  3. EV GATE  — enter only when expected value per contract, AFTER the
                taker fee and a required margin, is positive:

                    ev_cents = p*(100 - price) - (1-p)*price - fee(price)
                    require   ev_cents >= margin_cents

                This replaces the flat "edge >= 5%" gate. It is
                automatically strict in the middle of the price range
                (where the fee peaks at 1.75c) and permissive at the
                extremes (0.33c) — which is exactly the structural shape
                of where edge lives in this market.

  4. SIZE     — fractional Kelly on the model's own probability, then
                hard caps (per-trade, per-market, open positions, daily
                loss). Kelly is computed on the fee-adjusted payoff so
                sizing and the EV gate agree about the economics.

Exits: ONE rule, and it is deliberately minimal. A binary option pays
0 or 100 at settlement; selling early is only right when the model has
genuinely flipped AND there is enough time left for that flip to matter.
Everything else holds to settlement. The legacy loss-cut tiers are gone:
the June 6 post-mortem showed they converted winners into realized losses
against a drained book.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from btc15.core.fees import (
    DEFAULT_MULTIPLIER, TAKER_RATE, taker_fee_cents_per_contract,
)
from btc15.core.pricer import DEFAULT_GRID, MarketQuote, SliceGrid, price_band

log = logging.getLogger(__name__)


# ── Decisions ────────────────────────────────────────────────────────────────

@dataclass
class Decision:
    """What the policy wants done. `kind` is one of:
    'enter' | 'exit' | 'none'."""
    kind: str
    ticker: str
    side: Optional[str] = None
    contracts: int = 0
    limit_cents: Optional[int] = None
    reason: str = ""
    # Diagnostics — always populated, always logged, even for 'none'.
    ev_cents: Optional[float] = None
    edge: Optional[float] = None
    kelly_fraction: Optional[float] = None
    reject_gate: Optional[str] = None


@dataclass
class Position:
    ticker: str
    side: str
    contracts: int
    entry_cents: float          # average fill price, cents
    cost_usd: float             # cash out the door incl. fees
    fees_usd: float
    opened_ts: float
    trade_id: str
    strike: float = 0.0

    def mark_to_market_usd(self, bid_cents: Optional[float]) -> float:
        if bid_cents is None:
            return 0.0
        return (bid_cents - self.entry_cents) * self.contracts / 100.0


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class PolicyConfig:
    # 1. Window
    min_seconds: float = 30.0
    max_seconds: float = 840.0

    # 2. Slices — "phase:band" keys. Empty set == all slices enabled.
    enabled_slices: set = field(default_factory=set)

    # 3. EV gate
    ev_margin_cents: float = 0.75      # required EV per contract after fees
    min_confidence: float = 0.04       # model must have *some* directional view
    max_spread_cents: float = 8.0      # skip books too wide to price honestly

    # 4. Sizing
    kelly_fraction: float = 0.25
    max_single_trade_usd: float = 10.0
    min_single_trade_usd: float = 1.0
    max_open_positions: int = 3
    max_per_market_usd: float = 10.0
    daily_loss_limit_usd: float = 50.0

    # Exit
    exit_flip_min_ev_cents: float = 2.0   # model must want the OTHER side this much
    exit_min_seconds: float = 120.0       # ...and this much time must remain

    # Input-sanity guards on the vol nowcast. NOT strategy gates — these
    # exist because of an observed failure, not a hunch. On 2026-08-19 the
    # engine fired 0.2s after startup: sigma was pinned at its 0.20 floor
    # (no BRTI history yet), the model priced 0.996 against a market at
    # 0.905, it bought 10 contracts at 91c, and four seconds later — once
    # real ticks arrived and sigma rose — the same model wanted the other
    # side by 13.7c and flipped out for a loss. Both legs were sigma
    # instability. Neither was information.
    warmup_sec: float = 120.0          # BRTI history required before entering
    reject_clamped_sigma: bool = True  # never enter while sigma is at a clamp

    # Entry cadence. KXBTC15M lists exactly one market at a time, so
    # max_open_positions can never bind and one-shot-per-window is the
    # real constraint on trade count. max_entries_per_market = 1 is the
    # original hold-to-settle behavior; > 1 permits re-entry after an exit
    # (and, with it, scaling into a market as the model sharpens late).
    max_entries_per_market: int = 1
    entry_cooldown_sec: float = 0.0

    # Slice grid — the (phase x band) boundaries the slice gate reads.
    grid: SliceGrid = field(default_factory=lambda: DEFAULT_GRID)

    # Execution
    slippage_cents: int = 1               # limit padding on IOC entries

    # Fee model — see the rate-uncertainty note in core/fees.py. If Kalshi
    # prices crypto above the standard rate, everything below is mispriced,
    # so FeeCalibrator verifies these against real fills at runtime.
    fee_rate: float = TAKER_RATE
    fee_multiplier: float = DEFAULT_MULTIPLIER


def slice_key(phase: str, band: str) -> str:
    return f"{phase}:{band}"


# ── EV / Kelly math ──────────────────────────────────────────────────────────

def ev_cents_per_contract(
    prob_win: float,
    price_cents: float,
    fee_rate: float = TAKER_RATE,
    fee_multiplier: float = DEFAULT_MULTIPLIER,
) -> float:
    """Expected value in cents of buying one contract at `price_cents`,
    net of the taker fee. Payout is 100 on a win, 0 on a loss."""
    fee = taker_fee_cents_per_contract(price_cents, fee_rate, fee_multiplier)
    return prob_win * (100.0 - price_cents) - (1.0 - prob_win) * price_cents - fee


def kelly_fraction(
    prob_win: float,
    price_cents: float,
    fee_rate: float = TAKER_RATE,
    fee_multiplier: float = DEFAULT_MULTIPLIER,
) -> float:
    """Full-Kelly fraction of bankroll, computed on the fee-adjusted
    payoff so sizing agrees with the EV gate. Returns 0 when the bet is
    not +EV after fees."""
    if not (0.0 < price_cents < 100.0) or not (0.0 < prob_win < 1.0):
        return 0.0
    fee = taker_fee_cents_per_contract(price_cents, fee_rate, fee_multiplier)
    net_win = (100.0 - price_cents) - fee     # cents gained on a win
    loss = price_cents + fee                  # cents lost on a loss
    if net_win <= 0 or loss <= 0:
        return 0.0
    b = net_win / loss                        # net odds per unit risked
    q = 1.0 - prob_win
    f = (b * prob_win - q) / b
    return max(0.0, f)


# ── The policy ───────────────────────────────────────────────────────────────

class Policy:
    """Stateless w.r.t. market data; holds only the book of positions and
    the session's realized P&L (for the daily-loss guard)."""

    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self.positions: dict[str, Position] = {}
        self.realized_pnl_usd: float = 0.0
        self.halted: bool = False
        self.halt_reason: str = ""
        # Per-market entry budget and cooldown clock. Keyed by ticker and
        # never pruned during a session — a 15-minute market is gone long
        # before the dict is worth trimming.
        self.entries_by_ticker: dict[str, int] = {}
        self.last_close_ts: dict[str, float] = {}

    # ── Entries ──────────────────────────────────────────────────────────────

    def evaluate_entry(
        self, q: MarketQuote, bankroll_usd: float, now_ts: float | None = None,
    ) -> Decision:
        """One market, one scan. Returns an 'enter' or 'none' Decision;
        the reject_gate field always says which test failed.

        Pure with respect to market data: calling it never mutates the
        policy. That is what lets the engine shadow-evaluate on every scan
        even when auto_trade is off, so `reject_gate` telemetry exists in
        signal-only sessions instead of being uniformly null.
        """
        d = Decision(kind="none", ticker=q.ticker)
        if now_ts is None:
            now_ts = time.time()

        if self.halted:
            d.reject_gate = "halted"
            return d
        if q.degenerate:
            d.reject_gate = "degenerate_quote"
            return d
        if q.crossed:
            # Not a strategy gate — a data-sanity guard. See MarketQuote.crossed.
            d.reject_gate = "crossed_book"
            return d
        if self.cfg.warmup_sec > 0 and q.tick_span_sec < self.cfg.warmup_sec:
            d.reject_gate = "warmup"
            return d
        if self.cfg.reject_clamped_sigma and q.sigma_clamped:
            d.reject_gate = "sigma_clamped"
            return d
        # 1. WINDOW
        if not (self.cfg.min_seconds <= q.secs <= self.cfg.max_seconds):
            d.reject_gate = "window"
            return d
        if q.ticker in self.positions:
            d.reject_gate = "already_positioned"
            return d
        if self.entries_by_ticker.get(q.ticker, 0) >= self.cfg.max_entries_per_market:
            d.reject_gate = "max_entries_per_market"
            return d
        if self.cfg.entry_cooldown_sec > 0:
            last = self.last_close_ts.get(q.ticker)
            if last is not None and (now_ts - last) < self.cfg.entry_cooldown_sec:
                d.reject_gate = "entry_cooldown"
                return d
        if len(self.positions) >= self.cfg.max_open_positions:
            d.reject_gate = "max_open_positions"
            return d

        side = q.recommended_side
        if side is None:
            d.reject_gate = "no_side"
            return d
        if q.confidence < self.cfg.min_confidence:
            d.reject_gate = "min_confidence"
            return d

        price = q.exec_ask_cents(side)
        if price is None:
            d.reject_gate = "no_book"
            return d
        if not (1.0 <= price <= 99.0):
            d.reject_gate = "price_out_of_range"
            return d
        if q.yes_bid is not None and q.yes_ask is not None:
            spread = q.yes_ask - q.yes_bid
            if spread > self.cfg.max_spread_cents:
                d.reject_gate = "spread_too_wide"
                return d

        # 2. SLICE
        band = price_band(price, self.cfg.grid)
        key = slice_key(q.phase, band)
        if self.cfg.enabled_slices and key not in self.cfg.enabled_slices:
            d.reject_gate = f"slice_disabled[{key}]"
            return d

        # 3. EV GATE
        p = q.prob_win(side)
        entry_price = min(99.0, price + self.cfg.slippage_cents)
        ev = ev_cents_per_contract(
            p, entry_price, self.cfg.fee_rate, self.cfg.fee_multiplier,
        )
        d.ev_cents = round(ev, 3)
        mkt_p = q.market_prob_yes
        if mkt_p is not None:
            d.edge = round((p - mkt_p) if side == "yes" else (p - (1.0 - mkt_p)), 4)
        if ev < self.cfg.ev_margin_cents:
            d.reject_gate = "ev_margin"
            return d

        # 4. SIZE
        f = kelly_fraction(
            p, entry_price, self.cfg.fee_rate, self.cfg.fee_multiplier,
        ) * self.cfg.kelly_fraction
        d.kelly_fraction = round(f, 4)
        if f <= 0:
            d.reject_gate = "kelly_zero"
            return d
        dollars = min(
            f * bankroll_usd,
            self.cfg.max_single_trade_usd,
            self.cfg.max_per_market_usd,
            max(0.0, bankroll_usd),
        )
        if dollars < self.cfg.min_single_trade_usd:
            d.reject_gate = "below_min_size"
            return d
        contracts = int(dollars / (entry_price / 100.0))
        if contracts <= 0:
            d.reject_gate = "zero_contracts"
            return d

        d.kind = "enter"
        d.side = side
        d.contracts = contracts
        d.limit_cents = int(entry_price)
        d.reason = (
            f"ev={ev:.2f}c p={p:.3f} mkt={mkt_p:.3f} " if mkt_p is not None
            else f"ev={ev:.2f}c p={p:.3f} "
        ) + f"slice={key} kelly={f:.3f}"
        return d

    # ── Exits ────────────────────────────────────────────────────────────────

    def evaluate_exit(self, q: MarketQuote) -> Decision:
        """The single exit rule: leave only if the model now wants the
        OTHER side with real conviction and there is time for it to pay.

        Deliberately absent: loss cuts, profit takes, trailing stops. A
        binary settles at 0 or 100 — selling into a thin late book at 3c
        is dominated by holding whenever our subjective P(win) exceeds
        3%, and the June 6 post-mortem showed the loss-cut tiers were
        realizing losses on positions that went on to settle as wins.
        """
        d = Decision(kind="none", ticker=q.ticker)
        pos = self.positions.get(q.ticker)
        if pos is None:
            d.reject_gate = "no_position"
            return d
        if q.degenerate:
            d.reject_gate = "degenerate_quote"
            return d
        if q.crossed:
            d.reject_gate = "crossed_book"
            return d
        if q.secs < self.cfg.exit_min_seconds:
            d.reject_gate = "too_late_to_flip"
            return d

        other = "no" if pos.side == "yes" else "yes"
        price_other = q.exec_ask_cents(other)
        if price_other is None:
            d.reject_gate = "no_book"
            return d
        ev_other = ev_cents_per_contract(
            q.prob_win(other), price_other,
            self.cfg.fee_rate, self.cfg.fee_multiplier,
        )
        d.ev_cents = round(ev_other, 3)
        if ev_other < self.cfg.exit_flip_min_ev_cents:
            d.reject_gate = "flip_ev_insufficient"
            return d

        # Must be able to actually sell what we hold.
        exit_bid = q.exec_bid_cents(pos.side)
        if exit_bid is None or exit_bid <= 0:
            d.reject_gate = "no_exit_bid"
            return d

        d.kind = "exit"
        d.side = pos.side
        d.contracts = pos.contracts
        d.limit_cents = int(exit_bid)
        d.reason = f"model_flip ev_other={ev_other:.2f}c p_other={q.prob_win(other):.3f}"
        return d

    # ── Book-keeping ─────────────────────────────────────────────────────────

    def open_cost_basis_usd(self) -> float:
        return sum(p.cost_usd for p in self.positions.values())

    def record_open(self, pos: Position) -> None:
        self.positions[pos.ticker] = pos
        self.entries_by_ticker[pos.ticker] = self.entries_by_ticker.get(pos.ticker, 0) + 1

    def record_close(self, ticker: str, pnl_usd: float, now_ts: float | None = None) -> None:
        self.positions.pop(ticker, None)
        self.last_close_ts[ticker] = time.time() if now_ts is None else now_ts
        self.realized_pnl_usd += pnl_usd
        if (
            self.cfg.daily_loss_limit_usd > 0
            and self.realized_pnl_usd <= -abs(self.cfg.daily_loss_limit_usd)
            and not self.halted
        ):
            self.halted = True
            self.halt_reason = (
                f"daily loss limit (${self.realized_pnl_usd:+.2f} <= "
                f"-${abs(self.cfg.daily_loss_limit_usd):.2f})"
            )
            log.warning(f"[HALT] {self.halt_reason}")
