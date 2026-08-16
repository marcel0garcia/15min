"""Honest paper broker.

The old paper path assumed every order filled instantly at the displayed
ask, for unlimited size, with no fee — so paper P&L was a fantasy that
could never be reproduced live. This broker simulates the three things
that actually separate paper from live:

  1. DEPTH.  An IOC buy walks the displayed book level by level. If the
     top level only shows 3 contracts and we want 12, we fill 3 there and
     pay up for the rest. If the book runs out, we fill partially — the
     ledger records what we actually got, never what we wanted.

  2. QUEUE.  We are not first in line. A taker order crossing the spread
     is modeled as filling (we are paying up), but an optional
     `adverse_cents` widens our fill by that much to represent the tick
     that moves against us between decision and arrival. Maker/resting
     fills are NOT simulated as free money: they fill only when the
     market trades THROUGH our price, not merely to it.

  3. FEES.   Every taker fill pays the real Kalshi fee curve from
     core/fees.py — the same function the EV gate uses to decide.

Settlement is authoritative and never marks off a drained book: the
engine passes Kalshi's official result when available; the paper
fallback is our own final-minute TWAP of BRTI, which is the actual
settlement instrument.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from btc15.core.fees import DEFAULT_MULTIPLIER, TAKER_RATE, taker_fee_usd

log = logging.getLogger(__name__)


@dataclass
class Fill:
    ticker: str
    side: str                 # "yes" | "no"
    action: str               # "buy" | "sell"
    contracts: int            # what actually filled (0 == no fill)
    avg_price_cents: float
    gross_usd: float          # contracts * avg_price / 100
    fee_usd: float
    cash_delta_usd: float     # negative on buys, positive on sells (net of fee)
    requested_contracts: int
    trade_id: str
    ts: float
    note: str = ""

    @property
    def filled(self) -> bool:
        return self.contracts > 0


@dataclass
class LedgerEntry:
    ts: float
    kind: str                 # "entry" | "exit" | "settlement"
    ticker: str
    side: str
    contracts: int
    price_cents: float
    fee_usd: float
    cash_delta_usd: float
    pnl_usd: Optional[float]
    trade_id: str
    note: str = ""


@dataclass
class PaperBrokerConfig:
    starting_cash_usd: float = 100.0
    adverse_cents: float = 0.0        # extra cents paid vs displayed price
    allow_partial: bool = True
    # When the book has no displayed depth at all, refuse rather than
    # inventing a fill. Set True only for offline experiments.
    fill_without_depth: bool = False
    # Fee model — must match the policy's, or the EV gate and the
    # simulated cost of the trade it authorizes disagree.
    fee_rate: float = TAKER_RATE
    fee_multiplier: float = DEFAULT_MULTIPLIER


class PaperBroker:
    """Cash-accurate simulated broker. All monetary state lives here so
    the engine and the dashboard read one source of truth."""

    def __init__(self, cfg: PaperBrokerConfig):
        self.cfg = cfg
        self.cash_usd: float = cfg.starting_cash_usd
        self.ledger: list[LedgerEntry] = []
        self.fees_paid_usd: float = 0.0
        # Net of EVERY cost of the round trip — entry fee, exit fee, and
        # price. One convention, so the identity below always holds:
        #
        #     cash_usd == starting_cash_usd + realized_pnl_usd
        #
        # (fees_paid_usd is a diagnostic, NOT a second deduction). Mixing
        # gross-at-settlement with net-at-exit is exactly how a P&L number
        # becomes untrustworthy; tests/test_paper_session.py asserts this
        # identity over a full synthetic session.
        self.realized_pnl_usd: float = 0.0
        self.wins: int = 0
        self.losses: int = 0

    # ── Book helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _levels_for_buy(book: dict, side: str) -> list[tuple[float, float]]:
        """Ascending (price_cents, size) levels a BUY of `side` consumes.

        The cache stores the book in YES space: `yes_bids` maps
        price->size of resting YES buyers, `yes_asks` price->size of
        resting YES sellers. Buying NO at price X consumes the YES bid at
        (100 - X), so the NO ask ladder is the YES bid ladder mirrored.
        """
        if side == "yes":
            asks = book.get("yes_asks") or {}
            return sorted(((float(p), float(s)) for p, s in asks.items() if s > 0))
        bids = book.get("yes_bids") or {}
        return sorted(((100.0 - float(p), float(s)) for p, s in bids.items() if s > 0))

    @staticmethod
    def _levels_for_sell(book: dict, side: str) -> list[tuple[float, float]]:
        """Descending (price_cents, size) levels a SELL of `side` hits."""
        if side == "yes":
            bids = book.get("yes_bids") or {}
            return sorted(((float(p), float(s)) for p, s in bids.items() if s > 0), reverse=True)
        asks = book.get("yes_asks") or {}
        return sorted(((100.0 - float(p), float(s)) for p, s in asks.items() if s > 0), reverse=True)

    # ── Orders ───────────────────────────────────────────────────────────────

    def buy_ioc(
        self,
        *,
        ticker: str,
        side: str,
        contracts: int,
        limit_cents: float,
        book: dict,
        trade_id: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> Fill:
        """Walk the displayed ask ladder up to `limit_cents`."""
        ts = ts if ts is not None else time.time()
        trade_id = trade_id or uuid.uuid4().hex[:8]
        levels = self._levels_for_buy(book, side)

        if not levels and self.cfg.fill_without_depth:
            levels = [(limit_cents, float(contracts))]

        remaining = contracts
        filled = 0
        notional_cents = 0.0
        for price, size in levels:
            eff = price + self.cfg.adverse_cents
            if eff > limit_cents or remaining <= 0:
                break
            take = int(min(remaining, size))
            if take <= 0:
                continue
            filled += take
            notional_cents += take * eff
            remaining -= take

        if filled == 0:
            return Fill(
                ticker=ticker, side=side, action="buy", contracts=0,
                avg_price_cents=0.0, gross_usd=0.0, fee_usd=0.0,
                cash_delta_usd=0.0, requested_contracts=contracts,
                trade_id=trade_id, ts=ts, note="no_fill_depth_or_limit",
            )
        if filled < contracts and not self.cfg.allow_partial:
            return Fill(
                ticker=ticker, side=side, action="buy", contracts=0,
                avg_price_cents=0.0, gross_usd=0.0, fee_usd=0.0,
                cash_delta_usd=0.0, requested_contracts=contracts,
                trade_id=trade_id, ts=ts, note="partial_rejected",
            )

        avg = notional_cents / filled
        gross = notional_cents / 100.0
        fee = taker_fee_usd(avg, filled, self.cfg.fee_rate, self.cfg.fee_multiplier)
        cash_delta = -(gross + fee)

        # Cash constraint — never spend money we don't have.
        if -cash_delta > self.cash_usd + 1e-9:
            affordable = 0
            notional_cents = 0.0
            for price, size in levels:
                eff = price + self.cfg.adverse_cents
                if eff > limit_cents:
                    break
                for _ in range(int(min(size, contracts))):
                    trial_notional = notional_cents + eff
                    trial_avg = trial_notional / (affordable + 1)
                    trial_fee = taker_fee_usd(
                        trial_avg, affordable + 1,
                        self.cfg.fee_rate, self.cfg.fee_multiplier,
                    )
                    if trial_notional / 100.0 + trial_fee > self.cash_usd:
                        break
                    affordable += 1
                    notional_cents = trial_notional
                    if affordable >= contracts:
                        break
                if affordable >= contracts:
                    break
            if affordable == 0:
                return Fill(
                    ticker=ticker, side=side, action="buy", contracts=0,
                    avg_price_cents=0.0, gross_usd=0.0, fee_usd=0.0,
                    cash_delta_usd=0.0, requested_contracts=contracts,
                    trade_id=trade_id, ts=ts, note="insufficient_cash",
                )
            filled = affordable
            avg = notional_cents / filled
            gross = notional_cents / 100.0
            fee = taker_fee_usd(avg, filled, self.cfg.fee_rate, self.cfg.fee_multiplier)
            cash_delta = -(gross + fee)

        self.cash_usd += cash_delta
        self.fees_paid_usd += fee
        self.ledger.append(LedgerEntry(
            ts=ts, kind="entry", ticker=ticker, side=side, contracts=filled,
            price_cents=avg, fee_usd=fee, cash_delta_usd=cash_delta,
            pnl_usd=None, trade_id=trade_id,
            note=f"requested={contracts}",
        ))
        return Fill(
            ticker=ticker, side=side, action="buy", contracts=filled,
            avg_price_cents=avg, gross_usd=gross, fee_usd=fee,
            cash_delta_usd=cash_delta, requested_contracts=contracts,
            trade_id=trade_id, ts=ts,
            note="partial" if filled < contracts else "",
        )

    def sell_ioc(
        self,
        *,
        ticker: str,
        side: str,
        contracts: int,
        limit_cents: float,
        book: dict,
        entry_cents: float,
        entry_fee_usd: float = 0.0,
        trade_id: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> Fill:
        """Walk the displayed bid ladder down to `limit_cents`.

        `entry_fee_usd` is the fee already paid to open the contracts being
        closed. It is folded into realized P&L so that P&L is net of every
        cost of the round trip — see the note on `realized_pnl_usd`.
        """
        ts = ts if ts is not None else time.time()
        trade_id = trade_id or uuid.uuid4().hex[:8]
        levels = self._levels_for_sell(book, side)
        if not levels and self.cfg.fill_without_depth:
            levels = [(limit_cents, float(contracts))]

        remaining = contracts
        filled = 0
        notional_cents = 0.0
        for price, size in levels:
            eff = price - self.cfg.adverse_cents
            if eff < limit_cents or remaining <= 0:
                break
            take = int(min(remaining, size))
            if take <= 0:
                continue
            filled += take
            notional_cents += take * eff
            remaining -= take

        if filled == 0:
            return Fill(
                ticker=ticker, side=side, action="sell", contracts=0,
                avg_price_cents=0.0, gross_usd=0.0, fee_usd=0.0,
                cash_delta_usd=0.0, requested_contracts=contracts,
                trade_id=trade_id, ts=ts, note="no_fill_depth_or_limit",
            )

        avg = notional_cents / filled
        gross = notional_cents / 100.0
        fee = taker_fee_usd(avg, filled, self.cfg.fee_rate, self.cfg.fee_multiplier)
        cash_delta = gross - fee
        # Net of BOTH legs' fees. On a partial close, only the closed
        # portion's share of the entry fee is realized.
        entry_fee_share = entry_fee_usd * (filled / contracts) if contracts else 0.0
        pnl = (avg - entry_cents) * filled / 100.0 - fee - entry_fee_share

        self.cash_usd += cash_delta
        self.fees_paid_usd += fee
        self.realized_pnl_usd += pnl
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1
        self.ledger.append(LedgerEntry(
            ts=ts, kind="exit", ticker=ticker, side=side, contracts=filled,
            price_cents=avg, fee_usd=fee, cash_delta_usd=cash_delta,
            pnl_usd=pnl, trade_id=trade_id, note=f"requested={contracts}",
        ))
        return Fill(
            ticker=ticker, side=side, action="sell", contracts=filled,
            avg_price_cents=avg, gross_usd=gross, fee_usd=fee,
            cash_delta_usd=cash_delta, requested_contracts=contracts,
            trade_id=trade_id, ts=ts,
            note="partial" if filled < contracts else "",
        )

    def settle(
        self,
        *,
        ticker: str,
        side: str,
        contracts: int,
        entry_cents: float,
        result: str,             # "yes" | "no" — authoritative outcome
        entry_fee_usd: float = 0.0,
        trade_id: str = "",
        ts: Optional[float] = None,
    ) -> float:
        """Credit settlement. Kalshi charges NO fee on settlement, so the
        entry fee is the round trip's only cost. Returns P&L in USD, net
        of that fee."""
        ts = ts if ts is not None else time.time()
        payout_cents = 100.0 if result == side else 0.0
        proceeds = payout_cents * contracts / 100.0
        pnl = (payout_cents - entry_cents) * contracts / 100.0 - entry_fee_usd

        self.cash_usd += proceeds
        self.realized_pnl_usd += pnl
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1
        self.ledger.append(LedgerEntry(
            ts=ts, kind="settlement", ticker=ticker, side=side,
            contracts=contracts, price_cents=payout_cents, fee_usd=0.0,
            cash_delta_usd=proceeds, pnl_usd=pnl, trade_id=trade_id,
            note=f"result={result}",
        ))
        return pnl

    # ── Reporting ────────────────────────────────────────────────────────────

    @property
    def win_rate(self) -> Optional[float]:
        n = self.wins + self.losses
        return (self.wins / n) if n else None

    def equity_usd(self, open_positions_value_usd: float = 0.0) -> float:
        return self.cash_usd + open_positions_value_usd

    def summary(self) -> dict:
        return {
            "cash_usd": round(self.cash_usd, 4),
            "realized_pnl_usd": round(self.realized_pnl_usd, 4),
            "fees_paid_usd": round(self.fees_paid_usd, 4),
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "n_ledger": len(self.ledger),
        }
