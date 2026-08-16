"""Per-market quote: model probability next to the market's own price.

One function, one dataclass. The MarketQuote is the unit that flows
through the policy, the decision log, and the offline scorer — it always
carries BOTH the model's probability and the market's implied probability
so every downstream consumer can measure the model against the market.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from btc15.models.settlement_twap import twap_fair_value, accrued_average


def phase_of(secs: float) -> str:
    """Shared phase labels: early > 9m, mid 5-9m, prime 1.5-5m, late < 1.5m."""
    if secs > 540:
        return "early"
    if secs > 300:
        return "mid"
    if secs > 90:
        return "prime"
    return "late"


def price_band(price_cents: float) -> str:
    """Coarse bands for slice reporting: extreme / outer / middle."""
    p = min(price_cents, 100 - price_cents)   # distance-symmetric
    if p <= 15:
        return "extreme"
    if p <= 30:
        return "outer"
    return "middle"


@dataclass
class MarketQuote:
    ticker: str
    strike: float
    spot: float
    secs: float
    phase: str

    # Model
    prob_yes: float
    confidence: float          # |prob_yes - 0.5| * 2
    z_score: float
    sigma: float
    degenerate: bool

    # Market (cents; None when the book is empty/unknown)
    yes_bid: Optional[float]
    yes_ask: Optional[float]

    @property
    def mid_cents(self) -> Optional[float]:
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2.0
        return self.yes_bid if self.yes_bid is not None else self.yes_ask

    @property
    def market_prob_yes(self) -> Optional[float]:
        mid = self.mid_cents
        return mid / 100.0 if mid is not None else None

    @property
    def recommended_side(self) -> Optional[str]:
        if self.degenerate:
            return None
        if self.prob_yes > 0.5:
            return "yes"
        if self.prob_yes < 0.5:
            return "no"
        return None

    def prob_win(self, side: str) -> float:
        return self.prob_yes if side == "yes" else 1.0 - self.prob_yes

    def exec_ask_cents(self, side: str) -> Optional[float]:
        """Price a taker BUY of `side` pays right now (cents)."""
        if side == "yes":
            return self.yes_ask
        return 100.0 - self.yes_bid if self.yes_bid is not None else None

    def exec_bid_cents(self, side: str) -> Optional[float]:
        """Price a taker SELL of `side` receives right now (cents)."""
        if side == "yes":
            return self.yes_bid
        return 100.0 - self.yes_ask if self.yes_ask is not None else None


def quote_market(
    *,
    ticker: str,
    strike: float,
    spot: float,
    secs: float,
    sigma: float,
    ticks: Sequence[tuple[float, float]],
    now_ts: float,
    yes_bid: Optional[float],
    yes_ask: Optional[float],
) -> MarketQuote:
    """Price one market off the TWAP-settlement model."""
    acc_avg, acc_n = accrued_average(ticks, now_ts=now_ts, tau_seconds=secs)
    fv = twap_fair_value(
        spot=spot, strike=strike, sigma=sigma, tau_seconds=secs,
        accrued_avg=acc_avg, accrued_count=acc_n,
    )
    return MarketQuote(
        ticker=ticker, strike=strike, spot=spot, secs=secs,
        phase=phase_of(secs),
        prob_yes=fv.prob_yes, confidence=fv.confidence,
        z_score=fv.z_score if fv.z_score not in (float("inf"), float("-inf")) else (99.0 if fv.z_score > 0 else -99.0),
        sigma=sigma, degenerate=fv.degenerate,
        yes_bid=yes_bid, yes_ask=yes_ask,
    )
