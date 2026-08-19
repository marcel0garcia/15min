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


@dataclass(frozen=True)
class SliceGrid:
    """Boundaries of the (phase x band) grid that R1 measures and R2 gates on.

    These five numbers were hard-coded, which meant the entire measurement
    methodology rested on unmeasured constants: move `phase_prime_sec` from
    90 to 60 and a slice that "beats the market" can stop doing so. They are
    parameters now so the sweep can vary the grid itself, not just the
    policy evaluated on a fixed grid.
    """
    phase_early_sec: float = 540.0
    phase_mid_sec: float = 300.0
    phase_prime_sec: float = 90.0
    band_extreme_cents: float = 15.0
    band_outer_cents: float = 30.0


DEFAULT_GRID = SliceGrid()


def phase_of(secs: float, grid: SliceGrid = DEFAULT_GRID) -> str:
    """Phase label by seconds remaining. Defaults: early > 9m, mid 5-9m,
    prime 1.5-5m, late < 1.5m."""
    if secs > grid.phase_early_sec:
        return "early"
    if secs > grid.phase_mid_sec:
        return "mid"
    if secs > grid.phase_prime_sec:
        return "prime"
    return "late"


def price_band(price_cents: float, grid: SliceGrid = DEFAULT_GRID) -> str:
    """Coarse bands for slice reporting: extreme / outer / middle."""
    p = min(price_cents, 100 - price_cents)   # distance-symmetric
    if p <= grid.band_extreme_cents:
        return "extreme"
    if p <= grid.band_outer_cents:
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

    # Pricer internals — carried so the decision log can explain WHY a
    # probability was what it was. Without accrued_avg/locked_frac you
    # cannot tell a final-minute lock-in from an ordinary late quote when
    # reading the recordings back.
    accrued_avg: Optional[float] = None
    accrued_count: int = 0
    locked_frac: float = 0.0
    k_eff: Optional[float] = None

    # Health of the vol nowcast that produced prob_yes. sigma is the single
    # most load-bearing input to the model, so the policy needs to know when
    # it is untrustworthy rather than treating every sigma as equal.
    sigma_clamped: bool = False      # the floor or ceiling bound
    tick_span_sec: float = 0.0       # seconds of BRTI history behind sigma

    @property
    def mid_cents(self) -> Optional[float]:
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2.0
        return self.yes_bid if self.yes_bid is not None else self.yes_ask

    @property
    def locked(self) -> bool:
        """Zero-spread book: yes_bid == yes_ask.

        Normal and tradeable here, which is not obvious. `yes_ask` is
        derived as 100 - best_no_bid, so bid == ask means the best YES bid
        and the best NO bid sum to exactly 100 — fully collateralized, no
        arbitrage, just a spread of zero. It is common: 158 of 175 scans in
        the 19AUG18:26 session. An earlier version of this guard treated it
        as corrupt and rejected 60-95% of all scans.
        """
        return (
            self.yes_bid is not None
            and self.yes_ask is not None
            and self.yes_bid == self.yes_ask
        )

    @property
    def crossed(self) -> bool:
        """True only when the book is genuinely inconsistent — YES bid
        STRICTLY above YES ask, i.e. best_yes_bid + best_no_bid > 100.

        That is riskless arbitrage and cannot persist on the exchange, so
        when we see it our cache is stale, not the market. Observed live on
        2026-08-19 (bid 85 / ask 84), rare — about 3% of scans. Pricing off
        it would invent edge out of a bookkeeping error, so the policy
        refuses. Note the strictness: `>=` here would also reject every
        locked book. See `locked`.
        """
        return (
            self.yes_bid is not None
            and self.yes_ask is not None
            and self.yes_bid > self.yes_ask
        )

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
    grid: SliceGrid = DEFAULT_GRID,
    sigma_clamped: bool = False,
    tick_span_sec: float = 0.0,
) -> MarketQuote:
    """Price one market off the TWAP-settlement model."""
    acc_avg, acc_n = accrued_average(ticks, now_ts=now_ts, tau_seconds=secs)
    fv = twap_fair_value(
        spot=spot, strike=strike, sigma=sigma, tau_seconds=secs,
        accrued_avg=acc_avg, accrued_count=acc_n,
    )
    return MarketQuote(
        ticker=ticker, strike=strike, spot=spot, secs=secs,
        phase=phase_of(secs, grid),
        prob_yes=fv.prob_yes, confidence=fv.confidence,
        z_score=fv.z_score if fv.z_score not in (float("inf"), float("-inf")) else (99.0 if fv.z_score > 0 else -99.0),
        sigma=sigma, degenerate=fv.degenerate,
        yes_bid=yes_bid, yes_ask=yes_ask,
        accrued_avg=acc_avg, accrued_count=acc_n,
        locked_frac=float(fv.inputs.get("locked_frac") or 0.0),
        k_eff=fv.inputs.get("k_eff"),
        sigma_clamped=sigma_clamped, tick_span_sec=tick_span_sec,
    )
