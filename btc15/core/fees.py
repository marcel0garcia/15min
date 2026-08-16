"""Kalshi fee arithmetic — the house edge every trade must clear.

Taker fee per contract, as published:

    fee_usd = ceil_to_cent( rate * multiplier * contracts * p * (1 - p) )

with p the contract price in dollars. The curve peaks at p=0.50 and
collapses toward the extremes, which is the single most important
structural fact about where edge can live in this market:

    p=0.50 -> 1.75c/contract      p=0.95 -> 0.33c/contract

Maker fills are free on these markets.

⚠ RATE UNCERTAINTY — READ BEFORE TRADING LIVE
    0.07 is the documented rate for standard categories. Third-party
    summaries of the July 2026 fee revision describe a per-category
    multiplier (default 1) and suggest CRYPTO may price above the
    standard rate; Kalshi's own fee schedule could not be reached from
    this environment to confirm the KXBTC value.

    This matters more than it looks: if the real crypto rate is 2x, the
    EV gate under-charges by up to 1.75c/contract mid-range, which turns
    marginal winners into certain losers. So the rate is a CONFIG KNOB,
    not a constant, and `FeeCalibrator` below measures the true rate from
    observed fills and screams when reality disagrees with the model.

    Before the first live session: run one small trade, compare the fee
    Kalshi reports against `taker_fee_usd`, and set `core.fee_rate` to
    the measured value.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

TAKER_RATE = 0.07
DEFAULT_MULTIPLIER = 1.0


def taker_fee_usd(
    price_cents: float,
    contracts: int,
    rate: float = TAKER_RATE,
    multiplier: float = DEFAULT_MULTIPLIER,
) -> float:
    """Fee in dollars for a taker fill of `contracts` at `price_cents`.
    Rounded UP to the next cent, matching Kalshi's published rounding.

    The pre-ceil round to 9 decimals is load-bearing, not cosmetic: at
    p=0.50 the exact fee for 100 contracts is 1.75, but binary floating
    point evaluates it as 1.7500000000000002, and a naive ceil would bill
    1.76. That cent would be charged on every mid-price simulated fill,
    quietly biasing every backtest against us.
    """
    if contracts <= 0:
        return 0.0
    p = price_cents / 100.0
    raw_cents = rate * multiplier * contracts * p * (1.0 - p) * 100.0
    return math.ceil(round(raw_cents, 9)) / 100.0


def taker_fee_cents_per_contract(
    price_cents: float,
    rate: float = TAKER_RATE,
    multiplier: float = DEFAULT_MULTIPLIER,
) -> float:
    """Un-rounded per-contract fee in cents — the right marginal cost for
    EV math, where per-batch rounding is not the relevant quantity."""
    p = price_cents / 100.0
    return rate * multiplier * p * (1.0 - p) * 100.0


def implied_rate(fee_usd: float, price_cents: float, contracts: int) -> Optional[float]:
    """Back out the (rate x multiplier) product an observed fill implies.

    Inverse of taker_fee_usd, ignoring the ceil (which biases the implied
    rate slightly high on small fills — the calibrator accounts for this
    by requiring a minimum contract count before it trusts a sample).
    """
    if contracts <= 0:
        return None
    p = price_cents / 100.0
    denom = contracts * p * (1.0 - p)
    if denom <= 1e-12:
        return None
    return fee_usd / denom


@dataclass
class FeeCalibrator:
    """Compares fees Kalshi actually charged against what we modeled.

    An unnoticed fee-model error is the quietest way for a bot with real
    edge to lose money, so this runs on every live fill and escalates:
    a single divergent fill is logged, a persistent pattern is a WARNING
    telling the operator exactly what to set `core.fee_rate` to.
    """
    rate: float = TAKER_RATE
    multiplier: float = DEFAULT_MULTIPLIER
    min_contracts: int = 5          # below this, ceil-rounding dominates
    tolerance: float = 0.15         # 15% relative divergence is signal
    samples: list = field(default_factory=list)
    _warned: bool = False

    def observe(self, *, fee_usd: float, price_cents: float, contracts: int) -> Optional[float]:
        """Record an observed fill's fee. Returns the implied rate when the
        sample is usable, else None."""
        if contracts < self.min_contracts or fee_usd <= 0:
            return None
        r = implied_rate(fee_usd, price_cents, contracts)
        if r is None:
            return None
        self.samples.append(r)
        expected = self.rate * self.multiplier
        if abs(r - expected) / expected > self.tolerance:
            log.info(
                f"[FEE] observed rate {r:.4f} vs modeled {expected:.4f} "
                f"({contracts} @ {price_cents:.0f}c, fee ${fee_usd:.2f})"
            )
        self._maybe_warn()
        return r

    @property
    def observed_rate(self) -> Optional[float]:
        """Median implied rate across usable samples."""
        if not self.samples:
            return None
        s = sorted(self.samples)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

    def _maybe_warn(self) -> None:
        if self._warned or len(self.samples) < 5:
            return
        obs = self.observed_rate
        expected = self.rate * self.multiplier
        if obs and abs(obs - expected) / expected > self.tolerance:
            self._warned = True
            log.warning(
                f"[FEE] MODEL MISMATCH over {len(self.samples)} fills: Kalshi is "
                f"charging ~{obs:.4f} but the EV gate assumes {expected:.4f}. "
                f"Every entry decision is mispriced. Set core.fee_rate: {obs:.4f} "
                f"in config.yaml and restart."
            )

    def summary(self) -> dict:
        return {
            "modeled_rate": self.rate * self.multiplier,
            "observed_rate": self.observed_rate,
            "n_samples": len(self.samples),
        }
