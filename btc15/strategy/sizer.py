"""
Kelly Criterion position sizer for binary bets.

For a binary bet with:
  p  = probability of winning
  b  = net odds (what we win per $1 risked)

Kelly fraction: f* = (bp - q) / b  where q = 1 - p

On Kalshi:
  YES at price X cents → win $1, risk X cents
  b = (100 - X) / X   (net odds, profit per $1 risked)

We apply a fractional Kelly (typically 0.25x) to account for model uncertainty.
"""
from __future__ import annotations

import math
import logging

log = logging.getLogger(__name__)


def kelly_fraction_binary(
    prob_win: float,
    price_cents: float,
    fractional: float = 0.25,
    fee_rate: float = 0.07,
) -> float:
    """
    Returns the Kelly fraction of bankroll to bet.

    Args:
        prob_win:    Probability of winning (0–1).
        price_cents: Price paid in cents (1–99). E.g., 65 for YES at 0.65.
        fractional:  Kelly multiplier (0.25 = quarter Kelly).
        fee_rate:    Kalshi's fee on net profit (~7% of winnings).

    Returns:
        Fraction of bankroll to bet (0–1). May be 0 or negative (no bet).
    """
    if price_cents <= 0 or price_cents >= 100:
        return 0.0
    if prob_win <= 0 or prob_win >= 1:
        return 0.0

    # Net profit per $1 risked, after Kalshi's ~7% fee on winnings
    gross_profit_per_unit = (100 - price_cents) / price_cents
    net_profit_per_unit = gross_profit_per_unit * (1 - fee_rate)
    b = net_profit_per_unit

    q = 1 - prob_win
    full_kelly = (b * prob_win - q) / b

    if full_kelly <= 0:
        return 0.0

    return min(full_kelly * fractional, 0.20)  # hard cap at 20% of bankroll


def size_position(
    prob_win: float,
    price_cents: float,
    bankroll_usd: float,
    max_trade_usd: float,
    min_trade_usd: float,
    kelly_fraction: float = 0.25,
) -> int:
    """
    Compute number of contracts to buy.

    Args:
        prob_win:       P(win) from our model.
        price_cents:    Price per contract in cents (1–99).
        bankroll_usd:   Available capital.
        max_trade_usd:  Hard cap per trade.
        min_trade_usd:  Minimum trade size.
        kelly_fraction: Kelly multiplier.

    Returns:
        Number of contracts (each worth $0.01 at price_cents/100).
        Returns 0 if position size is below minimum.
    """
    frac = kelly_fraction_binary(prob_win, price_cents, kelly_fraction)
    if frac <= 0:
        return 0

    dollar_amount = frac * bankroll_usd
    dollar_amount = min(dollar_amount, max_trade_usd)

    if dollar_amount < min_trade_usd:
        return 0

    # Each contract costs price_cents / 100 dollars
    cost_per_contract = price_cents / 100
    if cost_per_contract <= 0:
        return 0

    contracts = int(dollar_amount / cost_per_contract)
    return max(contracts, 0)


def expected_value(
    prob_win: float,
    price_cents: float,
    contracts: int,
) -> float:
    """
    Expected profit in USD for a position.

    EV = contracts * (prob_win * profit_per_contract - prob_lose * cost_per_contract)
    """
    if contracts <= 0:
        return 0.0
    cost = price_cents / 100
    payout = 1.00  # $1 per contract on win
    ev_per_contract = prob_win * (payout - cost) - (1 - prob_win) * cost
    return contracts * ev_per_contract


def log_bet_info(
    ticker: str,
    side: str,
    prob_win: float,
    price_cents: float,
    contracts: int,
    bankroll_usd: float,
    kelly_fraction: float,
):
    ev = expected_value(prob_win, price_cents, contracts)
    cost = contracts * price_cents / 100
    log.info(
        f"[SIZER] {ticker} {side.upper()} | "
        f"P(win)={prob_win:.1%} | Price={price_cents}¢ | "
        f"Contracts={contracts} | Cost=${cost:.2f} | EV=${ev:.2f} | "
        f"Kelly={kelly_fraction:.0%}"
    )
