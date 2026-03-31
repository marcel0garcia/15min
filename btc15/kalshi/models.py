"""Kalshi API data models."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(str, Enum):
    YES = "yes"
    NO = "no"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    RESTING = "resting"
    CANCELED = "canceled"
    EXECUTED = "executed"
    PENDING = "pending"


class TimeInForce(str, Enum):
    GTC = "good_till_canceled"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"


class SelfTradePrevention(str, Enum):
    CANCEL_INCOMING = "taker_at_cross"
    CANCEL_RESTING = "maker"


class MarketStatus(str, Enum):
    OPEN = "open"
    ACTIVE = "active"       # Kalshi current API uses "active"
    CLOSED = "closed"
    SETTLED = "settled"
    UNOPENED = "unopened"
    FINALIZED = "finalized"


@dataclass
class Market:
    ticker: str
    series_ticker: str
    title: str
    status: MarketStatus
    yes_bid: float        # 0–100 cents
    yes_ask: float
    no_bid: float
    no_ask: float
    last_price: float
    volume: int
    open_interest: int
    strike_price: float   # USD strike for KXBTC
    close_time: datetime  # When this market expires
    result: Optional[str] = None  # "yes" | "no" | None

    @property
    def yes_mid(self) -> float:
        if self.yes_bid and self.yes_ask:
            return (self.yes_bid + self.yes_ask) / 2
        return self.last_price

    @property
    def seconds_remaining(self) -> float:
        now = datetime.utcnow().replace(tzinfo=None)
        close = self.close_time.replace(tzinfo=None)
        return max(0.0, (close - now).total_seconds())

    @property
    def minutes_remaining(self) -> float:
        return self.seconds_remaining / 60


@dataclass
class Order:
    order_id: str
    ticker: str
    side: Side
    order_type: OrderType
    count: int              # Number of contracts ($0.01 each)
    yes_price: int          # cents 1–99
    no_price: int
    status: OrderStatus
    filled_count: int = 0
    remaining_count: int = 0
    created_time: Optional[datetime] = None
    expiration_time: Optional[datetime] = None

    @property
    def fill_usd(self) -> float:
        """USD value of filled contracts."""
        price = self.yes_price if self.side == Side.YES else self.no_price
        return self.filled_count * price / 100


@dataclass
class Position:
    ticker: str
    side: Side
    contracts: int
    avg_price_cents: float   # What we paid (cents)
    current_yes_bid: float
    current_yes_ask: float
    market_status: MarketStatus = MarketStatus.OPEN
    result: Optional[str] = None

    @property
    def cost_usd(self) -> float:
        return self.contracts * self.avg_price_cents / 100

    @property
    def current_value_usd(self) -> float:
        if self.market_status == MarketStatus.SETTLED:
            if self.result == self.side.value:
                return self.contracts * 1.00
            return 0.0
        # Mark to market
        if self.side == Side.YES:
            bid = self.current_yes_bid / 100
        else:
            bid = (100 - self.current_yes_ask) / 100
        return self.contracts * bid

    @property
    def unrealized_pnl(self) -> float:
        return self.current_value_usd - self.cost_usd


@dataclass
class Orderbook:
    ticker: str
    yes_bids: list = field(default_factory=list)   # [(price_cents, size), ...]
    yes_asks: list = field(default_factory=list)
    timestamp: Optional[datetime] = None

    @property
    def best_yes_bid(self) -> Optional[float]:
        return max((p for p, _ in self.yes_bids), default=None)

    @property
    def best_yes_ask(self) -> Optional[float]:
        return min((p for p, _ in self.yes_asks), default=None)

    @property
    def yes_mid(self) -> Optional[float]:
        b, a = self.best_yes_bid, self.best_yes_ask
        if b and a:
            return (b + a) / 2
        return b or a

    @property
    def spread_cents(self) -> Optional[float]:
        b, a = self.best_yes_bid, self.best_yes_ask
        if b and a:
            return a - b
        return None


@dataclass
class Trade:
    """Our own trade record."""
    trade_id: str
    ticker: str
    side: Side
    contracts: int
    entry_price_cents: float
    exit_price_cents: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    settled: bool = False
    won: Optional[bool] = None

    @property
    def pnl_usd(self) -> Optional[float]:
        if self.exit_price_cents is None:
            return None
        if self.settled and self.won is not None:
            gross = self.contracts * (1.00 if self.won else 0.00)
        else:
            gross = self.contracts * self.exit_price_cents / 100
        cost = self.contracts * self.entry_price_cents / 100
        return gross - cost


@dataclass
class PortfolioBalance:
    available_balance_cents: int
    portfolio_value_cents: int

    @property
    def available_usd(self) -> float:
        return self.available_balance_cents / 100

    @property
    def portfolio_usd(self) -> float:
        return self.portfolio_value_cents / 100
