"""Shared types for price feeds.

Moved out of the legacy coinbase.py module so multiple feed implementations
(single-venue Coinbase WS, consolidated BRTI, future venues) can produce
the same OHLCBar / Tick shape without circular imports.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Coroutine, Optional


TickHandler = Callable[[float, float, int], Coroutine]


@dataclass
class Tick:
    price: float
    qty: float
    ts_ms: int

    @property
    def ts(self) -> float:
        return self.ts_ms / 1000


@dataclass
class OHLCBar:
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    ts: float
    interval_sec: int
    trade_count: int = 0

    @property
    def pct_change(self) -> float:
        return (self.close - self.open) / self.open if self.open else 0.0

    @property
    def body_pct(self) -> float:
        return (self.close - self.open) / self.open if self.open else 0.0


class BarAccumulator:
    """Rolls ticks into a single in-progress OHLC bar. Caller is
    responsible for detecting bar-close boundaries and starting a new
    accumulator for the next interval."""

    def __init__(self, ts_ms: int, interval_sec: int):
        self.ts_ms = ts_ms
        self.interval_sec = interval_sec
        self._open: Optional[float] = None
        self._high: float = float("-inf")
        self._low: float = float("inf")
        self._close: float = 0.0
        self._vol: float = 0.0
        self._vol_price: float = 0.0
        self._count: int = 0

    def add(self, tick: Tick) -> None:
        if self._open is None:
            self._open = tick.price
        self._high = max(self._high, tick.price)
        self._low = min(self._low, tick.price)
        self._close = tick.price
        self._vol += tick.qty
        self._vol_price += tick.qty * tick.price
        self._count += 1

    def to_ohlc(self) -> OHLCBar:
        vwap = self._vol_price / self._vol if self._vol > 0 else self._close
        return OHLCBar(
            open=self._open or self._close,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._vol,
            vwap=vwap,
            ts=self.ts_ms / 1000,
            interval_sec=self.interval_sec,
            trade_count=self._count,
        )
