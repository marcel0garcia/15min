"""Pure-numpy technical indicators — no external TA-lib required."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TechnicalSignals:
    """All computed signals for one moment in time."""
    rsi: Optional[float] = None            # 0–100
    macd: Optional[float] = None           # MACD line
    macd_signal: Optional[float] = None    # Signal line
    macd_hist: Optional[float] = None      # Histogram
    bb_upper: Optional[float] = None       # Bollinger upper
    bb_mid: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_pct: Optional[float] = None         # (price - lower) / (upper - lower)
    bb_zscore: Optional[float] = None      # (price - mid) / std
    ema_fast: Optional[float] = None       # EMA(9)
    ema_slow: Optional[float] = None       # EMA(21)
    ema_trend: Optional[float] = None      # +1 / -1 / 0
    vwap: Optional[float] = None
    price_vs_vwap: Optional[float] = None  # pct deviation from vwap
    momentum_1m: Optional[float] = None    # 1-bar return
    momentum_5m: Optional[float] = None    # 5-bar return
    momentum_15m: Optional[float] = None   # 15-bar return
    atr: Optional[float] = None            # Average True Range
    trend_slope: Optional[float] = None    # OLS slope of last 15 bars (normalized)
    trend_r2: Optional[float] = None       # R² of trend regression

    @property
    def momentum_score(self) -> float:
        """
        Composite momentum score: +1 = strong up, -1 = strong down, 0 = neutral.
        Uses RSI, MACD histogram, and short momentum.
        """
        score = 0.0
        n = 0

        if self.rsi is not None:
            # RSI: >55 bullish, <45 bearish
            score += np.clip((self.rsi - 50) / 25, -1, 1)
            n += 1

        if self.macd_hist is not None and self.atr:
            norm_hist = np.clip(self.macd_hist / (self.atr * 0.1 + 1e-8), -1, 1)
            score += norm_hist
            n += 1

        if self.bb_zscore is not None:
            # BB z-score: momentum is DIRECTION, not reversion
            score += np.clip(self.bb_zscore / 2, -1, 1)
            n += 1

        if self.ema_trend is not None:
            score += self.ema_trend
            n += 1

        return score / n if n > 0 else 0.0

    @property
    def directional_probability_up(self) -> float:
        """P(price moves up over next 15 min) from technical signals. [0, 1]"""
        return (self.momentum_score + 1) / 2


def compute_signals(
    closes: list[float],
    highs: Optional[list[float]] = None,
    lows: Optional[list[float]] = None,
    volumes: Optional[list[float]] = None,
    vwaps: Optional[list[float]] = None,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_sig: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> TechnicalSignals:
    """Compute all technical indicators from price history."""
    sig = TechnicalSignals()
    c = np.array(closes, dtype=float)
    n = len(c)

    if n < 2:
        return sig

    # ── RSI ─────────────────────────────────────────────────────────────
    if n >= rsi_period + 1:
        sig.rsi = _rsi(c, rsi_period)

    # ── MACD ────────────────────────────────────────────────────────────
    if n >= macd_slow + macd_sig:
        macd_line, signal_line, hist = _macd(c, macd_fast, macd_slow, macd_sig)
        sig.macd = macd_line
        sig.macd_signal = signal_line
        sig.macd_hist = hist

    # ── Bollinger Bands ──────────────────────────────────────────────────
    if n >= bb_period:
        upper, mid, lower = _bollinger(c, bb_period, bb_std)
        sig.bb_upper = upper
        sig.bb_mid = mid
        sig.bb_lower = lower
        price = c[-1]
        if upper != lower:
            sig.bb_pct = (price - lower) / (upper - lower)
            sig.bb_zscore = (price - mid) / (bb_std * np.std(c[-bb_period:]))

    # ── EMAs ─────────────────────────────────────────────────────────────
    if n >= 9:
        sig.ema_fast = _ema(c, 9)[-1]
    if n >= 21:
        sig.ema_slow = _ema(c, 21)[-1]
    if sig.ema_fast and sig.ema_slow:
        sig.ema_trend = np.sign(sig.ema_fast - sig.ema_slow)

    # ── ATR ──────────────────────────────────────────────────────────────
    if highs and lows and len(highs) >= 14:
        h = np.array(highs[-15:], dtype=float)
        l = np.array(lows[-15:], dtype=float)
        cp = c[-15:]
        sig.atr = _atr(h, l, cp, 14)

    # ── VWAP deviation ──────────────────────────────────────────────────
    if vwaps and len(vwaps) > 0:
        sig.vwap = vwaps[-1]
        if sig.vwap and sig.vwap > 0:
            sig.price_vs_vwap = (c[-1] - sig.vwap) / sig.vwap

    # ── Momentum ─────────────────────────────────────────────────────────
    if n >= 2:
        sig.momentum_1m = (c[-1] - c[-2]) / c[-2]
    if n >= 6:
        sig.momentum_5m = (c[-1] - c[-6]) / c[-6]
    if n >= 16:
        sig.momentum_15m = (c[-1] - c[-16]) / c[-16]

    # ── Trend regression ─────────────────────────────────────────────────
    if n >= 10:
        window = min(15, n)
        y = c[-window:]
        x = np.arange(window, dtype=float)
        slope, r2 = _ols_slope_r2(x, y)
        # Normalize slope as % per bar
        sig.trend_slope = slope / (np.mean(y) + 1e-8)
        sig.trend_r2 = r2

    return sig


# ── Pure-numpy indicator implementations ──────────────────────────────────────

def _ema(prices: np.ndarray, period: int) -> np.ndarray:
    k = 2 / (period + 1)
    ema = np.empty_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = prices[i] * k + ema[i - 1] * (1 - k)
    return ema


def _rsi(prices: np.ndarray, period: int) -> float:
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _macd(
    prices: np.ndarray, fast: int, slow: int, signal: int
) -> tuple[float, float, float]:
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line[-1] - signal_line[-1]
    return float(macd_line[-1]), float(signal_line[-1]), float(hist)


def _bollinger(
    prices: np.ndarray, period: int, num_std: float
) -> tuple[float, float, float]:
    window = prices[-period:]
    mid = float(np.mean(window))
    std = float(np.std(window))
    return mid + num_std * std, mid, mid - num_std * std


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
    tr_list = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_list.append(tr)
    return float(np.mean(tr_list[-period:]))


def _ols_slope_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    ss_xy = np.dot(xm, ym)
    ss_xx = np.dot(xm, xm)
    if ss_xx == 0:
        return 0.0, 0.0
    slope = ss_xy / ss_xx
    y_pred = np.mean(y) + slope * xm
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum(ym ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(max(0, r2))
