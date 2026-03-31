"""
Ensemble model: combines multiple probability sources into a final YES/NO
probability estimate and edge calculation vs. Kalshi orderbook price.

Models:
  1. binary_options_model  — Black-Scholes digital option approximation
  2. technical_momentum    — RSI/MACD/BB composite directional score
  3. trend_regression      — OLS slope + R² confidence
  4. ml_model              — LightGBM (needs training data; gracefully disabled)

Output:
  P(BTC at settlement > strike) if predicting YES
  P(BTC at settlement < strike) if predicting NO
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm

from btc15.models.technical import TechnicalSignals, compute_signals
from btc15.feeds.binance import OHLCBar

log = logging.getLogger(__name__)


@dataclass
class ModelOutput:
    """Final combined model output for one market."""
    ticker: str
    strike: float
    current_price: float
    seconds_remaining: float

    # Individual model probabilities P(settle > strike)
    prob_binary_options: Optional[float] = None
    prob_technical: Optional[float] = None
    prob_trend: Optional[float] = None
    prob_ml: Optional[float] = None

    # Combined
    prob_yes: float = 0.5      # P(settle > strike)
    prob_no: float = 0.5       # P(settle < strike) = 1 - prob_yes
    confidence: float = 0.0   # 0–1 how confident we are in our estimate

    # Edge vs Kalshi
    kalshi_yes_price: Optional[float] = None   # cents / 100
    edge_yes: Optional[float] = None           # prob_yes - kalshi_yes_price/100
    edge_no: Optional[float] = None            # prob_no  - (1 - kalshi_yes_price/100)
    recommended_side: Optional[str] = None    # "yes" | "no" | None

    @property
    def best_edge(self) -> Optional[float]:
        edges = []
        if self.edge_yes is not None:
            edges.append(self.edge_yes)
        if self.edge_no is not None:
            edges.append(self.edge_no)
        return max(edges) if edges else None

    @property
    def signal_str(self) -> str:
        if self.recommended_side is None:
            return "NEUTRAL"
        edge = self.best_edge or 0
        if edge > 0.10:
            return f"STRONG {self.recommended_side.upper()}"
        return f"WEAK {self.recommended_side.upper()}"


class EnsembleModel:
    """
    Combines all sub-models to produce probability estimates.
    Works immediately out-of-the-box with the binary options + technical models.
    ML model is optional and gracefully disabled if not trained.
    """

    def __init__(self, weights: dict, config=None):
        self.weights = weights
        self.cfg = config
        self._ml_model = None
        self._try_load_ml_model()

    def _try_load_ml_model(self):
        try:
            import joblib
            from pathlib import Path
            model_path = Path("data/ml_model.pkl")
            if model_path.exists():
                self._ml_model = joblib.load(model_path)
                log.info("ML model loaded from data/ml_model.pkl")
            else:
                log.info("No ML model found — using statistical models only")
        except Exception as e:
            log.debug(f"ML model load skipped: {e}")

    def predict(
        self,
        ticker: str,
        strike: float,
        current_price: float,
        seconds_remaining: float,
        annual_vol: float,
        bars: list[OHLCBar],
        kalshi_yes_bid: Optional[float] = None,
        kalshi_yes_ask: Optional[float] = None,
        min_edge: float = 0.04,
        min_confidence: float = 0.62,
    ) -> ModelOutput:
        out = ModelOutput(
            ticker=ticker,
            strike=strike,
            current_price=current_price,
            seconds_remaining=seconds_remaining,
        )

        # Kalshi mid-price (in 0–1 scale)
        if kalshi_yes_bid is not None and kalshi_yes_ask is not None:
            out.kalshi_yes_price = (kalshi_yes_bid + kalshi_yes_ask) / 2 / 100
        elif kalshi_yes_bid is not None:
            out.kalshi_yes_price = kalshi_yes_bid / 100

        active_weights: dict[str, float] = {}
        probs: dict[str, float] = {}

        # ── 1. Binary options model (Black-Scholes digital) ──────────────────
        p_bsm = self._binary_option_prob(
            spot=current_price,
            strike=strike,
            seconds_remaining=seconds_remaining,
            annual_vol=annual_vol,
        )
        if p_bsm is not None:
            out.prob_binary_options = p_bsm
            probs["binary_options_model"] = p_bsm
            active_weights["binary_options_model"] = self.weights.get("binary_options_model", 0.40)

        # ── 2. Technical momentum ────────────────────────────────────────────
        if len(bars) >= 5:
            closes = [b.close for b in bars]
            highs = [b.high for b in bars]
            lows = [b.low for b in bars]
            vols = [b.volume for b in bars]
            vwaps = [b.vwap for b in bars]
            signals: TechnicalSignals = compute_signals(
                closes=closes, highs=highs, lows=lows,
                volumes=vols, vwaps=vwaps,
            )
            p_tech = signals.directional_probability_up
            out.prob_technical = p_tech
            probs["technical_momentum"] = p_tech
            active_weights["technical_momentum"] = self.weights.get("technical_momentum", 0.30)

            # ── 3. Trend regression ──────────────────────────────────────────
            if signals.trend_slope is not None and signals.trend_r2 is not None:
                # Convert slope to probability using logistic function
                # slope_norm ∈ [-1, 1] roughly
                slope_norm = np.clip(signals.trend_slope * 500, -1, 1)
                r2_confidence = signals.trend_r2
                p_trend = (1 + slope_norm) / 2  # 0–1
                # Shrink toward 0.5 when R² is low
                p_trend = 0.5 + (p_trend - 0.5) * r2_confidence
                out.prob_trend = p_trend
                probs["trend_regression"] = p_trend
                active_weights["trend_regression"] = self.weights.get("trend_regression", 0.20)

        # ── 4. ML model ──────────────────────────────────────────────────────
        if self._ml_model is not None and len(bars) >= 20:
            try:
                features = self._build_ml_features(
                    strike, current_price, seconds_remaining, annual_vol, bars
                )
                p_ml = float(self._ml_model.predict_proba([features])[0][1])
                out.prob_ml = p_ml
                probs["ml_model"] = p_ml
                active_weights["ml_model"] = self.weights.get("ml_model", 0.10)
            except Exception as e:
                log.debug(f"ML prediction failed: {e}")

        # ── Weighted combination ──────────────────────────────────────────────
        if not probs:
            return out

        total_w = sum(active_weights.get(k, 0) for k in probs)
        if total_w == 0:
            return out

        prob_yes = sum(
            probs[k] * active_weights.get(k, 0) for k in probs
        ) / total_w
        out.prob_yes = float(np.clip(prob_yes, 0.01, 0.99))
        out.prob_no = 1.0 - out.prob_yes

        # Confidence = agreement between models (inverse of std dev)
        if len(probs) >= 2:
            model_probs = list(probs.values())
            std = float(np.std(model_probs))
            out.confidence = max(0.0, 1.0 - std * 4)  # std of 0.25 → confidence 0
        else:
            out.confidence = 0.5

        # ── Edge calculation ──────────────────────────────────────────────────
        if out.kalshi_yes_price is not None:
            out.edge_yes = out.prob_yes - out.kalshi_yes_price
            out.edge_no = out.prob_no - (1.0 - out.kalshi_yes_price)

            # Recommend side if edge AND confidence meet thresholds
            if out.confidence >= min_confidence:
                if out.edge_yes >= min_edge:
                    out.recommended_side = "yes"
                elif out.edge_no >= min_edge:
                    out.recommended_side = "no"

        return out

    def _binary_option_prob(
        self,
        spot: float,
        strike: float,
        seconds_remaining: float,
        annual_vol: float,
        drift: float = 0.0,
    ) -> Optional[float]:
        """
        P(S_T > K) using log-normal distribution (risk-neutral).

        For very short time horizons (<60s) this converges to step function,
        so we add a minimum time floor.
        """
        if seconds_remaining < 10 or spot <= 0 or strike <= 0:
            return None

        T = max(seconds_remaining, 30) / (365 * 24 * 3600)  # years
        sigma = max(annual_vol, 0.10)

        d2 = (math.log(spot / strike) + (drift - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        # P(S_T > K) = N(d2) under risk-neutral measure
        p = float(norm.cdf(d2))
        return np.clip(p, 0.01, 0.99)

    def _build_ml_features(
        self,
        strike: float,
        current_price: float,
        seconds_remaining: float,
        annual_vol: float,
        bars: list[OHLCBar],
    ) -> list[float]:
        """Build feature vector for ML model."""
        closes = [b.close for b in bars[-30:]]
        signals = compute_signals(closes)

        moneyness = math.log(current_price / strike)
        t_remaining = seconds_remaining / 900  # normalized 0–1

        return [
            moneyness,
            t_remaining,
            annual_vol,
            signals.rsi / 100 if signals.rsi else 0.5,
            signals.macd_hist or 0.0,
            signals.bb_pct or 0.5,
            signals.bb_zscore or 0.0,
            signals.momentum_1m or 0.0,
            signals.momentum_5m or 0.0,
            signals.momentum_15m or 0.0,
            signals.trend_slope or 0.0,
            signals.trend_r2 or 0.0,
            signals.ema_trend or 0.0,
        ]
