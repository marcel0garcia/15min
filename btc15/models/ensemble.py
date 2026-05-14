"""
Ensemble model: combines multiple probability sources into a final YES/NO
probability estimate and edge calculation vs. Kalshi orderbook price.

Models (in priority order):
  1. orderbook_imbalance   — real-time buy/sell pressure from live book depth
  2. technical_momentum    — RSI/MACD/BB composite directional score
  3. trend_regression      — OLS slope + R² confidence
  4. binary_options_model  — Black-Scholes digital option (sanity check at short horizons)
  5. ml_model              — LightGBM (needs training data; gracefully disabled)

Ensemble weights (default):
  orderbook_imbalance  0.25  ← highest alpha: forward-looking flow signal
  technical_momentum   0.35  ← strong for 15-min crypto moves
  trend_regression     0.20  ← medium-term context
  binary_options_model 0.10  ← sanity check; becomes step-function at <5 min
  ml_model             0.10  ← when trained; increases over time

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
    prob_orderbook: Optional[float] = None

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
    Works immediately out-of-the-box with orderbook + technical + BSM models.
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

    def _time_adjusted_weights(self, seconds_remaining: float) -> dict[str, float]:
        """Shift model weights based on time phase.

        Early (>480s): orderbook + momentum dominate. BSM is nearly useless
        (too much time, gentle probability curves).

        Prime (180-480s): balanced — all models contribute.

        Near settlement (<180s): BSM dominates — moneyness is king.
        Orderbook becomes thin-market noise, momentum matters less when
        time is so short.
        """
        base = dict(self.weights)
        if seconds_remaining > 480:
            # Early: boost flow/momentum, reduce BSM
            base["orderbook_imbalance"] = base.get("orderbook_imbalance", 0.25) + 0.05
            base["technical_momentum"] = base.get("technical_momentum", 0.35)
            base["trend_regression"] = base.get("trend_regression", 0.20) + 0.05
            base["binary_options_model"] = max(0.02, base.get("binary_options_model", 0.10) - 0.08)
        elif seconds_remaining < 180:
            # Near settlement: BSM (moneyness) is the single best predictor
            base["binary_options_model"] = base.get("binary_options_model", 0.10) + 0.30
            base["orderbook_imbalance"] = max(0.05, base.get("orderbook_imbalance", 0.25) - 0.15)
            base["technical_momentum"] = max(0.10, base.get("technical_momentum", 0.35) - 0.15)
            base["trend_regression"] = max(0.05, base.get("trend_regression", 0.20) - 0.10)
        # Prime (180-480s): use base weights as-is
        return base

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
        orderbook_bid_depth: float = 0.0,
        orderbook_ask_depth: float = 0.0,
        min_edge: float = 0.04,
        min_confidence: float = 0.55,
    ) -> ModelOutput:
        out = ModelOutput(
            ticker=ticker,
            strike=strike,
            current_price=current_price,
            seconds_remaining=seconds_remaining,
        )

        # Kalshi mid-price (in 0–1 scale).
        # Only use prices that are actually meaningful (> 0). A yes_ask of 0 means
        # the data is missing or the market is at an extreme — don't compute a fake mid.
        if kalshi_yes_bid is not None and kalshi_yes_ask is not None and kalshi_yes_ask > 0:
            out.kalshi_yes_price = (kalshi_yes_bid + kalshi_yes_ask) / 2 / 100
        elif kalshi_yes_bid is not None and kalshi_yes_bid > 0:
            out.kalshi_yes_price = kalshi_yes_bid / 100

        # Time-adaptive weights: BSM dominates near settlement, flow/momentum early
        phase_weights = self._time_adjusted_weights(seconds_remaining)

        active_weights: dict[str, float] = {}
        probs: dict[str, float] = {}

        # ── 1. Orderbook imbalance ────────────────────────────────────────────
        # Forward-looking flow signal: heavy YES bid pressure → buyers loading YES
        p_ob = self._orderbook_imbalance_prob(orderbook_bid_depth, orderbook_ask_depth)
        if p_ob is not None:
            out.prob_orderbook = p_ob
            probs["orderbook_imbalance"] = p_ob
            active_weights["orderbook_imbalance"] = phase_weights.get("orderbook_imbalance", 0.25)

        # ── 2. Technical momentum ─────────────────────────────────────────────
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
            active_weights["technical_momentum"] = phase_weights.get("technical_momentum", 0.35)

            # ── 3. Trend regression ───────────────────────────────────────────
            # Gate on R² ≥ 0.30 — a noisy fit shouldn't contribute at all.
            # Slope multiplier dropped from 500 to 150: the previous value
            # saturated the [-1, 1] clip on tiny 0.002 slopes, making the
            # trend signal effectively binary at 1-min bar resolution.
            if (signals.trend_slope is not None
                    and signals.trend_r2 is not None
                    and signals.trend_r2 >= 0.30):
                slope_norm = np.clip(signals.trend_slope * 150, -1, 1)
                r2_confidence = signals.trend_r2
                p_trend = (1 + slope_norm) / 2
                # Shrink toward 0.5 when R² is low
                p_trend = 0.5 + (p_trend - 0.5) * r2_confidence
                out.prob_trend = p_trend
                probs["trend_regression"] = p_trend
                active_weights["trend_regression"] = phase_weights.get("trend_regression", 0.20)

        # ── 4. Binary options model (Black-Scholes digital) ───────────────────
        # Reduced weight: near-useful as directional sanity check only.
        # At <5 min remaining, d2 converges to step-function on moneyness.
        p_bsm = self._binary_option_prob(
            spot=current_price,
            strike=strike,
            seconds_remaining=seconds_remaining,
            annual_vol=annual_vol,
        )
        if p_bsm is not None:
            out.prob_binary_options = p_bsm
            probs["binary_options_model"] = p_bsm
            active_weights["binary_options_model"] = phase_weights.get("binary_options_model", 0.10)

        # ── 5. ML model ───────────────────────────────────────────────────────
        if self._ml_model is not None and len(bars) >= 20:
            try:
                features = self._build_ml_features(
                    strike, current_price, seconds_remaining, annual_vol, bars
                )
                p_ml = float(self._ml_model.predict_proba([features])[0][1])
                out.prob_ml = p_ml
                probs["ml_model"] = p_ml
                active_weights["ml_model"] = phase_weights.get("ml_model", 0.10)
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

        # ── Confidence ────────────────────────────────────────────────────────
        # Conviction-weighted: distance of weighted mean from 0.5 (the actual edge),
        # shrunk by directional disagreement between sub-models. The old std-dev
        # formula rewarded clusters near 0.5 — we now require both edge AND
        # agreement, which kills the marginal noise that was firing trades.
        if len(probs) >= 2:
            model_probs = list(probs.values())
            weights_arr = [active_weights[k] for k in probs]
            w_sum = sum(weights_arr) or 1.0
            weighted_mean = sum(p * w for p, w in zip(model_probs, weights_arr)) / w_sum
            conviction = abs(weighted_mean - 0.5) * 2  # 0 (neutral) → 1 (extreme)
            same_side = sum(
                1 for p in model_probs if (p > 0.5) == (weighted_mean > 0.5)
            )
            agreement = same_side / len(model_probs)  # 1.0 = unanimous
            base_confidence = conviction * (0.5 + 0.5 * agreement)
        else:
            base_confidence = 0.0

        # Volume velocity boost: if last 3 bars' avg volume > 1.3x overall avg,
        # something is happening — directional move is more likely to continue.
        # Lower trigger (1.3x) and higher cap (0.15) so this actually moves
        # the needle on real volume spikes; old 0.08 cap was nearly invisible.
        vol_boost = 0.0
        if len(bars) >= 5:
            recent_avg_vol = sum(b.volume for b in bars[-3:]) / 3
            overall_avg_vol = sum(b.volume for b in bars[-20:]) / min(len(bars), 20)
            if overall_avg_vol > 0 and recent_avg_vol > 0:
                velocity = recent_avg_vol / overall_avg_vol
                if velocity > 1.3:
                    vol_boost = min(0.15, (velocity - 1.3) * 0.10)

        out.confidence = min(0.99, base_confidence + vol_boost)

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

    def _orderbook_imbalance_prob(
        self,
        bid_depth: float,
        ask_depth: float,
    ) -> Optional[float]:
        """
        Convert orderbook depth imbalance into a directional probability.

        Heavy YES bid depth → buyers loading up → bullish → P(YES) > 0.5
        Heavy YES ask depth → sellers → bearish → P(YES) < 0.5

        Threshold: require at least 10 total contracts to avoid noise from
        thin markets where 1 contract creates 100% imbalance.

        Mapping (linear amplification, capped):
          ratio=0.70 (70% bid) → P(YES) ≈ 0.78
          ratio=0.50 (balanced) → P(YES) = 0.50
          ratio=0.30 (70% ask) → P(YES) ≈ 0.22
        """
        total = bid_depth + ask_depth
        if total < 10:
            return None  # too thin; no signal

        ratio = bid_depth / total  # 0–1; 0.5 = balanced
        # Amplify signal by 1.4× but cap at extremes
        p = 0.5 + (ratio - 0.5) * 1.4
        return float(np.clip(p, 0.10, 0.90))

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

        Note: at <5 minutes remaining, this function approaches a step-function
        on moneyness (spot vs strike), so its weight is intentionally low (0.10).
        It still provides value as a sanity check: if spot is far above strike
        with 8+ min left, BSM will correctly flag high P(YES).
        """
        if seconds_remaining < 10 or spot <= 0 or strike <= 0:
            return None

        T = max(seconds_remaining, 30) / (365 * 24 * 3600)  # years
        sigma = max(annual_vol, 0.10)

        d2 = (math.log(spot / strike) + (drift - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        p = float(norm.cdf(d2))
        return float(np.clip(p, 0.01, 0.99))

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

        # Guard against degenerate markets (strike=0 or price=0 from a stale/malformed snapshot).
        # Matches the same guard used in _bsm_prob_yes above.
        if strike > 0 and current_price > 0:
            moneyness = math.log(current_price / strike)
        else:
            moneyness = 0.0
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
