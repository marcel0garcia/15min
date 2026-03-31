"""
Historical data bootstrap — trains the ML model WITHOUT waiting for live trades.

Fetches years of 1-minute BTC/USD bars from Kraken REST (free, no auth),
reconstructs what our model features would have been at each 15-min window,
labels the outcome (did price go up?), and trains LightGBM on all of it.

Usage:
    python -m btc15.models.bootstrap             # 6 months (default, ~18k windows)
    python -m btc15.models.bootstrap --months 12 # 1 year
    python -m btc15.models.bootstrap --months 24 # 2 years (~73k windows)

Each 15-min window generates 3 training samples (entry at t=0, t=5, t=10 min),
so 6 months → ~55,000 samples.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time
from pathlib import Path

import aiohttp
import numpy as np

log = logging.getLogger(__name__)

DATA_PATH = Path("data/training_data.npz")
# Binance.US REST — works from US IPs, deep history, 1000 bars per call
BINANCE_US_URL = "https://api.binance.us/api/v3/klines"
PAGE_SIZE = 1000         # Binance.US max per call
RATE_LIMIT_SEC = 0.20    # 5 req/sec, safe
ENTRY_OFFSETS = [0, 5, 10]  # Simulate entering at 0, 5, and 10 min into each window


async def fetch_1min_bars(months: int = 6) -> list[list]:
    """
    Fetch 1-minute OHLCV bars from Binance.US going back `months` months.
    Returns list of [timestamp_s, open, high, low, close, volume].
    """
    all_bars: list[list] = []
    start_ms = int((time.time() - months * 30 * 24 * 3600) * 1000)
    pages = 0

    async with aiohttp.ClientSession() as session:
        current_start_ms = start_ms
        while True:
            try:
                async with session.get(
                    BINANCE_US_URL,
                    params={
                        "symbol": "BTCUSDT",
                        "interval": "1m",
                        "limit": PAGE_SIZE,
                        "startTime": current_start_ms,
                    },
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        log.warning(f"Binance.US error {resp.status}: {text[:100]}, retrying...")
                        await asyncio.sleep(2)
                        continue
                    raw = await resp.json()

                if not raw or not isinstance(raw, list):
                    break

                # Binance kline: [open_time_ms, open, high, low, close, volume, ...]
                # Normalize to [timestamp_s, open, high, low, close, volume]
                bars = [[k[0] // 1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
                        for k in raw]

                all_bars.extend(bars)
                pages += 1
                last_bar_ms = raw[-1][0]  # open_time of last bar

                if pages % 50 == 0:
                    import datetime
                    dt = datetime.datetime.utcfromtimestamp(bars[0][0])
                    total_bars = len(all_bars)
                    windows = total_bars // 15
                    samples = windows * len(ENTRY_OFFSETS)
                    print(f"  [{pages} pages] {total_bars:,} bars  "
                          f"currently at {dt.strftime('%Y-%m-%d')} "
                          f"→ ~{samples:,} samples so far...")

                # Stop if last page (fewer bars than requested or caught up to now)
                if len(raw) < PAGE_SIZE:
                    break
                if last_bar_ms >= int(time.time() * 1000) - 120_000:
                    break

                # Advance to the bar after the last one we got
                current_start_ms = last_bar_ms + 60_000  # +1 minute
                await asyncio.sleep(RATE_LIMIT_SEC)

            except asyncio.TimeoutError:
                log.warning("Request timed out, retrying...")
                await asyncio.sleep(2)
            except Exception as e:
                log.error(f"Fetch error: {e}, retrying...")
                await asyncio.sleep(3)

    # Deduplicate by timestamp
    seen: set[int] = set()
    deduped = []
    for bar in all_bars:
        ts = bar[0]
        if ts not in seen:
            seen.add(ts)
            deduped.append(bar)
    deduped.sort(key=lambda x: x[0])
    return deduped


def bars_to_closes(bars: list[list]) -> list[float]:
    # bars format: [timestamp_s, open, high, low, close, volume]
    return [float(b[4]) for b in bars]


def realized_vol(closes: list[float], lookback: int = 30) -> float:
    """Annualized realized vol from 1-min log returns."""
    if len(closes) < 5:
        return 0.80
    arr = np.array(closes[-lookback:])
    returns = np.diff(np.log(arr))
    bars_per_year = 365 * 24 * 60  # 1-min bars
    return float(np.std(returns) * np.sqrt(bars_per_year))


def build_features(
    closes_before_entry: list[float],
    strike: float,
    current_price: float,
    seconds_remaining: float,
    annual_vol: float,
) -> list[float]:
    """
    Reproduce the exact same feature vector as EnsembleModel._build_ml_features.
    Must stay in sync with that method.
    """
    from btc15.models.technical import compute_signals

    signals = compute_signals(closes_before_entry[-30:])
    moneyness = math.log(current_price / strike) if strike > 0 else 0.0
    t_remaining = seconds_remaining / 900.0  # normalized 0–1

    return [
        moneyness,
        t_remaining,
        annual_vol,
        (signals.rsi / 100) if signals.rsi is not None else 0.5,
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


def build_training_data(bars: list[list]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raw OHLCV bars into (X, y) training arrays.

    For each 15-minute window, simulate entering at t=0, t=5, t=10 minutes.
    Label = 1 if close at end of window >= close at start (YES resolves).
    """
    closes = bars_to_closes(bars)
    n = len(closes)

    WINDOW = 15   # bars per window (1 min each)
    LOOKBACK = 35 # bars of history needed before entry for features

    X_list = []
    y_list = []
    skipped = 0

    # Iterate over each 15-min window
    for window_start in range(LOOKBACK, n - WINDOW, WINDOW):
        strike = closes[window_start]        # price at window open
        settlement = closes[window_start + WINDOW - 1]  # price at close
        outcome = 1 if settlement >= strike else 0

        vol = realized_vol(closes[:window_start], lookback=30)

        for offset in ENTRY_OFFSETS:
            entry_idx = window_start + offset
            if entry_idx >= n - 1:
                continue

            history = closes[entry_idx - LOOKBACK:entry_idx + 1]
            if len(history) < 15:
                skipped += 1
                continue

            current_price = closes[entry_idx]
            seconds_remaining = (WINDOW - offset) * 60.0

            try:
                features = build_features(
                    closes_before_entry=history,
                    strike=strike,
                    current_price=current_price,
                    seconds_remaining=seconds_remaining,
                    annual_vol=vol,
                )
                X_list.append(features)
                y_list.append(outcome)
            except Exception:
                skipped += 1
                continue

    if skipped:
        log.debug(f"Skipped {skipped} windows (insufficient history)")

    return np.array(X_list, dtype=float), np.array(y_list, dtype=int)


def train_on_data(X: np.ndarray, y: np.ndarray) -> bool:
    try:
        import joblib
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, accuracy_score
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        return False

    print(f"\nTraining LightGBM on {len(X):,} samples...")
    print(f"Class balance: {y.mean():.1%} YES / {1-y.mean():.1%} NO")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=20,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"Validation AUC:      {val_auc:.4f}  (0.5=random, 1.0=perfect)")
    print(f"Validation Accuracy: {val_acc:.1%}")

    # Feature importance
    feature_names = [
        "moneyness", "t_remaining", "annual_vol",
        "rsi", "macd_hist", "bb_pct", "bb_zscore",
        "mom_1m", "mom_5m", "mom_15m",
        "trend_slope", "trend_r2", "ema_trend",
    ]
    importances = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: -x[1]
    )
    print("\nTop feature importances:")
    for name, imp in importances[:7]:
        bar = "█" * int(imp / max(i for _, i in importances) * 20)
        print(f"  {name:15s} {bar} {imp:.0f}")

    model_path = Path("data/ml_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    return True


async def main(months: int = 6):
    import datetime
    start_date = datetime.datetime.utcnow() - datetime.timedelta(days=months * 30)
    print(f"Fetching 1-minute BTC/USD bars from Kraken")
    print(f"Range: {start_date.strftime('%Y-%m-%d')} → now  ({months} months)")
    print(f"Pages needed: ~{months * 30 * 24 * 60 // PAGE_SIZE:,}  (each = 12 hrs)")
    print(f"Estimated time: ~{months * 30 * 24 * 60 // PAGE_SIZE * RATE_LIMIT_SEC:.0f} seconds\n")

    t0 = time.time()
    bars = await fetch_1min_bars(months=months)
    elapsed = time.time() - t0
    print(f"\nFetched {len(bars):,} bars in {elapsed:.1f}s")

    if len(bars) < 1000:
        print("Not enough data fetched. Check your internet connection.")
        return

    print("Building training samples...")
    X, y = build_training_data(bars)
    print(f"Generated {len(X):,} training samples from {len(bars) // 15:,} windows")

    # Also save in the same format as the live collector so they accumulate together
    DATA_PATH.parent.mkdir(exist_ok=True)
    if DATA_PATH.exists():
        existing = np.load(DATA_PATH, allow_pickle=True)
        X = np.vstack([existing["X"], X])
        y = np.append(existing["y"], y)
        print(f"Merged with {len(existing['X']):,} existing live samples → {len(X):,} total")
    np.savez(DATA_PATH, X=X, y=y)

    train_on_data(X, y)
    total = time.time() - t0
    print(f"\nDone in {total:.0f}s total. Run ./run.sh to start trading with the trained model.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Bootstrap ML model from historical BTC data")
    parser.add_argument("--months", type=int, default=6, help="Months of history to fetch (default 6)")
    args = parser.parse_args()
    asyncio.run(main(months=args.months))
