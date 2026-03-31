"""
LightGBM model trainer and predictor.

Run `python -m btc15.models.ml_model` to train on collected data.
The bot collects training data automatically while running — the more it runs,
the better the ML model becomes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

MODEL_PATH = Path("data/ml_model.pkl")
DATA_PATH = Path("data/training_data.npz")


def train_model(min_samples: int = 500) -> bool:
    """Train LightGBM on collected training data. Returns True if successful."""
    try:
        import joblib
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        log.error(f"ML training dependencies missing: {e}")
        return False

    if not DATA_PATH.exists():
        log.warning(f"No training data at {DATA_PATH}. Run the bot first to collect data.")
        return False

    data = np.load(DATA_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    if len(X) < min_samples:
        log.warning(f"Only {len(X)} samples, need {min_samples}. Keep collecting data.")
        return False

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    log.info(f"ML model trained: {len(X)} samples, val AUC = {val_auc:.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    log.info(f"ML model saved to {MODEL_PATH}")
    return True


def collect_sample(features: list[float], outcome: int):
    """
    Append a training sample (features + 0/1 label) to the data store.
    Called by the strategy engine after each market settles.
    """
    DATA_PATH.parent.mkdir(exist_ok=True)
    x = np.array(features, dtype=float)
    if DATA_PATH.exists():
        existing = np.load(DATA_PATH, allow_pickle=True)
        X = np.vstack([existing["X"], x.reshape(1, -1)])
        y = np.append(existing["y"], outcome)
    else:
        X = x.reshape(1, -1)
        y = np.array([outcome])
    np.savez(DATA_PATH, X=X, y=y)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = train_model(min_samples=100)
    if not success:
        print("Training failed or insufficient data. See logs above.")
    else:
        print(f"Model trained and saved to {MODEL_PATH}")
