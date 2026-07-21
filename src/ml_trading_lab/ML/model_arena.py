"""Multi-Model Arena — trains XGBoost, LightGBM, CatBoost on the same dataset and compares them.

Each model is trained independently on the same feature matrix and evaluated on the same
held-out validation split. Results are ranked by ROC AUC.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.model_selection import train_test_split

from ml_trading_lab.ML.xgboost_model import XGBoostModel
from ml_trading_lab.ML.lightgbm_model import LightGBMModel
from ml_trading_lab.ML.catboost_model import CatBoostModel


MODEL_REGISTRY = {
    "XGBoost":  XGBoostModel,
    "LightGBM": LightGBMModel,
    "CatBoost": CatBoostModel,
}

_DEFAULT_PARAMS = {
    "XGBoost":  {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05},
    "LightGBM": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
    "CatBoost": {"iterations": 200, "depth": 5, "learning_rate": 0.05},
}


class ModelArena:
    """Train multiple ML models on the same dataset and compare their performance."""

    def __init__(self, val_fraction: float = 0.20, random_state: int = 42) -> None:
        self.val_fraction = val_fraction
        self.random_state = random_state
        # Stored results per model
        self.results: Dict[str, Dict[str, Any]] = {}
        self.trained_models: Dict[str, Any] = {}

    def train_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        progress_callback=None,
        custom_params: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Train all registered models and return a ranked comparison dict.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary target (1=win, 0=loss).
            feature_names: Names corresponding to X columns.
            progress_callback: Optional callable(pct: int, msg: str).
            custom_params: Override default hyperparameters per model name.

        Returns:
            Dict of {model_name: metrics_dict} sorted by AUC descending.
        """
        params = {**_DEFAULT_PARAMS, **(custom_params or {})}
        results: Dict[str, Dict[str, Any]] = {}

        # Validation split (chronological — last val_fraction rows)
        n_val = max(1, int(len(X) * self.val_fraction))
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        model_names = list(MODEL_REGISTRY.keys())
        for idx, name in enumerate(model_names):
            pct_start = int(idx / len(model_names) * 90)
            if progress_callback:
                progress_callback(pct_start, f"Training {name}...")

            t0 = time.time()
            try:
                ModelCls = MODEL_REGISTRY[name]
                model = ModelCls(parameters=params.get(name, {}))
                model.fit(X_train, y_train)

                # Predict on validation
                probs_val  = model.predict_proba(X_val)
                preds_val  = (probs_val >= 0.5).astype(int)

                # Metrics
                auc = float(roc_auc_score(y_val, probs_val)) if len(np.unique(y_val)) > 1 else 0.5
                acc = float(accuracy_score(y_val, preds_val))
                ll  = float(log_loss(y_val, probs_val)) if len(np.unique(y_val)) > 1 else 1.0

                importances = model.get_feature_importances(feature_names)
                elapsed = round(time.time() - t0, 2)

                results[name] = {
                    "auc":        round(auc, 4),
                    "accuracy":   round(acc, 4),
                    "log_loss":   round(ll, 4),
                    "train_size": len(X_train),
                    "val_size":   len(X_val),
                    "n_features": len(feature_names),
                    "train_time": elapsed,
                    "importances": dict(list(importances.items())[:15]),
                    "status":     "trained",
                }
                self.trained_models[name] = model

                if progress_callback:
                    progress_callback(
                        pct_start + int(85 / len(model_names)),
                        f"✅ {name} done — AUC={auc:.3f}, Acc={acc:.1%}, Time={elapsed}s"
                    )

            except Exception as e:
                results[name] = {"status": "error", "error": str(e), "auc": 0.0}
                if progress_callback:
                    progress_callback(pct_start, f"❌ {name} failed: {e}")

        # Rank by AUC
        self.results = dict(
            sorted(results.items(), key=lambda x: x[1].get("auc", 0), reverse=True)
        )
        if progress_callback:
            progress_callback(100, "All models trained and ranked.")
        return self.results

    def predict_all(self, X_row: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Get win probability from every trained model for the same feature row.

        Returns dict: {model_name: {"win_prob": float, "verdict": str, "auc": float}}
        """
        predictions: Dict[str, Dict[str, Any]] = {}
        for name, model in self.trained_models.items():
            try:
                prob = float(model.predict_proba(X_row)[0])
                predictions[name] = {
                    "win_prob":  round(prob, 4),
                    "verdict":   "BUY" if prob >= 0.55 else "SKIP",
                    "auc":       self.results.get(name, {}).get("auc", None),
                    "status":    "ok",
                }
            except Exception as e:
                predictions[name] = {"status": "error", "error": str(e), "win_prob": None}
        return predictions

    def best_model_name(self) -> Optional[str]:
        """Return the name of the model with the highest AUC."""
        if not self.results:
            return None
        return next(iter(self.results))

    def best_model(self):
        """Return the best trained model object."""
        name = self.best_model_name()
        return self.trained_models.get(name) if name else None

    def leaderboard(self) -> List[Dict[str, Any]]:
        """Return a ranked list of model results suitable for display."""
        board = []
        for rank, (name, metrics) in enumerate(self.results.items(), start=1):
            board.append({
                "rank":       rank,
                "model":      name,
                "auc":        metrics.get("auc"),
                "accuracy":   metrics.get("accuracy"),
                "log_loss":   metrics.get("log_loss"),
                "train_time": metrics.get("train_time"),
                "status":     metrics.get("status", "unknown"),
            })
        return board
