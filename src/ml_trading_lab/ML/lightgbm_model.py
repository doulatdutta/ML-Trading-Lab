"""Full LightGBM model wrapper for strategy setup win-probability classification."""

import os
from typing import Dict, Any, List, Optional
import numpy as np


class LightGBMModel:
    """Wrapper around LGBMClassifier for predicting trade setup outcomes."""

    framework = "lightgbm"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")

        params = parameters or {}
        self.model = lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
            min_child_samples=params.get("min_child_samples", 10),
            random_state=params.get("random_state", 42),
            verbosity=-1,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit LGBMClassifier on training data."""
        self.model.fit(X, y)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of the positive class (TP hit before SL hit)."""
        probs = self.model.predict_proba(X)
        if probs.shape[1] > 1:
            return probs[:, 1]
        return probs[:, 0]

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        """Return feature→importance mapping sorted descending."""
        importances = self.model.feature_importances_
        if len(importances) != len(feature_names):
            return {f"feature_{i}": float(v) for i, v in enumerate(importances)}
        total = importances.sum() or 1.0
        feat_dict = {name: float(val / total) for name, val in zip(feature_names, importances)}
        return dict(sorted(feat_dict.items(), key=lambda x: x[1], reverse=True))

    def save_model(self, path: str) -> None:
        """Save model to disk in LightGBM text format."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.booster_.save_model(path)

    def load_model(self, path: str) -> None:
        """Load model state from disk."""
        import lightgbm as lgb
        booster = lgb.Booster(model_file=path)
        self.model.booster_ = booster
        self._fitted = True
