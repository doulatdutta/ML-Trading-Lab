"""Full CatBoost model wrapper for strategy setup win-probability classification."""

import os
from typing import Dict, Any, List, Optional
import numpy as np


class CatBoostModel:
    """Wrapper around CatBoostClassifier for predicting trade setup outcomes."""

    framework = "catboost"

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError("catboost not installed. Run: pip install catboost")

        params = parameters or {}
        self.model = CatBoostClassifier(
            iterations=params.get("iterations", 200),
            depth=params.get("depth", 5),
            learning_rate=params.get("learning_rate", 0.05),
            l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
            subsample=params.get("subsample", 0.8),
            random_seed=params.get("random_seed", 42),
            verbose=0,
            allow_writing_files=False,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit CatBoostClassifier on training data."""
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
        importances = self.model.get_feature_importance()
        if len(importances) != len(feature_names):
            return {f"feature_{i}": float(v) for i, v in enumerate(importances)}
        total = importances.sum() or 1.0
        feat_dict = {name: float(val / total) for name, val in zip(feature_names, importances)}
        return dict(sorted(feat_dict.items(), key=lambda x: x[1], reverse=True))

    def save_model(self, path: str) -> None:
        """Save model to disk in CatBoost format."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)

    def load_model(self, path: str) -> None:
        """Load model state from disk."""
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        self._fitted = True
