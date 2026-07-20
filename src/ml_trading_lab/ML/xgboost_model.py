"""XGBoost model wrapper for strategy setup validation classification."""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import xgboost as xgb


class XGBoostModel:
    """Wrapper class around XGBClassifier for predicting trade setup outcomes."""

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Initialize XGBClassifier with default or custom hyperparameters."""
        params = parameters or {}
        # Safe default regularization parameters to prevent overfitting
        self.model = xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=params.get("random_state", 42),
            eval_metric="logloss",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the XGBoost classifier on the training feature matrix X and target labels y."""
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the win probability (TP hit before SL hit) for setups in X.

        Returns a 1D numpy array of probabilities.
        """
        # predict_proba returns probabilities for each class: shape (N, 2)
        # We return the probability for the positive class (TP hit = index 1)
        probs = self.model.predict_proba(X)
        if probs.shape[1] > 1:
            return probs[:, 1]
        return probs[:, 0]

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        """Return a dictionary mapping feature names to their importance scores."""
        importances = self.model.feature_importances_
        if len(importances) != len(feature_names):
            # Fallback/Truncate if length mismatch
            return {f"feature_{i}": float(val) for i, val in enumerate(importances)}
        
        feat_dict = {name: float(val) for name, val in zip(feature_names, importances)}
        # Sort by importance descending
        return dict(sorted(feat_dict.items(), key=lambda x: x[1], reverse=True))

    def save_model(self, path: str) -> None:
        """Save the underlying XGBoost model to disk in JSON format."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)

    def load_model(self, path: str) -> None:
        """Load the model state from a saved JSON file."""
        self.model.load_model(path)
