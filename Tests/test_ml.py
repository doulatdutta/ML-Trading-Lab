"""Tests for XGBoostModel wrapping and ModelManager validation and promotion rules."""

import os
import numpy as np

from ml_trading_lab.ML import XGBoostModel, ModelManager


def test_xgboost_model_fit_predict() -> None:
    """Verify that XGBoostModel wrapper can fit and predict probabilities correctly."""
    # Create simple binary classification data
    np.random.seed(42)
    X_train = np.random.normal(loc=0.0, scale=1.0, size=(100, 4))
    # y depends on the first feature
    y_train = (X_train[:, 0] > 0).astype(int)

    X_val = np.random.normal(loc=0.0, scale=1.0, size=(20, 4))

    model = XGBoostModel(parameters={"max_depth": 2, "n_estimators": 10})
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)

    # Output should be 1D array of shape (20,)
    assert probs.shape == (20,)
    # Probabilities must lie in [0, 1]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Verify feature importances map correctly
    feature_names = ["feat_A", "feat_B", "feat_C", "feat_D"]
    importances = model.get_feature_importances(feature_names)
    assert len(importances) == 4
    assert "feat_A" in importances
    assert isinstance(importances["feat_A"], float)


def test_xgboost_model_save_load(tmp_path) -> None:
    """Verify that XGBoostModel can be saved and loaded back with identical predictions."""
    np.random.seed(42)
    X_train = np.random.normal(loc=0.0, scale=1.0, size=(80, 3))
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_val = np.random.normal(loc=0.0, scale=1.0, size=(10, 3))

    model = XGBoostModel(parameters={"n_estimators": 10})
    model.fit(X_train, y_train)
    original_probs = model.predict_proba(X_val)

    # Save to temp file
    save_path = os.path.join(str(tmp_path), "model.json")
    model.save_model(save_path)
    assert os.path.exists(save_path)

    # Load into new model instance
    loaded_model = XGBoostModel()
    loaded_model.load_model(save_path)
    loaded_probs = loaded_model.predict_proba(X_val)

    # Predictions must match exactly
    assert np.allclose(original_probs, loaded_probs)


def test_model_manager_promotion() -> None:
    """Verify that ModelManager promotes models meeting the AUC threshold and rejects others."""
    manager = ModelManager()

    # Define validation target (3 class 0, 3 class 1)
    y_val = np.array([0, 0, 0, 1, 1, 1])
    X_val = np.zeros((6, 2))

    # Stub class to simulate model predictions
    class StubModel:
        def __init__(self, probabilities: np.ndarray) -> None:
            self.probabilities = probabilities
        
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return self.probabilities

    # 1. Bad model prediction (ROC AUC = 0.5, below 0.52 threshold)
    bad_probs = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    bad_candidate = StubModel(bad_probs)
    
    evidence_bad = {
        "X_val": X_val,
        "y_val": y_val,
        "auc_threshold": 0.52
    }
    assert manager.promote_if_validated(bad_candidate, evidence_bad) is False

    # 2. Perfect model prediction (ROC AUC = 1.0, above 0.52 threshold)
    good_probs = np.array([0.1, 0.1, 0.2, 0.8, 0.9, 0.9])
    good_candidate = StubModel(good_probs)
    
    evidence_good = {
        "X_val": X_val,
        "y_val": y_val,
        "auc_threshold": 0.52
    }
    assert manager.promote_if_validated(good_candidate, evidence_good) is True
