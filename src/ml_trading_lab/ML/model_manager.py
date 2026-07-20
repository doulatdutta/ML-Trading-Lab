"""Model version promotion and lifecycle management using scikit-learn."""

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from typing import Dict, Any


class ModelManager:
    """Promote models only after predefined validation checks pass."""

    def __init__(self) -> None:
        """Initialize ModelManager with empty metadata records."""
        self.promotion_log: list[Dict[str, Any]] = []

    def promote_if_validated(self, candidate: Any, evidence: Dict[str, Any]) -> bool:
        """Evaluate validation metrics and promote the candidate model if threshold is met.

        evidence must contain:
        - X_val: validation feature matrix (numpy array)
        - y_val: validation binary outcomes (numpy array)
        - auc_threshold: float (default = 0.52)
        """
        X_val = evidence.get("X_val")
        y_val = evidence.get("y_val")
        auc_threshold = evidence.get("auc_threshold", 0.52)

        if X_val is None or y_val is None:
            raise ValueError("Validation data X_val and y_val must be provided in evidence.")

        # Ensure both classes are present in the validation labels (ROC AUC requires binary target diversity)
        unique_classes = np.unique(y_val)
        if len(unique_classes) < 2:
            print(f"Validation target diversity insufficient: only {unique_classes} present. Promotion rejected.")
            return False

        # 1. Compute validation predictions
        probs = candidate.predict_proba(X_val)

        # 2. Compute metrics
        val_auc = roc_auc_score(y_val, probs)
        val_loss = log_loss(y_val, probs)

        is_promoted = val_auc >= auc_threshold

        # Log promotion event
        log_entry = {
            "timestamp": np.datetime64("now"),
            "val_auc": float(val_auc),
            "val_logloss": float(val_loss),
            "auc_threshold": float(auc_threshold),
            "promoted": is_promoted
        }
        self.promotion_log.append(log_entry)

        print(f"Validation Completed. ROC AUC: {val_auc:.4f} (Threshold: {auc_threshold:.2f}). Log Loss: {val_loss:.4f}.")
        if is_promoted:
            print("Model candidate successfully PROMOTED to active baseline.")
        else:
            print("Model candidate REJECTED (insufficient statistical validation edge).")

        return is_promoted
