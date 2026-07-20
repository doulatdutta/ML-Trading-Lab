"""Model version promotion contract."""


class ModelManager:
    """Promote models only after predefined out-of-sample checks pass."""

    def promote_if_validated(self, candidate: object, evidence: object) -> bool:
        """Evaluate promotion evidence; implementation intentionally deferred."""
        raise NotImplementedError("Model promotion policy is not implemented yet.")
