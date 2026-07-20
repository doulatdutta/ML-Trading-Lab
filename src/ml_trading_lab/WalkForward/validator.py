"""Walk-forward validation boundary."""


class WalkForwardValidator:
    """Evaluate candidates across sequential unseen time windows."""

    def validate(self, candidate: object, dataset: object) -> object:
        """Return validation evidence; implementation is deferred."""
        raise NotImplementedError("Walk-forward validation is not implemented yet.")
