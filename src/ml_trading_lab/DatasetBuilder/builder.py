"""Create chronological ML datasets from strategy events."""


class DatasetBuilder:
    """Attach outcomes to point-in-time feature rows without feature leakage."""

    def build(self, features: object, events: object) -> object:
        """Return a labeled dataset; implementation is deferred."""
        raise NotImplementedError("Dataset labeling is not implemented yet.")
