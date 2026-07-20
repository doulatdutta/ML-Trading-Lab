"""Live advisor boundary."""


class LiveAdvisor:
    """Provide advisory scores only; it must not submit orders."""

    def score(self, setup: object) -> object:
        """Return a future advisory response without order execution."""
        raise NotImplementedError("Live scoring is not implemented yet.")
