"""EA artifact-generation boundary."""


class EAGenerator:
    """Produce reviewable MQL5 artifacts only after validation approval."""

    def generate(self, approved_candidate: object) -> str:
        """Return an EA artifact path when MQL5 templates are implemented."""
        raise NotImplementedError("EA generation is not implemented yet.")
