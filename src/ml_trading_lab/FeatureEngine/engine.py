"""Feature-engineering boundary."""


class FeatureEngine:
    """Build features from information available at each event time only."""

    def transform(self, market_data: object) -> object:
        """Return feature rows aligned with source timestamps."""
        raise NotImplementedError("Feature engineering is not implemented yet.")
