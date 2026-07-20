"""Optimization boundary."""


class StrategyOptimizer:
    """Propose reviewable candidates; never alter a deployed EA directly."""

    def propose(self, baseline: object, constraints: object) -> list[object]:
        """Return candidate changes after implementation is added."""
        raise NotImplementedError("Optimization is not implemented yet.")
