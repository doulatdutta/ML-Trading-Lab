"""Strategy definition contracts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyDefinition:
    """A named, human-reviewable strategy specification, not live logic."""

    name: str
    version: str
