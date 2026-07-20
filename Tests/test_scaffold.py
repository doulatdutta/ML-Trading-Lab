"""Smoke tests for package boundaries."""

from ml_trading_lab import __version__
from ml_trading_lab.ML import ModelManager
from ml_trading_lab.StrategyEngine import StrategyDefinition


def test_package_is_importable() -> None:
    """The initial package exposes its version."""
    assert __version__ == "0.1.0"


def test_core_placeholders_are_available() -> None:
    """The initial contracts can be imported without optional dependencies."""
    assert ModelManager is not None
    assert StrategyDefinition(name="baseline", version="0.1").name == "baseline"
