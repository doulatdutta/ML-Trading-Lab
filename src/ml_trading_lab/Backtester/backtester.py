"""Backtesting boundary."""


class Backtester:
    """Simulate explicitly defined rules with spread and slippage assumptions."""

    def run(self, strategy: object, market_data: object) -> object:
        """Return a backtest report; implementation is deferred."""
        raise NotImplementedError("Backtesting is not implemented yet.")
