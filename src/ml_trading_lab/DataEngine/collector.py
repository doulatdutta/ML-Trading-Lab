"""Interfaces for importing historical and future MT5 market data."""

from typing import Protocol


class MarketDataCollector(Protocol):
    """Defines a point-in-time-safe market data source.

    Concrete implementations may read CSV, Parquet, a database, or the MT5
    Python API. They must return only data available at the requested time.
    """

    def load_bars(self, symbol: str, timeframe: str, start: str, end: str) -> object:
        """Return OHLCV/spread bars for the requested period."""
        ...
