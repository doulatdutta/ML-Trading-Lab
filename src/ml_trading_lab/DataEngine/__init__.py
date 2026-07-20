"""Market-data ingestion contracts and MT5 integration boundary."""

from .collector import MarketDataCollector
from .csv_collector import CSVMarketDataCollector

__all__ = ["MarketDataCollector", "CSVMarketDataCollector"]
