"""CSV-based and synthetic fallback market data collector implementation using Polars."""

import os
from datetime import datetime
import numpy as np
import polars as pl
from typing import Optional

from .collector import MarketDataCollector


class CSVMarketDataCollector(MarketDataCollector):
    """Loads historical bar data from CSV/Parquet files or generates synthetic data if missing."""

    def __init__(self, raw_directory: str = "data/raw") -> None:
        """Initialize the collector with a path to raw files."""
        self.raw_directory = raw_directory

    def load_bars(self, symbol: str, timeframe: str, start: Optional[str] = None, end: Optional[str] = None) -> pl.DataFrame:
        """Load OHLCV bars for the requested period from CSV/Parquet or fallback to synthetic data.

        Timestamps in start/end are string format, e.g., '2023-01-01'.
        """
        # Formulate file path candidates
        csv_filename = f"{symbol}_{timeframe}.csv"
        parquet_filename = f"{symbol}_{timeframe}.parquet"

        csv_path = os.path.join(self.raw_directory, csv_filename)
        parquet_path = os.path.join(self.raw_directory, parquet_filename)

        df: Optional[pl.DataFrame] = None

        if os.path.exists(parquet_path):
            df = pl.read_parquet(parquet_path)
        elif os.path.exists(csv_path):
            df = pl.read_csv(csv_path)

        if df is not None:
            # Ensure timestamp is datetime type
            if df["timestamp"].dtype == pl.String:
                df = df.with_columns(pl.col("timestamp").str.to_datetime())
            
            # Filter by start/end timestamps if provided
            if start and end:
                start_dt = datetime.strptime(start, "%Y-%m-%d")
                end_dt = datetime.strptime(f"{end} 23:59:59", "%Y-%m-%d %H:%M:%S")
                df = df.filter(
                    (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt)
                )
            return df

        # Fallback to generating synthetic data if files do not exist
        print(f"Historical file not found at {csv_path} or {parquet_path}. Generating synthetic {symbol} {timeframe} data...")
        start_dt = datetime.strptime(start, "%Y-%m-%d") if start else datetime(2023, 1, 1)
        end_dt = datetime.strptime(f"{end} 23:59:59", "%Y-%m-%d %H:%M:%S") if end else datetime(2023, 1, 31, 23, 59, 59)
        df_synthetic = self._generate_synthetic_bars(symbol, timeframe, start_dt, end_dt)

        # Make sure directory exists and write file for subsequent usage
        os.makedirs(self.raw_directory, exist_ok=True)
        df_synthetic.write_csv(csv_path)

        return df_synthetic

    def _generate_synthetic_bars(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> pl.DataFrame:
        """Generates realistic synthetic OHLCV data using a random walk model."""
        # Determine frequency
        freq_map = {
            "M1": "1m",
            "M5": "5m",
            "M15": "15m",
            "M30": "30m",
            "H1": "1h",
            "H4": "4h",
            "D1": "1d"
        }
        interval = freq_map.get(timeframe, "15m")

        # Generate datetime index using Polars datetime_range
        dates_all = pl.datetime_range(start_dt, end_dt, interval=interval, eager=True)
        
        # Filter out weekends (Saturday=6, Sunday=7 in Polars)
        dates = dates_all.filter(dates_all.dt.weekday() < 6)
        
        n_bars = len(dates)
        if n_bars == 0:
            raise ValueError(f"No bars generated for range {start_dt} to {end_dt} with timeframe {timeframe}")

        # Gold-like parameters if symbol is XAUUSD, otherwise standard Forex
        if "XAU" in symbol.upper():
            start_price = 2300.0
            volatility = 0.001  # 0.1% per bar (standard M15 variation)
            spread_mean = 1.5   # 1.5 dollars/points
        else:
            start_price = 1.1000
            volatility = 0.0003
            spread_mean = 0.00015

        # Simulating log returns
        np.random.seed(42)  # reproducible research
        returns = np.random.normal(loc=0.0, scale=volatility, size=n_bars)
        price_path = start_price * np.exp(np.cumsum(returns))

        opens = np.zeros(n_bars)
        closes = np.zeros(n_bars)
        highs = np.zeros(n_bars)
        lows = np.zeros(n_bars)

        # Set first open
        opens[0] = start_price
        closes[0] = price_path[0]
        
        for i in range(1, n_bars):
            opens[i] = closes[i - 1]
            closes[i] = price_path[i]

        # Generate highs and lows relative to opens and closes
        high_noise = np.abs(np.random.normal(loc=0.0, scale=volatility * 0.5, size=n_bars))
        low_noise = np.abs(np.random.normal(loc=0.0, scale=volatility * 0.5, size=n_bars))

        highs = np.maximum(opens, closes) + (opens * high_noise)
        lows = np.minimum(opens, closes) - (opens * low_noise)

        # Tick volume
        tick_volumes = np.random.randint(100, 1500, size=n_bars).astype(np.int64)

        # Spreads (in points/pips)
        spread_noise = np.random.normal(loc=0.0, scale=spread_mean * 0.1, size=n_bars)
        spreads = np.maximum(spread_mean * 0.5, spread_mean + spread_noise)

        # Construct Polars DataFrame
        df = pl.DataFrame({
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": tick_volumes,
            "spread": spreads,
        })

        return df
