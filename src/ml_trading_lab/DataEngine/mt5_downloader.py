"""MT5 bulk historical data downloader — downloads 1-year of bars and saves to CSV cache."""

import os
from datetime import datetime, timedelta
from typing import Optional
import polars as pl


class MT5DataDownloader:
    """Download historical OHLCV bars from a connected MT5 terminal and cache locally."""

    def __init__(self, raw_directory: str = "Config/raw") -> None:
        self.raw_directory = raw_directory
        os.makedirs(raw_directory, exist_ok=True)

    def download(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "M1",
        years: float = 1.0,
        overwrite: bool = False,
        progress_callback=None,
    ) -> pl.DataFrame:
        """Download bars for the given symbol/timeframe covering the last N years.

        Saves result to Config/raw/{symbol}_{timeframe}.csv.
        Returns the downloaded DataFrame.
        """
        try:
            import MetaTrader5 as mt5
        except ImportError:
            raise RuntimeError("MetaTrader5 package not installed.")

        csv_path = os.path.join(self.raw_directory, f"{symbol}_{timeframe}.csv")
        if os.path.exists(csv_path) and not overwrite:
            if progress_callback:
                progress_callback(100, f"Cache exists at {csv_path} — use overwrite=True to re-download.")
            return pl.read_csv(csv_path).with_columns(
                pl.col("timestamp").str.to_datetime()
            )

        _TF_MAP = {
            "M1":  mt5.TIMEFRAME_M1,
            "M3":  mt5.TIMEFRAME_M3,
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
        }
        tf_const = _TF_MAP.get(timeframe)
        if tf_const is None:
            raise ValueError(f"Unknown timeframe: {timeframe}. Valid: {list(_TF_MAP)}")

        if progress_callback:
            progress_callback(5, "Initializing MT5 connection...")

        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

        # Ensure symbol is selected in Market Watch
        if not mt5.symbol_select(symbol, True):
            # Check if symbol exists in MT5 at all
            sym_info = mt5.symbol_info(symbol)
            if sym_info is None:
                # Find active symbols to provide guidance
                active_symbols = [s.name for s in mt5.symbols_get() if s.select]
                mt5.shutdown()
                raise RuntimeError(
                    f"Symbol '{symbol}' not found in MetaTrader 5. Please verify if your broker uses "
                    f"a suffix (e.g. XAUUSD.m, XAUUSD.raw, GOLD). "
                    f"Active Market Watch symbols: {active_symbols[:10]}"
                )

        end_dt = datetime.now()
        
        # Try fetching with progressively smaller windows if we get invalid params
        # (often caused by 'Max bars in chart' limits in MT5 options or lack of history from broker)
        ranges_to_try = [years * 365, 180, 90, 30, 10]
        success = False
        import time
        rates = None
        
        for days in ranges_to_try:
            start_dt = end_dt - timedelta(days=int(days))
            if progress_callback:
                progress_callback(15, f"Downloading {symbol} {timeframe} ({int(days)} days)...")
                
            # MT5 history sync is asynchronous; add retries with sleep to allow data download
            for attempt in range(5):
                rates = mt5.copy_rates_range(symbol, tf_const, start_dt, end_dt)
                if rates is not None and len(rates) > 0:
                    success = True
                    break
                if progress_callback:
                    progress_callback(20 + attempt * 10, f"Syncing history from broker ({int(days)} days)... attempt {attempt + 1}/5")
                time.sleep(1.0)
                
            if success:
                break
                
        mt5.shutdown()
        
        if not success or rates is None or len(rates) == 0:
            raise RuntimeError(
                f"No bars returned for {symbol} {timeframe} even after falling back to shorter windows. "
                f"Please check: \n"
                f"1. Is MT5 logged into an active trading account?\n"
                f"2. Open the {symbol} M1 chart in MT5 and press 'Home' key repeatedly to force download history.\n"
                f"3. In MT5, go to Tools -> Options -> Charts and set 'Max bars in chart' to 'Unlimited'."
            )

        if progress_callback:
            progress_callback(70, f"Received {len(rates):,} bars — processing...")

        import numpy as np
        timestamps = [datetime.fromtimestamp(r["time"]) for r in rates]
        df = pl.DataFrame({
            "timestamp": timestamps,
            "open":        [float(r["open"])  for r in rates],
            "high":        [float(r["high"])  for r in rates],
            "low":         [float(r["low"])   for r in rates],
            "close":       [float(r["close"]) for r in rates],
            "tick_volume": [int(r["tick_volume"]) for r in rates],
            "spread":      [float(r["spread"]) for r in rates],
        })

        if progress_callback:
            progress_callback(90, f"Saving {len(df):,} bars to {csv_path}...")

        df.write_csv(csv_path)

        if progress_callback:
            progress_callback(100, f"✅ Done — {len(df):,} {symbol} {timeframe} bars saved to {csv_path}")

        return df
