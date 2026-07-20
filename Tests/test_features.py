"""Tests for FeatureEngine calculation correctness and look-ahead bias safety using Polars."""

import numpy as np
import polars as pl
import pytest
from datetime import datetime

from ml_trading_lab.FeatureEngine import FeatureEngine
from ml_trading_lab.DataEngine import CSVMarketDataCollector


@pytest.fixture
def sample_market_data() -> pl.DataFrame:
    """Generates a small predictable Polars DataFrame for testing features."""
    start_dt = datetime(2023, 1, 1, 0, 0, 0)
    end_dt = datetime(2023, 1, 2, 0, 0, 0)
    dates = pl.datetime_range(start_dt, end_dt, interval="1m", eager=True)
    n_bars = len(dates)
    df = pl.DataFrame(
        {
            "timestamp": dates,
            "open": np.ones(n_bars) * 100.0,
            "high": np.ones(n_bars) * 101.0,
            "low": np.ones(n_bars) * 99.0,
            "close": np.ones(n_bars) * 100.0,
            "tick_volume": np.ones(n_bars) * 500,
            "spread": np.ones(n_bars) * 2.0,
        }
    )
    return df


def test_feature_engine_constant_input(sample_market_data: pl.DataFrame) -> None:
    """Verify that on flat input, features compute without error and match expected properties."""
    engine = FeatureEngine()
    features = engine.transform(sample_market_data)

    # All EMAs should converge to the constant close value (100.0)
    assert np.isclose(features["ema_fast"][50], 100.0)
    assert np.isclose(features["sm_basis"][50], 100.0)
    assert np.isclose(features["emaUp"][50], 100.0)
    assert np.isclose(features["emaDn"][50], 100.0)

    # Since prices are constant, BB std is 0. Upper, middle, lower bands should equal close
    bb_middle_clean = features["bb_middle"].drop_nulls()
    bb_upper_clean = features["bb_upper"].drop_nulls()
    bb_lower_clean = features["bb_lower"].drop_nulls()
    bb_width_clean = features["bb_width"].drop_nulls()

    assert np.isclose(bb_middle_clean[0], 100.0)
    assert np.isclose(bb_upper_clean[0], 100.0)
    assert np.isclose(bb_lower_clean[0], 100.0)
    assert np.isclose(bb_width_clean[0], 0.0)


def test_feature_engine_lookahead_bias(sample_market_data: pl.DataFrame) -> None:
    """Verify that changing a value at index T does not change features at t < T."""
    engine = FeatureEngine()
    
    # Get original features
    features_original = engine.transform(sample_market_data)
    
    # Modify the last row of raw data
    modified_data = sample_market_data.clone()
    # Modify close of the last bar
    last_idx = len(modified_data) - 1
    modified_data[last_idx, "close"] = 200.0
    modified_data[last_idx, "high"] = 205.0
    
    # Get modified features
    features_modified = engine.transform(modified_data)
    
    # Assert that all rows except the last one are identical
    cols_to_compare = [
        "ema_fast", "sm_basis", "emaUp", "emaDn", "totalGap",
        "bb_middle", "bb_upper", "bb_lower", "bb_width",
        "atr", "bb_position"
    ]
    
    for col in cols_to_compare:
        # Compare all rows except the last one
        original_slice = features_original[col].head(last_idx)
        modified_slice = features_modified[col].head(last_idx)
        
        # Check that non-null values match exactly
        original_clean = original_slice.drop_nulls()
        modified_clean = modified_slice.drop_nulls()
        
        assert original_clean.equals(modified_clean), f"Look-ahead bias detected in feature: {col}"


def test_csv_collector_synthetic_generation(tmp_path) -> None:
    """Verify that the CSVMarketDataCollector successfully generates synthetic data when files are missing."""
    collector = CSVMarketDataCollector(raw_directory=str(tmp_path))
    
    symbol = "XAUUSD"
    timeframe = "M1"
    start = "2023-01-02"
    end = "2023-01-06"
    
    df = collector.load_bars(symbol, timeframe, start, end)
    
    # Check that df is not empty and has correct columns
    assert not df.is_empty()
    assert all(col in df.columns for col in ["timestamp", "open", "high", "low", "close", "tick_volume", "spread"])
    
    # Check that time index filters correctly
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(f"{end} 23:59:59", "%Y-%m-%d %H:%M:%S")
    assert df["timestamp"].min() >= start_dt
    assert df["timestamp"].max() <= end_dt
    
    # Check that file was saved to raw_directory
    saved_file = tmp_path / f"{symbol}_{timeframe}.csv"
    assert saved_file.exists()
