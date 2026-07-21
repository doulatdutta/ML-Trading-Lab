"""Tests to verify strategy alignment with M1/M3 BB-EMA crossover logic."""

import polars as pl
import pytest
from datetime import datetime, timedelta
import numpy as np

from ml_trading_lab.FeatureEngine.engine import FeatureEngine
from ml_trading_lab.StrategyEngine.strategy import EMASmoothingBBStrategy, TradeSetup
from ml_trading_lab.DatasetBuilder.builder import DatasetBuilder


@pytest.fixture
def base_alignment_data() -> pl.DataFrame:
    """Creates flat mock primary data of 50 bars."""
    start_dt = datetime(2023, 1, 1, 9, 0, 0)
    dates = pl.datetime_range(start_dt, start_dt + timedelta(hours=2), interval="1m", eager=True)
    n_bars = len(dates)
    
    df = pl.DataFrame({
        "timestamp": dates,
        "open": np.ones(n_bars) * 102.0,
        "high": np.ones(n_bars) * 103.0,
        "low": np.ones(n_bars) * 101.0,
        "close": np.ones(n_bars) * 102.0,
        "tick_volume": np.ones(n_bars) * 500,
        "spread": np.ones(n_bars) * 2.0,
    })
    return df


def test_feature_engine_crossovers(base_alignment_data: pl.DataFrame) -> None:
    """Verify that FeatureEngine calculates outer EMA Bands, slopes, and totalGap correctly."""
    engine = FeatureEngine()
    features = engine.transform(base_alignment_data)
    
    # On flat data:
    # EMA fast converges to close (102.0)
    # sm_basis (rolling mean of ema_fast) converges to close (102.0)
    # sm_std is 0.0, so emaUp and emaDn should converge to close (102.0)
    assert np.isclose(features["emaUp"][30], 102.0)
    assert np.isclose(features["emaDn"][30], 102.0)
    assert np.isclose(features["totalGap"][30], 0.0)


def test_strategy_alignment_crossovers(base_alignment_data: pl.DataFrame) -> None:
    """Verify strategy detects M1 crossovers and M3 trend alignment accurately."""
    df_m1 = base_alignment_data.clone()
    
    # Configure a Long crossover breakout at index 25
    # Long condition requires:
    # 1. bbDn[1] <= emaDn[1] and bbDn[0] > emaDn[0] (crossover)
    # 2. bbUp rising
    # 3. totalGap expanding
    # 4. totalGap squeezed
    # For testing, we can directly set these values at index 25 to force trigger:
    # We populate the required columns in the DataFrame
    df_features = FeatureEngine().transform(df_m1)
    df_features = df_features.with_columns([
        pl.lit(True).alias("m3_longAlign"),
        pl.lit(True).alias("m3_shortAlign")
    ])
    
    # Setup crossover at index 25:
    # bb_lower[24] = 100.0, emaDn[24] = 101.0 (bbDn <= emaDn is True)
    # bb_lower[25] = 102.0, emaDn[25] = 101.5 (bbDn > emaDn is True)
    # bb_upper[25] = 104.0, bbUp_sl[25] = 103.0 (bbUp rising is True)
    # totalGap[24] = 1.0, totalGap[25] = 2.0 (expanding & squeezed <= 5.0 is True)
    df_features[24, "bb_lower"] = 100.0
    df_features[24, "emaDn"] = 101.0
    df_features[25, "bb_lower"] = 102.0
    df_features[25, "emaDn"] = 101.5
    
    df_features[25, "bb_upper"] = 104.0
    df_features[25, "bbUp_sl"] = 103.0
    
    df_features[24, "totalGap"] = 1.0
    df_features[25, "totalGap"] = 2.0
    
    strategy = EMASmoothingBBStrategy(parameters={"sqz_threshold": 5.0})
    setups = strategy.detect_setups(df_features, "XAUUSD", "M1")
    
    assert len(setups) == 1
    assert setups[0].direction == "long"


def test_dataset_builder_early_exits(base_alignment_data: pl.DataFrame) -> None:
    """Verify that DatasetBuilder performs early exits when slope exit triggers."""
    df = base_alignment_data.clone()
    
    # Setup setup at index 20
    setup = TradeSetup(
        setup_id="test_long_exit",
        timestamp=df[20, "timestamp"],
        symbol="XAUUSD",
        timeframe="M1",
        direction="long",
        entry_price=102.0,
        stop_loss=90.0,
        take_profit=115.0,
        features={}
    )
    
    df_features = FeatureEngine().transform(df)
    # Add dummy exit columns
    # Configure longExit true at index 25:
    # emaUp[1] > emaUp_1_sl and emaUp[0] < emaUp_sl
    # (emaUp[24] > emaUp_1_sl[25] and emaUp[25] < emaUp_sl[25])
    df_features[24, "emaUp"] = 105.0
    df_features[25, "emaUp_1_sl"] = 104.0 # 105.0 > 104.0 is True
    
    df_features[25, "emaUp"] = 101.0
    df_features[25, "emaUp_sl"] = 102.0 # 101.0 < 102.0 is True
    
    # Ensure SL/TP are not hit:
    df_features[25, "high"] = 103.0
    df_features[25, "low"] = 101.0
    # Early exit price is close at index 25
    df_features[25, "close"] = 107.0
    
    builder = DatasetBuilder()
    labeled = builder.build(df_features, [setup])
    
    assert len(labeled) == 1
    # Exit price matches close at index 25
    assert labeled[0, "exit_price"] == 107.0
    assert labeled[0, "bars_to_exit"] == 5 # 25 - 20 = 5 bars
    # realized_r = (107 - 102) / (102 - 90) = 5 / 12 = 0.4166...
    assert np.isclose(labeled[0, "realized_r"], 5.0 / 12.0)
