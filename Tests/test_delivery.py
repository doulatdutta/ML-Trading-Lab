"""Tests for StrategyOptimizer, WalkForwardValidator, LiveAdvisor, and EAGenerator."""

import os
import polars as pl
import pytest
from datetime import datetime, timedelta
import numpy as np

from ml_trading_lab.Optimizer.optimizer import StrategyOptimizer
from ml_trading_lab.WalkForward.validator import WalkForwardValidator
from ml_trading_lab.LiveAdvisor.advisor import LiveAdvisor
from ml_trading_lab.EA_Generator.generator import EAGenerator
from ml_trading_lab.StrategyEngine.strategy import TradeSetup, EMASmoothingBBStrategy


@pytest.fixture
def delivery_market_data() -> pl.DataFrame:
    """Creates a base flat market DataFrame of 100 bars for validation testing."""
    start_dt = datetime(2023, 1, 1, 9, 0, 0)
    dates = pl.datetime_range(start_dt, start_dt + timedelta(hours=25), interval="1m", eager=True)
    n_bars = len(dates)
    
    df = pl.DataFrame({
        "timestamp": dates,
        "open": np.ones(n_bars) * 102.0,
        "high": np.ones(n_bars) * 103.0,
        "low": np.ones(n_bars) * 101.0,
        "close": np.ones(n_bars) * 102.0,
        "tick_volume": np.ones(n_bars) * 500,
        "spread": np.ones(n_bars) * 2.0,
        "bb_upper": np.ones(n_bars) * 103.0,
        "bb_lower": np.ones(n_bars) * 102.0,
        "emaUp": np.ones(n_bars) * 103.0,
        "emaDn": np.ones(n_bars) * 101.5,
        "totalGap": np.ones(n_bars) * 2.0,
        "bbUp_sl": np.ones(n_bars) * 103.0,
        "bbDn_sl": np.ones(n_bars) * 101.0,
        "atr": np.ones(n_bars) * 2.0,
        "m3_longAlign": np.ones(n_bars, dtype=bool),
        "m3_shortAlign": np.ones(n_bars, dtype=bool),
    })
    return df


def test_strategy_optimizer(delivery_market_data: pl.DataFrame) -> None:
    """Verify that StrategyOptimizer evaluates candidates over a grid search."""
    df = delivery_market_data.clone()
    
    # Configure a Long crossover breakout at index 20:
    df[19, "bb_lower"] = 100.0
    df[19, "emaDn"] = 101.0
    df[20, "bb_lower"] = 102.0
    df[20, "emaDn"] = 101.5
    df[20, "bb_upper"] = 104.0
    df[20, "bbUp_sl"] = 103.0
    df[19, "totalGap"] = 1.0
    df[20, "totalGap"] = 2.0
    df[20, "close"] = 103.0
    
    # Configure a win outcome at index 25
    df[25, "high"] = 110.0
    
    optimizer = StrategyOptimizer()
    grid = {
        "sqz_threshold": [5.0],
        "atr_sl_mult": [1.5],
        "atr_tp_mult": [3.0]
    }
    
    best_params, best_report = optimizer.propose(df, grid, "XAUUSD", "M1")
    
    assert best_params["sqz_threshold"] == 5.0
    assert best_report["total_trades"] == 1
    assert best_report["wins"] == 1


def test_walk_forward_validator(delivery_market_data: pl.DataFrame) -> None:
    """Verify that WalkForwardValidator slices data and compares candidates against baseline."""
    df = delivery_market_data.clone()
    
    # Configure setups in both splits
    # Setup at index 20
    df[19, "bb_lower"] = 100.0
    df[19, "emaDn"] = 101.0
    df[20, "bb_lower"] = 102.0
    df[20, "emaDn"] = 101.5
    df[20, "bb_upper"] = 104.0
    df[20, "bbUp_sl"] = 103.0
    df[19, "totalGap"] = 1.0
    df[20, "totalGap"] = 2.0
    df[20, "close"] = 103.0
    df[25, "high"] = 110.0
    
    # Setup at index 70
    df[69, "bb_lower"] = 100.0
    df[69, "emaDn"] = 101.0
    df[70, "bb_lower"] = 102.0
    df[70, "emaDn"] = 101.5
    df[70, "bb_upper"] = 104.0
    df[70, "bbUp_sl"] = 103.0
    df[69, "totalGap"] = 1.0
    df[70, "totalGap"] = 2.0
    df[70, "close"] = 103.0
    df[75, "high"] = 110.0

    validator = WalkForwardValidator()
    grid = {
        "sqz_threshold": [5.0],
        "atr_sl_mult": [1.5],
        "atr_tp_mult": [3.0]
    }
    
    baseline = {
        "sqz_threshold": 5.0,
        "atr_sl_mult": 1.5,
        "atr_tp_mult": 3.0
    }
    
    report = validator.validate(df, grid, baseline, n_splits=2, symbol="XAUUSD", timeframe="M1")
    
    assert "approved" in report
    assert "walk_forward_metrics" in report
    assert "baseline_metrics" in report
    assert report["folds_evaluated"] == 2


def test_live_advisor_mocked() -> None:
    """Verify that LiveAdvisor evaluates mock rates data successfully."""
    # Create M1 and M3 histories
    start_dt = datetime(2023, 1, 1, 9, 0, 0)
    dates_m1 = pl.datetime_range(start_dt, start_dt + timedelta(hours=10), interval="1m", eager=True)
    n_m1 = len(dates_m1)
    
    df_m1 = pl.DataFrame({
        "timestamp": dates_m1,
        "open": np.ones(n_m1) * 102.0,
        "high": np.ones(n_m1) * 103.0,
        "low": np.ones(n_m1) * 101.0,
        "close": np.ones(n_m1) * 102.0,
        "tick_volume": np.ones(n_m1) * 500,
        "spread": np.ones(n_m1) * 2.0,
    })
    
    # Configure a close crossover setup at index -2
    df_m1[n_m1 - 2, "close"] = 105.0
    
    dates_m3 = pl.datetime_range(start_dt, start_dt + timedelta(hours=10), interval="3m", eager=True)
    n_m3 = len(dates_m3)
    df_m3 = pl.DataFrame({
        "timestamp": dates_m3,
        "open": np.ones(n_m3) * 102.0,
        "high": np.ones(n_m3) * 103.0,
        "low": np.ones(n_m3) * 101.0,
        "close": np.ones(n_m3) * 105.0,
        "tick_volume": np.ones(n_m3) * 500,
        "spread": np.ones(n_m3) * 2.0,
    })
    df_m3[0, "close"] = 100.0

    from unittest.mock import patch
    
    setup = TradeSetup(
        setup_id="long_123",
        timestamp=df_m1[-2, "timestamp"],
        symbol="XAUUSD",
        timeframe="M1",
        direction="long",
        entry_price=105.0,
        stop_loss=100.0,
        take_profit=115.0,
        features={"bb_width_percentile": 0.1, "bb_position": 0.5}
    )

    params = {
        "bb_std": 0.05,
        "sm_std": 0.05,
        "sqz_threshold": 10.0,
        "atr_sl_mult": 1.5,
        "atr_tp_mult": 3.0
    }
    
    advisor = LiveAdvisor()
    with patch.object(EMASmoothingBBStrategy, "detect_setups", return_value=[setup]):
        result = advisor.score(parameters=params, mock_data={"M1": df_m1, "M3": df_m3})
    
    assert result["status"] == "active_setup"
    assert result["direction"] == "long"
    assert "win_probability" in result


def test_ea_generator(tmp_path) -> None:
    """Verify that EAGenerator correctly outputs versioned MQL5 files."""
    generator = EAGenerator()
    params = {
        "bb_width_threshold": 0.22,
        "atr_sl_mult": 1.25,
        "atr_tp_mult": 3.25,
        "ema_fast_period": 18,
        "ema_slow_period": 48
    }
    
    out_file = os.path.join(str(tmp_path), "Test_EA.mq5")
    generator.generate(params, out_file)
    
    assert os.path.exists(out_file)
    with open(out_file, "r") as f:
        content = f.read()
        assert "InpBBWidthThreshold  = 0.22" in content
        assert "InpATRSLMultiplier   = 1.25" in content
        assert "InpATRTPMultiplier   = 3.25" in content
        assert "InpEMAFastPeriod     = 18" in content
        assert "InpEMASlowPeriod     = 48" in content
