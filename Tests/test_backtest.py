"""Tests for StrategyEngine setup detection, DatasetBuilder labeling, and Backtester calculations using Polars."""

import polars as pl
import pytest
from datetime import datetime, timedelta
import numpy as np

from ml_trading_lab.StrategyEngine.strategy import EMASmoothingBBStrategy, TradeSetup
from ml_trading_lab.DatasetBuilder.builder import DatasetBuilder
from ml_trading_lab.Backtester.backtester import Backtester


@pytest.fixture
def base_market_data() -> pl.DataFrame:
    """Creates a base flat market DataFrame of 50 bars.

    We set high=103.0, low=101.0, close=102.0.
    For an entry of 103.0 with ATR=2.0, the SL is 100.0 and TP is 109.0.
    These settings ensure the trade is safe from SL/TP hits on flat bars.
    """
    start_dt = datetime(2023, 1, 1, 9, 0, 0)
    dates = pl.datetime_range(start_dt, start_dt + timedelta(hours=12), interval="1m", eager=True)
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


def test_strategy_setup_detection(base_market_data: pl.DataFrame) -> None:
    """Verify that setup detection triggers correctly under crossover and squeeze conditions."""
    df = base_market_data.clone()
    
    # Configure a Long crossover breakout at index 20:
    # 1. bbDn[1] <= emaDn[1] and bbDn[0] > emaDn[0] (crossover)
    df[19, "bb_lower"] = 100.0
    df[19, "emaDn"] = 101.0
    df[20, "bb_lower"] = 102.0
    df[20, "emaDn"] = 101.5
    # 2. bbUp rising (bb_upper[20] > bbUp_sl[20])
    df[20, "bb_upper"] = 104.0
    df[20, "bbUp_sl"] = 103.0
    # 3. totalGap expanding (totalGap[20] > totalGap[19])
    df[19, "totalGap"] = 1.0
    df[20, "totalGap"] = 2.0
    # 4. close at 20
    df[20, "close"] = 103.0
    
    strategy = EMASmoothingBBStrategy(parameters={"sqz_threshold": 5.0})
    setups = strategy.detect_setups(df, "XAUUSD", "M1")
    
    assert len(setups) == 1
    assert setups[0].direction == "long"
    assert setups[0].entry_price == 103.0
    # SL = Entry - 1.5 * ATR = 103.0 - 1.5 * 2.0 = 100.0
    assert np.isclose(setups[0].stop_loss, 100.0)
    # TP = Entry + 3.0 * ATR = 103.0 + 3.0 * 2.0 = 109.0
    assert np.isclose(setups[0].take_profit, 109.0)


def test_outcome_labeling_tp_first(base_market_data: pl.DataFrame) -> None:
    """Verify that TP hit before SL is labeled correctly."""
    df = base_market_data.clone()
    
    # Define setup at index 20 (Long, entry=103.0, SL=100.0, TP=109.0)
    setup = TradeSetup(
        setup_id="test_long",
        timestamp=df[20, "timestamp"],
        symbol="XAUUSD",
        timeframe="M1",
        direction="long",
        entry_price=103.0,
        stop_loss=100.0,
        take_profit=109.0,
        features={}
    )
    
    # Configure outcome at index 25: High reaches 110.0 (TP is 109.0)
    df[25, "high"] = 110.0
    
    builder = DatasetBuilder()
    labeled = builder.build(df, [setup])
    
    assert len(labeled) == 1
    assert labeled[0, "tp_before_sl"] is True
    assert np.isclose(labeled[0, "realized_r"], 2.0) # (109 - 103) / 3 = 2.0 R-multiple
    assert labeled[0, "exit_price"] == 109.0
    assert labeled[0, "bars_to_exit"] == 5 # 25 - 20 = 5 bars


def test_outcome_labeling_sl_first(base_market_data: pl.DataFrame) -> None:
    """Verify that SL hit before TP is labeled correctly."""
    df = base_market_data.clone()
    
    # Define setup at index 20 (Long, entry=103.0, SL=100.0, TP=109.0)
    setup = TradeSetup(
        setup_id="test_long",
        timestamp=df[20, "timestamp"],
        symbol="XAUUSD",
        timeframe="M1",
        direction="long",
        entry_price=103.0,
        stop_loss=100.0,
        take_profit=109.0,
        features={}
    )
    
    # Configure outcome at index 25: Low hits 99.0 (SL is 100.0)
    df[25, "low"] = 99.0
    
    builder = DatasetBuilder()
    labeled = builder.build(df, [setup])
    
    assert len(labeled) == 1
    assert labeled[0, "tp_before_sl"] is False
    assert np.isclose(labeled[0, "realized_r"], -1.0)
    assert labeled[0, "exit_price"] == 100.0


def test_outcome_labeling_same_bar_crossing(base_market_data: pl.DataFrame) -> None:
    """Verify that same bar crossing both SL and TP defaults conservatively to SL hit."""
    df = base_market_data.clone()
    
    setup = TradeSetup(
        setup_id="test_long",
        timestamp=df[20, "timestamp"],
        symbol="XAUUSD",
        timeframe="M1",
        direction="long",
        entry_price=103.0,
        stop_loss=100.0,
        take_profit=109.0,
        features={}
    )
    
    # Configure outcome at index 25: High is 111.0 (hits TP), Low is 98.0 (hits SL)
    df[25, "high"] = 111.0
    df[25, "low"] = 98.0
    
    builder = DatasetBuilder()
    labeled = builder.build(df, [setup])
    
    assert len(labeled) == 1
    assert labeled[0, "tp_before_sl"] is False
    assert np.isclose(labeled[0, "realized_r"], -1.0)


def test_backtester_metrics(base_market_data: pl.DataFrame) -> None:
    """Verify backtester output calculations (Expectancy, Win Rate, Profit Factor, Drawdown)."""
    df = base_market_data.clone()
    
    # Configure 3 setup points (Longs) at index 10, 20, 30
    for idx in [10, 20, 30]:
        df[idx - 1, "bb_lower"] = 100.0
        df[idx - 1, "emaDn"] = 101.0
        df[idx, "bb_lower"] = 102.0
        df[idx, "emaDn"] = 101.5
        df[idx, "bb_upper"] = 104.0
        df[idx, "bbUp_sl"] = 103.0
        df[idx - 1, "totalGap"] = 1.0
        df[idx, "totalGap"] = 2.0
        df[idx, "close"] = 103.0
    
    # Configure outcomes:
    # Trade 1 (index 10): Wins (+2.0 R) at index 15
    df[15, "high"] = 110.0
    
    # Trade 2 (index 20): Loses (-1.0 R) at index 25
    df[25, "low"] = 99.0
    
    # Trade 3 (index 30): Wins (+2.0 R) at index 35
    df[35, "high"] = 110.0
    
    strategy = EMASmoothingBBStrategy(parameters={"sqz_threshold": 5.0})
    backtester = Backtester()
    
    report = backtester.run(strategy, df, "XAUUSD", "M1")
    
    assert report["total_trades"] == 3
    assert report["wins"] == 2
    assert report["losses"] == 1
    assert np.isclose(report["win_rate"], 2/3)
    # total realized R = 2.0 - 1.0 + 2.0 = 3.0 R
    assert np.isclose(report["total_realized_r"], 3.0)
    # expectancy = 3.0 / 3 = 1.0 R
    assert np.isclose(report["expectancy"], 1.0)
    # profit factor = (2.0 + 2.0) / 1.0 = 4.0
    assert np.isclose(report["profit_factor"], 4.0)
    # max drawdown: starts at 0, goes to +2.0, drops to +1.0 (drawdown of 1.0), rises to +3.0. Max drawdown = 1.0 R
    assert np.isclose(report["max_drawdown_r"], 1.0)
