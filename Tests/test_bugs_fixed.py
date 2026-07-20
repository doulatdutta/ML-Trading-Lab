import os
import json
import numpy as np
import polars as pl
import pytest
from datetime import datetime, timedelta

from ml_trading_lab.FeatureEngine.engine import FeatureEngine
from ml_trading_lab.LiveAdvisor.advisor import LiveAdvisor
from ml_trading_lab.StrategyEngine.strategy import EMASmoothingBBStrategy, TradeSetup
from ml_trading_lab.ML.xgboost_model import XGBoostModel

def test_total_gap_does_not_cancel():
    # Create non-flat mock market data
    # Standard deviation of close should be non-zero
    dates = pl.datetime_range(datetime(2023, 1, 1), datetime(2023, 1, 2), interval="1h", eager=True)
    n = len(dates)
    # Alternating prices to ensure std is not 0
    close = [100.0 if i % 2 == 0 else 105.0 for i in range(n)]
    
    df = pl.DataFrame({
        "timestamp": dates,
        "open": close,
        "high": [c + 1.0 for c in close],
        "low": [c - 1.0 for c in close],
        "close": close,
        "tick_volume": [100] * n,
        "spread": [1.0] * n
    })
    
    fe = FeatureEngine(parameters={"bb_period": 5, "bb_std": 1.0, "sm_period": 5, "sm_std": 2.5})
    features = fe.transform(df)
    
    # Verify that totalGap is not always zero or identical to 2 * (sm_basis - bb_middle)
    # Upper gap = emaUp - bb_upper
    # Lower gap = bb_lower - emaDn
    # totalGap = Upper gap + Lower gap
    # On symmetric bands, it is 2 * (sm_std * sm_std_series - bb_std * bb_std_series)
    # Check that it contains standard deviation terms and changes based on volatility
    gap_values = features["totalGap"].drop_nulls()
    assert len(gap_values) > 0
    # The gap should be non-zero on volatile data (unless standard deviations perfectly balance, which is rare)
    assert not np.allclose(gap_values.to_numpy(), 0.0)
    
    # Calculate mathematically: (emaUp - bb_upper) + (bb_lower - emaDn)
    idx = 10
    ema_up = features["emaUp"][idx]
    ema_dn = features["emaDn"][idx]
    bb_up = features["bb_upper"][idx]
    bb_dn = features["bb_lower"][idx]
    expected_gap = (ema_up - bb_up) + (bb_dn - ema_dn)
    assert np.isclose(features["totalGap"][idx], expected_gap)


def test_live_advisor_companion_features(tmp_path):
    # 1. Save dummy model and features companion file
    model_path = os.path.join(tmp_path, "test_model.json")
    feat_path = os.path.join(tmp_path, "test_model_features.json")
    
    # Save empty xgboost model structure
    from xgboost import XGBClassifier
    clf = XGBClassifier()
    # Fit on small dummy data to allow saving
    clf.fit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 1]))
    clf.save_model(model_path)
    
    custom_features = ["bb_width_percentile", "totalGap"]
    with open(feat_path, "w") as f:
        json.dump(custom_features, f)
        
    # 2. Init LiveAdvisor with the custom model
    advisor = LiveAdvisor(model_path=model_path)
    assert advisor.model_loaded
    assert advisor.feature_names == custom_features
    
    # 3. Score a mock setup with custom features
    # Create mock M1 and M3 rates
    start_dt = datetime(2023, 1, 1, 9, 0, 0)
    dates_m1 = pl.datetime_range(start_dt, start_dt + timedelta(hours=2), interval="1m", eager=True)
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
    # Add a mock setup at index -2
    df_m1[n_m1 - 2, "close"] = 105.0
    
    df_m3 = pl.DataFrame({
        "timestamp": pl.datetime_range(start_dt, start_dt + timedelta(hours=2), interval="3m", eager=True),
        "open": 102.0, "high": 103.0, "low": 101.0, "close": 105.0, "tick_volume": 500, "spread": 2.0
    })
    
    # Mock strategy.detect_setups to return a setup with specific feature values
    setup = TradeSetup(
        setup_id="test_long",
        timestamp=df_m1[-2, "timestamp"],
        symbol="XAUUSD",
        timeframe="M1",
        direction="long",
        entry_price=105.0,
        stop_loss=100.0,
        take_profit=115.0,
        features={"bb_width_percentile": 0.5, "totalGap": 2.5}
    )
    
    from unittest.mock import patch
    with patch.object(EMASmoothingBBStrategy, "detect_setups", return_value=[setup]):
        res = advisor.score("XAUUSD", "M1", mock_data={"M1": df_m1, "M3": df_m3})
        
    assert res["status"] == "active_setup"
    assert "win_probability" in res
