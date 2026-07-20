"""Tests for the RuleExtractor class."""

import polars as pl
import numpy as np

from ml_trading_lab.ML.rule_extractor import RuleExtractor


def test_rule_extractor_basic() -> None:
    """Verify that RuleExtractor trains on labeled data and extracts MQL5 expression and python filter."""
    np.random.seed(42)
    # Generate mock labeled dataset
    n_samples = 50
    data = {
        "tp_before_sl": np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]).tolist(),
        "feature_bb_width_pct": np.random.uniform(0.0, 1.0, size=n_samples).tolist(),
        "feature_ema_slope": np.random.normal(0.0, 0.1, size=n_samples).tolist(),
        "feature_atr_percentile": np.random.uniform(0.0, 1.0, size=n_samples).tolist(),
        "feature_session_london": np.random.choice([0.0, 1.0], size=n_samples).tolist(),
        "feature_session_ny": np.random.choice([0.0, 1.0], size=n_samples).tolist(),
        "feature_totalGap": np.random.uniform(1.0, 10.0, size=n_samples).tolist(),
    }
    labeled_df = pl.DataFrame(data)

    extractor = RuleExtractor(max_depth=3, min_win_rate=0.55, min_samples=3)
    mql5_expr, py_filter, paths = extractor.extract_rules(labeled_df)

    # Output MQL5 expression should be a string
    assert isinstance(mql5_expr, str)
    assert mql5_expr != ""

    # python filter must be a callable
    assert callable(py_filter)

    # If rules were discovered, the expression must contain variables or be 'true'
    if mql5_expr != "true":
        assert any(var in mql5_expr for var in ["bbWidthPct", "emaSlope", "atrPercentile", "sessionLondon", "sessionNY", "totalGap[1]"])

    # Test the python filter on a mock setup row
    mock_row = {
        "bb_width_pct": 0.5,
        "ema_slope": 0.01,
        "atr_percentile": 0.5,
        "session_london": 1.0,
        "session_ny": 0.0,
        "totalGap": 5.0,
    }
    res = py_filter(mock_row)
    assert isinstance(res, bool)
