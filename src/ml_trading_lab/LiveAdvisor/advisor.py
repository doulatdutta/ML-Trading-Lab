"""Live MetaTrader 5 advisory scoring system."""

import os
from typing import Dict, Any, Optional
import polars as pl
import numpy as np
from dotenv import load_dotenv

# Try importing MetaTrader 5 safely
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from ml_trading_lab.FeatureEngine.engine import FeatureEngine
from ml_trading_lab.StrategyEngine.strategy import EMASmoothingBBStrategy
from ml_trading_lab.ML.xgboost_model import XGBoostModel


DEFAULT_FEATURE_COLS = [
    "bb_width_percentile", "bb_position", "ema_fast",
    "totalGap", "atr", "hour", "weekday",
    "session_asian", "session_london", "session_ny", "session_overlap",
    "bbUp_sl", "bbDn_sl",
    "liq_sweep_bull_20", "liq_sweep_bear_20", "liq_sweep_bull_50", "liq_sweep_bear_50",
]


class LiveAdvisor:
    """Connects to live/demo MT5 terminal, fetches chart history, and evaluates setup probabilities."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize and load the trained XGBoost model."""
        load_dotenv()
        self.model = XGBoostModel()
        self.feature_names = []
        if model_path and os.path.exists(model_path):
            self.model.load_model(model_path)
            self.model_loaded = True
            
            # Load companion feature names JSON file
            feat_path = model_path.replace(".json", "_features.json")
            if os.path.exists(feat_path):
                import json
                try:
                    with open(feat_path, "r") as f:
                        self.feature_names = json.load(f)
                except Exception as e:
                    print(f"Error loading companion feature names: {e}")
        else:
            self.model_loaded = False

    def connect_mt5(self) -> bool:
        """Initialize connection to MetaTrader 5 using credentials from environmental variables."""
        if mt5 is None:
            print("MetaTrader5 python package not available.")
            return False

        # Load environment variables
        login_str = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        path = os.getenv("MT5_TERMINAL_PATH")

        # Try connecting
        if login_str and password and server:
            try:
                login = int(login_str)
                # If terminal path is defined, launch it
                if path:
                    connected = mt5.initialize(path=path, login=login, password=password, server=server)
                else:
                    connected = mt5.initialize(login=login, password=password, server=server)
            except Exception as e:
                print(f"Error parsing MT5 credentials: {e}")
                connected = False
        else:
            # Fallback to active terminal connection
            connected = mt5.initialize()

        if not connected:
            print(f"MT5 Initialization failed: {mt5.last_error()}")
            return False
            
        print("Successfully connected to MetaTrader 5 terminal.")
        return True

    def fetch_rates(self, symbol: str, timeframe: str, count: int = 500) -> Optional[pl.DataFrame]:
        """Fetch chronological bar rates from MT5 terminal for the given symbol and timeframe."""
        if mt5 is None:
            return None

        # Map timeframe string to MT5 constants
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M3": mt5.TIMEFRAME_M3,
        }
        mt5_tf = tf_map.get(timeframe)
        if mt5_tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        if rates is None or len(rates) == 0:
            print(f"No rates found for {symbol} on {timeframe}.")
            return None

        # Convert structured numpy array to Polars DataFrame
        df = pl.DataFrame(rates)
        df = df.rename({"time": "timestamp"})
        # Convert Unix epoch timestamp (seconds) to datetime
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Int64).mul(1000).cast(pl.Datetime("ms"))
        )
        return df

    def score(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "M1",
        parameters: Optional[Dict[str, Any]] = None,
        mock_data: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """Score the latest setups for a symbol and return advisory trade rating.

        Supports mock_data dictionary for offline testing.
        """
        # 1. Fetch rates
        if mock_data is not None:
            df_m1 = mock_data.get("M1")
            df_m3 = mock_data.get("M3")
        else:
            if not self.connect_mt5():
                return {"status": "error", "message": "Failed to connect to MT5"}
            
            df_m1 = self.fetch_rates(symbol, "M1")
            df_m3 = self.fetch_rates(symbol, "M3")
            
            # Shut down connection after query
            mt5.shutdown()

        if df_m1 is None or df_m3 is None or df_m1.is_empty() or df_m3.is_empty():
            return {"status": "error", "message": "Failed to fetch rates history"}

        # 2. Build features
        params = parameters or {}
        feature_engine = FeatureEngine(parameters=params)
        df_features = feature_engine.transform(df_m1)
        df_combined = feature_engine.join_htf_trend(df_features, df_m3)

        # 3. Detect setups
        strategy = EMASmoothingBBStrategy(parameters=params)
        setups = strategy.detect_setups(df_combined, symbol, timeframe)

        if not setups:
            return {
                "status": "no_setup",
                "message": "No active setup detected at the current bar.",
                "symbol": symbol,
                "timeframe": timeframe,
            }

        # Check if the latest setup is on the very last bar (completed or active)
        latest_setup = setups[-1]

        # Allow matching the last bar or the bar before the last bar
        is_active = latest_setup.timestamp >= df_combined["timestamp"][-2]

        if not is_active:
            return {
                "status": "no_setup",
                "message": f"Latest setup is stale (timestamp: {latest_setup.timestamp}), no current bar setup.",
                "symbol": symbol,
                "timeframe": timeframe,
            }

        # 4. Extract features and evaluate with model
        feature_names = self.feature_names if self.model_loaded and self.feature_names else DEFAULT_FEATURE_COLS
        
        # Build features input row
        features_dict = latest_setup.features
        # Ensure we check both prefixed and raw forms of the feature names
        X_row = []
        for name in feature_names:
            val = features_dict.get(name)
            if val is None:
                if name.startswith("feature_"):
                    val = features_dict.get(name.replace("feature_", ""))
                else:
                    val = features_dict.get(f"feature_{name}")
            if val is None:
                val = 0.0
            X_row.append(float(val))

        X_input = np.array([X_row])

        if self.model_loaded:
            win_prob = float(self.model.predict_proba(X_input)[0])
        else:
            # Fallback to safe prior probability baseline (e.g. 0.5) if model is not trained yet
            win_prob = 0.5

        # Promotion check: win probability > 0.55 makes it highly favorable
        advisory_action = "favorable" if win_prob >= 0.55 else "unfavorable"

        return {
            "status": "active_setup",
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": latest_setup.direction,
            "entry_price": latest_setup.entry_price,
            "stop_loss": latest_setup.stop_loss,
            "take_profit": latest_setup.take_profit,
            "timestamp": latest_setup.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "win_probability": win_prob,
            "advisory_action": advisory_action,
        }
