"""Strategy definition and setup/outcome contracts."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
import polars as pl


@dataclass(frozen=True)
class StrategyDefinition:
    """A named, human-reviewable strategy specification with settings."""

    name: str
    version: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeSetup:
    """Represents a potential strategy setup at a specific point in time."""

    setup_id: str
    timestamp: datetime
    symbol: str
    timeframe: str
    direction: str  # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeOutcome:
    """Represents the future outcome of a strategy setup."""

    setup_id: str
    realized_r: float  # e.g., -1.0 for full SL, +2.0 for hitting TP
    tp_before_sl: bool  # True if TP was reached before SL
    exit_price: float
    exit_timestamp: datetime
    bars_to_exit: int


class EMASmoothingBBStrategy:
    """Implements setup detection for the M1/M3 BB-EMA State crossover strategy using Polars."""

    def __init__(self, parameters: Optional[Dict[str, Any]] = None, ml_filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> None:
        """Initialize with configuration parameters."""
        self.parameters = parameters or {}
        # Crossover specific inputs
        self.sqz_threshold = self.parameters.get("sqz_threshold", 5.0)
        self.atr_sl_mult = self.parameters.get("atr_sl_mult", 1.5)
        self.atr_tp_mult = self.parameters.get("atr_tp_mult", 3.0)
        self.ml_filter_fn = ml_filter_fn

    def detect_setups(self, df: pl.DataFrame, symbol: str, timeframe: str) -> list[TradeSetup]:
        """Detect long and short setups in the feature DataFrame."""
        # Ensure we have all necessary columns
        required_cols = [
            "close", "bb_upper", "bb_lower", "emaUp", "emaDn",
            "totalGap", "bbUp_sl", "bbDn_sl", "atr", "timestamp"
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Check alignment parameters
        m3_longAlign = pl.col("m3_longAlign") if "m3_longAlign" in df.columns else pl.lit(True)
        m3_shortAlign = pl.col("m3_shortAlign") if "m3_shortAlign" in df.columns else pl.lit(True)

        # Long crossover condition:
        # BB Lower crosses above EMA BB Lower + BB Upper rising + expanding + squeezed (on completed bar)
        long_cond = (
            (df["bb_lower"].shift(1) <= df["emaDn"].shift(1)) &
            (df["bb_lower"] > df["emaDn"]) &
            (df["bb_upper"] > df["bbUp_sl"]) &
            (df["totalGap"] > df["totalGap"].shift(1)) &
            (df["totalGap"].shift(1) <= self.sqz_threshold) &
            m3_longAlign
        )

        # Short crossover condition:
        # BB Upper crosses below EMA BB Upper + BB Lower falling + expanding + squeezed (on completed bar)
        short_cond = (
            (df["bb_upper"].shift(1) >= df["emaUp"].shift(1)) &
            (df["bb_upper"] < df["emaUp"]) &
            (df["bb_lower"] < df["bbDn_sl"]) &
            (df["totalGap"] > df["totalGap"].shift(1)) &
            (df["totalGap"].shift(1) <= self.sqz_threshold) &
            m3_shortAlign
        )

        # Select matching setups
        long_df = df.filter(long_cond)
        short_df = df.filter(short_cond)
        
        setups = []
        
        # Populate long setups
        for row in long_df.iter_rows(named=True):
            atr_val = row["atr"]
            if atr_val is None or np.isnan(atr_val):
                continue
            
            # Apply ML rules filter if present
            if self.ml_filter_fn is not None:
                if not self.ml_filter_fn(row):
                    continue

            entry = row["close"]
            sl = entry - self.atr_sl_mult * atr_val
            tp = entry + self.atr_tp_mult * atr_val
            setup_id = f"long_{row['timestamp'].strftime('%Y%m%d_%H%M%S')}"
            
            setups.append(
                TradeSetup(
                    setup_id=setup_id,
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    timeframe=timeframe,
                    direction="long",
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    features={k: v for k, v in row.items() if k not in ["timestamp"]}
                )
            )
            
        # Populate short setups
        for row in short_df.iter_rows(named=True):
            atr_val = row["atr"]
            if atr_val is None or np.isnan(atr_val):
                continue

            # Apply ML rules filter if present
            if self.ml_filter_fn is not None:
                if not self.ml_filter_fn(row):
                    continue

            entry = row["close"]
            sl = entry + self.atr_sl_mult * atr_val
            tp = entry - self.atr_tp_mult * atr_val
            setup_id = f"short_{row['timestamp'].strftime('%Y%m%d_%H%M%S')}"
            
            setups.append(
                TradeSetup(
                    setup_id=setup_id,
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    timeframe=timeframe,
                    direction="short",
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    features={k: v for k, v in row.items() if k not in ["timestamp"]}
                )
            )
            
        # Sort setups chronologically
        setups.sort(key=lambda x: x.timestamp)
        return setups

    def detect_exits(self, df: pl.DataFrame) -> Tuple[pl.Series, pl.Series]:
        """Detect long and short exit signals in the feature DataFrame."""
        required = ["emaUp", "emaUp_1_sl", "emaUp_sl", "emaDn", "emaDn_1_sl", "emaDn_sl"]
        for col in required:
            if col not in df.columns:
                empty = pl.Series("exit", [False] * len(df))
                return empty, empty

        m3_longAlign = pl.col("m3_longAlign") if "m3_longAlign" in df.columns else pl.lit(True)
        m3_shortAlign = pl.col("m3_shortAlign") if "m3_shortAlign" in df.columns else pl.lit(True)

        long_exit_slope = (pl.col("emaUp").shift(1) > pl.col("emaUp_1_sl")) & (pl.col("emaUp") < pl.col("emaUp_sl"))
        long_exit = long_exit_slope & (~m3_longAlign if "m3_longAlign" in df.columns else pl.lit(True))

        short_exit_slope = (pl.col("emaDn").shift(1) < pl.col("emaDn_1_sl")) & (pl.col("emaDn") > pl.col("emaDn_sl"))
        short_exit = short_exit_slope & (~m3_shortAlign if "m3_shortAlign" in df.columns else pl.lit(True))

        # Evaluate expressions to series
        return df.select(long_exit).to_series(), df.select(short_exit).to_series()


