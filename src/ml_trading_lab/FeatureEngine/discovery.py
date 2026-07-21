"""Feature Discovery Engine — auto-derives 80–120 signals from EMA Smoothing BB + BB indicators.

Every feature is computed strictly without look-ahead bias:
- Rolling windows use only past bars (no center=True)
- Slopes use shift(N) to reference N bars ago
- All columns computed via Polars lazy expressions
"""

import polars as pl
from typing import Dict, Any, Optional, List


class FeatureDiscoveryEngine:
    """Auto-derive a rich feature matrix from raw OHLCV + EMA/BB indicator values.

    Input DataFrame must already have base indicator columns produced by FeatureEngine:
        bb_middle, bb_upper, bb_lower, bb_std_series
        ema_fast, sm_basis, emaUp, emaDn, totalGap
        atr, bbUp_sl, bbDn_sl
        timestamp, close, open, high, low, tick_volume

    This engine layers on top of those base columns to produce 80–120 additional features.
    """

    def __init__(self, slope_win: int = 3, pct_win: int = 50, lag_bars: int = 3) -> None:
        """
        Args:
            slope_win: N bars over which to compute slopes / derivatives.
            pct_win:   Window for rolling percentile calculations.
            lag_bars:  Number of lagged bar copies to include per key feature.
        """
        self.slope_win = slope_win
        self.pct_win   = pct_win
        self.lag_bars  = lag_bars

    # ── Public API ──────────────────────────────────────────────────────────

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return df enriched with all discovered features.

        Does NOT drop original columns — appends new ones only.
        """
        df = self._bb_features(df)
        df = self._ema_band_features(df)
        df = self._gap_features(df)
        df = self._atr_features(df)
        df = self._price_position_features(df)
        df = self._cross_interaction_features(df)
        df = self._time_session_features(df)
        df = self._lag_features(df)
        return df

    def feature_names(self, df: pl.DataFrame) -> List[str]:
        """Return list of all feature column names (excluding raw OHLCV and timestamp)."""
        raw = {"timestamp", "open", "high", "low", "close", "tick_volume", "spread"}
        return [c for c in df.columns if c not in raw]

    # ── BB Features ─────────────────────────────────────────────────────────

    def _bb_features(self, df: pl.DataFrame) -> pl.DataFrame:
        w = self.slope_win
        p = self.pct_win

        df = df.with_columns([
            # Band width in price units
            (pl.col("bb_upper") - pl.col("bb_lower")).alias("bb_width"),
        ])

        df = df.with_columns([
            # Width slope (positive = expanding, negative = squeezing)
            (pl.col("bb_width") - pl.col("bb_width").shift(w)).alias("bb_width_slope"),
            # Rolling min/max for percentile
            pl.col("bb_width").rolling_min(window_size=p).alias("_bw_min"),
            pl.col("bb_width").rolling_max(window_size=p).alias("_bw_max"),
        ])

        df = df.with_columns([
            # Width percentile 0–1 (0 = tightest squeeze, 1 = widest expansion)
            (
                (pl.col("bb_width") - pl.col("_bw_min")) /
                (pl.col("_bw_max") - pl.col("_bw_min") + 1e-9)
            ).alias("bb_width_pct"),
            # Width acceleration (2nd derivative)
            (pl.col("bb_width_slope") - pl.col("bb_width_slope").shift(w)).alias("bb_width_accel"),
            # Distance of close from upper/lower bands
            (pl.col("close") - pl.col("bb_upper")).alias("close_to_bb_upper"),
            (pl.col("close") - pl.col("bb_lower")).alias("close_to_bb_lower"),
            # Normalised close position within band (0=lower, 1=upper)
            (
                (pl.col("close") - pl.col("bb_lower")) /
                (pl.col("bb_upper") - pl.col("bb_lower") + 1e-9)
            ).alias("price_in_bb"),
        ])

        # BB middle slope
        df = df.with_columns([
            (pl.col("bb_middle") - pl.col("bb_middle").shift(w)).alias("bb_middle_slope"),
        ])

        # Squeeze duration — count consecutive bars where bb_width_pct < 0.20
        # Implemented as a lagged cumulative counter (approximation via rolling sum of squeeze flags)
        df = df.with_columns([
            (pl.col("bb_width_pct") < 0.20).cast(pl.Int32).alias("_squeeze_flag"),
        ])
        df = df.with_columns([
            pl.col("_squeeze_flag").rolling_sum(window_size=20).alias("bb_squeeze_duration"),
        ])

        # Breakout strength — how far close is from band relative to ATR
        if "atr" in df.columns:
            df = df.with_columns([
                (pl.col("close_to_bb_upper") / (pl.col("atr") + 1e-9)).alias("bb_upper_breakout_str"),
                (pl.col("close_to_bb_lower") / (pl.col("atr") + 1e-9)).alias("bb_lower_breakout_str"),
            ])

        # Drop temp columns
        df = df.drop([c for c in ["_bw_min", "_bw_max", "_squeeze_flag"] if c in df.columns])
        return df

    # ── EMA Smoothing Band Features ─────────────────────────────────────────

    def _ema_band_features(self, df: pl.DataFrame) -> pl.DataFrame:
        w = self.slope_win
        p = self.pct_win

        # EMA slope + acceleration + curvature
        df = df.with_columns([
            (pl.col("ema_fast") - pl.col("ema_fast").shift(w)).alias("ema_slope"),
        ])
        df = df.with_columns([
            (pl.col("ema_slope") - pl.col("ema_slope").shift(w)).alias("ema_accel"),
        ])
        df = df.with_columns([
            (pl.col("ema_accel") - pl.col("ema_accel").shift(w)).alias("ema_curvature"),
        ])

        # Normalised EMA slope (as % of price)
        df = df.with_columns([
            (pl.col("ema_slope") / (pl.col("close") + 1e-9)).alias("ema_slope_pct"),
        ])

        # EMA band width (outer bands)
        df = df.with_columns([
            (pl.col("emaUp") - pl.col("emaDn")).alias("ema_band_width"),
        ])
        df = df.with_columns([
            (pl.col("ema_band_width") - pl.col("ema_band_width").shift(w)).alias("ema_band_width_slope"),
            pl.col("ema_band_width").rolling_min(window_size=p).alias("_ebw_min"),
            pl.col("ema_band_width").rolling_max(window_size=p).alias("_ebw_max"),
        ])
        df = df.with_columns([
            (
                (pl.col("ema_band_width") - pl.col("_ebw_min")) /
                (pl.col("_ebw_max") - pl.col("_ebw_min") + 1e-9)
            ).alias("ema_band_width_pct"),
        ])

        # Distance of close from EMA bands
        df = df.with_columns([
            (pl.col("close") - pl.col("emaUp")).alias("close_to_ema_upper"),
            (pl.col("close") - pl.col("emaDn")).alias("close_to_ema_lower"),
            # Position within EMA bands (0=lower, 1=upper)
            (
                (pl.col("close") - pl.col("emaDn")) /
                (pl.col("emaUp") - pl.col("emaDn") + 1e-9)
            ).alias("price_in_ema_band"),
        ])

        # EMA band slopes
        df = df.with_columns([
            (pl.col("emaUp") - pl.col("emaUp").shift(w)).alias("ema_upper_slope"),
            (pl.col("emaDn") - pl.col("emaDn").shift(w)).alias("ema_lower_slope"),
        ])

        df = df.drop([c for c in ["_ebw_min", "_ebw_max"] if c in df.columns])
        return df

    # ── Gap (Outer-Inner interaction) Features ───────────────────────────────

    def _gap_features(self, df: pl.DataFrame) -> pl.DataFrame:
        w = self.slope_win
        p = self.pct_win

        if "totalGap" not in df.columns:
            return df

        df = df.with_columns([
            (pl.col("totalGap") - pl.col("totalGap").shift(w)).alias("totalGap_slope"),
        ])
        df = df.with_columns([
            (pl.col("totalGap_slope") - pl.col("totalGap_slope").shift(w)).alias("totalGap_accel"),
            pl.col("totalGap").rolling_min(window_size=p).alias("_tg_min"),
            pl.col("totalGap").rolling_max(window_size=p).alias("_tg_max"),
        ])
        df = df.with_columns([
            (
                (pl.col("totalGap") - pl.col("_tg_min")) /
                (pl.col("_tg_max") - pl.col("_tg_min") + 1e-9)
            ).alias("totalGap_pct"),
        ])

        # Ratio of inner band width to outer band width
        if "bb_width" in df.columns and "ema_band_width" in df.columns:
            df = df.with_columns([
                (pl.col("bb_width") / (pl.col("ema_band_width") + 1e-9)).alias("inner_outer_ratio"),
            ])

        df = df.drop([c for c in ["_tg_min", "_tg_max"] if c in df.columns])
        return df

    # ── ATR Features ────────────────────────────────────────────────────────

    def _atr_features(self, df: pl.DataFrame) -> pl.DataFrame:
        w = self.slope_win
        p = self.pct_win

        if "atr" not in df.columns:
            return df

        df = df.with_columns([
            (pl.col("atr") - pl.col("atr").shift(w)).alias("atr_slope"),
            pl.col("atr").rolling_min(window_size=p).alias("_atr_min"),
            pl.col("atr").rolling_max(window_size=p).alias("_atr_max"),
            (pl.col("atr") / (pl.col("close") + 1e-9)).alias("atr_pct_price"),
        ])
        df = df.with_columns([
            (
                (pl.col("atr") - pl.col("_atr_min")) /
                (pl.col("_atr_max") - pl.col("_atr_min") + 1e-9)
            ).alias("atr_percentile"),
        ])
        df = df.with_columns([
            (pl.col("atr_slope") - pl.col("atr_slope").shift(w)).alias("atr_accel"),
        ])
        # Volatility regime: 0=low, 1=medium, 2=high
        df = df.with_columns([
            pl.when(pl.col("atr_percentile") < 0.33).then(0)
              .when(pl.col("atr_percentile") < 0.67).then(1)
              .otherwise(2)
              .alias("vol_regime"),
        ])

        df = df.drop([c for c in ["_atr_min", "_atr_max"] if c in df.columns])
        return df

    # ── Price Position Features ──────────────────────────────────────────────

    def _price_position_features(self, df: pl.DataFrame) -> pl.DataFrame:
        w = self.slope_win
        p = self.pct_win

        # Price slope / momentum
        df = df.with_columns([
            (pl.col("close") - pl.col("close").shift(w)).alias("price_slope"),
        ])
        df = df.with_columns([
            (pl.col("price_slope") - pl.col("price_slope").shift(w)).alias("price_accel"),
        ])

        # Candle body features
        df = df.with_columns([
            (pl.col("close") - pl.col("open")).alias("candle_body"),
            (pl.col("high") - pl.col("low")).alias("candle_range"),
        ])
        df = df.with_columns([
            (pl.col("candle_body") / (pl.col("candle_range") + 1e-9)).alias("candle_body_pct"),
            ((pl.col("high") - pl.col("close").shift(1)) / (pl.col("candle_range") + 1e-9)).alias("upper_wick_pct"),
            ((pl.col("close").shift(1) - pl.col("low"))  / (pl.col("candle_range") + 1e-9)).alias("lower_wick_pct"),
        ])

        # Rolling high/low percentile position
        df = df.with_columns([
            pl.col("close").rolling_min(window_size=20).alias("_lo20"),
            pl.col("close").rolling_max(window_size=20).alias("_hi20"),
        ])
        df = df.with_columns([
            (
                (pl.col("close") - pl.col("_lo20")) /
                (pl.col("_hi20") - pl.col("_lo20") + 1e-9)
            ).alias("price_20bar_pct"),
        ])
        df = df.drop([c for c in ["_lo20", "_hi20"] if c in df.columns])
        return df

    # ── Cross-Interaction Features ───────────────────────────────────────────

    def _cross_interaction_features(self, df: pl.DataFrame) -> pl.DataFrame:
        # EMA momentum × volatility regime
        if "ema_slope" in df.columns and "atr_percentile" in df.columns:
            df = df.with_columns([
                (pl.col("ema_slope") * pl.col("atr_percentile")).alias("ema_slope_x_atr_pct"),
            ])

        # Squeeze × EMA acceleration
        if "bb_squeeze_duration" in df.columns and "ema_accel" in df.columns:
            df = df.with_columns([
                (pl.col("bb_squeeze_duration") * pl.col("ema_accel")).alias("squeeze_x_ema_accel"),
            ])

        # Gap × BB width percentile
        if "totalGap_pct" in df.columns and "bb_width_pct" in df.columns:
            df = df.with_columns([
                (pl.col("totalGap_pct") * pl.col("bb_width_pct")).alias("gap_pct_x_bb_pct"),
            ])

        # Price in band × EMA slope
        if "price_in_bb" in df.columns and "ema_slope" in df.columns:
            df = df.with_columns([
                (pl.col("price_in_bb") * pl.col("ema_slope")).alias("price_in_bb_x_ema_slope"),
            ])

        # Band width ratio × ATR percentile
        if "inner_outer_ratio" in df.columns and "atr_percentile" in df.columns:
            df = df.with_columns([
                (pl.col("inner_outer_ratio") * pl.col("atr_percentile")).alias("inner_outer_x_atr_pct"),
            ])

        return df

    # ── Time & Session Features ──────────────────────────────────────────────

    def _time_session_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if "timestamp" not in df.columns:
            return df
        if "hour" not in df.columns:
            df = df.with_columns([
                pl.col("timestamp").dt.hour().alias("hour"),
                pl.col("timestamp").dt.weekday().alias("weekday"),
            ])
        if "session_london" not in df.columns:
            df = df.with_columns([
                ((pl.col("hour") >= 0) & (pl.col("hour") < 8)).cast(pl.Int64).alias("session_asian"),
                ((pl.col("hour") >= 8) & (pl.col("hour") < 16)).cast(pl.Int64).alias("session_london"),
                ((pl.col("hour") >= 12) & (pl.col("hour") < 20)).cast(pl.Int64).alias("session_ny"),
                ((pl.col("hour") >= 12) & (pl.col("hour") < 16)).cast(pl.Int64).alias("session_overlap"),
            ])

        # Minutes to key session opens (cyclically encoded)
        import numpy as np
        df = df.with_columns([
            pl.col("timestamp").dt.minute().alias("_minute"),
        ])
        df = df.with_columns([
            (pl.col("hour") * 60 + pl.col("_minute")).alias("_min_of_day"),
        ])
        # London open = 8:00 = 480, NY open = 13:00 = 780
        df = df.with_columns([
            (
                ((pl.col("_min_of_day") - 480) % 1440).cast(pl.Float32) / 1440.0
            ).alias("mins_to_london_open_norm"),
            (
                ((pl.col("_min_of_day") - 780) % 1440).cast(pl.Float32) / 1440.0
            ).alias("mins_to_ny_open_norm"),
        ])
        # Hour-of-day sine/cosine encoding (captures cyclical nature)
        df = df.with_columns([
            (pl.col("hour").cast(pl.Float32) / 24.0 * 2 * 3.14159265).alias("_hour_rad"),
        ])
        # Note: Polars doesn't have sin/cos natively in expressions, so we use a map
        hours = df["_hour_rad"].to_list()
        sins  = [float(__import__("math").sin(h)) for h in hours]
        coss  = [float(__import__("math").cos(h)) for h in hours]
        df = df.with_columns([
            pl.Series("hour_sin", sins),
            pl.Series("hour_cos", coss),
        ])
        df = df.drop([c for c in ["_minute", "_min_of_day", "_hour_rad"] if c in df.columns])
        return df

    # ── Lag Features ────────────────────────────────────────────────────────

    def _lag_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add N-bar lagged copies of the most important features."""
        key_features = [
            "ema_slope", "ema_accel", "bb_width_pct", "totalGap",
            "atr_percentile", "price_in_bb", "price_in_ema_band",
            "bb_squeeze_duration", "candle_body_pct",
            "rsi_14", "adx_14", "close_to_vwap_atr",
            "liq_sweep_bull_20", "liq_sweep_bear_20",
            "liq_sweep_lower_wick_ratio", "liq_sweep_upper_wick_ratio"
        ]
        available = [f for f in key_features if f in df.columns]
        lag_exprs = []
        for feat in available:
            for lag in range(1, self.lag_bars + 1):
                lag_exprs.append(
                    pl.col(feat).shift(lag).alias(f"{feat}_lag{lag}")
                )
        if lag_exprs:
            df = df.with_columns(lag_exprs)
        return df
