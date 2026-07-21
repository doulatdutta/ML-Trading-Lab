"""Feature-engineering boundary and calculations using Polars."""

import numpy as np
import polars as pl
from typing import Dict, Any, Optional


class FeatureEngine:
    """Build features from information available at each event time only, using Polars."""

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the FeatureEngine with optional strategy parameters."""
        self.parameters = parameters or {}
        # Strategy specific settings mapped from user EA inputs
        self.bb_period = self.parameters.get("bb_period", 20)
        self.bb_std = self.parameters.get("bb_std", 1.0)
        self.ema_fast_period = self.parameters.get("ema_fast_period", 20)
        self.sm_period = self.parameters.get("sm_period", 20)
        self.sm_std = self.parameters.get("sm_std", 2.5)
        self.slope_len = self.parameters.get("slope_len", 3)
        self.atr_period = self.parameters.get("atr_period", 14)
        self.ema_trend_period = self.parameters.get("ema_trend_period", 20)

        # Modular Feature Flags (Disableable)
        self.enable_liquidity_sweeps = self.parameters.get("enable_liquidity_sweeps", True)
        self.enable_rsi_adx = self.parameters.get("enable_rsi_adx", True)
        self.enable_vwap = self.parameters.get("enable_vwap", True)
        self.enable_mtf = self.parameters.get("enable_mtf", True)

    def transform(self, market_data: pl.DataFrame) -> pl.DataFrame:
        """Return feature rows aligned with source timestamps without look-ahead bias.

        market_data must have columns: timestamp, open, high, low, close, tick_volume, spread.
        """
        # Ensure we do not modify the original dataframe in place
        df = market_data.clone()

        # 1. Bollinger Bands (Inner)
        df = df.with_columns([
            pl.col("close").rolling_mean(window_size=self.bb_period).alias("bb_middle"),
            pl.col("close").rolling_std(window_size=self.bb_period).alias("bb_std_series"),
        ])
        
        df = df.with_columns([
            (pl.col("bb_middle") + self.bb_std * pl.col("bb_std_series")).alias("bb_upper"),
            (pl.col("bb_middle") - self.bb_std * pl.col("bb_std_series")).alias("bb_lower"),
        ])

        # 2. EMA Bollinger Bands (Outer)
        df = df.with_columns([
            pl.col("close").ewm_mean(span=self.ema_fast_period, adjust=False).alias("ema_fast"),
        ])

        df = df.with_columns([
            pl.col("ema_fast").rolling_mean(window_size=self.sm_period).alias("sm_basis"),
            pl.col("ema_fast").rolling_std(window_size=self.sm_period).alias("sm_std_series"),
        ])

        df = df.with_columns([
            (pl.col("sm_basis") + self.sm_std * pl.col("sm_std_series")).alias("emaUp"),
            (pl.col("sm_basis") - self.sm_std * pl.col("sm_std_series")).alias("emaDn"),
        ])

        # 3. Gap Width Squeeze & Expansion
        df = df.with_columns([
            ((pl.col("emaUp") - pl.col("bb_upper")) + (pl.col("bb_lower") - pl.col("emaDn"))).alias("totalGap")
        ])

        # Bollinger Band Width Percentile (retained as standard ML feature)
        df = df.with_columns([
            (pl.col("bb_upper") - pl.col("bb_lower")).alias("bb_width"),
        ])
        df = df.with_columns([
            pl.col("bb_width").rolling_map(
                lambda x: float(x.rank()[-1]) / len(x) if not x.is_empty() and not np.isnan(x.to_numpy()).any() else None,
                window_size=self.bb_period
            ).alias("bb_width_percentile"),
            ((pl.col("close") - pl.col("bb_lower")) / (pl.col("bb_width") + 1e-8)).alias("bb_position")
        ])

        # 4. Slope Lookback values over InpSlopeLen (slope_len)
        df = df.with_columns([
            pl.col("bb_upper").shift(self.slope_len).alias("bbUp_sl"),
            pl.col("bb_lower").shift(self.slope_len).alias("bbDn_sl"),
            pl.col("emaUp").shift(self.slope_len).alias("emaUp_sl"),
            pl.col("emaDn").shift(self.slope_len).alias("emaDn_sl"),
            pl.col("emaUp").shift(1 + self.slope_len).alias("emaUp_1_sl"),
            pl.col("emaDn").shift(1 + self.slope_len).alias("emaDn_1_sl"),
        ])

        # 5. ATR (Average True Range)
        df = df.with_columns([
            pl.col("close").shift(1).alias("close_prev")
        ])
        
        high_low = pl.col("high") - pl.col("low")
        high_close_prev = (pl.col("high") - pl.col("close_prev")).abs()
        low_close_prev = (pl.col("low") - pl.col("close_prev")).abs()
        
        df = df.with_columns([
            pl.max_horizontal(high_low, high_close_prev, low_close_prev).alias("tr")
        ])
        
        df = df.with_columns([
            pl.col("tr").rolling_mean(window_size=self.atr_period).alias("atr")
        ])
        
        # Clean up intermediate columns
        df = df.drop(["close_prev", "tr", "bb_std_series", "sm_std_series"])

        # 5.5. Liquidity Sweep features (Gated by enable_liquidity_sweeps)
        if self.enable_liquidity_sweeps:
            df = df.with_columns([
                pl.col("high").shift(1).rolling_max(window_size=20).alias("high_prev_max_20"),
                pl.col("low").shift(1).rolling_min(window_size=20).alias("low_prev_min_20"),
                pl.col("high").shift(1).rolling_max(window_size=50).alias("high_prev_max_50"),
                pl.col("low").shift(1).rolling_min(window_size=50).alias("low_prev_min_50"),
            ])

            # Wick and Candle metrics for sweep rejection analysis
            candle_range = (pl.col("high") - pl.col("low")) + 1e-9
            upper_wick = pl.col("high") - pl.max_horizontal("open", "close")
            lower_wick = pl.min_horizontal("open", "close") - pl.col("low")

            df = df.with_columns([
                ((pl.col("low") < pl.col("low_prev_min_20")) & (pl.col("close") > pl.col("low_prev_min_20"))).cast(pl.Float64).alias("liq_sweep_bull_20"),
                ((pl.col("high") > pl.col("high_prev_max_20")) & (pl.col("close") < pl.col("high_prev_max_20"))).cast(pl.Float64).alias("liq_sweep_bear_20"),
                ((pl.col("low") < pl.col("low_prev_min_50")) & (pl.col("close") > pl.col("low_prev_min_50"))).cast(pl.Float64).alias("liq_sweep_bull_50"),
                ((pl.col("high") > pl.col("high_prev_max_50")) & (pl.col("close") < pl.col("high_prev_max_50"))).cast(pl.Float64).alias("liq_sweep_bear_50"),

                # Sweep Depth in ATR multiples
                ((pl.col("low_prev_min_20") - pl.col("low")) / (pl.col("atr") + 1e-8)).alias("liq_sweep_depth_bull_20"),
                ((pl.col("high") - pl.col("high_prev_max_20")) / (pl.col("atr") + 1e-8)).alias("liq_sweep_depth_bear_20"),

                # Wick rejection ratios
                (lower_wick / candle_range).alias("liq_sweep_lower_wick_ratio"),
                (upper_wick / candle_range).alias("liq_sweep_upper_wick_ratio"),

                # Fair Value Gap (FVG) detection
                ((pl.col("low") > pl.col("high").shift(2))).cast(pl.Float64).alias("fvg_bullish"),
                ((pl.col("high") < pl.col("low").shift(2))).cast(pl.Float64).alias("fvg_bearish"),
            ])

            # Drop intermediate max/min columns
            df = df.drop(["high_prev_max_20", "low_prev_min_20", "high_prev_max_50", "low_prev_min_50"])

        # 5.6 RSI and ADX indicators (Gated by enable_rsi_adx)
        if self.enable_rsi_adx:
            # RSI 14
            diff = pl.col("close") - pl.col("close").shift(1)
            gain = pl.when(diff > 0).then(diff).otherwise(0.0)
            loss = pl.when(diff < 0).then(-diff).otherwise(0.0)

            df = df.with_columns([
                gain.rolling_mean(window_size=14).alias("_avg_gain"),
                loss.rolling_mean(window_size=14).alias("_avg_loss"),
            ])
            df = df.with_columns([
                (100.0 - (100.0 / (1.0 + (pl.col("_avg_gain") / (pl.col("_avg_loss") + 1e-9))))).alias("rsi_14"),
            ])
            df = df.drop(["_avg_gain", "_avg_loss"])

            # ADX 14 (Directional Movement & Trend Strength)
            up_move = pl.col("high") - pl.col("high").shift(1)
            down_move = pl.col("low").shift(1) - pl.col("low")

            plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
            minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)

            df = df.with_columns([
                plus_dm.rolling_mean(window_size=14).alias("_plus_di_raw"),
                minus_dm.rolling_mean(window_size=14).alias("_minus_di_raw"),
            ])
            df = df.with_columns([
                (100.0 * (pl.col("_plus_di_raw") / (pl.col("atr") + 1e-9))).alias("plus_di_14"),
                (100.0 * (pl.col("_minus_di_raw") / (pl.col("atr") + 1e-9))).alias("minus_di_14"),
            ])
            df = df.with_columns([
                ((pl.col("plus_di_14") - pl.col("minus_di_14")).abs() / (pl.col("plus_di_14") + pl.col("minus_di_14") + 1e-9) * 100.0).alias("_dx")
            ])
            df = df.with_columns([
                pl.col("_dx").rolling_mean(window_size=14).alias("adx_14"),
            ])
            df = df.drop(["_plus_di_raw", "_minus_di_raw", "_dx"])

        # 5.7 VWAP calculation (Gated by enable_vwap)
        if self.enable_vwap:
            volume = pl.col("tick_volume").fill_null(1.0)
            typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
            tp_vol = typical_price * volume

            df = df.with_columns([
                tp_vol.alias("_tp_vol"),
                volume.alias("_vol"),
            ])

            # Rolling 200 bar VWAP approximation or session cumulative VWAP
            df = df.with_columns([
                (pl.col("_tp_vol").rolling_sum(window_size=100) / (pl.col("_vol").rolling_sum(window_size=100) + 1e-9)).alias("vwap_100"),
            ])
            df = df.with_columns([
                ((pl.col("close") - pl.col("vwap_100")) / (pl.col("atr") + 1e-8)).alias("close_to_vwap_atr"),
            ])
            df = df.drop(["_tp_vol", "_vol"])

        # 6. Session and Time Context
        df = df.with_columns([
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.weekday().alias("weekday"), # Monday is 1, Sunday is 7 in Polars
        ])
        
        # Normalize weekday to match pandas standard (0 is Monday, 6 is Sunday)
        df = df.with_columns([
            (pl.col("weekday") - 1).alias("weekday")
        ])

        df = df.with_columns([
            ((pl.col("hour") >= 0) & (pl.col("hour") < 8)).cast(pl.Int64).alias("session_asian"),
            ((pl.col("hour") >= 8) & (pl.col("hour") < 16)).cast(pl.Int64).alias("session_london"),
            ((pl.col("hour") >= 12) & (pl.col("hour") < 20)).cast(pl.Int64).alias("session_ny"),
            ((pl.col("hour") >= 12) & (pl.col("hour") < 16)).cast(pl.Int64).alias("session_overlap"),
        ])

        return df


    def join_htf_trend(self, df_m1: pl.DataFrame, df_m3: pl.DataFrame) -> pl.DataFrame:
        """Join M3 trend alignment signals to M1 bars with zero look-ahead bias."""
        # 1. Calculate Bollinger Bands and EMA Bands on M3 DataFrame
        m3_calc = df_m3.with_columns([
            pl.col("close").rolling_mean(window_size=self.bb_period).alias("m3_bbBasis"),
            pl.col("close").rolling_std(window_size=self.bb_period).alias("m3_bb_std_series"),
            pl.col("close").ewm_mean(span=self.ema_fast_period, adjust=False).alias("m3_ema"),
        ])

        m3_calc = m3_calc.with_columns([
            (pl.col("m3_bbBasis") + self.bb_std * pl.col("m3_bb_std_series")).alias("m3_bbUp"),
            (pl.col("m3_bbBasis") - self.bb_std * pl.col("m3_bb_std_series")).alias("m3_bbDn"),
            pl.col("m3_ema").rolling_mean(window_size=self.sm_period).alias("m3_smBasis"),
            pl.col("m3_ema").rolling_std(window_size=self.sm_period).alias("m3_sm_std_series"),
        ])

        m3_calc = m3_calc.with_columns([
            (pl.col("m3_smBasis") + self.sm_std * pl.col("m3_sm_std_series")).alias("m3_emaUp"),
            (pl.col("m3_smBasis") - self.sm_std * pl.col("m3_sm_std_series")).alias("m3_emaDn"),
        ])

        # 2. Evaluate Alignment
        m3_trend = m3_calc.with_columns([
            ((pl.col("m3_bbUp") > pl.col("m3_emaUp")) & (pl.col("m3_bbDn") > pl.col("m3_emaDn"))).alias("m3_longAlign"),
            ((pl.col("m3_bbUp") < pl.col("m3_emaUp")) & (pl.col("m3_bbDn") < pl.col("m3_emaDn"))).alias("m3_shortAlign"),
        ])

        # Shift trend alignment by 1 bar to get the previous completed M3 bar alignment state
        m3_trend = m3_trend.with_columns([
            pl.col("m3_longAlign").shift(1),
            pl.col("m3_shortAlign").shift(1),
        ]).select(["timestamp", "m3_longAlign", "m3_shortAlign"])

        # 3. Truncate M1 timestamp to 3m to align with M3 timestamps
        df_m1_truncated = df_m1.with_columns(
            pl.col("timestamp").dt.truncate("3m").alias("m3_timestamp")
        )

        # 4. Join M1 with M3
        joined = df_m1_truncated.join(
            m3_trend,
            left_on="m3_timestamp",
            right_on="timestamp",
            how="left"
        )

        # Drop the join key and fill nulls with True
        joined = joined.drop("m3_timestamp")
        joined = joined.with_columns([
            pl.col("m3_longAlign").fill_null(True),
            pl.col("m3_shortAlign").fill_null(True),
        ])

        return joined


