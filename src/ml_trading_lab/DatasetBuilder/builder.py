"""Build chronological ML datasets from strategy events and future outcomes."""

import polars as pl
from typing import List

from ml_trading_lab.StrategyEngine.strategy import TradeSetup


class DatasetBuilder:
    """Attach outcomes to point-in-time feature rows without feature leakage."""

    def build(self, df: pl.DataFrame, setups: List[TradeSetup]) -> pl.DataFrame:
        """Evaluate each setup's future outcome and return a labeled dataset.

        df: The Polars DataFrame of engineered features (must be chronologically ordered).
        setups: The list of detected TradeSetup events.
        """
        if not setups:
            # Return an empty DataFrame with the expected schema
            return pl.DataFrame()

        # Build lookup dict for timestamps to indices
        timestamp_to_idx = {ts: idx for idx, ts in enumerate(df["timestamp"])}

        # Detect early exit signals
        from ml_trading_lab.StrategyEngine.strategy import EMASmoothingBBStrategy
        strategy = EMASmoothingBBStrategy()
        long_exits, short_exits = strategy.detect_exits(df)
        long_exits_np = long_exits.to_numpy()
        short_exits_np = short_exits.to_numpy()

        # Extract columns as numpy arrays or list for fast lookup
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        timestamps = df["timestamp"]
        n_bars = len(df)

        labeled_records = []

        for setup in setups:
            ts = setup.timestamp
            if ts not in timestamp_to_idx:
                continue

            idx = timestamp_to_idx[ts]
            entry = setup.entry_price
            sl = setup.stop_loss
            tp = setup.take_profit
            direction = setup.direction

            realized_r = 0.0
            tp_before_sl = False
            exit_price = entry
            exit_ts = ts
            bars_to_exit = 0
            resolved = False

            # Scan forward from the next bar
            for k in range(idx + 1, n_bars):
                high_k = highs[k]
                low_k = lows[k]
                close_k = closes[k]
                ts_k = timestamps[k]

                if direction == "long":
                    tp_hit = high_k >= tp
                    sl_hit = low_k <= sl
                    early_exit = bool(long_exits_np[k])

                    if tp_hit and sl_hit:
                        # Conservatively assume SL hit first
                        realized_r = -1.0
                        tp_before_sl = False
                        exit_price = sl
                        exit_ts = ts_k
                        bars_to_exit = k - idx
                        resolved = True
                        break
                    elif sl_hit:
                        realized_r = -1.0
                        tp_before_sl = False
                        exit_price = sl
                        exit_ts = ts_k
                        bars_to_exit = k - idx
                        resolved = True
                        break
                    elif tp_hit:
                        realized_r = (tp - entry) / abs(entry - sl)
                        tp_before_sl = True
                        exit_price = tp
                        exit_ts = ts_k
                        bars_to_exit = k - idx
                        resolved = True
                        break
                    elif early_exit:
                        realized_r = (close_k - entry) / abs(entry - sl)
                        tp_before_sl = close_k >= tp
                        exit_price = close_k
                        exit_ts = ts_k
                        bars_to_exit = k - idx
                        resolved = True
                        break

                else:  # short
                    tp_hit = low_k <= tp
                    sl_hit = high_k >= sl
                    early_exit = bool(short_exits_np[k])

                    if tp_hit and sl_hit:
                        # Conservatively assume SL hit first
                        realized_r = -1.0
                        tp_before_sl = False
                        exit_price = sl
                        exit_ts = ts_k
                        bars_to_exit = k - idx
                        resolved = True
                        break
                    elif sl_hit:
                        realized_r = -1.0
                        tp_before_sl = False
                        exit_price = sl
                        exit_ts = ts_k
                        bars_to_exit = k - idx
                        resolved = True
                        break
                    elif tp_hit:
                        realized_r = (entry - tp) / abs(sl - entry)
                        tp_before_sl = True
                        exit_price = tp
                        exit_ts = ts_k
                        bars_to_exit = k - idx
                        resolved = True
                        break
                    elif early_exit:
                        realized_r = (entry - close_k) / abs(sl - entry)
                        tp_before_sl = close_k <= tp
                        exit_price = close_k
                        exit_ts = ts_k
                        bars_to_exit = k - idx
                        resolved = True
                        break

            # Handle unresolved trades at the end of the history
            if not resolved:
                final_close = closes[-1]
                final_ts = timestamps[-1]
                bars_to_exit = n_bars - 1 - idx
                exit_ts = final_ts
                exit_price = final_close
                tp_before_sl = False
                if direction == "long":
                    realized_r = (final_close - entry) / abs(entry - sl)
                else:
                    realized_r = (entry - final_close) / abs(sl - entry)

            # Record labeled setup
            record = {
                "setup_id": setup.setup_id,
                "timestamp": setup.timestamp,
                "direction": setup.direction,
                "entry_price": entry,
                "stop_loss": sl,
                "take_profit": tp,
                "tp_before_sl": tp_before_sl,
                "realized_r": float(realized_r),
                "exit_price": float(exit_price),
                "exit_timestamp": exit_ts,
                "bars_to_exit": int(bars_to_exit),
            }

            # Add features to the record
            for feat_name, feat_val in setup.features.items():
                record[f"feature_{feat_name}"] = feat_val

            labeled_records.append(record)

        if not labeled_records:
            return pl.DataFrame()

        return pl.DataFrame(labeled_records)
