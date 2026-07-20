"""Walk-forward validation over sequential rolling chronological windows."""

from typing import Dict, Any, List
import polars as pl

from ml_trading_lab.Optimizer.optimizer import StrategyOptimizer
from ml_trading_lab.Backtester.backtester import Backtester
from ml_trading_lab.StrategyEngine.strategy import EMASmoothingBBStrategy


class WalkForwardValidator:
    """Evaluate strategy candidates across chronological out-of-sample windows."""

    def validate(
        self,
        market_data: pl.DataFrame,
        parameter_grid: Dict[str, List[Any]],
        baseline_params: Dict[str, Any],
        n_splits: int = 3,
        train_ratio: float = 0.7,
        symbol: str = "XAUUSD",
        timeframe: str = "M15",
        ml_filter_fn = None,
    ) -> Dict[str, Any]:
        """Perform rolling chronological splits and compare optimized walk-forward vs baseline.

        Returns validation evidence report.
        """
        # Ensure data is sorted by timestamp
        df = market_data.sort("timestamp")
        n_bars = len(df)
        if n_bars < 50:
            raise ValueError("Insufficient data bars for walk-forward validation splits.")

        # Compute split size
        split_size = n_bars // n_splits
        
        optimizer = StrategyOptimizer()
        backtester = Backtester()

        wf_reports: List[Dict[str, Any]] = []
        base_reports: List[Dict[str, Any]] = []

        for i in range(n_splits):
            # Rolling window boundaries
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, n_bars)
            window_df = df[start_idx:end_idx]

            # In-sample (train) and Out-of-sample (test) division
            n_win = len(window_df)
            train_end = int(n_win * train_ratio)
            
            train_df = window_df[0:train_end]
            test_df = window_df[train_end:n_win]

            if train_df.is_empty() or test_df.is_empty():
                continue

            # 1. Optimize on in-sample (train) slice
            best_params, _ = optimizer.propose(train_df, parameter_grid, symbol, timeframe)

            # 2. Test candidate out-of-sample
            opt_strategy = EMASmoothingBBStrategy(parameters=best_params, ml_filter_fn=ml_filter_fn)
            opt_test_report = backtester.run(opt_strategy, test_df, symbol, timeframe)
            wf_reports.append(opt_test_report)

            # 3. Test baseline out-of-sample
            base_strategy = EMASmoothingBBStrategy(parameters=baseline_params)
            base_test_report = backtester.run(base_strategy, test_df, symbol, timeframe)
            base_reports.append(base_test_report)

        # Aggregate metrics across splits
        total_wf_trades = sum(r["total_trades"] for r in wf_reports)
        total_wf_r = sum(r["total_realized_r"] for r in wf_reports)
        wf_wins = sum(r["wins"] for r in wf_reports)
        wf_win_rate = float(wf_wins / total_wf_trades) if total_wf_trades > 0 else 0.0

        total_base_trades = sum(r["total_trades"] for r in base_reports)
        total_base_r = sum(r["total_realized_r"] for r in base_reports)
        base_wins = sum(r["wins"] for r in base_reports)
        base_win_rate = float(base_wins / total_base_trades) if total_base_trades > 0 else 0.0

        # Promotion rule: optimized candidate outperforms or matches the baseline in total realized R-multiple
        approved = total_wf_r > total_base_r

        return {
            "approved": approved,
            "walk_forward_metrics": {
                "total_trades": total_wf_trades,
                "total_realized_r": total_wf_r,
                "win_rate": wf_win_rate,
            },
            "baseline_metrics": {
                "total_trades": total_base_trades,
                "total_realized_r": total_base_r,
                "win_rate": base_win_rate,
            },
            "folds_evaluated": len(wf_reports),
        }
