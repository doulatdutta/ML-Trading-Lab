"""Optimization of strategy parameters on in-sample historical data."""

import itertools
from typing import Dict, Any, List, Tuple
import polars as pl

from ml_trading_lab.StrategyEngine.strategy import EMASmoothingBBStrategy
from ml_trading_lab.Backtester.backtester import Backtester


class StrategyOptimizer:
    """Propose reviewable parameter candidates based on in-sample backtest metrics."""

    def propose(
        self,
        market_data: pl.DataFrame,
        parameter_grid: Dict[str, List[Any]],
        symbol: str = "XAUUSD",
        timeframe: str = "M15",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return the best parameter combination and corresponding backtest report.

        We optimize for total_realized_r, requiring a minimum trade count (default >= 1).
        """
        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())
        permutations = list(itertools.product(*values))

        best_params = {}
        best_report: Dict[str, Any] = {}
        best_metric = -float("inf")

        backtester = Backtester()

        for perm in permutations:
            params = dict(zip(keys, perm))
            
            # Instantiate strategy with these candidates
            strategy = EMASmoothingBBStrategy(parameters=params)
            report = backtester.run(strategy, market_data, symbol, timeframe)

            # Optimization target: total realized R-multiple
            metric = report["total_realized_r"]
            
            # We want to maximize total R-multiple, requiring at least one trade
            if report["total_trades"] >= 1 and metric > best_metric:
                best_metric = metric
                best_params = params
                best_report = report

        # If no configuration yielded any trades, fallback to the first permutation
        if not best_params and permutations:
            best_params = dict(zip(keys, permutations[0]))
            strategy = EMASmoothingBBStrategy(parameters=best_params)
            best_report = backtester.run(strategy, market_data, symbol, timeframe)

        return best_params, best_report
