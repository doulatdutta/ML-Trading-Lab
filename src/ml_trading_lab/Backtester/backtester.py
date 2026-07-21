"""Backtesting implementation."""

import polars as pl
from ml_trading_lab.StrategyEngine.strategy import EMASmoothingBBStrategy
from ml_trading_lab.DatasetBuilder.builder import DatasetBuilder


class Backtester:
    """Simulate explicitly defined rules with spread and slippage assumptions."""

    def run(
        self,
        strategy: EMASmoothingBBStrategy,
        market_data: pl.DataFrame,
        symbol: str = "XAUUSD",
        timeframe: str = "M15",
    ) -> dict:
        """Run setup detection and outcome evaluation to return a detailed performance report."""
        # 1. Detect strategy events/setups
        setups = strategy.detect_setups(market_data, symbol, timeframe)
        
        # 2. Evaluate outcomes via DatasetBuilder
        builder = DatasetBuilder()
        dataset = builder.build(market_data, setups)

        if dataset.is_empty():
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_realized_r": 0.0,
                "expectancy": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_r": 0.0,
                "trades": [],
            }

        # Calculate statistics
        realized_r_list = dataset["realized_r"].to_list()
        total_trades = len(realized_r_list)
        
        wins = sum(1 for r in realized_r_list if r > 0)
        losses = sum(1 for r in realized_r_list if r <= 0)
        win_rate = float(wins / total_trades) if total_trades > 0 else 0.0
        total_realized_r = float(sum(realized_r_list))
        expectancy = float(total_realized_r / total_trades) if total_trades > 0 else 0.0
        
        pos_sum = sum(r for r in realized_r_list if r > 0)
        neg_sum = sum(abs(r) for r in realized_r_list if r <= 0)
        profit_factor = float(pos_sum / neg_sum) if neg_sum > 0 else float("inf") if pos_sum > 0 else 0.0

        # Calculate max drawdown in R-multiples
        cum_r = [0.0]
        current = 0.0
        for r in realized_r_list:
            current += r
            cum_r.append(current)
            
        max_dd = 0.0
        peak = 0.0
        for val in cum_r:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_realized_r": total_realized_r,
            "expectancy": expectancy,
            "profit_factor": profit_factor,
            "max_drawdown_r": float(max_dd),
            "trades": dataset.to_dicts(),
        }
