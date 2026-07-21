"""Monte Carlo Validator — stress-tests a strategy's trade log by random permutation.

Answers the question: "If these results were just random luck, how likely?"

Method:
1. Take the realized R-multiple sequence from a backtest.
2. Randomly shuffle trade order N times (10,000 default).
3. Measure the distribution of: total R, max drawdown, Sharpe-like ratio.
4. Compute the 5th percentile worst-case metrics.
5. Check if strategy passes the funded account drawdown limit.
"""

import random
from typing import List, Dict, Any, Optional
import statistics


class MonteCarloValidator:
    """Validate backtest results through random permutation testing."""

    def __init__(
        self,
        n_simulations: int = 10_000,
        max_drawdown_limit_r: float = 20.0,
        confidence_level: float = 0.95,
        random_seed: int = 42,
    ) -> None:
        """
        Args:
            n_simulations:        Number of random shuffle iterations.
            max_drawdown_limit_r: Account drawdown limit in R-multiples.
                                  Setups with SL = 1R → 20R limit ≈ 20× average risk.
            confidence_level:     e.g. 0.95 = 95th percentile check.
            random_seed:          For reproducibility.
        """
        self.n_simulations       = n_simulations
        self.max_drawdown_limit  = max_drawdown_limit_r
        self.confidence_level    = confidence_level
        self.random_seed         = random_seed

    def validate(
        self,
        realized_r: List[float],
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Run Monte Carlo on the given trade R-multiple sequence.

        Args:
            realized_r:        List of per-trade R-multiples (positive=win, negative=loss).
            progress_callback: Optional callable(pct: int, msg: str).

        Returns:
            Dictionary with simulation statistics and pass/fail verdict.
        """
        if len(realized_r) < 5:
            return {
                "approved": False,
                "reason": "Insufficient trades for Monte Carlo (minimum 5 required).",
                "n_trades": len(realized_r),
            }

        rng = random.Random(self.random_seed)
        trades = list(realized_r)

        sim_total_r:    List[float] = []
        sim_max_dd:     List[float] = []
        sim_win_rate:   List[float] = []

        report_interval = max(1, self.n_simulations // 20)

        for sim in range(self.n_simulations):
            rng.shuffle(trades)
            total_r, max_dd, wins = _simulate(trades)
            sim_total_r.append(total_r)
            sim_max_dd.append(max_dd)
            sim_win_rate.append(wins / len(trades))
            if progress_callback and sim % report_interval == 0:
                pct = int(sim / self.n_simulations * 90)
                progress_callback(pct, f"Monte Carlo: {sim:,}/{self.n_simulations:,} simulations...")

        sim_total_r.sort()
        sim_max_dd.sort()
        sim_win_rate.sort()

        # Percentile helpers
        def pct(lst: List[float], p: float) -> float:
            idx = max(0, min(len(lst) - 1, int(len(lst) * p)))
            return lst[idx]

        tail = 1.0 - self.confidence_level  # e.g. 0.05 for 95% confidence

        worst_5pct_dd      = pct(sim_max_dd, 1.0 - tail)      # 95th pct of drawdown
        best_5pct_total_r  = pct(sim_total_r, tail)            # 5th pct of total R
        median_total_r     = pct(sim_total_r, 0.5)
        median_win_rate    = pct(sim_win_rate, 0.5)

        # Actual (original order) metrics for reference
        orig_total_r, orig_max_dd, orig_wins = _simulate(realized_r)

        # Probability of getting this total_r by luck (fraction of sims that beat it randomly)
        prob_by_luck = sum(1 for r in sim_total_r if r >= orig_total_r) / len(sim_total_r)

        # Approval rules
        dd_passes = worst_5pct_dd <= self.max_drawdown_limit
        edge_passes = prob_by_luck < 0.10  # less than 10% chance it's random luck

        approved = dd_passes and edge_passes

        if progress_callback:
            verdict = "✅ APPROVED" if approved else "❌ REJECTED"
            progress_callback(100, f"Monte Carlo complete — {verdict}")

        return {
            "approved":              approved,
            "n_simulations":         self.n_simulations,
            "n_trades":              len(realized_r),
            "confidence_level":      self.confidence_level,
            # Original sequence metrics
            "original": {
                "total_r":   round(orig_total_r, 3),
                "max_dd":    round(orig_max_dd, 3),
                "win_rate":  round(orig_wins / len(realized_r), 4),
            },
            # Simulation statistics
            "sim_stats": {
                "worst_5pct_max_dd":    round(worst_5pct_dd, 3),
                "best_5pct_total_r":    round(best_5pct_total_r, 3),
                "median_total_r":       round(median_total_r, 3),
                "median_win_rate":      round(median_win_rate, 4),
                "prob_by_luck":         round(prob_by_luck, 4),
                "dd_limit_r":           self.max_drawdown_limit,
            },
            # Rejection reasons
            "checks": {
                "max_dd_within_limit":  dd_passes,
                "result_not_by_luck":   edge_passes,
            },
        }


def _simulate(trades: List[float]):
    """Compute total_r, max_drawdown, wins for a sequence of R-multiples."""
    total_r = 0.0
    peak    = 0.0
    max_dd  = 0.0
    wins    = 0
    for r in trades:
        total_r += r
        if total_r > peak:
            peak = total_r
        dd = peak - total_r
        if dd > max_dd:
            max_dd = dd
        if r > 0:
            wins += 1
    return total_r, max_dd, wins
