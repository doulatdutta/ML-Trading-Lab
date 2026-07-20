"""Evolution Engine — orchestrates the full self-improvement pipeline as a single callable loop.

Pipeline:
    1. Load / refresh market data (CSV cache or MT5 download)
    2. Feature Discovery Engine → 80-120 features
    3. Build labeled dataset (win/loss outcomes)
    4. Model Arena → train XGBoost + LightGBM + CatBoost
    5. Best model → Optimizer proposes new parameters (adaptive retry loop)
    6. Walk-Forward Validator → 3-fold chronological validation
    7. Monte Carlo Validator → 10,000 permutation stress-test
    8. If approved → EA Generator produces versioned MQL5 file
    9. Emit activity log events at every stage

Adaptive Retry Logic (NEW):
    If Monte Carlo or Walk-Forward rejects:
        - Automatically retry with tighter parameter grids
        - Progressively raise the ML probability filter threshold
        - Log every attempt to the dashboard
        - Stop as soon as a configuration is approved

The engine is designed to run in a background thread so the dashboard stays responsive.
"""

import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import polars as pl


# ── Adaptive retry configurations ────────────────────────────────────────────
# Each attempt uses a progressively tighter parameter space and higher ML
# probability threshold. The system automatically tries all of them before
# giving up.

_RETRY_CONFIGS = [
    {
        "label": "Attempt 1: Wide Crossover (No ML filter)",
        "grid": {"sqz_threshold": [3.0, 5.0, 8.0], "atr_sl_mult": [1.5, 2.0], "atr_tp_mult": [3.0, 4.0]},
        "ml_percentile_keep": 1.0,
    },
    {
        "label": "Attempt 2: Wide Crossover + ML (Keep top 90%)",
        "grid": {"sqz_threshold": [3.0, 5.0, 8.0], "atr_sl_mult": [1.5, 2.0], "atr_tp_mult": [3.0, 4.0]},
        "ml_percentile_keep": 0.90,
    },
    {
        "label": "Attempt 3: Moderate Crossover + ML (Keep top 80%)",
        "grid": {"sqz_threshold": [2.5, 4.0, 6.0], "atr_sl_mult": [1.25, 1.5, 1.75], "atr_tp_mult": [2.5, 3.0, 3.5]},
        "ml_percentile_keep": 0.80,
    },
    {
        "label": "Attempt 4: Moderate Crossover + ML (Keep top 70%)",
        "grid": {"sqz_threshold": [2.5, 4.0, 6.0], "atr_sl_mult": [1.25, 1.5, 1.75], "atr_tp_mult": [2.5, 3.0, 3.5]},
        "ml_percentile_keep": 0.70,
    },
    {
        "label": "Attempt 5: Tighter Squeeze + ML (Keep top 60%)",
        "grid": {"sqz_threshold": [2.0, 3.0, 4.0], "atr_sl_mult": [1.0, 1.25, 1.5], "atr_tp_mult": [2.0, 2.5, 3.0]},
        "ml_percentile_keep": 0.60,
    },
    {
        "label": "Attempt 6: Tighter Squeeze + ML (Keep top 50%)",
        "grid": {"sqz_threshold": [2.0, 3.0, 4.0], "atr_sl_mult": [1.0, 1.25, 1.5], "atr_tp_mult": [2.0, 2.5, 3.0]},
        "ml_percentile_keep": 0.50,
    },
    {
        "label": "Attempt 7: Ultra Selective + ML (Keep top 40%)",
        "grid": {"sqz_threshold": [1.5, 2.0, 2.5], "atr_sl_mult": [1.0, 1.25], "atr_tp_mult": [1.5, 2.0, 2.5]},
        "ml_percentile_keep": 0.40,
    },
    {
        "label": "Attempt 8: Ultra Selective + ML (Keep top 30%)",
        "grid": {"sqz_threshold": [1.5, 2.0, 2.5], "atr_sl_mult": [1.0, 1.25], "atr_tp_mult": [1.5, 2.0, 2.5]},
        "ml_percentile_keep": 0.30,
    },
    {
        "label": "Attempt 9: Extreme Squeeze + ML (Keep top 20%)",
        "grid": {"sqz_threshold": [1.0, 1.5, 2.0], "atr_sl_mult": [1.0, 1.25], "atr_tp_mult": [1.25, 1.5, 2.0]},
        "ml_percentile_keep": 0.20,
    },
    {
        "label": "Attempt 10: Elite Setup Only + ML (Keep top 10%)",
        "grid": {"sqz_threshold": [1.0, 1.5], "atr_sl_mult": [1.0, 1.2], "atr_tp_mult": [1.2, 1.5, 1.8]},
        "ml_percentile_keep": 0.10,
    },
]


class EvolutionEngine:
    """Run one full self-improvement cycle and return a structured report.

    When validation is rejected the engine automatically retries with
    tighter parameters and a higher ML-filter probability threshold,
    logging every attempt to the dashboard in real time.
    """

    def __init__(
        self,
        raw_directory: str = "Config/raw",
        model_directory: str = "Config/models",
        ea_directory:   str = "MQL5/Experts",
    ) -> None:
        self.raw_directory   = raw_directory
        self.model_directory = model_directory
        self.ea_directory    = ea_directory

    def run(
        self,
        symbol:    str = "XAUUSD",
        timeframe: str = "M1",
        strategy_params: Optional[Dict[str, Any]] = None,
        log: Optional[Callable[[str, str, str], None]] = None,
        progress: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute the full evolution pipeline with adaptive retry.

        Steps 1–4 run once (data + feature engineering + model training).
        Steps 5–8 (optimizer → walk-forward → monte carlo → EA) are retried
        automatically with progressively tighter parameters if rejected.
        """
        def _log(level, source, msg):
            if log:
                log(level, source, msg)

        def _prog(pct, msg):
            if progress:
                progress(pct, msg)

        params = strategy_params or {
            "sqz_threshold": 5.0, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0,
            "bb_period": 20, "bb_std": 1.0, "sm_std": 2.5,
        }
        report: Dict[str, Any] = {
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol, "timeframe": timeframe,
        }

        # ── Step 1: Load Data ─────────────────────────────────────────────
        _log("ML", "EvolutionEngine", f"Step 1/8 — Loading {symbol} {timeframe} data...")
        _prog(2, "Loading market data...")
        try:
            from ml_trading_lab.DataEngine.csv_collector import CSVMarketDataCollector
            collector = CSVMarketDataCollector(raw_directory=self.raw_directory)
            df_m1 = collector.load_bars(symbol, "M1")
            df_m3 = collector.load_bars(symbol, "M3")
            _log("INFO", "EvolutionEngine", f"Loaded {len(df_m1):,} M1 bars + {len(df_m3):,} M3 bars")
        except Exception as e:
            _log("ERROR", "EvolutionEngine", f"Data load failed: {e}")
            return {**report, "approved": False, "error": f"Data load failed: {e}"}

        # ── Step 2: Feature Engineering ───────────────────────────────────
        _log("ML", "EvolutionEngine", "Step 2/8 — Feature Discovery Engine (80–120 features)...")
        _prog(10, "Running Feature Discovery Engine...")
        try:
            from ml_trading_lab.FeatureEngine.engine import FeatureEngine
            from ml_trading_lab.FeatureEngine.discovery import FeatureDiscoveryEngine
            fe = FeatureEngine(parameters=params)
            df_feats = fe.transform(df_m1)
            df_feats = fe.join_htf_trend(df_feats, df_m3)
            discovery = FeatureDiscoveryEngine()
            df_feats  = discovery.transform(df_feats)
            n_features = len(discovery.feature_names(df_feats))
            _log("INFO", "EvolutionEngine", f"Feature matrix ready — {n_features} features")
        except Exception as e:
            _log("ERROR", "EvolutionEngine", f"Feature engineering failed: {e}")
            return {**report, "approved": False, "error": f"Feature engineering failed: {e}"}

        # ── Step 3: Build Labeled Dataset ─────────────────────────────────
        _log("ML", "EvolutionEngine", "Step 3/8 — Detecting setups and labeling outcomes...")
        _prog(20, "Building labeled dataset...")
        try:
            from ml_trading_lab.StrategyEngine.strategy import EMASmoothingBBStrategy
            from ml_trading_lab.DatasetBuilder.builder import DatasetBuilder
            strategy = EMASmoothingBBStrategy(parameters=params)
            setups   = strategy.detect_setups(df_feats, symbol, "M1")
            labeled  = DatasetBuilder().build(df_feats, setups)
            n_wins   = int(sum(labeled["tp_before_sl"].to_list()))
            _log("INFO", "EvolutionEngine", f"Labeled {len(labeled)} setups ({n_wins} wins / {len(labeled)-n_wins} losses)")
            if len(labeled) < 10:
                _log("ERROR", "EvolutionEngine", "Not enough labeled setups to train (< 10). Aborting.")
                return {**report, "approved": False, "error": "Not enough setups"}
        except Exception as e:
            _log("ERROR", "EvolutionEngine", f"Dataset building failed: {e}")
            return {**report, "approved": False, "error": f"Dataset building failed: {e}"}

        # ── Step 4: Model Arena ───────────────────────────────────────────
        _log("ML", "EvolutionEngine", "Step 4/8 — Training Model Arena (XGBoost + LightGBM + CatBoost)...")
        _prog(32, "Training all models...")
        arena = None
        feat_cols: List[str] = []
        try:
            X, y, feat_cols = _extract_Xy(labeled, df_feats)
            if X is None or len(X) < 10:
                raise ValueError("Feature extraction returned empty array")

            from ml_trading_lab.ML.model_arena import ModelArena
            arena = ModelArena(val_fraction=0.20)
            arena_results = arena.train_all(X, y, feat_cols)
            best_name = arena.best_model_name()
            best_auc  = arena_results[best_name]["auc"]
            _log("ML", "EvolutionEngine",
                 f"Arena complete — Best: {best_name} AUC={best_auc:.3f}")
            report["arena"] = arena_results
            report["best_model"] = best_name
        except Exception as e:
            _log("ERROR", "EvolutionEngine", f"Model Arena failed: {e}")
            return {**report, "approved": False, "error": f"Model Arena failed: {e}"}

        # ── Step 4.5: Rule Extraction ─────────────────────────────────────
        _log("ML", "EvolutionEngine", "Step 4.5/8 — Running Rule Discovery Extractor...")
        try:
            from ml_trading_lab.ML.rule_extractor import RuleExtractor
            extractor = RuleExtractor()
            mql5_rules, base_python_filter, rule_paths = extractor.extract_rules(labeled)
            _log("INFO", "EvolutionEngine", f"Discovered ML rules: {mql5_rules}")
            report["ml_rules_expression"] = mql5_rules
            report["ml_rule_paths"] = rule_paths
        except Exception as e:
            _log("ERROR", "EvolutionEngine", f"Rule discovery failed (non-fatal): {e}")
            mql5_rules = "true"
            base_python_filter = lambda x: True
            report["ml_rules_expression"] = "true"
            report["ml_rule_paths"] = []

        # ── Steps 5–8: Adaptive Retry Loop ────────────────────────────────
        # Try each config in order. Stop as soon as we get approval.
        _log("ML", "EvolutionEngine",
             f"Starting adaptive optimization — will try up to {len(_RETRY_CONFIGS)} configurations...")

        approved        = False
        best_params     = {}
        best_bt         = {}
        wf_report       = {}
        mc_report       = {}
        ea_path         = None

        for attempt_idx, cfg in enumerate(_RETRY_CONFIGS):
            attempt_num  = attempt_idx + 1
            cfg_label    = cfg["label"]
            grid         = cfg["grid"]
            ml_percentile = cfg.get("ml_percentile_keep", 1.0)

            _log("ML", "EvolutionEngine",
                 f"— Attempt {attempt_num}/{len(_RETRY_CONFIGS)}: {cfg_label}")
            _prog(40 + int(attempt_idx * 5.5), f"Attempt {attempt_num}: {cfg_label}...")

            # Build ML filter function using calibrated percentile
            ml_filter_fn = None
            if arena and ml_percentile < 1.0:
                probs = arena.best_model().predict_proba(X)
                threshold_val = float(np.percentile(probs, 100 * (1 - ml_percentile)))
                ml_filter_fn = _build_ml_filter(arena.best_model(), feat_cols, threshold_val)
                _log("INFO", "EvolutionEngine",
                     f"  ML calibrated (keep top {ml_percentile:.0%}) → threshold ≥ {threshold_val:.4f}")

            # ── Step 5: Optimizer ─────────────────────────────────────────
            try:
                from ml_trading_lab.Optimizer.optimizer import StrategyOptimizer
                from ml_trading_lab.Backtester.backtester import Backtester

                optimizer = StrategyOptimizer()
                best_params, best_bt = optimizer.propose(df_feats, grid, symbol, "M1")
                _log("INFO", "EvolutionEngine",
                     f"  Optimizer → sqz={best_params.get('sqz_threshold')}, "
                     f"sl={best_params.get('atr_sl_mult')}, tp={best_params.get('atr_tp_mult')}, "
                     f"trades={best_bt.get('total_trades')}, R={best_bt.get('total_realized_r', 0):.2f}")

                # ML-filtered backtest
                if ml_filter_fn:
                    strategy_filtered = EMASmoothingBBStrategy(parameters=best_params, ml_filter_fn=ml_filter_fn)
                    bt_filtered = Backtester().run(strategy_filtered, df_feats, symbol, "M1")
                    _log("INFO", "EvolutionEngine",
                         f"  ML-filtered → trades={bt_filtered['total_trades']}, "
                         f"WR={bt_filtered['win_rate']:.0%}, R={bt_filtered['total_realized_r']:.2f}")
                    report["ml_filtered_backtest"] = {k: v for k, v in bt_filtered.items() if k != "trades"}
                    # Use ML-filtered trades for MC validation (more honest)
                    mc_r_vals = [t.get("realized_r", 0.0) for t in bt_filtered.get("trades", []) if isinstance(t, dict)]
                else:
                    mc_r_vals = [t.get("realized_r", 0.0) for t in best_bt.get("trades", []) if isinstance(t, dict)]

                report["optimized_params"] = best_params
                report["optimizer_backtest"] = {k: v for k, v in best_bt.items() if k != "trades"}

            except Exception as e:
                _log("ERROR", "EvolutionEngine", f"  Optimizer failed on attempt {attempt_num}: {e}")
                continue

            # ── Step 6: Walk-Forward ──────────────────────────────────────
            try:
                from ml_trading_lab.WalkForward.validator import WalkForwardValidator
                _log("VALIDATION", "EvolutionEngine",
                     f"  Running Walk-Forward (attempt {attempt_num})...")
                validator = WalkForwardValidator()
                wf_report = validator.validate(
                    df_feats, grid, best_params, n_splits=3, symbol=symbol, timeframe="M1",
                    ml_filter_fn=ml_filter_fn
                )
                wf_approved = wf_report.get("approved", False)
                wf_r = wf_report.get("walk_forward_metrics", {}).get("total_realized_r", 0)
                _log("VALIDATION", "EvolutionEngine",
                     f"  Walk-Forward: {'✅ APPROVED' if wf_approved else '❌ REJECTED'} "
                     f"(R={wf_r:.2f})")
                report["walk_forward"] = wf_report
            except Exception as e:
                _log("ERROR", "EvolutionEngine", f"  Walk-Forward failed: {e}")
                wf_approved = False
                wf_report = {"approved": False, "error": str(e)}

            # ── Step 7: Monte Carlo ───────────────────────────────────────
            try:
                from ml_trading_lab.WalkForward.monte_carlo import MonteCarloValidator
                _log("VALIDATION", "EvolutionEngine",
                     f"  Running Monte Carlo on {len(mc_r_vals)} trades (attempt {attempt_num})...")
                mc = MonteCarloValidator(n_simulations=10_000, max_drawdown_limit_r=20.0)
                mc_report = mc.validate(mc_r_vals)
                mc_approved = mc_report.get("approved", False)
                dd95 = mc_report.get("sim_stats", {}).get("worst_5pct_max_dd", 0)
                luck = mc_report.get("sim_stats", {}).get("prob_by_luck", 1.0)
                _log("VALIDATION", "EvolutionEngine",
                     f"  Monte Carlo: {'✅ APPROVED' if mc_approved else '❌ REJECTED'} "
                     f"(Worst-5%DD={dd95:.1f}R, LuckProb={luck:.0%})")
                report["monte_carlo"] = mc_report
            except Exception as e:
                _log("ERROR", "EvolutionEngine", f"  Monte Carlo failed: {e}")
                mc_approved = False
                mc_report = {"approved": False, "error": str(e)}

            approved = wf_approved and mc_approved

            if approved:
                _log("ML", "EvolutionEngine",
                     f"✅ Attempt {attempt_num} APPROVED — proceeding to EA generation")
                break
            else:
                reasons = []
                if not wf_approved: reasons.append("Walk-Forward")
                if not mc_approved: reasons.append("Monte Carlo")
                reason_str = " + ".join(reasons)
                if attempt_num < len(_RETRY_CONFIGS):
                    next_cfg = _RETRY_CONFIGS[attempt_idx + 1]
                    _log("ML", "EvolutionEngine",
                         f"⟳ Attempt {attempt_num} rejected ({reason_str}). "
                         f"Retrying with '{next_cfg['label']}'...")
                else:
                    _log("VALIDATION", "EvolutionEngine",
                         f"❌ All {len(_RETRY_CONFIGS)} attempts exhausted — strategy not approved.")

        # ── Step 8: EA Generation ─────────────────────────────────────────
        report["approved"] = approved

        if approved:
            _log("ML", "EvolutionEngine", "Step 8/8 — Generating versioned MQL5 EA...")
            _prog(92, "Generating EA...")
            try:
                from ml_trading_lab.EA_Generator.generator import EAGenerator
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                os.makedirs(self.ea_directory, exist_ok=True)
                ea_path = os.path.join(self.ea_directory, f"ML_EA_evo_{ts}.mq5")
                EAGenerator().generate(best_params, ea_path, ml_rules_expression=mql5_rules)
                report["ea_path"] = ea_path
                _log("ML", "EvolutionEngine", f"✅ EA generated at {ea_path}")
            except Exception as e:
                _log("ERROR", "EvolutionEngine", f"EA generation failed: {e}")
                report["ea_path"] = None
        else:
            reasons = []
            if not wf_report.get("approved", False):
                reasons.append("Walk-Forward rejected (candidate doesn't beat baseline)")
            if not mc_report.get("approved", False):
                reasons.append("Monte Carlo rejected (DD risk or random-luck concern)")
            report["rejection_reasons"] = reasons

        report["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _prog(100, f"Evolution cycle complete — {'APPROVED ✅' if approved else 'REJECTED ❌'}")
        return report


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_Xy(labeled: pl.DataFrame, df_feats: pl.DataFrame):
    """Extract feature matrix and targets from labeled dataset."""
    raw_cols = {"timestamp", "open", "high", "low", "close",
                "tick_volume", "spread", "tp_before_sl", "realized_r",
                "direction", "entry_price", "stop_loss", "take_profit",
                "setup_id", "exit_price", "exit_timestamp", "bars_to_exit"}

    feat_cols = [c for c in labeled.columns if c not in raw_cols
                 and labeled[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]
    if not feat_cols:
        feat_cols = [c for c in labeled.columns if c.startswith("feature_")]
    if not feat_cols:
        return None, None, []

    X = labeled.select(feat_cols).fill_null(0.0).to_numpy()
    y = labeled["tp_before_sl"].cast(pl.Int32).to_numpy()
    return X, y, feat_cols


def _build_ml_filter(best_model, feat_cols: List[str], threshold_val: float):
    """Build a callable filter using the best trained classifier model.

    The returned function accepts a feature row dict and returns True
    if the model's win-probability for that row is >= threshold_val.
    """
    if best_model is None:
        return None

    def _filter_fn(row: dict) -> bool:
        try:
            # Build feature vector matching the training column order
            x_row = []
            for col in feat_cols:
                # Remove "feature_" prefix for lookup in row
                lookup_col = col.replace("feature_", "")
                val = row.get(lookup_col)
                if val is None:
                    val = row.get(col, 0.0)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
                x_row.append(float(val))

            X = np.array(x_row, dtype=np.float32).reshape(1, -1)
            prob = float(best_model.predict_proba(X)[0])
            return prob >= threshold_val
        except Exception:
            return True  # on error, don't block the trade

    return _filter_fn
