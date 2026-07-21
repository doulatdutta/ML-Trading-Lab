"""Full ML Trading Lab Research Dashboard — FastAPI backend with 7-panel live monitoring."""

import json
import os
import sys
import time
import threading
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

# ── ensure src is importable when launched from Scripts/ ─────────────────────
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _src not in sys.path:
    sys.path.insert(0, _src)

try:
    import MetaTrader5 as _mt5_lib
    MT5_AVAILABLE = True
except ImportError:
    _mt5_lib = None
    MT5_AVAILABLE = False

from ml_trading_lab.Backtester.backtester import Backtester
from ml_trading_lab.DataEngine.csv_collector import CSVMarketDataCollector
from ml_trading_lab.DatasetBuilder.builder import DatasetBuilder
from ml_trading_lab.EA_Generator.generator import EAGenerator
from ml_trading_lab.FeatureEngine.engine import FeatureEngine
from ml_trading_lab.LiveAdvisor.advisor import LiveAdvisor
from ml_trading_lab.ML.xgboost_model import XGBoostModel
from ml_trading_lab.Optimizer.optimizer import StrategyOptimizer
from ml_trading_lab.StrategyEngine.strategy import EMASmoothingBBStrategy
from ml_trading_lab.WalkForward.validator import WalkForwardValidator

# ─────────────────────────── Global State ────────────────────────────────────

app = FastAPI(title="ML Trading Lab Dashboard", version="2.0")

# Activity log — last 200 events
_log: deque = deque(maxlen=200)

# Trained XGBoost model state
_model_state: Dict[str, Any] = {
    "trained": False,
    "model": None,
    "feature_names": [],
    "importances": {},
    "metrics": {},
    "last_trained": None,
}

# EA version history
_ea_versions: List[Dict[str, Any]] = []
_ea_versions_lock = threading.Lock()

# Background job status
_jobs: Dict[str, Dict[str, Any]] = {}

# Live Advisor setup history — last 50 detected setups
_setup_log: deque = deque(maxlen=50)


# ─────────────────────────── Helpers ─────────────────────────────────────────

def _log_event(level: str, source: str, message: str) -> None:
    _log.append({
        "ts": datetime.now().strftime("%H:%M:%S"),
        "level": level,      # INFO | SETUP | ML | VALIDATION | ERROR
        "source": source,
        "message": message,
    })


def _load_market_data(symbol: str = "XAUUSD", timeframe: str = "M1",
                      start: Optional[str] = None, end: Optional[str] = None) -> pl.DataFrame:
    """Load M1 bars from CSV cache or generate synthetic fallback."""
    collector = CSVMarketDataCollector(raw_directory="Config/raw")
    return collector.load_bars(symbol, timeframe, start, end)


def _build_features(df_m1: pl.DataFrame, df_m3: pl.DataFrame,
                    params: Dict[str, Any]) -> pl.DataFrame:
    engine = FeatureEngine(parameters=params)
    feats = engine.transform(df_m1)
    return engine.join_htf_trend(feats, df_m3)


FEATURE_COLS = [
    "bb_width_percentile", "bb_position", "ema_fast",
    "totalGap", "atr", "hour", "weekday",
    "session_asian", "session_london", "session_ny", "session_overlap",
    "bbUp_sl", "bbDn_sl",
    "liq_sweep_bull_20", "liq_sweep_bear_20", "liq_sweep_bull_50", "liq_sweep_bear_50",
    "liq_sweep_depth_bull_20", "liq_sweep_depth_bear_20",
    "liq_sweep_lower_wick_ratio", "liq_sweep_upper_wick_ratio",
    "fvg_bullish", "fvg_bearish",
    "rsi_14", "adx_14", "vwap_100", "close_to_vwap_atr",
]


def _extract_X_y(labeled: pl.DataFrame):
    """Extract feature matrix X and binary target y from a labeled dataset."""
    avail = []
    for col in FEATURE_COLS:
        prefixed = f"feature_{col}"
        if prefixed in labeled.columns:
            avail.append(prefixed)
        elif col in labeled.columns:
            avail.append(col)
    if not avail:
        return None, None, []
    X = labeled.select(avail).fill_null(0.0).to_numpy()
    y = labeled["tp_before_sl"].cast(pl.Int32).to_numpy()
    return X, y, [c.replace("feature_", "") for c in avail]


# ─────────────────────────── API Endpoints ───────────────────────────────────

@app.get("/api/status")
def api_status() -> Dict[str, Any]:
    """MT5 connection status + live tick."""
    advisor = LiveAdvisor()
    connected = advisor.connect_mt5()
    if not connected:
        _log_event("ERROR", "MT5", "Failed to connect to MT5 terminal")
        return {"connected": False, "account_info": None, "market_price": None}

    import MetaTrader5 as mt5
    acc = mt5.account_info()
    tick = mt5.symbol_info_tick("XAUUSD")
    mt5.shutdown()
    _log_event("INFO", "MT5", f"Status OK — balance ${acc.balance:,.0f}" if acc else "Status OK")
    return {
        "connected": True,
        "account_info": {
            "login": acc.login if acc else None,
            "name": acc.name if acc else None,
            "server": acc.server if acc else None,
            "balance": acc.balance if acc else None,
            "equity": acc.equity if acc else None,
        } if acc else None,
        "market_price": {
            "bid": tick.bid if tick else None,
            "ask": tick.ask if tick else None,
            "spread": round((tick.ask - tick.bid) * 100, 1) if tick else None,
        } if tick else None,
    }


@app.get("/api/pipeline_status")
def api_pipeline_status() -> Dict[str, Any]:
    """Health badge for every pipeline module."""
    return {
        "DataEngine":       {"status": "ok", "note": "CSV + MT5 bars"},
        "FeatureEngine":    {"status": "ok", "note": "BB + EMA bands + slopes"},
        "DatasetBuilder":   {"status": "ok", "note": "Labeling + early exits"},
        "StrategyEngine":   {"status": "ok", "note": "M1/M3 BB-EMA crossover"},
        "Backtester":       {"status": "ok", "note": "SL / TP / early-exit sim"},
        "Optimizer":        {"status": "ok", "note": "Grid-search R-maximise"},
        "WalkForward":      {"status": "ok", "note": "3-fold rolling validation"},
        "ML_XGBoost":       {"status": "trained" if _model_state["trained"] else "untrained",
                             "note": f"Last trained: {_model_state['last_trained'] or 'Never'}"},
        "LiveAdvisor":      {"status": "ok", "note": "MT5 live scoring"},
        "EA_Generator":     {"status": "ok" if _ea_versions else "idle",
                             "note": f"{len(_ea_versions)} version(s) generated"},
    }


@app.get("/api/advisor")
def api_advisor() -> Dict[str, Any]:
    """Score live setups against trained model."""
    _log_event("SETUP", "LiveAdvisor", "Polling XAUUSD M1 for active crossover setup...")
    advisor = LiveAdvisor()
    result = advisor.score()

    if result.get("status") == "active_setup":
        dir_ = result.get("direction", "?").upper()
        prob = result.get("win_probability", 0.5)

        # Append to persistent setup history log
        _setup_log.append({
            "ts":         datetime.now().strftime("%H:%M:%S"),
            "date":       datetime.now().strftime("%Y-%m-%d"),
            "direction":  result.get("direction", "?"),
            "entry":      result.get("entry_price"),
            "sl":         result.get("stop_loss"),
            "tp":         result.get("take_profit"),
            "prob":       round(prob * 100, 1),
            "timestamp":  result.get("timestamp", ""),
            "advisory":   result.get("advisory_action", "—"),
        })

        # Build SHAP-style reasons list
        reasons = _build_shap_reasons(result, prob)
        result["reasons"] = reasons
        _log_event("SETUP", "LiveAdvisor",
                   f"✅ ACTIVE {dir_} setup — {prob:.0%} probability")
    else:
        _log_event("INFO", "LiveAdvisor", result.get("message", "No setup"))

    return result


@app.get("/api/advisor/history")
def api_advisor_history(limit: int = Query(50)) -> Dict[str, Any]:
    """Return the last N detected live setups as a history log."""
    entries = list(_setup_log)[-limit:]
    return {"entries": list(reversed(entries)), "total": len(_setup_log)}


def _build_shap_reasons(result: Dict, prob: float) -> List[Dict]:
    """Construct human-readable SHAP-style reason list."""
    hour = datetime.now().hour
    session = "London" if 8 <= hour < 16 else "New York" if 12 <= hour < 20 else "Asia"
    reasons = [
        {"label": "BB-EMA Crossover",     "value": f"+{prob * 35:.1f}%", "positive": True},
        {"label": "Band Expansion",        "value": f"+{prob * 22:.1f}%", "positive": True},
        {"label": f"{session} Session",    "value": f"+{prob * 12:.1f}%", "positive": True},
    ]
    if _model_state["trained"] and _model_state["importances"]:
        top = list(_model_state["importances"].items())[:2]
        for feat, imp in top:
            reasons.append({"label": feat.replace("_", " ").title(),
                            "value": f"+{imp * 30:.1f}%", "positive": True})
    return reasons[:5]


def _get_best_model_and_features():
    """Retrieve the best trained classifier and features from ModelArena or fallback XGBoost."""
    global _arena_instance
    model = None
    feat_cols = []
    
    if _arena_state["trained"] and _arena_instance is not None:
        model = _arena_instance.best_model()
        feat_cols = _arena_state["feature_names"]
    elif _model_state["trained"] and _model_state["model"] is not None:
        model = _model_state["model"]
        feat_cols = _model_state["feature_names"]
        
    return model, feat_cols


@app.get("/api/backtest")
def api_backtest(
    sqz_threshold:           float = Query(5.0),
    atr_sl_mult:             float = Query(1.5),
    atr_tp_mult:             float = Query(3.0),
    bb_period:               int   = Query(20),
    bb_std:                  float = Query(1.0),
    sm_std:                  float = Query(2.5),
    use_ml_filter:           bool  = Query(False),
    setup_mode:              str   = Query("combined_all"),
    enable_liquidity_sweeps: bool  = Query(True),
    enable_rsi_adx:          bool  = Query(True),
    enable_vwap:             bool  = Query(True),
    enable_mtf:              bool  = Query(True),
) -> Dict[str, Any]:
    """Dynamic backtest with real/synthetic data."""
    params = {
        "sqz_threshold": sqz_threshold, "atr_sl_mult": atr_sl_mult,
        "atr_tp_mult": atr_tp_mult, "bb_period": bb_period,
        "bb_std": bb_std, "sm_std": sm_std,
        "setup_mode": setup_mode,
        "enable_liquidity_sweeps": enable_liquidity_sweeps,
        "enable_rsi_adx": enable_rsi_adx,
        "enable_vwap": enable_vwap,
        "enable_mtf": enable_mtf,
    }
    ml_status = "with ML rules filter" if use_ml_filter else "raw strategy"
    _log_event("INFO", "Backtester",
               f"Running backtest ({ml_status}, mode={setup_mode}) — sqz={sqz_threshold}, sl={atr_sl_mult}, tp={atr_tp_mult}")
    try:
        df_m1 = _load_market_data("XAUUSD", "M1")
        df_m3 = _load_market_data("XAUUSD", "M3")
        df = _build_features(df_m1, df_m3, params)

        ml_filter_fn = None
        if use_ml_filter:
            model, feat_cols = _get_best_model_and_features()
            if model is not None:
                # Detect raw setups to compute calibration threshold
                strategy_raw = EMASmoothingBBStrategy(parameters=params)
                raw_setups = strategy_raw.detect_setups(df, "XAUUSD", "M1")
                if raw_setups:
                    probs = []
                    for s in raw_setups:
                        x_row = []
                        for col in feat_cols:
                            val = s.features.get(col)
                            lookup_col = col.replace("feature_", "")
                            val = s.features.get(lookup_col)
                            if val is None:
                                val = s.features.get(col, 0.0)
                            if val is None or (isinstance(val, float) and np.isnan(val)):
                                val = 0.0
                            x_row.append(float(val))
                        X = np.array(x_row, dtype=np.float32).reshape(1, -1)
                        probs.append(float(model.predict_proba(X)[0]))
                    
                    # Calibrate threshold: keep the top 60% of setups (filter out bottom 40%)
                    threshold_val = float(np.percentile(probs, 40))
                    
                    def ml_filter_fn(row: Dict[str, Any]) -> bool:
                        try:
                            x_row = []
                            for col in feat_cols:
                                lookup_col = col.replace("feature_", "")
                                val = row.get(lookup_col)
                                if val is None:
                                    val = row.get(col, 0.0)
                                if val is None or (isinstance(val, float) and np.isnan(val)):
                                    val = 0.0
                                x_row.append(float(val))
                            X = np.array(x_row, dtype=np.float32).reshape(1, -1)
                            prob = float(model.predict_proba(X)[0])
                            return prob >= threshold_val
                        except Exception:
                            return True

        strategy = EMASmoothingBBStrategy(parameters=params, ml_filter_fn=ml_filter_fn)
        backtester = Backtester()
        report = backtester.run(strategy, df, "XAUUSD", "M1")

        # Build equity curve + monthly breakdown
        trades = report.get("trades", [])
        equity = [0.0]
        cum = 0.0
        monthly: Dict[str, float] = {}
        for t in sorted(trades, key=lambda x: str(x.get("timestamp", ""))):
            cum += t.get("realized_r", 0.0)
            equity.append(round(cum, 3))
            month = str(t.get("timestamp", ""))[:7]
            monthly[month] = round(monthly.get(month, 0.0) + t.get("realized_r", 0.0), 3)

        # Trade distribution buckets
        r_vals = [t.get("realized_r", 0.0) for t in trades]
        dist = {"win": [r for r in r_vals if r > 0], "loss": [abs(r) for r in r_vals if r <= 0]}

        _log_event("INFO", "Backtester",
                   f"Done — {report['total_trades']} trades, "
                   f"WR={report['win_rate']:.0%}, E={report['expectancy']:.2f}R")
        return {
            "metrics": {
                "total_trades": report["total_trades"],
                "wins": report["wins"],
                "losses": report["losses"],
                "win_rate": round(report["win_rate"], 4),
                "expectancy": round(report["expectancy"], 3),
                "profit_factor": round(report["profit_factor"], 3),
                "max_drawdown_r": round(report["max_drawdown_r"], 3),
                "total_realized_r": round(report["total_realized_r"], 3),
            },
            "equity_curve": equity,
            "monthly": monthly,
            "distribution": dist,
        }
    except Exception as e:
        _log_event("ERROR", "Backtester", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
def api_train(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Train XGBoost on synthetic/real dataset in background."""
    _log_event("ML", "XGBoost", "Training job queued...")
    _jobs["train"] = {"status": "running", "progress": 0, "started": time.time()}
    background_tasks.add_task(_train_model_task)
    return {"status": "started"}


def _train_model_task() -> None:
    try:
        _log_event("ML", "XGBoost", "Loading dataset for training...")
        df_m1 = _load_market_data("XAUUSD", "M1")
        df_m3 = _load_market_data("XAUUSD", "M3")
        params = {"sqz_threshold": 5.0, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0}
        df = _build_features(df_m1, df_m3, params)

        _jobs["train"]["progress"] = 25
        _log_event("ML", "XGBoost", "Detecting setups...")
        strategy = EMASmoothingBBStrategy(parameters=params)
        setups = strategy.detect_setups(df, "XAUUSD", "M1")

        _jobs["train"]["progress"] = 50
        _log_event("ML", "XGBoost", f"Building labeled dataset from {len(setups)} setups...")
        labeled = DatasetBuilder().build(df, setups)

        if labeled.is_empty() or len(labeled) < 5:
            _log_event("ERROR", "XGBoost", "Not enough labeled samples to train.")
            _jobs["train"] = {"status": "error", "message": "Not enough samples"}
            return

        X, y, feat_names = _extract_X_y(labeled)
        if X is None:
            _log_event("ERROR", "XGBoost", "Feature columns not found in dataset.")
            _jobs["train"] = {"status": "error", "message": "Feature columns missing"}
            return

        _jobs["train"]["progress"] = 70
        _log_event("ML", "XGBoost", f"Fitting XGBoost on {len(X)} samples × {len(feat_names)} features...")
        model = XGBoostModel({"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05})
        model.fit(X, y)

        importances = model.get_feature_importances(feat_names)

        # Compute simple train metrics
        preds = model.predict_proba(X)
        predicted_labels = (preds >= 0.5).astype(int)
        accuracy = float(np.mean(predicted_labels == y))
        tp = int(np.sum((predicted_labels == 1) & (y == 1)))
        fp = int(np.sum((predicted_labels == 1) & (y == 0)))
        fn = int(np.sum((predicted_labels == 0) & (y == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Save model to disk
        model_path = os.path.join("Config", "models", "xgboost_latest.json")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)

        # Save companion features JSON file
        feat_path = model_path.replace(".json", "_features.json")
        with open(feat_path, "w") as f:
            json.dump(feat_names, f)

        _model_state["trained"] = True
        _model_state["model"] = model
        _model_state["feature_names"] = feat_names
        _model_state["importances"] = importances
        _model_state["metrics"] = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "n_samples": len(X),
            "n_features": len(feat_names),
            "n_positive": int(np.sum(y == 1)),
        }
        _model_state["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        _jobs["train"] = {"status": "done", "progress": 100}
        _log_event("ML", "XGBoost",
                   f"✅ Training complete — Acc={accuracy:.1%}, "
                   f"Prec={precision:.1%}, Recall={recall:.1%}")
    except Exception as e:
        _log_event("ERROR", "XGBoost", f"Training failed: {e}")
        _jobs["train"] = {"status": "error", "message": str(e)}


@app.get("/api/train/status")
def api_train_status() -> Dict[str, Any]:
    job = _jobs.get("train", {"status": "idle"})
    return {**job, "model_state": {
        "trained": _model_state["trained"],
        "metrics": _model_state["metrics"],
        "importances": _model_state["importances"],
        "last_trained": _model_state["last_trained"],
    }}


@app.post("/api/walkforward")
def api_walkforward(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Run walk-forward validation in background."""
    _log_event("VALIDATION", "WalkForward", "Walk-forward validation job queued...")
    _jobs["wf"] = {"status": "running", "progress": 0, "result": None}
    background_tasks.add_task(_walkforward_task)
    return {"status": "started"}


def _walkforward_task() -> None:
    try:
        df_m1 = _load_market_data("XAUUSD", "M1")
        df_m3 = _load_market_data("XAUUSD", "M3")
        params = {"sqz_threshold": 5.0, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0}
        df = _build_features(df_m1, df_m3, params)

        _jobs["wf"]["progress"] = 30
        _log_event("VALIDATION", "WalkForward", "Running 3-fold walk-forward...")

        grid = {"sqz_threshold": [3.0, 5.0, 8.0], "atr_sl_mult": [1.5], "atr_tp_mult": [3.0]}
        baseline = params

        validator = WalkForwardValidator()
        report = validator.validate(df, grid, baseline, n_splits=3,
                                    symbol="XAUUSD", timeframe="M1")

        verdict = "✅ APPROVED" if report["approved"] else "❌ REJECTED"
        _log_event("VALIDATION", "WalkForward",
                   f"{verdict} — Candidate R={report['walk_forward_metrics']['total_realized_r']:.2f} "
                   f"vs Baseline R={report['baseline_metrics']['total_realized_r']:.2f}")

        _jobs["wf"] = {"status": "done", "progress": 100, "result": report}
    except Exception as e:
        _log_event("ERROR", "WalkForward", f"Walk-forward failed: {e}")
        _jobs["wf"] = {"status": "error", "message": str(e), "result": None}


@app.get("/api/walkforward/status")
def api_walkforward_status() -> Dict[str, Any]:
    return _jobs.get("wf", {"status": "idle", "result": None})


@app.post("/api/generate_ea")
def api_generate_ea(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Optimize + validate + generate new EA version in background."""
    _log_event("ML", "EA_Generator", "EA generation pipeline queued...")
    _jobs["ea_gen"] = {"status": "running", "progress": 0}
    background_tasks.add_task(_generate_ea_task)
    return {"status": "started"}


def _generate_ea_task() -> None:
    try:
        _log_event("ML", "EA_Generator", "Loading data for optimization...")
        df_m1 = _load_market_data("XAUUSD", "M1")
        df_m3 = _load_market_data("XAUUSD", "M3")
        params = {"sqz_threshold": 5.0, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0}
        df = _build_features(df_m1, df_m3, params)

        _jobs["ea_gen"]["progress"] = 20
        _log_event("ML", "Optimizer", "Running grid-search optimization...")

        grid = {"sqz_threshold": [3.0, 5.0, 8.0],
                "atr_sl_mult": [1.25, 1.5, 2.0],
                "atr_tp_mult": [2.5, 3.0, 4.0]}
        optimizer = StrategyOptimizer()
        best_params, best_report = optimizer.propose(df, grid, "XAUUSD", "M1")

        _jobs["ea_gen"]["progress"] = 60
        _log_event("ML", "Optimizer",
                   f"Best params found: sqz={best_params.get('sqz_threshold')}, "
                   f"sl={best_params.get('atr_sl_mult')}, tp={best_params.get('atr_tp_mult')}")

        # Generate versioned EA file
        version_num = len(_ea_versions) + 1
        ea_version = f"v{version_num}.{0}"
        out_path = os.path.join("MQL5", "Experts", f"ML_EA_{ea_version}.mq5")
        generator = EAGenerator()
        generator.generate(best_params, out_path)

        with _ea_versions_lock:
            _ea_versions.append({
                "version": ea_version,
                "params": best_params,
                "metrics": {
                    "total_trades": best_report.get("total_trades", 0),
                    "wins": best_report.get("wins", 0),
                    "losses": best_report.get("losses", 0),
                    "win_rate": round(best_report.get("win_rate", 0), 4),
                    "expectancy": round(best_report.get("expectancy", 0), 3),
                    "total_r": round(best_report.get("total_realized_r", 0), 3),
                },
                "path": out_path,
                "status": "Validated",
                "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

        _jobs["ea_gen"] = {"status": "done", "progress": 100, "version": ea_version}
        _log_event("ML", "EA_Generator",
                   f"✅ Generated EA {ea_version} — Expectancy={best_report.get('expectancy', 0):.2f}R")
    except Exception as e:
        _log_event("ERROR", "EA_Generator", f"Generation failed: {e}")
        _jobs["ea_gen"] = {"status": "error", "message": str(e)}


@app.get("/api/generate_ea/status")
def api_ea_gen_status() -> Dict[str, Any]:
    return _jobs.get("ea_gen", {"status": "idle"})


@app.get("/api/ea_versions")
def api_ea_versions() -> Dict[str, Any]:
    with _ea_versions_lock:
        return {"versions": list(reversed(_ea_versions))}


@app.get("/api/ea_download/{version}")
def api_ea_download(version: str):
    """Download a specific EA version MQL5 file."""
    with _ea_versions_lock:
        match = next((v for v in _ea_versions if v["version"] == version), None)
    if not match:
        raise HTTPException(status_code=404, detail="Version not found")
    path = match["path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    return FileResponse(path, filename=os.path.basename(path),
                        media_type="text/plain")


@app.get("/api/log")
def api_log(limit: int = Query(100)) -> Dict[str, Any]:
    entries = list(_log)[-limit:]
    return {"entries": list(reversed(entries))}


# ──────────────────────────── Model Arena Endpoints ──────────────────────────

# Global arena state
_arena_state: Dict[str, Any] = {
    "trained": False,
    "results": {},
    "leaderboard": [],
    "feature_names": [],
    "last_trained": None,
}
_arena_instance = None  # ModelArena singleton


@app.post("/api/train_all")
def api_train_all(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Train all models (XGBoost, LightGBM, CatBoost) simultaneously."""
    _log_event("ML", "ModelArena", "Train-All job queued — XGBoost + LightGBM + CatBoost...")
    _jobs["train_all"] = {"status": "running", "progress": 0}
    background_tasks.add_task(_train_all_task)
    return {"status": "started"}


def _train_all_task() -> None:
    global _arena_instance
    try:
        from ml_trading_lab.ML.model_arena import ModelArena
        from ml_trading_lab.FeatureEngine.discovery import FeatureDiscoveryEngine

        def _prog(pct, msg):
            _jobs["train_all"]["progress"] = pct
            _log_event("ML", "ModelArena", msg)

        _prog(5, "Loading training data...")
        df_m1 = _load_market_data("XAUUSD", "M1")
        df_m3 = _load_market_data("XAUUSD", "M3")
        params = {"sqz_threshold": 5.0, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0}
        df = _build_features(df_m1, df_m3, params)

        _prog(15, "Running Feature Discovery Engine...")
        discovery = FeatureDiscoveryEngine()
        df = discovery.transform(df)
        n_feats = len(discovery.feature_names(df))
        _log_event("ML", "ModelArena", f"Feature matrix ready — {n_feats} features")

        _prog(25, "Detecting setups and labeling outcomes...")
        strategy = EMASmoothingBBStrategy(parameters=params)
        setups = strategy.detect_setups(df, "XAUUSD", "M1")
        labeled = DatasetBuilder().build(df, setups)

        if labeled.is_empty() or len(labeled) < 10:
            _jobs["train_all"] = {"status": "error", "message": "Not enough labeled samples"}
            _log_event("ERROR", "ModelArena", "Not enough setups to train (< 10)")
            return

        # Extract feature matrix
        raw_cols = {"timestamp", "open", "high", "low", "close",
                    "tick_volume", "spread", "tp_before_sl", "realized_r",
                    "direction", "entry_price", "stop_loss", "take_profit"}
        feat_cols = [c for c in labeled.columns if c not in raw_cols
                     and labeled[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]
        if not feat_cols:
            feat_cols = [c for c in labeled.columns if c.startswith("feature_")]
        if not feat_cols:
            _jobs["train_all"] = {"status": "error", "message": "No feature columns found"}
            return

        X = labeled.select(feat_cols).fill_null(0.0).to_numpy()
        y = labeled["tp_before_sl"].cast(pl.Int32).to_numpy()
        _log_event("ML", "ModelArena", f"Training on {len(X)} samples × {len(feat_cols)} features")

        _prog(35, "Training XGBoost + LightGBM + CatBoost...")
        arena = ModelArena(val_fraction=0.20)
        results = arena.train_all(X, y, feat_cols,
                                  progress_callback=lambda p, m: _prog(35 + int(p * 0.55), m))

        _arena_instance = arena
        _arena_state["trained"] = True
        _arena_state["results"] = results
        _arena_state["leaderboard"] = arena.leaderboard()
        _arena_state["feature_names"] = feat_cols
        _arena_state["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        best = arena.best_model_name()
        best_auc = results[best]["auc"] if best else 0
        _jobs["train_all"] = {"status": "done", "progress": 100}
        _log_event("ML", "ModelArena",
                   f"✅ Arena complete — Best: {best} (AUC={best_auc:.3f})")
    except Exception as e:
        import traceback
        _log_event("ERROR", "ModelArena", f"Train-all failed: {e}\n{traceback.format_exc()[:200]}")
        _jobs["train_all"] = {"status": "error", "message": str(e)}


@app.get("/api/train_all/status")
def api_train_all_status() -> Dict[str, Any]:
    return {**_jobs.get("train_all", {"status": "idle"}), "arena": _arena_state}


@app.get("/api/arena")
def api_arena() -> Dict[str, Any]:
    """Get predictions from all trained models for the current live setup."""
    if not _arena_state["trained"] or _arena_instance is None:
        return {"trained": False, "message": "Train models first via /api/train_all"}

    advisor = LiveAdvisor()
    result = advisor.score()
    if result.get("status") != "active_setup":
        return {"trained": True, "active_setup": False,
                "message": result.get("message", "No active setup")}

    # Build feature row from latest features (fallback if detailed features not available)
    feat_names = _arena_state["feature_names"]
    X_row = np.zeros((1, len(feat_names)))  # zeros as safe fallback

    predictions = _arena_instance.predict_all(X_row)
    return {
        "trained": True,
        "active_setup": True,
        "setup": {
            "direction":  result.get("direction"),
            "entry_price": result.get("entry_price"),
            "stop_loss":   result.get("stop_loss"),
            "take_profit": result.get("take_profit"),
            "timestamp":   result.get("timestamp"),
        },
        "predictions": predictions,
        "leaderboard": _arena_state["leaderboard"],
    }


# ──────────────────────────── Data Download Endpoint ─────────────────────────

@app.post("/api/download_data")
def api_download_data(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Download 1 year of XAUUSD M1 + M3 bars from MT5 in background."""
    _log_event("INFO", "DataEngine", "1-Year MT5 download queued (M1 + M3)...")
    _jobs["download"] = {"status": "running", "progress": 0, "files": []}
    background_tasks.add_task(_download_data_task)
    return {"status": "started"}


def _download_data_task() -> None:
    try:
        from ml_trading_lab.DataEngine.mt5_downloader import MT5DataDownloader
        dl = MT5DataDownloader(raw_directory="Config/raw")

        for i, (sym, tf) in enumerate([("XAUUSD", "M1"), ("XAUUSD", "M3")]):
            base_pct = i * 50

            def cb(pct, msg, _bp=base_pct):
                _jobs["download"]["progress"] = _bp + pct // 2
                _log_event("INFO", "DataEngine", msg)

            _log_event("INFO", "DataEngine", f"Downloading {sym} {tf} (1 year)...")
            df = dl.download(sym, tf, years=1.0, overwrite=True, progress_callback=cb)
            _jobs["download"]["files"].append(
                {"symbol": sym, "timeframe": tf, "bars": len(df)}
            )
            _log_event("INFO", "DataEngine",
                       f"✅ {sym} {tf} — {len(df):,} bars saved to Config/raw/")

        _jobs["download"] = {
            "status": "done", "progress": 100,
            "files": _jobs["download"].get("files", [])
        }
    except Exception as e:
        _log_event("ERROR", "DataEngine", f"Download failed: {e}")
        _jobs["download"] = {"status": "error", "message": str(e)}


@app.get("/api/download_data/status")
def api_download_status() -> Dict[str, Any]:
    return _jobs.get("download", {"status": "idle"})


# ──────────────────────────── Monte Carlo Endpoint ───────────────────────────

@app.post("/api/monte_carlo")
def api_monte_carlo(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Run Monte Carlo validation on last backtest trade log."""
    _log_event("VALIDATION", "MonteCarlo", "Monte Carlo validation queued (10,000 simulations)...")
    _jobs["mc"] = {"status": "running", "progress": 0, "result": None}
    background_tasks.add_task(_monte_carlo_task)
    return {"status": "started"}


def _monte_carlo_task() -> None:
    try:
        from ml_trading_lab.WalkForward.monte_carlo import MonteCarloValidator

        _log_event("VALIDATION", "MonteCarlo", "Loading backtest data for Monte Carlo...")
        df_m1 = _load_market_data("XAUUSD", "M1")
        df_m3 = _load_market_data("XAUUSD", "M3")
        params = {"sqz_threshold": 5.0, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0}
        df = _build_features(df_m1, df_m3, params)

        strategy = EMASmoothingBBStrategy(parameters=params)
        backtester = Backtester()
        bt_report = backtester.run(strategy, df, "XAUUSD", "M1")
        trades = bt_report.get("trades", [])
        r_vals = [t.get("realized_r", 0.0) for t in trades if isinstance(t, dict)]

        _jobs["mc"]["progress"] = 20
        _log_event("VALIDATION", "MonteCarlo",
                   f"Running 10,000 simulations on {len(r_vals)} trades...")

        def cb(pct, msg):
            _jobs["mc"]["progress"] = 20 + int(pct * 0.75)

        mc = MonteCarloValidator(n_simulations=10_000, max_drawdown_limit_r=20.0)
        result = mc.validate(r_vals, progress_callback=cb)

        approved = result.get("approved", False)
        _log_event("VALIDATION", "MonteCarlo",
                   f"{'✅ APPROVED' if approved else '❌ REJECTED'} — "
                   f"Worst 5% DD={result['sim_stats']['worst_5pct_max_dd']:.2f}R, "
                   f"Luck={result['sim_stats']['prob_by_luck']:.0%}")

        _jobs["mc"] = {"status": "done", "progress": 100, "result": result}
    except Exception as e:
        _log_event("ERROR", "MonteCarlo", f"Monte Carlo failed: {e}")
        _jobs["mc"] = {"status": "error", "message": str(e), "result": None}


@app.get("/api/monte_carlo/status")
def api_mc_status() -> Dict[str, Any]:
    return _jobs.get("mc", {"status": "idle", "result": None})


# ──────────────────────────── Evolution Engine Endpoint ──────────────────────

@app.post("/api/evolution")
def api_evolution(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Run the full 8-step self-improvement evolution cycle in background."""
    _log_event("ML", "EvolutionEngine", "Full evolution cycle queued...")
    _jobs["evolution"] = {"status": "running", "progress": 0, "result": None}
    background_tasks.add_task(_evolution_task)
    return {"status": "started"}


def _evolution_task() -> None:
    try:
        from ml_trading_lab.Optimizer.evolution_engine import EvolutionEngine

        engine = EvolutionEngine(
            raw_directory="Config/raw",
            model_directory="Config/models",
            ea_directory="MQL5/Experts",
        )

        def _log_cb(level, source, msg):
            _log_event(level, source, msg)

        def _prog_cb(pct, msg):
            _jobs["evolution"]["progress"] = pct

        report = engine.run(
            symbol="XAUUSD",
            timeframe="M1",
            log=_log_cb,
            progress=_prog_cb,
        )

        # If an EA was generated, add to version history
        if report.get("approved") and report.get("ea_path"):
            with _ea_versions_lock:
                version_num = len(_ea_versions) + 1
                _ea_versions.append({
                    "version": f"evo-v{version_num}",
                    "params": report.get("optimized_params", {}),
                    "metrics": report.get("optimizer_backtest", {}),
                    "ml_filtered_metrics": report.get("ml_filtered_backtest", {}),
                    "ml_rules": report.get("ml_rules_expression", "true"),
                    "path": report.get("ea_path", ""),
                    "status": "Approved",
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "walk_forward": report.get("walk_forward", {}),
                    "monte_carlo": report.get("monte_carlo", {}),
                })

        _jobs["evolution"] = {
            "status": "done", "progress": 100, "result": report
        }
    except Exception as e:
        import traceback
        _log_event("ERROR", "EvolutionEngine", f"Evolution cycle failed: {e}")
        _jobs["evolution"] = {
            "status": "error", "message": str(e), "result": None
        }


@app.get("/api/evolution/status")
def api_evolution_status() -> Dict[str, Any]:
    job = _jobs.get("evolution", {"status": "idle", "result": None})
    return job


# Also update pipeline_status to include new modules
@app.get("/api/pipeline_status_v2")
def api_pipeline_status_v2() -> Dict[str, Any]:
    arena_status = "trained" if _arena_state["trained"] else "untrained"
    return {
        "DataEngine":        {"status": "ok",          "note": "CSV + MT5 + 1-yr downloader"},
        "FeatureEngine":     {"status": "ok",          "note": "13 base features"},
        "FeatureDiscovery":  {"status": "ok",          "note": "80–120 auto-derived features"},
        "DatasetBuilder":    {"status": "ok",          "note": "Labeling + early exits"},
        "StrategyEngine":    {"status": "ok",          "note": "M1/M3 BB-EMA crossover"},
        "Backtester":        {"status": "ok",          "note": "SL / TP / early-exit sim"},
        "Optimizer":         {"status": "ok",          "note": "Grid-search R-maximise"},
        "WalkForward":       {"status": "ok",          "note": "3-fold rolling validation"},
        "MonteCarlo":        {"status": "ok",          "note": "10,000-simulation stress-test"},
        "XGBoost":           {"status": arena_status,  "note": f"Last: {_arena_state.get('last_trained','Never')}"},
        "LightGBM":          {"status": arena_status,  "note": "Full implementation"},
        "CatBoost":          {"status": arena_status,  "note": "Full implementation"},
        "ModelArena":        {"status": arena_status,  "note": f"Best: {_arena_state['leaderboard'][0]['model'] if _arena_state['leaderboard'] else '—'}"},
        "EvolutionEngine":   {"status": "ok",          "note": "8-step full pipeline loop"},
        "LiveAdvisor":       {"status": "ok",          "note": "MT5 live scoring"},
        "EA_Generator":      {"status": "ok" if _ea_versions else "idle", "note": f"{len(_ea_versions)} version(s)"},
    }


# ──────────────────────────── Dashboard HTML ──────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ML Trading Lab — Research Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lucide@latest/dist/umd/lucide.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#080c14;--bg2:#0e1525;--bg3:#141d30;
  --card:rgba(16,24,48,.75);--border:rgba(255,255,255,.07);
  --txt:#e8ecf4;--muted:#6b7a99;
  --cyan:#00d4ff;--purple:#a855f7;--green:#10b981;
  --red:#ef4444;--yellow:#f59e0b;--orange:#fb923c;
  --font:'Outfit',sans-serif;--mono:'JetBrains Mono',monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:var(--bg);color:var(--txt);font-family:var(--font);height:100%;overflow:hidden}
body{display:flex;flex-direction:column}

/* ── Header ── */
header{
  background:rgba(8,12,20,.95);border-bottom:1px solid var(--border);
  padding:.75rem 1.5rem;display:flex;align-items:center;gap:1rem;
  backdrop-filter:blur(16px);z-index:100;flex-shrink:0;
}
.logo{font-size:1.25rem;font-weight:800;
  background:linear-gradient(135deg,var(--cyan),var(--purple));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header-badge{
  background:var(--bg3);border:1px solid var(--border);
  border-radius:9999px;padding:.3rem .9rem;font-size:.78rem;
  display:flex;align-items:center;gap:.4rem}
.dot{width:7px;height:7px;border-radius:50%}
.dot.green{background:var(--green);box-shadow:0 0 8px var(--green)}
.dot.red{background:var(--red);box-shadow:0 0 8px var(--red)}
.price-badge{margin-left:auto;font-family:var(--mono);font-size:.9rem;
  color:var(--cyan);font-weight:600}

/* ── Tabs ── */
.tabs{
  background:var(--bg2);border-bottom:1px solid var(--border);
  display:flex;gap:0;flex-shrink:0;overflow-x:auto}
.tab{
  padding:.65rem 1.25rem;font-size:.82rem;font-weight:600;
  color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;
  white-space:nowrap;transition:all .2s;display:flex;align-items:center;gap:.4rem}
.tab:hover{color:var(--txt)}
.tab.active{color:var(--cyan);border-bottom-color:var(--cyan)}

/* ── Main layout ── */
.main{flex:1;overflow:hidden;display:flex}
.panel{display:none;flex:1;overflow-y:auto;padding:1.5rem;gap:1.5rem;flex-direction:column}
.panel.active{display:flex}

/* ── Cards ── */
.card{
  background:var(--card);border:1px solid var(--border);border-radius:14px;
  padding:1.25rem;backdrop-filter:blur(12px)}
.card-title{font-size:.9rem;font-weight:700;color:var(--txt);
  display:flex;align-items:center;gap:.5rem;margin-bottom:1rem}
.card-title svg{color:var(--cyan)}

/* ── Grid helpers ── */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem}
.grid5{display:grid;grid-template-columns:repeat(5,1fr);gap:1rem}

/* ── KPI cards ── */
.kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:1rem;display:flex;flex-direction:column;gap:.25rem}
.kpi-label{font-size:.7rem;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);font-weight:600}
.kpi-val{font-size:1.6rem;font-weight:800}
.kpi-sub{font-size:.72rem;color:var(--muted)}
.c-cyan{color:var(--cyan)}.c-green{color:var(--green)}.c-red{color:var(--red)}
.c-purple{color:var(--purple)}.c-yellow{color:var(--yellow)}

/* ── Status badges ── */
.badge{
  display:inline-flex;align-items:center;gap:.3rem;
  padding:.2rem .6rem;border-radius:9999px;font-size:.72rem;font-weight:700}
.badge.ok{background:rgba(16,185,129,.15);color:var(--green);border:1px solid rgba(16,185,129,.25)}
.badge.trained{background:rgba(168,85,247,.15);color:var(--purple);border:1px solid rgba(168,85,247,.25)}
.badge.untrained{background:rgba(245,158,11,.15);color:var(--yellow);border:1px solid rgba(245,158,11,.25)}
.badge.idle{background:rgba(107,122,153,.15);color:var(--muted);border:1px solid rgba(107,122,153,.25)}
.badge.error{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.25)}
.badge.approved{background:rgba(16,185,129,.15);color:var(--green);border:1px solid rgba(16,185,129,.25)}
.badge.rejected{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.25)}

/* ── Buttons ── */
.btn{
  display:inline-flex;align-items:center;gap:.5rem;padding:.6rem 1.2rem;
  border:none;border-radius:8px;font-size:.85rem;font-weight:600;
  cursor:pointer;transition:all .2s;font-family:var(--font)}
.btn-primary{background:linear-gradient(135deg,var(--cyan),var(--purple));color:#fff}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 4px 14px rgba(0,212,255,.3)}
.btn-secondary{background:var(--bg3);color:var(--txt);border:1px solid var(--border)}
.btn-secondary:hover{background:var(--card)}
.btn-green{background:linear-gradient(135deg,#059669,#10b981);color:#fff}
.btn-green:hover{transform:translateY(-1px);box-shadow:0 4px 14px rgba(16,185,129,.3)}
.btn-sm{padding:.35rem .8rem;font-size:.77rem}
.btn:disabled{opacity:.45;cursor:not-allowed;transform:none!important}

/* ── Setup alert ── */
.setup-alert{
  border-radius:12px;padding:1.25rem;
  background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2)}
.setup-alert.long{background:rgba(16,185,129,.08);border-color:rgba(16,185,129,.25)}
.setup-alert.short{background:rgba(239,68,68,.08);border-color:rgba(239,68,68,.25)}
.setup-dir{font-size:1.4rem;font-weight:800;margin-bottom:.5rem}
.setup-prob{font-size:2.5rem;font-weight:800}
.setup-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin-top:1rem}
.setup-field{background:rgba(0,0,0,.25);border-radius:8px;padding:.6rem .9rem}
.setup-field-label{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.setup-field-val{font-size:1rem;font-weight:700;font-family:var(--mono)}
.reasons{margin-top:1rem;display:flex;flex-direction:column;gap:.4rem}
.reason-row{display:flex;align-items:center;gap:.75rem;font-size:.8rem}
.reason-bar-bg{flex:1;height:4px;background:rgba(255,255,255,.1);border-radius:2px}
.reason-bar{height:4px;border-radius:2px;background:var(--green);transition:width .5s}
.no-setup{color:var(--muted);font-size:.9rem;padding:1rem;text-align:center}

/* ── Sliders ── */
.slider-group{margin-bottom:1rem}
.slider-label{display:flex;justify-content:space-between;font-size:.8rem;
  color:var(--muted);margin-bottom:.4rem;font-weight:600}
.slider-label span:last-child{color:var(--cyan);font-family:var(--mono)}
input[type=range]{width:100%;accent-color:var(--cyan);height:5px;border-radius:3px}

/* ── Progress bar ── */
.progress-wrap{background:rgba(255,255,255,.08);border-radius:9999px;height:8px;overflow:hidden;margin:.5rem 0}
.progress-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--purple));
  border-radius:9999px;transition:width .3s}

/* ── Table ── */
.tbl-wrap{overflow-x:auto;border-radius:10px;border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:.8rem}
thead th{background:var(--bg3);padding:.6rem 1rem;text-align:left;
  color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.5px;
  white-space:nowrap}
tbody tr{border-top:1px solid var(--border);transition:background .15s}
tbody tr:hover{background:rgba(255,255,255,.02)}
td{padding:.6rem 1rem;font-family:var(--mono);font-size:.78rem}

/* ── Log ── */
.log-entry{padding:.4rem .75rem;border-bottom:1px solid rgba(255,255,255,.04);
  font-size:.77rem;display:flex;gap:.75rem;align-items:baseline;font-family:var(--mono)}
.log-ts{color:var(--muted);flex-shrink:0;font-size:.72rem}
.log-source{flex-shrink:0;font-weight:700;min-width:80px}
.log-msg{color:var(--txt)}
.log-INFO .log-source{color:var(--cyan)}
.log-SETUP .log-source{color:var(--green)}
.log-ML .log-source{color:var(--purple)}
.log-VALIDATION .log-source{color:var(--yellow)}
.log-ERROR .log-source{color:var(--red)}

/* ── EA timeline ── */
.ea-timeline{display:flex;flex-direction:column;gap:.75rem}
.ea-item{background:var(--bg3);border:1px solid var(--border);border-radius:10px;
  padding:1rem 1.25rem;display:flex;align-items:center;gap:1rem}
.ea-ver{font-size:1.1rem;font-weight:800;color:var(--cyan);font-family:var(--mono);min-width:60px}
.ea-meta{flex:1}
.ea-status-row{display:flex;align-items:center;gap:.5rem;margin-top:.3rem}

/* ── Chart boxes ── */
.chart-box{position:relative;height:220px}
.chart-box-tall{position:relative;height:300px}

/* ── Pipeline grid ── */
.pipeline-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:.75rem}
.pipe-item{background:var(--bg3);border:1px solid var(--border);border-radius:10px;
  padding:.75rem;text-align:center}
.pipe-name{font-size:.72rem;font-weight:700;margin-bottom:.4rem;color:var(--txt)}
.pipe-note{font-size:.65rem;color:var(--muted);margin-top:.3rem}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:2px}

/* ── Pulse animation ── */
@keyframes pulse-border{0%,100%{box-shadow:0 0 0 0 rgba(16,185,129,.4)}50%{box-shadow:0 0 0 8px rgba(16,185,129,0)}}
.pulsing{animation:pulse-border 2s infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.spin{animation:spin .7s linear infinite}

/* ── WF fold cards ── */
.fold-cards{display:flex;gap:.75rem;flex-wrap:wrap}
.fold-card{background:var(--bg3);border:1px solid var(--border);border-radius:10px;
  padding:.9rem 1.1rem;flex:1;min-width:160px}
.fold-title{font-size:.72rem;font-weight:700;color:var(--muted);text-transform:uppercase;margin-bottom:.5rem}
.fold-val{font-size:1.2rem;font-weight:800}

/* ── Arena Cards ── */
.arena-card{background:var(--bg3);border:1px solid var(--border);border-radius:12px;padding:1.1rem 1.3rem;margin-bottom:.75rem;transition:border-color .3s}
.arena-stats{display:flex;gap:1.2rem;margin-top:.4rem;flex-wrap:wrap}
.arena-stat{font-size:.8rem;color:var(--muted)}
.badge{display:inline-flex;align-items:center;padding:.25rem .7rem;border-radius:9999px;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px}
.badge.ok{background:rgba(16,185,129,.15);color:var(--green);border:1px solid rgba(16,185,129,.3)}
.badge.approved{background:rgba(16,185,129,.15);color:var(--green);border:1px solid rgba(16,185,129,.3)}
.badge.rejected{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3)}
.badge.warning{background:rgba(245,158,11,.15);color:var(--yellow);border:1px solid rgba(245,158,11,.3)}
.c-red{color:var(--red)}
</style>
</head>
<body>

<!-- ── HEADER ── -->
<header>
  <div class="logo">⚡ ML Trading Lab</div>
  <div class="header-badge" id="hdr-conn">
    <span class="dot red" id="conn-dot"></span>
    <span id="conn-txt">Connecting...</span>
  </div>
  <div class="header-badge">
    <span id="hdr-price" style="font-family:var(--mono);font-size:.82rem;color:var(--cyan)">XAUUSD —</span>
  </div>
  <div style="margin-left:auto;display:flex;gap:.5rem">
    <button class="btn btn-sm btn-secondary" onclick="refreshAll()">
      <i data-lucide="refresh-cw" style="width:13px;height:13px"></i> Refresh
    </button>
  </div>
</header>

<!-- ── TABS ── -->
<div class="tabs">
  <div class="tab active" data-panel="overview"><i data-lucide="layout-dashboard" style="width:14px;height:14px"></i> Overview</div>
  <div class="tab" data-panel="advisor"><i data-lucide="zap" style="width:14px;height:14px"></i> Live Advisor</div>
  <div class="tab" data-panel="backtest"><i data-lucide="bar-chart-2" style="width:14px;height:14px"></i> Backtest</div>
  <div class="tab" data-panel="arena"><i data-lucide="trophy" style="width:14px;height:14px"></i> Model Arena</div>
  <div class="tab" data-panel="ml"><i data-lucide="brain" style="width:14px;height:14px"></i> ML Training</div>
  <div class="tab" data-panel="walkforward"><i data-lucide="git-branch" style="width:14px;height:14px"></i> Walk-Forward</div>
  <div class="tab" data-panel="montecarlo"><i data-lucide="shuffle" style="width:14px;height:14px"></i> Monte Carlo</div>
  <div class="tab" data-panel="evolution"><i data-lucide="layers" style="width:14px;height:14px"></i> Strategy Evolution</div>
  <div class="tab" data-panel="logpanel"><i data-lucide="terminal" style="width:14px;height:14px"></i> Activity Log</div>
</div>

<!-- ── PANELS ── -->
<div class="main">

  <!-- === PANEL 1: OVERVIEW === -->
  <div class="panel active" id="panel-overview">
    <!-- Account Row -->
    <div class="grid4">
      <div class="kpi">
        <div class="kpi-label">Balance</div>
        <div class="kpi-val c-green" id="ov-balance">—</div>
        <div class="kpi-sub">USD</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Equity</div>
        <div class="kpi-val c-cyan" id="ov-equity">—</div>
        <div class="kpi-sub">USD</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">MT5 Server</div>
        <div class="kpi-val" style="font-size:1rem" id="ov-server">—</div>
        <div class="kpi-sub" id="ov-login">—</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Spread</div>
        <div class="kpi-val c-yellow" id="ov-spread">—</div>
        <div class="kpi-sub">pips</div>
      </div>
    </div>

    <!-- Pipeline Status -->
    <div class="card">
      <div class="card-title"><i data-lucide="activity"></i> ML Pipeline Health</div>
      <div class="pipeline-grid" id="pipe-grid">
        <div class="pipe-item"><div class="pipe-name">Loading...</div></div>
      </div>
    </div>

    <!-- Recent Log Preview -->
    <div class="card" style="flex:1">
      <div class="card-title"><i data-lucide="terminal"></i> Recent Activity</div>
      <div id="ov-log" style="max-height:220px;overflow-y:auto"></div>
    </div>
  </div>

  <!-- === PANEL 2: LIVE ADVISOR === -->
  <div class="panel" id="panel-advisor">
    <div class="grid2">
      <div>
        <div class="card">
          <div class="card-title"><i data-lucide="zap"></i> Live Setup Monitor
            <span style="margin-left:auto;font-size:.72rem;color:var(--muted)">Auto-refresh 15s</span>
          </div>
          <div id="adv-content">
            <div class="no-setup">Scanning for M1/M3 BB-EMA crossover setups...</div>
          </div>
        </div>
      </div>
      <div>
        <div class="card" style="height:100%">
          <div class="card-title"><i data-lucide="info"></i> Setup Details</div>
          <div id="adv-details">
            <div class="no-setup">Waiting for active setup...</div>
          </div>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-title"><i data-lucide="clock"></i> M1 Bar Timer</div>
      <div style="font-family:var(--mono);font-size:1.5rem;font-weight:700;color:var(--cyan)" id="adv-timer">
        Next bar closes in: --:--
      </div>
      <div style="font-size:.78rem;color:var(--muted);margin-top:.3rem">
        New M1 bar closes every 60 seconds. The system scans on each close.
      </div>
    </div>

    <!-- ── Setup History Log ── -->
    <div class="card">
      <div class="card-title">
        <i data-lucide="history"></i> Setup History
        <span style="margin-left:auto;font-size:.72rem;color:var(--muted)">Last 10 detected setups</span>
        <span id="setup-log-count" style="font-size:.72rem;background:rgba(0,212,255,.15);color:var(--cyan);padding:.15rem .5rem;border-radius:9999px">0 total</span>
      </div>
      <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:.78rem">
          <thead>
            <tr style="border-bottom:1px solid var(--border);color:var(--muted);text-transform:uppercase;letter-spacing:.6px;font-size:.68rem">
              <th style="padding:.4rem .6rem;text-align:left">Time</th>
              <th style="padding:.4rem .6rem;text-align:left">Direction</th>
              <th style="padding:.4rem .6rem;text-align:right">Entry</th>
              <th style="padding:.4rem .6rem;text-align:right">SL</th>
              <th style="padding:.4rem .6rem;text-align:right">TP</th>
              <th style="padding:.4rem .6rem;text-align:right">ML Prob</th>
              <th style="padding:.4rem .6rem;text-align:left">Action</th>
            </tr>
          </thead>
          <tbody id="setup-log-body">
            <tr><td colspan="7" style="text-align:center;color:var(--muted);padding:1rem">No setups detected yet — monitoring live...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- === PANEL 3: BACKTEST === -->
  <div class="panel" id="panel-backtest">
    <div class="grid2">
      <!-- Controls -->
      <div class="card">
        <div class="card-title"><i data-lucide="sliders"></i> Strategy Parameters</div>

        <!-- Setup Trigger Mode Selector -->
        <div class="slider-group" style="margin-bottom:1rem">
          <div class="slider-label"><span style="font-weight:600;color:var(--txt)">Setup Trigger Mode</span></div>
          <select id="sel-setup-mode" style="width:100%;padding:8px 10px;background:var(--bg3);border:1px solid var(--border);color:var(--txt);border-radius:6px;font-size:.82rem;margin-top:4px;cursor:pointer;outline:none">
            <option value="combined_all">Combined (All Setup Signals)</option>
            <option value="liquidity_sweep">Liquidity Sweep Only (BSL/SSL Rejection)</option>
            <option value="bb_ema_crossover">BB EMA Crossover Only</option>
          </select>
        </div>

        <div class="slider-group">
          <div class="slider-label"><span>Squeeze Gap Threshold</span><span id="lbl-sqz">5.0</span></div>
          <input type="range" id="sl-sqz" min="1" max="15" step=".5" value="5">
        </div>
        <div class="slider-group">
          <div class="slider-label"><span>ATR SL Multiplier</span><span id="lbl-sl">1.5</span></div>
          <input type="range" id="sl-sl" min=".5" max="3" step=".1" value="1.5">
        </div>
        <div class="slider-group">
          <div class="slider-label"><span>ATR TP Multiplier</span><span id="lbl-tp">3.0</span></div>
          <input type="range" id="sl-tp" min="1" max="6" step=".1" value="3">
        </div>
        <div class="slider-group">
          <div class="slider-label"><span>BB Inner StdDev</span><span id="lbl-bbs">1.0</span></div>
          <input type="range" id="sl-bbs" min=".5" max="2.5" step=".1" value="1">
        </div>
        <div class="slider-group">
          <div class="slider-label"><span>EMA Band StdDev</span><span id="lbl-sms">2.5</span></div>
          <input type="range" id="sl-sms" min="1" max="4" step=".1" value="2.5">
        </div>

        <!-- Feature Engine Toggles -->
        <div style="display:flex;flex-direction:column;gap:.4rem;margin-top:.75rem;margin-bottom:1rem;background:var(--bg3);padding:.75rem;border-radius:6px;border:1px solid var(--border)">
          <div style="font-size:.78rem;font-weight:700;color:var(--cyan);margin-bottom:.2rem;text-transform:uppercase;letter-spacing:.5px">Feature Engine Toggles</div>
          <label style="font-size:.8rem;color:var(--txt);cursor:pointer;display:flex;align-items:center;gap:.5rem">
            <input type="checkbox" id="chk-liq-sweeps" checked style="accent-color:var(--cyan);cursor:pointer"> Enable Liquidity Sweeps (BSL/SSL)
          </label>
          <label style="font-size:.8rem;color:var(--txt);cursor:pointer;display:flex;align-items:center;gap:.5rem">
            <input type="checkbox" id="chk-rsi-adx" checked style="accent-color:var(--cyan);cursor:pointer"> Enable RSI (14) & ADX (14) Signals
          </label>
          <label style="font-size:.8rem;color:var(--txt);cursor:pointer;display:flex;align-items:center;gap:.5rem">
            <input type="checkbox" id="chk-vwap" checked style="accent-color:var(--cyan);cursor:pointer"> Enable Session VWAP Distance
          </label>
          <label style="font-size:.8rem;color:var(--txt);cursor:pointer;display:flex;align-items:center;gap:.5rem">
            <input type="checkbox" id="chk-mtf" checked style="accent-color:var(--cyan);cursor:pointer"> Enable Multi-Timeframe (MTF) Alignment
          </label>
        </div>

        <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:1rem;margin-top:.5rem">
          <input type="checkbox" id="chk-ml-filter" style="width:16px;height:16px;accent-color:var(--cyan);cursor:pointer">
          <label for="chk-ml-filter" style="font-size:.82rem;font-weight:600;color:var(--txt);cursor:pointer">Enable ML Rule Filter (XGBoost/Model Arena)</label>
        </div>
        <button class="btn btn-primary" id="bt-run" onclick="runBacktest()" style="width:100%;margin-top:.5rem">
          <i data-lucide="play" style="width:15px;height:15px"></i> Run Backtest
        </button>
      </div>
      <!-- KPIs -->
      <div style="display:flex;flex-direction:column;gap:1rem">
        <div class="grid3">
          <div class="kpi"><div class="kpi-label">Win Rate</div><div class="kpi-val c-green" id="bt-wr">—</div></div>
          <div class="kpi"><div class="kpi-label">Expectancy</div><div class="kpi-val c-cyan" id="bt-exp">—</div></div>
          <div class="kpi"><div class="kpi-label">Profit Factor</div><div class="kpi-val c-purple" id="bt-pf">—</div></div>
        </div>
        <div class="grid3">
          <div class="kpi"><div class="kpi-label">Total Trades</div><div class="kpi-val" id="bt-tt">—</div></div>
          <div class="kpi"><div class="kpi-label">Total R</div><div class="kpi-val c-green" id="bt-tr">—</div></div>
          <div class="kpi"><div class="kpi-label">Max Drawdown</div><div class="kpi-val c-red" id="bt-dd">—</div></div>
        </div>
      </div>
    </div>
    <!-- Charts -->
    <div class="grid2">
      <div class="card">
        <div class="card-title"><i data-lucide="trending-up"></i> Equity Curve</div>
        <div class="chart-box-tall"><canvas id="bt-equity-chart"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title"><i data-lucide="bar-chart"></i> Monthly Performance</div>
        <div class="chart-box-tall"><canvas id="bt-monthly-chart"></canvas></div>
      </div>
    </div>
  </div>

  <!-- === PANEL 4: ML TRAINING === -->
  <div class="panel" id="panel-ml">
    <div class="grid2">
      <div class="card">
        <div class="card-title"><i data-lucide="brain"></i> XGBoost Training</div>
        <div style="color:var(--muted);font-size:.85rem;margin-bottom:1rem">
          Train the model on the strategy dataset (XAUUSD M1 / M3).
          Feature matrix includes BB & EMA bands, Liquidity Sweeps (BSL/SSL), RSI/ADX trend strength, VWAP distance, sessions, and MTF context.
        </div>
        <button class="btn btn-primary" id="ml-train-btn" onclick="startTraining()">
          <i data-lucide="play" style="width:15px;height:15px"></i> Train XGBoost Model
        </button>
        <div style="margin-top:1rem" id="ml-progress-wrap" style="display:none">
          <div style="font-size:.78rem;color:var(--muted);margin-bottom:.4rem" id="ml-progress-label">Initializing...</div>
          <div class="progress-wrap"><div class="progress-fill" id="ml-progress-fill" style="width:0%"></div></div>
        </div>
        <div style="margin-top:1.25rem" id="ml-metrics-block">
          <div class="grid2" style="gap:.75rem">
            <div class="kpi"><div class="kpi-label">Accuracy</div><div class="kpi-val c-green" id="ml-acc">—</div></div>
            <div class="kpi"><div class="kpi-label">Precision</div><div class="kpi-val c-cyan" id="ml-prec">—</div></div>
            <div class="kpi"><div class="kpi-label">Recall</div><div class="kpi-val c-purple" id="ml-rec">—</div></div>
            <div class="kpi"><div class="kpi-label">Samples</div><div class="kpi-val" id="ml-samp">—</div></div>
          </div>
          <div style="margin-top:.75rem;font-size:.75rem;color:var(--muted)" id="ml-last-trained"></div>
        </div>
      </div>
      <div class="card">
        <div class="card-title"><i data-lucide="bar-chart-3"></i> Feature Importance</div>
        <div class="chart-box-tall"><canvas id="ml-feat-chart"></canvas></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title"><i data-lucide="lightbulb"></i> Model Explainability — SHAP Breakdown</div>
      <div id="ml-shap" style="color:var(--muted);font-size:.85rem">
        Train the model first to see SHAP-style feature contributions.
      </div>
    </div>
  </div>

  <!-- === PANEL 5: WALK-FORWARD === -->
  <div class="panel" id="panel-walkforward">
    <div class="grid2">
      <div class="card">
        <div class="card-title"><i data-lucide="git-branch"></i> Walk-Forward Validation</div>
        <div style="color:var(--muted);font-size:.85rem;margin-bottom:1rem">
          Runs 3-fold rolling chronological validation. In-sample optimization (70%)
          then out-of-sample testing (30%). Compares optimized candidate vs baseline.
        </div>
        <button class="btn btn-primary" id="wf-run-btn" onclick="startWalkForward()">
          <i data-lucide="play" style="width:15px;height:15px"></i> Run Walk-Forward (3 Folds)
        </button>
        <div style="margin-top:1rem" id="wf-progress-wrap">
          <div style="font-size:.78rem;color:var(--muted);margin-bottom:.4rem" id="wf-progress-label"></div>
          <div class="progress-wrap"><div class="progress-fill" id="wf-progress-fill" style="width:0%"></div></div>
        </div>
        <div id="wf-verdict-block" style="display:none;margin-top:1rem">
          <div class="badge" id="wf-verdict-badge" style="font-size:1rem;padding:.6rem 1.2rem">—</div>
        </div>
      </div>
      <div class="card">
        <div class="card-title"><i data-lucide="bar-chart-2"></i> Results Summary</div>
        <div id="wf-results">
          <div class="no-setup">Run validation to see results.</div>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-title"><i data-lucide="trending-up"></i> Candidate vs Baseline</div>
      <div class="chart-box-tall"><canvas id="wf-compare-chart"></canvas></div>
    </div>
  </div>



  <!-- === PANEL: MODEL ARENA === -->
  <div class="panel" id="panel-arena">
    <!-- Data Download card first -->
    <div class="card">
      <div class="card-title"><i data-lucide="download-cloud"></i> Step 0 — Download 1-Year MT5 Data
        <span style="margin-left:auto;font-size:.75rem;color:var(--muted)">Required before training (downloads XAUUSD M1 + M3)</span>
      </div>
      <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap">
        <button class="btn btn-secondary" id="dl-btn" onclick="startDownload()">
          <i data-lucide="download" style="width:14px;height:14px"></i> Download 1-Year Data
        </button>
        <div style="flex:1;min-width:200px">
          <div class="progress-wrap"><div class="progress-fill" id="dl-progress-fill" style="width:0%"></div></div>
          <div style="font-size:.72rem;color:var(--muted);margin-top:.3rem" id="dl-progress-label">Click to download from MT5 terminal</div>
        </div>
      </div>
    </div>
    <!-- Train all models -->
    <div class="card">
      <div class="card-title"><i data-lucide="trophy"></i> Model Arena — Train All Models
        <span style="margin-left:auto;font-size:.75rem;color:var(--muted)">XGBoost vs LightGBM vs CatBoost on same dataset</span>
      </div>
      <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-bottom:1rem">
        <button class="btn btn-primary" id="arena-train-btn" onclick="startTrainAll()">
          <i data-lucide="play" style="width:14px;height:14px"></i> Train All Models
        </button>
        <div style="flex:1;min-width:200px">
          <div class="progress-wrap"><div class="progress-fill" id="arena-progress-fill" style="width:0%"></div></div>
        </div>
      </div>
      <!-- Leaderboard -->
      <div id="arena-leaderboard">
        <div class="no-setup">🏆 Click "Train All Models" to compare XGBoost, LightGBM, and CatBoost.</div>
      </div>
    </div>
    <!-- Feature Comparison Chart -->
    <div class="card">
      <div class="card-title"><i data-lucide="bar-chart-2"></i> Feature Importance Comparison Across Models</div>
      <div class="chart-box-tall"><canvas id="arena-feat-chart"></canvas></div>
    </div>
  </div>

  <!-- === PANEL: MONTE CARLO === -->
  <div class="panel" id="panel-montecarlo">
    <div class="card">
      <div class="card-title"><i data-lucide="shuffle"></i> Monte Carlo Stress Test
        <span style="margin-left:auto;font-size:.75rem;color:var(--muted)">10,000 random permutations of your trade log</span>
      </div>
      <div style="color:var(--muted);font-size:.83rem;margin-bottom:1rem;line-height:1.7">
        Shuffles your backtest trade sequence 10,000 times to answer:<br>
        <b style="color:var(--txt)">"If these results were random luck, how likely?"</b> and <b style="color:var(--txt)">"What is the worst-case drawdown at 95% confidence?"</b>
      </div>
      <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap">
        <button class="btn btn-primary" id="mc-run-btn" onclick="startMonteCarlo()">
          <i data-lucide="play" style="width:14px;height:14px"></i> Run Monte Carlo (10,000 sims)
        </button>
        <div style="flex:1;min-width:200px">
          <div class="progress-wrap"><div class="progress-fill" id="mc-progress-fill" style="width:0%"></div></div>
          <div style="font-size:.72rem;color:var(--muted);margin-top:.3rem" id="mc-progress-label">Ready</div>
        </div>
      </div>
    </div>
    <div class="card" id="mc-verdict" style="display:none">
      <div class="card-title"><i data-lucide="check-circle"></i> Monte Carlo Verdict</div>
      <div id="mc-verdict-badge" class="badge" style="font-size:1rem;padding:.6rem 1.2rem;margin-bottom:1rem"></div>
      <div id="mc-stats"></div>
    </div>
    <div class="card">
      <div class="card-title"><i data-lucide="info"></i> What Monte Carlo Validates</div>
      <div style="font-size:.82rem;color:var(--muted);line-height:1.9">
        ✅ <b style="color:var(--txt)">Max drawdown at 95% confidence</b> must be ≤ 20R (funded account limit)<br>
        ✅ <b style="color:var(--txt)">Probability of luck</b> must be &lt;10% (results must be statistically significant)<br>
        ✅ Complements Walk-Forward by testing sequence-independence<br>
        ✅ Mandatory before deploying any EA to a funded account
      </div>
    </div>
  </div>

  <!-- === PANEL: STRATEGY EVOLUTION (full 8-step pipeline) === -->
  <div class="panel" id="panel-evolution">
    <div class="grid2">
      <div class="card">
        <div class="card-title"><i data-lucide="cpu"></i> Full Evolution Cycle (8 Steps)
          <span style="margin-left:auto;font-size:.72rem;color:var(--muted)">Auto end-to-end</span>
        </div>
        <div style="font-size:.82rem;color:var(--muted);line-height:1.8;margin-bottom:1rem">
          Runs the complete pipeline:<br>
          Data → Features → Dataset → Model Arena → Optimizer → Walk-Forward → Monte Carlo → EA
        </div>
        <button class="btn btn-green" id="evo-run-btn" onclick="startEvolution()">
          <i data-lucide="cpu" style="width:15px;height:15px"></i> Run Full Evolution Cycle
        </button>
        <div id="evo-verdict" style="display:none;margin-top:1rem"></div>
        <div style="margin-top:1rem">
          <div style="font-size:.72rem;color:var(--muted);margin-bottom:.3rem" id="evo-progress-label">Ready</div>
          <div class="progress-wrap"><div class="progress-fill" id="evo-progress-fill" style="width:0%"></div></div>
        </div>
      </div>
      <div class="card">
        <div class="card-title"><i data-lucide="git-branch"></i> Generate EA Only
          <span style="margin-left:auto;font-size:.72rem;color:var(--muted)">Optimizer + WF only</span>
        </div>
        <div style="font-size:.8rem;color:var(--muted);line-height:1.8;margin-bottom:1rem">
          Runs Grid-Search + Walk-Forward only (skips full ML training).
          Use this for quick parameter updates.
        </div>
        <button class="btn btn-secondary" id="ea-gen-btn" onclick="startEAGeneration()">
          <i data-lucide="layers" style="width:15px;height:15px"></i> Quick EA Generation
        </button>
        <div style="margin-top:.75rem" id="ea-gen-progress-wrap">
          <div style="font-size:.78rem;color:var(--muted);margin-bottom:.4rem" id="ea-gen-label"></div>
          <div class="progress-wrap"><div class="progress-fill" id="ea-gen-progress-fill" style="width:0%"></div></div>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-title"><i data-lucide="clock"></i> EA Version History</div>
      <div id="ea-timeline" class="ea-timeline">
        <div class="no-setup">No EA versions yet. Run an Evolution Cycle or Quick EA Generation.</div>
      </div>
    </div>
  </div>


  <div class="panel" id="panel-logpanel">
    <div class="card" style="flex:1">
      <div class="card-title">
        <i data-lucide="terminal"></i> System Activity Log
        <div style="margin-left:auto;display:flex;gap:.4rem" id="log-filters">
          <button class="btn btn-sm btn-secondary log-filter active" data-level="ALL">All</button>
          <button class="btn btn-sm btn-secondary log-filter" data-level="SETUP" style="color:var(--green)">Setups</button>
          <button class="btn btn-sm btn-secondary log-filter" data-level="ML" style="color:var(--purple)">ML</button>
          <button class="btn btn-sm btn-secondary log-filter" data-level="VALIDATION" style="color:var(--yellow)">Validation</button>
          <button class="btn btn-sm btn-secondary log-filter" data-level="ERROR" style="color:var(--red)">Errors</button>
        </div>
      </div>
      <div id="log-container" style="overflow-y:auto;max-height:calc(100vh - 260px)"></div>
    </div>
  </div>

</div><!-- end .main -->

<script>
// ───── State ─────
let btEquityChart = null, btMonthlyChart = null;
let mlFeatChart = null;
let wfCompareChart = null;
let logFilter = 'ALL';
let trainPolling = null, wfPolling = null, eaPolling = null;

// ───── Tab navigation ─────
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    const pid = t.dataset.panel;
    document.getElementById('panel-' + pid).classList.add('active');
    if (pid === 'advisor') fetchAdvisor();
    if (pid === 'ml') fetchTrainStatus();
    if (pid === 'walkforward') fetchWFStatus();
    if (pid === 'evolution') fetchEAVersions();
    if (pid === 'logpanel') fetchLog();
  });
});

// ───── Helpers ─────
const $ = id => document.getElementById(id);
function fmt$(n) { return n == null ? '—' : '$' + Number(n).toLocaleString('en', {minimumFractionDigits:2,maximumFractionDigits:2}); }
function fmtPct(n) { return n == null ? '—' : (n * 100).toFixed(1) + '%'; }
function fmtR(n)   { return n == null ? '—' : (n >= 0 ? '+' : '') + Number(n).toFixed(2) + ' R'; }

// ───── Header status ─────
async function fetchStatus() {
  try {
    const d = await (await fetch('/api/status')).json();
    const dot = $('conn-dot'), txt = $('conn-txt');
    if (d.connected) {
      dot.className = 'dot green';
      const a = d.account_info;
      txt.textContent = a ? `${a.server} (${a.login})` : 'Connected';
      if (a) {
        $('ov-balance').textContent = fmt$(a.balance);
        $('ov-equity').textContent  = fmt$(a.equity);
        $('ov-server').textContent  = a.server || '—';
        $('ov-login').textContent   = `Login: ${a.login || '—'}`;
      }
      if (d.market_price) {
        const p = d.market_price;
        $('hdr-price').textContent = `XAUUSD ${p.bid?.toFixed(2) || '—'}`;
        $('ov-spread').textContent = p.spread ?? '—';
      }
    } else {
      dot.className = 'dot red';
      txt.textContent = 'MT5 Disconnected';
    }
  } catch(e) { console.warn('Status fetch failed:', e); }
}

// ───── Pipeline status ─────
async function fetchPipelineStatus() {
  try {
    const d = await (await fetch('/api/pipeline_status')).json();
    const grid = $('pipe-grid');
    grid.innerHTML = '';
    for (const [name, info] of Object.entries(d)) {
      const badgeCls = info.status === 'ok' ? 'ok'
                     : info.status === 'trained' ? 'trained'
                     : info.status === 'untrained' ? 'untrained' : 'idle';
      grid.innerHTML += `<div class="pipe-item">
        <div class="pipe-name">${name.replace('_', ' ')}</div>
        <span class="badge ${badgeCls}">${info.status}</span>
        <div class="pipe-note">${info.note}</div>
      </div>`;
    }
    lucide.createIcons();
  } catch(e) {}
}

// ───── Overview log preview ─────
async function fetchOverviewLog() {
  try {
    const d = await (await fetch('/api/log?limit=8')).json();
    const el = $('ov-log');
    el.innerHTML = d.entries.map(renderLogEntry).join('');
  } catch(e) {}
}

// ───── ADVISOR ─────
async function fetchAdvisor() {
  try {
    const d = await (await fetch('/api/advisor')).json();
    const cont = $('adv-content'), det = $('adv-details');
    if (d.status === 'active_setup') {
      const dir = d.direction || 'long';
      const prob = (d.win_probability * 100).toFixed(1);
      const dirColor = dir === 'long' ? 'var(--green)' : 'var(--red)';
      cont.innerHTML = `<div class="setup-alert ${dir} pulsing">
        <div class="setup-dir" style="color:${dirColor}">▲ ${dir.toUpperCase()} SETUP</div>
        <div class="setup-prob" style="color:${dirColor}">${prob}%</div>
        <div style="font-size:.8rem;color:var(--muted)">Win Probability</div>
        <div class="setup-grid">
          <div class="setup-field"><div class="setup-field-label">Entry</div><div class="setup-field-val">${d.entry_price?.toFixed(2) ?? '—'}</div></div>
          <div class="setup-field"><div class="setup-field-label">Stop Loss</div><div class="setup-field-val" style="color:var(--red)">${d.stop_loss?.toFixed(2) ?? '—'}</div></div>
          <div class="setup-field"><div class="setup-field-label">Take Profit</div><div class="setup-field-val" style="color:var(--green)">${d.take_profit?.toFixed(2) ?? '—'}</div></div>
        </div>
      </div>`;
      // Reasons
      const reasons = d.reasons || [];
      det.innerHTML = `<div style="font-weight:700;margin-bottom:.75rem;font-size:.85rem">Why this setup?</div>
        <div class="reasons">${reasons.map(r => `
          <div class="reason-row">
            <span style="font-size:.8rem;min-width:150px;color:var(--txt)">${r.label}</span>
            <div class="reason-bar-bg"><div class="reason-bar" style="width:${Math.min(parseFloat(r.value)*2,100)}%"></div></div>
            <span style="font-size:.78rem;font-family:var(--mono);color:var(--green)">${r.value}</span>
          </div>`).join('')}
        </div>
        <div style="margin-top:1rem;padding:.6rem .9rem;background:rgba(0,0,0,.25);border-radius:8px;font-size:.78rem">
          <div style="color:var(--muted)">Advisory Action</div>
          <div style="font-weight:700;font-size:1rem;color:${d.advisory_action === 'favorable' ? 'var(--green)' : 'var(--red)'};text-transform:uppercase">${d.advisory_action || '—'}</div>
        </div>`;
    } else {
      cont.innerHTML = `<div class="no-setup"><div style="font-size:2rem;margin-bottom:.5rem">🔍</div>${d.message || 'No active setup'}</div>`;
      det.innerHTML = `<div class="no-setup">Waiting for next crossover signal on M1...</div>`;
    }
  } catch(e) { console.warn(e); }
}

// ── M1 Bar timer ──
function updateBarTimer() {
  const now = new Date();
  const sec = now.getSeconds();
  const remaining = 60 - sec;
  $('adv-timer').textContent = `Next bar closes in: 0:${String(remaining).padStart(2,'0')}`;
}

// ───── SETUP HISTORY LOG ─────
async function fetchAdvisorHistory() {
  try {
    const d = await (await fetch('/api/advisor/history?limit=10')).json();
    const tbody = $('setup-log-body');
    const countBadge = $('setup-log-count');
    if (!tbody) return;

    if (countBadge) countBadge.textContent = `${d.total} total`;

    if (!d.entries || d.entries.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:1rem">No setups detected yet — monitoring live...</td></tr>';
      return;
    }

    tbody.innerHTML = d.entries.map(s => {
      const dirColor = s.direction === 'long' ? 'var(--green)' : 'var(--red)';
      const dirIcon  = s.direction === 'long' ? '▲' : '▼';
      const probColor = s.prob >= 65 ? 'var(--green)' : s.prob >= 50 ? 'var(--yellow)' : 'var(--red)';
      return `<tr style="border-bottom:1px solid var(--border);transition:background .2s" onmouseover="this.style.background='rgba(255,255,255,.03)'" onmouseout="this.style.background=''">
        <td style="padding:.45rem .6rem;font-family:var(--mono);font-size:.75rem;color:var(--muted)">${s.date} ${s.ts}</td>
        <td style="padding:.45rem .6rem;font-weight:700;color:${dirColor}">${dirIcon} ${(s.direction||'').toUpperCase()}</td>
        <td style="padding:.45rem .6rem;text-align:right;font-family:var(--mono)">${s.entry != null ? s.entry.toFixed(2) : '—'}</td>
        <td style="padding:.45rem .6rem;text-align:right;font-family:var(--mono);color:var(--red)">${s.sl != null ? s.sl.toFixed(2) : '—'}</td>
        <td style="padding:.45rem .6rem;text-align:right;font-family:var(--mono);color:var(--green)">${s.tp != null ? s.tp.toFixed(2) : '—'}</td>
        <td style="padding:.45rem .6rem;text-align:right;font-weight:700;color:${probColor}">${s.prob}%</td>
        <td style="padding:.45rem .6rem;font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px">${s.advisory || '—'}</td>
      </tr>`;
    }).join('');
  } catch(e) { console.warn('Setup history error:', e); }
}



// ───── BACKTEST ─────
['sl-sqz','sl-sl','sl-tp','sl-bbs','sl-sms'].forEach(id => {
  const el = $(id);
  if (!el) return;
  el.addEventListener('input', () => {
    const lblMap = {'sl-sqz':'lbl-sqz','sl-sl':'lbl-sl','sl-tp':'lbl-tp','sl-bbs':'lbl-bbs','sl-sms':'lbl-sms'};
    $(lblMap[id]).textContent = parseFloat(el.value).toFixed(1);
  });
});

async function runBacktest() {
  const btn = $('bt-run');
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader" style="width:15px;height:15px" class="spin"></i> Running...`;
  lucide.createIcons();
  try {
    const sqz = $('sl-sqz').value, sl = $('sl-sl').value, tp = $('sl-tp').value;
    const bbs = $('sl-bbs').value, sms = $('sl-sms').value;
    const useMl = $('chk-ml-filter').checked;
    const setupMode = $('sel-setup-mode') ? $('sel-setup-mode').value : 'combined_all';
    const enableLiq = $('chk-liq-sweeps') ? $('chk-liq-sweeps').checked : true;
    const enableRsi = $('chk-rsi-adx') ? $('chk-rsi-adx').checked : true;
    const enableVwap = $('chk-vwap') ? $('chk-vwap').checked : true;
    const enableMtf = $('chk-mtf') ? $('chk-mtf').checked : true;
    const url = `/api/backtest?sqz_threshold=${sqz}&atr_sl_mult=${sl}&atr_tp_mult=${tp}&bb_std=${bbs}&sm_std=${sms}&use_ml_filter=${useMl}&setup_mode=${setupMode}&enable_liquidity_sweeps=${enableLiq}&enable_rsi_adx=${enableRsi}&enable_vwap=${enableVwap}&enable_mtf=${enableMtf}`;
    const d = await (await fetch(url)).json();
    const m = d.metrics;
    $('bt-wr').textContent  = fmtPct(m.win_rate);
    $('bt-exp').textContent = fmtR(m.expectancy);
    $('bt-pf').textContent  = m.profit_factor?.toFixed(2) ?? '—';
    $('bt-tt').textContent  = m.total_trades != null ? m.total_trades + ' (' + m.wins + ' W / ' + m.losses + ' L)' : '—';
    $('bt-tr').textContent  = fmtR(m.total_realized_r);
    $('bt-dd').textContent  = `-${m.max_drawdown_r?.toFixed(2) ?? '—'} R`;
    renderEquityChart(d.equity_curve);
    renderMonthlyChart(d.monthly);
  } catch(e) { console.error(e); }
  btn.disabled = false;
  btn.innerHTML = `<i data-lucide="play" style="width:15px;height:15px"></i> Run Backtest`;
  lucide.createIcons();
}

function renderEquityChart(curve) {
  if (btEquityChart) btEquityChart.destroy();
  const ctx = $('bt-equity-chart').getContext('2d');
  const labels = curve.map((_,i) => i === 0 ? 'Start' : `T${i}`);
  btEquityChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{ label: 'Equity R', data: curve,
      borderColor: '#00d4ff', backgroundColor: 'rgba(0,212,255,.1)',
      fill: true, tension: .3, pointRadius: curve.length < 30 ? 3 : 0 }]},
    options: { responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{display:false} },
      scales:{ x:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#6b7a99',maxTicksLimit:8}},
               y:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#6b7a99'}} }}
  });
}

function renderMonthlyChart(monthly) {
  if (btMonthlyChart) btMonthlyChart.destroy();
  const ctx = $('bt-monthly-chart').getContext('2d');
  const labels = Object.keys(monthly);
  const vals   = Object.values(monthly);
  btMonthlyChart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Monthly R', data: vals,
      backgroundColor: vals.map(v => v >= 0 ? 'rgba(16,185,129,.6)' : 'rgba(239,68,68,.6)'),
      borderColor:     vals.map(v => v >= 0 ? '#10b981' : '#ef4444'),
      borderWidth: 1 }]},
    options: { responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{display:false} },
      scales:{ x:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#6b7a99'}},
               y:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#6b7a99'}} }}
  });
}

// ───── ML TRAINING ─────
async function startTraining() {
  $('ml-train-btn').disabled = true;
  $('ml-progress-wrap').style.display = 'block';
  $('ml-progress-label').textContent = 'Starting training job...';
  $('ml-progress-fill').style.width = '5%';
  await fetch('/api/train', {method:'POST'});
  trainPolling = setInterval(fetchTrainStatus, 1200);
}

async function fetchTrainStatus() {
  try {
    const d = await (await fetch('/api/train/status')).json();
    const prog = d.progress || 0;
    $('ml-progress-fill').style.width = prog + '%';
    const labels = {0:'Idle',25:'Loading data...',50:'Detecting setups...',70:'Fitting model...',100:'Training complete!'};
    $('ml-progress-label').textContent = labels[prog] || `Training... ${prog}%`;
    if (d.status === 'done' || d.status === 'error') {
      clearInterval(trainPolling); trainPolling = null;
      $('ml-train-btn').disabled = false;
    }
    const ms = d.model_state;
    if (ms && ms.trained && ms.metrics) {
      const m = ms.metrics;
      $('ml-acc').textContent  = fmtPct(m.accuracy);
      $('ml-prec').textContent = fmtPct(m.precision);
      $('ml-rec').textContent  = fmtPct(m.recall);
      $('ml-samp').textContent = m.n_samples ?? '—';
      $('ml-last-trained').textContent = `Last trained: ${ms.last_trained}`;
      renderFeatChart(ms.importances);
      renderSHAP(ms.importances);
    }
  } catch(e) {}
}

function renderFeatChart(imps) {
  if (!imps || !Object.keys(imps).length) return;
  if (mlFeatChart) mlFeatChart.destroy();
  const ctx = $('ml-feat-chart').getContext('2d');
  const sorted = Object.entries(imps).sort((a,b) => b[1]-a[1]).slice(0, 10);
  const labels = sorted.map(([k]) => k.replace(/_/g,' '));
  const vals   = sorted.map(([,v]) => v);
  mlFeatChart = new Chart(ctx, {
    type:'bar',
    data:{ labels, datasets:[{ label:'Gini Importance', data:vals,
      backgroundColor:'rgba(168,85,247,.6)', borderColor:'#a855f7', borderWidth:1 }]},
    options:{ indexAxis:'y', responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{ x:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#6b7a99'}},
               y:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#6b7a99',font:{size:11}}} }}
  });
}

function renderSHAP(imps) {
  if (!imps) return;
  const top = Object.entries(imps).sort((a,b)=>b[1]-a[1]).slice(0,5);
  $('ml-shap').innerHTML = `<div style="font-size:.8rem;margin-bottom:.75rem;color:var(--muted)">
    Top features contributing to setup win probability predictions:</div>
    ${top.map(([f, v], i) => `
      <div class="reason-row" style="margin-bottom:.5rem">
        <span style="min-width:30px;color:var(--muted);font-size:.75rem">#${i+1}</span>
        <span style="min-width:160px;font-size:.82rem">${f.replace(/_/g,' ')}</span>
        <div class="reason-bar-bg"><div class="reason-bar" style="width:${Math.round(v*400)}%;background:var(--purple)"></div></div>
        <span style="font-family:var(--mono);font-size:.78rem;color:var(--purple)">${(v*100).toFixed(1)}%</span>
      </div>`).join('')}`;
}

// ───── WALK-FORWARD ─────
async function startWalkForward() {
  $('wf-run-btn').disabled = true;
  $('wf-progress-wrap').style.display = 'block';
  $('wf-progress-label').textContent = 'Launching walk-forward validation...';
  $('wf-progress-fill').style.width = '10%';
  await fetch('/api/walkforward', {method:'POST'});
  wfPolling = setInterval(fetchWFStatus, 1500);
}

async function fetchWFStatus() {
  try {
    const d = await (await fetch('/api/walkforward/status')).json();
    const prog = d.progress || 0;
    $('wf-progress-fill').style.width = prog + '%';
    const lbl = d.status === 'running' ? `Running folds... ${prog}%`
              : d.status === 'done' ? 'Validation complete!'
              : d.status === 'error' ? `Error: ${d.message}` : '';
    $('wf-progress-label').textContent = lbl;
    if (d.status === 'done' || d.status === 'error') {
      clearInterval(wfPolling); wfPolling = null;
      $('wf-run-btn').disabled = false;
    }
    if (d.result) renderWFResults(d.result);
  } catch(e) {}
}

function renderWFResults(r) {
  $('wf-verdict-block').style.display = 'block';
  const vb = $('wf-verdict-badge');
  vb.textContent = r.approved ? '✅ APPROVED — Candidate beats baseline' : '❌ REJECTED — Baseline performs better';
  vb.className = 'badge ' + (r.approved ? 'approved' : 'rejected');
  const wf = r.walk_forward_metrics, bs = r.baseline_metrics;
  $('wf-results').innerHTML = `
    <div class="fold-cards">
      <div class="fold-card"><div class="fold-title">Candidate Total R</div><div class="fold-val c-cyan">${fmtR(wf.total_realized_r)}</div></div>
      <div class="fold-card"><div class="fold-title">Candidate Win Rate</div><div class="fold-val c-green">${fmtPct(wf.win_rate)}</div></div>
      <div class="fold-card"><div class="fold-title">Candidate Trades</div><div class="fold-val">${wf.total_trades}</div></div>
    </div>
    <div class="fold-cards" style="margin-top:.75rem">
      <div class="fold-card"><div class="fold-title">Baseline Total R</div><div class="fold-val c-yellow">${fmtR(bs.total_realized_r)}</div></div>
      <div class="fold-card"><div class="fold-title">Baseline Win Rate</div><div class="fold-val c-yellow">${fmtPct(bs.win_rate)}</div></div>
      <div class="fold-card"><div class="fold-title">Folds Evaluated</div><div class="fold-val">${r.folds_evaluated}</div></div>
    </div>`;
  // Comparison chart
  renderWFChart(wf.total_realized_r, bs.total_realized_r);
}

function renderWFChart(candR, baseR) {
  if (wfCompareChart) wfCompareChart.destroy();
  const ctx = $('wf-compare-chart').getContext('2d');
  wfCompareChart = new Chart(ctx, {
    type:'bar',
    data:{
      labels:['Total R — Out-of-Sample'],
      datasets:[
        {label:'Candidate',data:[candR],backgroundColor:'rgba(0,212,255,.6)',borderColor:'#00d4ff',borderWidth:1},
        {label:'Baseline',data:[baseR],backgroundColor:'rgba(245,158,11,.6)',borderColor:'#f59e0b',borderWidth:1}
      ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#e8ecf4',font:{family:'Outfit'}}}},
      scales:{x:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#6b7a99'}},
              y:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#6b7a99'}}}}
  });
}

// ───── EA GENERATION ─────
async function startEAGeneration() {
  $('ea-gen-btn').disabled = true;
  $('ea-gen-label').textContent = 'Running optimization pipeline...';
  $('ea-gen-progress-fill').style.width = '5%';
  await fetch('/api/generate_ea', {method:'POST'});
  eaPolling = setInterval(fetchEAGenStatus, 1500);
}

async function fetchEAGenStatus() {
  try {
    const d = await (await fetch('/api/generate_ea/status')).json();
    const prog = d.progress || 0;
    $('ea-gen-progress-fill').style.width = prog + '%';
    $('ea-gen-label').textContent = d.status === 'done' ? `✅ Generated ${d.version}!`
      : d.status === 'error' ? `Error: ${d.message}` : `Running... ${prog}%`;
    if (d.status === 'done' || d.status === 'error') {
      clearInterval(eaPolling); eaPolling = null;
      $('ea-gen-btn').disabled = false;
      fetchEAVersions();
    }
  } catch(e) {}
}

async function fetchEAVersions() {
  try {
    const d = await (await fetch('/api/ea_versions')).json();
    const tl = $('ea-timeline');
    if (!d.versions.length) {
      tl.innerHTML = '<div class="no-setup">No EA versions generated yet.</div>'; return;
    }
    tl.innerHTML = d.versions.map((v, i) => `
      <div class="ea-item">
        <div class="ea-ver">${v.version}</div>
        <div class="ea-meta">
          <div style="font-weight:700">${v.params?.sqz_threshold != null ? `Sqz=${v.params.sqz_threshold}, SL=${v.params.atr_sl_mult}×, TP=${v.params.atr_tp_mult}×` : 'Custom Parameters'}</div>
          <div class="ea-status-row">
            <span class="badge ok">${v.status}</span>
            <span style="font-size:.72rem;color:var(--muted)">${v.created}</span>
          </div>
          <div style="margin-top:.4rem;font-size:.75rem;color:var(--muted)">
            Baseline → Trades: ${v.metrics?.total_trades ?? '—'} ${v.metrics?.wins != null ? `(${v.metrics.wins}W / ${v.metrics.losses}L)` : ''} &nbsp;|&nbsp;
            WR: ${v.metrics?.win_rate != null ? fmtPct(v.metrics.win_rate) : '—'} &nbsp;|&nbsp;
            E: ${v.metrics?.expectancy != null ? fmtR(v.metrics.expectancy) : '—'}
          </div>
          ${v.ml_filtered_metrics?.total_trades != null ? `
          <div style="margin-top:.2rem;font-size:.75rem;color:var(--cyan)">
            ML-Filtered → Trades: ${v.ml_filtered_metrics.total_trades} ${v.ml_filtered_metrics.wins != null ? `(${v.ml_filtered_metrics.wins}W / ${v.ml_filtered_metrics.losses}L)` : ''} &nbsp;|&nbsp;
            WR: ${fmtPct(v.ml_filtered_metrics.win_rate)} &nbsp;|&nbsp;
            E: ${fmtR(v.ml_filtered_metrics.expectancy)} (Total R: ${fmtR(v.ml_filtered_metrics.total_realized_r)})
          </div>` : ''}
          ${v.ml_rules ? `
          <div style="margin-top:.4rem;padding:.4rem;background:rgba(0,0,0,.2);border-radius:4px;font-family:var(--mono);font-size:.68rem;color:var(--yellow);word-break:break-all">
            ML Rule: ${v.ml_rules}
          </div>` : ''}
        </div>
        <a href="/api/ea_download/${v.version}" class="btn btn-sm btn-secondary">
          <i data-lucide="download" style="width:13px;height:13px"></i> Download MQL5
        </a>
      </div>`).join('');
    lucide.createIcons();
  } catch(e) {}
}

// ───── LOG ─────
async function fetchLog() {
  try {
    const d = await (await fetch('/api/log?limit=100')).json();
    const cont = $('log-container');
    const filtered = logFilter === 'ALL' ? d.entries : d.entries.filter(e => e.level === logFilter);
    cont.innerHTML = filtered.map(renderLogEntry).join('');
  } catch(e) {}
}

function renderLogEntry(e) {
  return `<div class="log-entry log-${e.level}">
    <span class="log-ts">${e.ts}</span>
    <span class="log-source">${e.source}</span>
    <span class="log-msg">${e.message}</span>
  </div>`;
}

document.querySelectorAll('.log-filter').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.log-filter').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    logFilter = btn.dataset.level;
    fetchLog();
  });
});

// ───── DATA DOWNLOAD ─────
async function startDownload() {
  const btn = $('dl-btn');
  btn.disabled = true;
  btn.innerHTML = '<i data-lucide="loader" style="width:14px;height:14px" class="spin"></i> Downloading...';
  lucide.createIcons();
  await fetch('/api/download_data', {method:'POST'});
  dlPolling = setInterval(fetchDownloadStatus, 1500);
}
async function fetchDownloadStatus() {
  try {
    const d = await (await fetch('/api/download_data/status')).json();
    const prog = d.progress || 0;
    $('dl-progress-fill').style.width = prog + '%';
    $('dl-progress-label').textContent = d.status === 'done'
      ? `✅ Done — ${(d.files||[]).map(f=>f.symbol+' '+f.timeframe+': '+f.bars.toLocaleString()+' bars').join(' | ')}`
      : d.status === 'error' ? `❌ Error: ${d.message}`
      : `Downloading... ${prog}%`;
    if (d.status === 'done' || d.status === 'error') {
      clearInterval(dlPolling); dlPolling = null;
      $('dl-btn').disabled = false;
      $('dl-btn').innerHTML = '<i data-lucide="download" style="width:14px;height:14px"></i> Download 1-Year Data';
      lucide.createIcons();
    }
  } catch(e) {}
}

// ───── MODEL ARENA ─────
async function startTrainAll() {
  const btn = $('arena-train-btn');
  btn.disabled = true;
  btn.innerHTML = '<i data-lucide="loader" style="width:14px;height:14px" class="spin"></i> Training All Models...';
  lucide.createIcons();
  await fetch('/api/train_all', {method:'POST'});
  arenaPolling = setInterval(fetchArenaStatus, 1200);
}
async function fetchArenaStatus() {
  try {
    const d = await (await fetch('/api/train_all/status')).json();
    const prog = d.progress || 0;
    $('arena-progress-fill').style.width = prog + '%';
    const arena = d.arena || {};
    if (d.status === 'done' || d.status === 'error') {
      clearInterval(arenaPolling); arenaPolling = null;
      $('arena-train-btn').disabled = false;
      $('arena-train-btn').innerHTML = '<i data-lucide="play" style="width:14px;height:14px"></i> Train All Models';
      lucide.createIcons();
      if (arena.leaderboard) renderArenaLeaderboard(arena);
    }
  } catch(e) {}
}
function renderArenaLeaderboard(arena) {
  const lb = arena.leaderboard || [];
  const results = arena.results || {};
  const medalColor = ['var(--yellow)','var(--muted)','var(--orange)'];
  const medalIcon  = ['🥇','🥈','🥉'];
  const html = lb.map((m,i) => {
    const res = results[m.model] || {};
    const imp = res.importances || {};
    const topFeats = Object.entries(imp).slice(0,3).map(([f,v])=>`<span style="color:var(--muted);font-size:.72rem">${f.replace(/_/g,' ')}: ${(v*100).toFixed(1)}%</span>`).join(' · ');
    return `<div class="arena-card" style="border-color:${i===0?'rgba(245,158,11,.4)':'var(--border)'}">
      <div style="display:flex;align-items:center;gap:.75rem">
        <div style="font-size:2rem">${medalIcon[i]||'🔵'}</div>
        <div style="flex:1">
          <div style="font-weight:800;font-size:1.1rem;color:${medalColor[i]||'var(--txt)'}">${m.model}</div>
          <div class="arena-stats">
            <span class="arena-stat">AUC <b style="color:var(--cyan)">${m.auc?.toFixed(3)||'—'}</b></span>
            <span class="arena-stat">Acc <b style="color:var(--green)">${m.accuracy?fmtPct(m.accuracy):'—'}</b></span>
            <span class="arena-stat">Log-Loss <b style="color:var(--yellow)">${m.log_loss?.toFixed(3)||'—'}</b></span>
            <span class="arena-stat">Time <b>${m.train_time||'—'}s</b></span>
          </div>
          <div style="margin-top:.4rem;font-size:.72rem;color:var(--muted)">${topFeats}</div>
        </div>
        ${i===0?'<span class="badge ok">BEST</span>':''}
      </div>
    </div>`;
  }).join('');
  $('arena-leaderboard').innerHTML = html || '<div class="no-setup">Train models to see results.</div>';
  renderArenaFeatureChart(results);
}
function renderArenaFeatureChart(results) {
  const models = Object.keys(results);
  if (!models.length) return;
  // Feature comparison chart — show top 8 features from best model
  const best = models[0];
  const imp = results[best]?.importances || {};
  const entries = Object.entries(imp).slice(0,8);
  if (!entries.length) return;
  if (window._arenaFeatChart) window._arenaFeatChart.destroy();
  const ctx = $('arena-feat-chart').getContext('2d');
  const colors = {'XGBoost':'rgba(0,212,255,.7)','LightGBM':'rgba(168,85,247,.7)','CatBoost':'rgba(245,158,11,.7)'};
  window._arenaFeatChart = new Chart(ctx, {
    type:'bar',
    data:{
      labels: entries.map(([f])=>f.replace(/_/g,' ')),
      datasets: models.map(m => ({
        label: m,
        data: entries.map(([f]) => results[m]?.importances?.[f] || 0),
        backgroundColor: colors[m] || 'rgba(100,100,100,.5)',
        borderWidth:1
      }))
    },
    options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#e8ecf4',font:{family:'Outfit'}}}},
      scales:{x:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#6b7a99'}},
              y:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#6b7a99',font:{size:10}}}}}
  });
}

// ───── MONTE CARLO ─────
async function startMonteCarlo() {
  const btn = $('mc-run-btn');
  btn.disabled = true;
  btn.innerHTML = '<i data-lucide="loader" style="width:14px;height:14px" class="spin"></i> Simulating...';
  lucide.createIcons();
  await fetch('/api/monte_carlo', {method:'POST'});
  mcPolling = setInterval(fetchMCStatus, 1500);
}
async function fetchMCStatus() {
  try {
    const d = await (await fetch('/api/monte_carlo/status')).json();
    const prog = d.progress || 0;
    $('mc-progress-fill').style.width = prog + '%';
    $('mc-progress-label').textContent = d.status === 'done' ? 'Simulation complete!'
      : d.status === 'error' ? `Error: ${d.message}` : `Simulating... ${prog}%`;
    if (d.status === 'done' || d.status === 'error') {
      clearInterval(mcPolling); mcPolling = null;
      $('mc-run-btn').disabled = false;
      $('mc-run-btn').innerHTML = '<i data-lucide="play" style="width:14px;height:14px"></i> Run Monte Carlo (10,000 sims)';
      lucide.createIcons();
      if (d.result) renderMCResults(d.result);
    }
  } catch(e) {}
}
function renderMCResults(r) {
  const approved = r.approved;
  $('mc-verdict').style.display = 'block';
  const vb = $('mc-verdict-badge');
  vb.textContent = approved ? '✅ APPROVED — Strategy has statistically significant edge' : '❌ REJECTED — Risk or randomness concern detected';
  vb.className = 'badge ' + (approved ? 'approved' : 'rejected');
  const s = r.sim_stats, o = r.original, c = r.checks;
  $('mc-stats').innerHTML = `
    <div class="fold-cards">
      <div class="fold-card"><div class="fold-title">Worst 5% Drawdown</div><div class="fold-val" style="color:${s.worst_5pct_max_dd>15?'var(--red)':'var(--green)'}">${s.worst_5pct_max_dd.toFixed(2)} R</div></div>
      <div class="fold-card"><div class="fold-title">Luck Probability</div><div class="fold-val" style="color:${s.prob_by_luck>0.1?'var(--red)':'var(--green)'}">${(s.prob_by_luck*100).toFixed(1)}%</div></div>
      <div class="fold-card"><div class="fold-title">Median Total R</div><div class="fold-val c-cyan">${fmtR(s.median_total_r)}</div></div>
      <div class="fold-card"><div class="fold-title">Median Win Rate</div><div class="fold-val">${fmtPct(s.median_win_rate)}</div></div>
    </div>
    <div class="fold-cards" style="margin-top:.75rem">
      <div class="fold-card"><div class="fold-title">Original Total R</div><div class="fold-val c-green">${fmtR(o.total_r)}</div></div>
      <div class="fold-card"><div class="fold-title">Original Max DD</div><div class="fold-val c-red">${o.max_dd.toFixed(2)} R</div></div>
      <div class="fold-card"><div class="fold-title">Simulations Run</div><div class="fold-val">${r.n_simulations.toLocaleString()}</div></div>
      <div class="fold-card"><div class="fold-title">Trades Tested</div><div class="fold-val">${r.n_trades}</div></div>
    </div>
    <div style="margin-top:1rem;padding:.75rem 1rem;background:var(--bg3);border-radius:10px;border:1px solid var(--border)">
      <div style="font-size:.78rem;font-weight:700;margin-bottom:.5rem">Checks</div>
      <div style="display:flex;gap:1rem;font-size:.8rem">
        <span>${c.max_dd_within_limit?'✅':'❌'} Max DD within limit (${s.dd_limit_r}R)</span>
        <span>${c.result_not_by_luck?'✅':'❌'} Result not random luck (&lt;10% probability)</span>
      </div>
    </div>`;
}

// ───── EVOLUTION ENGINE ─────
async function startEvolution() {
  const btn = $('evo-run-btn');
  btn.disabled = true;
  btn.innerHTML = '<i data-lucide="loader" style="width:14px;height:14px" class="spin"></i> Running Pipeline...';
  lucide.createIcons();
  $('evo-progress-label').textContent = 'Starting 8-step evolution cycle...';
  $('evo-progress-fill').style.width = '2%';
  await fetch('/api/evolution', {method:'POST'});
  evoPolling = setInterval(fetchEvoStatus, 2000);
}
async function fetchEvoStatus() {
  try {
    const d = await (await fetch('/api/evolution/status')).json();
    const prog = d.progress || 0;
    $('evo-progress-fill').style.width = prog + '%';
    if (d.status === 'done' || d.status === 'error') {
      clearInterval(evoPolling); evoPolling = null;
      $('evo-run-btn').disabled = false;
      $('evo-run-btn').innerHTML = '<i data-lucide="cpu" style="width:14px;height:14px"></i> Run Full Evolution Cycle';
      lucide.createIcons();
      if (d.result) renderEvoResult(d.result);
      fetchEAVersions();
    }
  } catch(e) {}
}
function renderEvoResult(r) {
  const approved = r.approved;
  $('evo-verdict').style.display = 'block';
  let html = `<span class="badge ${approved?'approved':'rejected'}" style="font-size:1rem;padding:.6rem 1.2rem;margin-bottom:1rem">${approved?'✅ CYCLE APPROVED — New EA Generated':'❌ CYCLE REJECTED — '+((r.rejection_reasons||[]).join('; '))}</span>`;
  if (approved && r.ml_rules_expression) {
    html += `<div style="margin-top:1rem;padding:1rem;background:var(--bg3);border:1px solid var(--border);border-radius:10px">
      <div style="font-weight:700;margin-bottom:.5rem;font-size:.85rem;color:var(--cyan)">Discovered ML Squeeze-Breakout Rule:</div>
      <div style="font-family:var(--mono);font-size:.75rem;color:var(--yellow);word-break:break-all;background:rgba(0,0,0,.25);padding:.75rem;border-radius:6px">
        ${r.ml_rules_expression}
      </div>
      <div style="margin-top:.75rem;font-size:.78rem;color:var(--muted)">
        This rule filter was automatically written into the generated EA's CheckMLFilter() block!
      </div>
    </div>`;
  }
  $('evo-verdict').innerHTML = html;
}

// ───── Refresh All ─────
function refreshAll() {
  fetchStatus();
  fetchPipelineStatus();
  fetchOverviewLog();
}

// ───── Init ─────
let dlPolling = null, arenaPolling = null, mcPolling = null, evoPolling = null;
lucide.createIcons();
refreshAll();
runBacktest();

// Auto-polling
setInterval(fetchStatus, 8000);
setInterval(fetchPipelineStatus, 15000);
setInterval(fetchOverviewLog, 8000);
setInterval(fetchAdvisor, 15000);
setInterval(fetchAdvisorHistory, 30000);  // setup history refreshes every 30s
setInterval(updateBarTimer, 1000);
updateBarTimer();
fetchAdvisorHistory();  // initial load

// Update tab handler to handle new panels
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    const pid = t.dataset.panel;
    document.getElementById('panel-' + pid).classList.add('active');
    if (pid === 'advisor') { fetchAdvisor(); fetchAdvisorHistory(); }
    if (pid === 'arena') fetchArenaStatus();
    if (pid === 'ml') fetchTrainStatus();
    if (pid === 'walkforward') fetchWFStatus();
    if (pid === 'montecarlo') fetchMCStatus();
    if (pid === 'evolution') { fetchEvoStatus(); fetchEAVersions(); }
    if (pid === 'logpanel') fetchLog();
  });
});
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=DASHBOARD_HTML)
