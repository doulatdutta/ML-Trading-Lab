# ML Trading Lab — Project Brief

## 1. Project purpose

ML Trading Lab is a personal **research and EA-improvement platform** for MetaTrader 5 (MT5).

Its goal is not to predict every next candle or allow an AI model to trade freely. Its goal is to help improve a strategy you define—such as an EMA smoothing Bollinger Band and Bollinger Band setup—by learning **when the strategy has a stronger statistical edge**.

The system will:

1. Read historical price and indicator data.
2. Detect the setups defined by the trader.
3. Record the market conditions present at every setup.
4. Measure each setup's outcome.
5. Train ML models such as XGBoost to identify useful conditions and filters.
6. Propose bounded changes to the EA.
7. Validate each proposal using out-of-sample and walk-forward testing.
8. Keep an improvement only when it performs better than the approved baseline.

The first live use is advisory/demo only. A funded account must never receive an automatically changed strategy without independent validation and manual approval.

---

## 2. Core principle

> You define the trading idea. Machine learning learns the market conditions in which that idea works best.

For example, the initial strategy may require an EMA condition and Bollinger Band expansion. ML should not replace that idea with an unexplained buy/sell signal. Instead, it may discover a reviewable filter such as:

```text
Take the existing setup only when:
- Bollinger Band width is expanding,
- EMA slope and acceleration are positive,
- volatility is above its recent percentile threshold,
- the trade is in the London session, and
- higher-timeframe trend agrees.
```

That proposed filter must be tested against an unchanged baseline EA before it can be considered.

---

## 3. Initial scope

### Included in the foundation

- Modular Python project structure.
- Configuration template and safe defaults.
- Contracts for market data, features, datasets, strategies, models, optimization, backtesting, walk-forward validation, advisory output, and EA generation.
- Placeholders for XGBoost, LightGBM, CatBoost, and model lifecycle management.
- Test scaffold and Git repository.

### Explicitly not implemented yet

- Trading rules.
- MT5 terminal connection.
- Order placement.
- Broker credentials or account details.
- Model training.
- Backtesting calculations.
- Automatic EA modification or deployment.

---

## 4. Architecture

```text
MT5 history / exported custom-indicator values
                    |
                    v
              DataEngine
                    |
                    v
             FeatureEngine
                    |
                    v
            DatasetBuilder
                    |
                    v
     StrategyEngine + ML experiments
                    |
                    v
      Optimizer proposes candidates
                    |
                    v
 Backtester -> WalkForward validation
                    |
          approved by evidence?
             |              |
            yes             no
             |              |
             v              v
 LiveAdvisor / EA release  reject candidate
```

### Python package modules

| Module | Responsibility |
| --- | --- |
| `DataEngine` | Import bars, ticks, spreads, and later MT5/custom-indicator data. |
| `FeatureEngine` | Create point-in-time-safe features from raw market data. |
| `DatasetBuilder` | Join strategy events with future outcomes to create labeled datasets. |
| `StrategyEngine` | Hold versioned, human-readable strategy definitions and candidate filters. |
| `ML` | Train and compare XGBoost, LightGBM, CatBoost, and baseline models. |
| `Optimizer` | Search only within explicitly approved parameter and filter ranges. |
| `Backtester` | Simulate entry, stop loss, take profit, costs, spread, and slippage. |
| `WalkForward` | Test candidates over sequential future periods not used for fitting. |
| `LiveAdvisor` | Score valid live setups; advisory only in the first deployment stage. |
| `Dashboard` | Future reports for model quality, backtests, feature importance, and candidates. |
| `EA_Generator` | Future generation of reviewable MQL5 release artifacts. |

---

## 5. Data design

The main asset of the project will be a trustworthy historical dataset, not the first ML model.

Each row should represent a potential strategy setup, including setups that would later lose. The data must include values known exactly when the setup appeared and outcomes known only afterward.

### Example initial input features

#### Price and candle context

- Open, high, low, close, range, tick volume, and spread.
- Candle body percentage, upper wick percentage, lower wick percentage.
- Consecutive bullish/bearish candles.
- Recent high/low distance and price momentum.

#### EMA features

- EMA values and distance between fast/slow EMAs.
- EMA slope.
- EMA rate of change.
- EMA acceleration / curvature.
- Time since crossover.
- Price distance from EMA.

#### Bollinger Band features

- Upper, middle, and lower band values.
- Band width.
- Band-width percentile compared with recent history.
- Band-width expansion/contraction rate.
- Squeeze duration.
- Price position inside/outside the bands.
- Number of bars outside a band.

#### Market and time context

- ATR and ATR percentile.
- Session: Asian, London, New York, or overlap.
- Hour, weekday, month.
- Higher-timeframe direction.
- Optional later features: liquidity sweep, market-structure shift, fair value gap, volume, correlated instruments, and news-distance flag.

### Outcome labels

Start with simple, unambiguous labels. Possible outputs are:

- `tp_before_sl`: whether the defined take profit was reached before the stop loss.
- `realized_r`: realized reward/risk multiple.
- `mfe_r`: maximum favorable excursion in R.
- `mae_r`: maximum adverse excursion in R.
- `bars_to_exit`: duration until the trade outcome.

Labels must be based on fixed, documented entry/exit assumptions. Changing labels changes the research question and needs a new dataset version.

---

## 6. Machine-learning approach

### Initial model plan

1. Begin with a simple baseline: no ML filter and transparent backtest.
2. Train XGBoost as the first candidate classifier/regressor.
3. Compare it with LightGBM and CatBoost only after the dataset is stable.
4. Prefer the simplest model that improves unseen-period performance.

### Model targets

The initial model should estimate probabilities and expected outcomes, for example:

- Probability that TP is reached before SL.
- Expected `realized_r`.
- Expected MFE and MAE.
- Confidence based on validation performance and calibration.

### Explainability

Model outputs must be explainable before they influence an EA. Use:

- Feature importance.
- SHAP values when appropriate.
- A saved record of training period, features, labels, parameters, and results.

The model must produce evidence that a condition improves results; it must not become a black box that silently controls funded-account risk.

---

## 7. Candidate EA evolution workflow

```text
Approved EA baseline (v1.0)
        |
        v
Historical setup dataset
        |
        v
ML + optimizer proposes a bounded condition
        |
        v
Candidate EA rule set (v1.1 candidate)
        |
        v
Backtest with realistic costs
        |
        v
Walk-forward and robustness checks
        |
        +-- fails --> reject and retain v1.0
        |
        +-- passes --> manual review -> demo trial -> approved release
```

Examples of valid candidate changes:

- Change an EMA or Bollinger Band parameter within a pre-approved range.
- Add a session filter.
- Add a volatility threshold.
- Add a higher-timeframe alignment filter.
- Change a stop/target rule only within specified risk limits.

Examples of invalid autonomous changes:

- Removing risk controls.
- Increasing position size without explicit approval.
- Changing funded-account execution directly.
- Deploying because in-sample backtest profit looked high.

---

## 8. Validation standards

Every candidate must meet these gates:

1. **No look-ahead bias** — features must be available at the original trade decision time.
2. **Chronological splits** — never randomly split time-series data.
3. **Locked baseline** — compare each candidate to the currently approved EA under identical assumptions.
4. **Cost assumptions** — include spread, commission, slippage, and realistic trading hours.
5. **Out-of-sample test** — reserve data unseen during optimization.
6. **Walk-forward validation** — repeatedly fit on the past and test on the next period.
7. **Robustness checks** — test reasonable variations in costs, parameters, and time periods.
8. **Demo trial** — observe live/demo behavior before real-money deployment.
9. **Manual approval** — the trader decides whether a validated candidate becomes an EA release.

Success should not be judged by win rate alone. Primary measurements should include expectancy, profit factor, drawdown, trade count, stability across periods, and risk-adjusted return.

---

## 9. MT5 and MQL5 integration plan

### Stage A: historical data

- Export/import historical bars and indicator values from MT5.
- Store raw data separately from processed datasets.
- Verify timezone, symbol suffix, timeframe, missing bars, spread, and broker data quality.

### Stage B: custom indicators

- Recreate the EMA smoothing Bollinger Band and Bollinger Band calculations in a controlled Python feature pipeline, or export their exact MQL5 values.
- Compare Python and MT5 indicator outputs bar by bar before using them for research.

### Stage C: advisory integration

- MT5/EA detects a valid user-defined setup.
- It sends or exports the current feature snapshot.
- Python model returns an advisory probability and explanation.
- No order is placed automatically in the initial implementation.

### Stage D: EA release artifacts

- Generate or update a versioned MQL5 EA only from an explicitly approved candidate rule set.
- Compile and test it in the MT5 Strategy Tester.
- Retain the previous approved EA version for rollback.

---

## 10. Delivery roadmap

### Milestone 0 — Scaffold (complete)

- Project folders and package boundaries created.
- Configuration template created.
- Git initialized.
- Smoke tests passing.

### Milestone 1 — Define one strategy

- Document exact EMA smoothing BB + BB entry conditions.
- Define symbol, timeframe, trade direction, entry timing, stop loss, take profit, and exit rules.
- Define what counts as a setup.

### Milestone 2 — Historical data and features

- Import a single symbol/timeframe history.
- Build and verify EMA/BB features.
- Compare values against MT5.
- Save a versioned feature dataset.

### Milestone 3 — Dataset and baseline backtest

- Create labels for every detected setup.
- Implement a simple, transparent baseline backtest.
- Report basic performance under documented costs.

### Milestone 4 — First XGBoost experiment

- Train on an earlier historical period.
- Validate on a later period.
- Measure calibration and feature importance.
- Create advisory-only predictions.

### Milestone 5 — Candidate filters

- Convert only supported, understandable patterns into candidate filters.
- Compare candidates to the baseline using walk-forward testing.
- Reject unstable improvements.

### Milestone 6 — MT5 demo advisor

- Connect the validated scorer to MT5.
- Record every advisory decision and real-time outcome.
- Run a meaningful demo period.

### Milestone 7 — Approved EA release process

- Generate versioned MQL5 artifacts for manually approved rules.
- Use the Strategy Tester, then demo validation.
- Maintain a rollback path and deployment log.

---

## 11. Repository layout

```text
ML-Trading-Lab/
├── Config/                  # Local-safe template configuration
├── Docs/                    # Architecture and research documentation
├── Scripts/                 # Thin command-line entry points
├── Tests/                   # Automated checks
├── src/ml_trading_lab/
│   ├── DataEngine/
│   ├── FeatureEngine/
│   ├── DatasetBuilder/
│   ├── StrategyEngine/
│   ├── ML/
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── catboost_model.py
│   │   └── model_manager.py
│   ├── Optimizer/
│   ├── Backtester/
│   ├── WalkForward/
│   ├── LiveAdvisor/
│   ├── Dashboard/
│   └── EA_Generator/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 12. Current status and next decision

The foundation is ready. The next work should be narrowly scoped to one strategy and one market, beginning with the exact rules for the EMA smoothing Bollinger Band + Bollinger Band observation.

Before coding the data importer, specify:

1. Symbol and broker symbol name.
2. Primary timeframe and higher timeframe, if used.
3. Long and short entry rules.
4. Exact EMA and Bollinger Band settings.
5. Stop loss, take profit, and exit definition.
6. Whether the first objective is win probability, expected R, or both.

Once these are defined, the project can build a correct dataset and test the idea without adding uncontrolled automation.
