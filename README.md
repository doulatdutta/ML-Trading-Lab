# ML Trading Lab

ML Trading Lab is a **research-first** foundation for improving user-defined MT5 strategies with validated machine learning. It deliberately contains no trading rules, broker credentials, execution logic, or trained models.

The intended lifecycle is:

```text
MT5 history / custom indicators
        -> feature generation -> labeled dataset -> model experiments
        -> candidate strategy filters -> backtest -> walk-forward validation
        -> demo advisor -> manually approved EA release
```

## Safety principle

Machine learning may propose filters or parameter candidates; it must never silently change or deploy an EA. Every candidate should pass out-of-sample and walk-forward testing, then demo validation, before any funded-account use.

## Architecture

| Area | Responsibility |
| --- | --- |
| `DataEngine` | Import historical prices and later connect to MT5. |
| `FeatureEngine` | Transform raw bars and indicator values into stable, point-in-time features. |
| `DatasetBuilder` | Build labeled training datasets without look-ahead bias. |
| `StrategyEngine` | Express transparent, user-owned strategy rules and candidates. |
| `ML` | Train, score, compare, and version statistical models. |
| `Optimizer` | Propose bounded parameter/filter experiments. |
| `Backtester` | Simulate entries, exits, costs, and risk assumptions. |
| `WalkForward` | Validate on chronological, unseen periods. |
| `LiveAdvisor` | Read-only live scoring and recommendations. |
| `Dashboard` | Future reporting interface. |
| `EA_Generator` | Future MQL5 templates and release artifacts. |

## Quick start

1. Install Python 3.11 or newer.
2. In this folder, create and activate a virtual environment.
3. Run `pip install -e .[dev]`.
4. Run `pytest` to check the scaffold.

Optional dependencies are intentionally separated:

- `pip install -e .[dev,ml]` for model experimentation.
- `pip install -e .[mt5]` only on the Windows machine that runs MetaTrader 5.

## Configuration

Copy `Config/settings.example.yaml` to `Config/settings.yaml` and adapt local paths. The real settings file is ignored by Git. Do not store account numbers, passwords, or broker credentials in this project.

## Suggested first milestone

Implement one non-executing historical data importer, then define the exact EMA and Bollinger Band features and outcome labels for a single symbol/timeframe. Do not add optimization or live trading until that data pipeline is verified.
