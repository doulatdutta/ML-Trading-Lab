# Architecture Notes

## Boundaries

The Python application owns data preparation, experiments, validation, and advisory output. MQL5 will eventually own chart-side indicator collection and the EA release artifact. Their integration should use versioned files or explicit interfaces, not hidden shared state.

## Non-negotiable validation gates

1. Feature availability is checked at the original decision timestamp.
2. Data is split chronologically, never randomly.
3. Backtests include stated spread, commission, and slippage assumptions.
4. Candidate changes are compared to a locked baseline on unseen periods.
5. Human approval and demo trading precede any funded-account deployment.
