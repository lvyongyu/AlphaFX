# AlphaFX Design Plan

## Product Positioning

AlphaFX is an explainable macro-factor research platform for AUD/USD directional signals.

It should not be positioned as:

- AI Forex Bot.
- Automated FX profit system.
- Fully automated trading robot.
- Live trading execution platform.

The app should help a researcher answer:

- What is the current AUD/USD directional signal?
- Which macro factors support or challenge it?
- How did similar signals perform historically?
- How robust is the signal out of sample?
- What are the main risks and failure modes?

## Design Principles

- AUD/USD only until the workflow is stable.
- Quant outputs first, explanation second.
- LLM/AI text never creates the signal.
- Probability should come from historical calibration where possible.
- Backtests must expose assumptions, costs, and trade records.
- Simple transparent models before complex ML.
- Keep every incremental step runnable and tested.
- No live trading until research, paper logging, and validation are mature.

## Incremental Implementation Strategy

### Step 1: Documentation And Positioning

Update public-facing language:

- Explainable macro-factor research platform.
- Research and paper trading only.
- Not financial advice.
- Not a profit guarantee.
- No live trading in V1/V2.

No code behavior changes in this step.

### Step 2: Signal Diagnostics

Add forward-return analysis for existing rule signals.

Keep it read-only:

- Compute historical forward returns for 20/40/60 trading days.
- Group by bullish, bearish, and neutral.
- Show average return, median return, hit rate, win/loss ratio, Sharpe, drawdown, and profit factor.
- Add tests for forward-return calculations.

### Step 3: Calibrated Probability

Replace hardcoded probability as the primary output.

Simple approach:

- Use prior historical outcomes for the same signal class.
- Calibrate on a chosen horizon, initially 20 trading days.
- Fall back to the score map only when sample size is too small.
- Label probabilities as `calibrated` or `fallback`.

Avoid:

- Using future data for the current signal.
- Overfitting by score bucket with tiny samples.

### Step 4: Trade-Level Backtest

Implement explicit trade simulation:

- Signal date.
- Next-trading-day entry.
- Holding-period exit.
- Position.
- Entry/exit price.
- Costs.
- Realised PnL.

Keep one strategy mode at first: current rule signal.

### Step 5: Benchmarks And Cost Model

Add transparent baselines:

- Buy and hold.
- Always flat.
- Random fixed-seed baseline.
- Moving-average crossover.
- Naive momentum.

Add cost controls:

- Transaction cost.
- Spread.
- Slippage.
- Rollover placeholder.

### Step 6: Walk-Forward Validation

Add rolling validation after the basic backtest is stable:

- 3-year train/calibration window.
- 6-month test window.
- Roll forward.
- Aggregate out-of-sample results.

### Step 7: Risk And Journal

Upgrade risk suggestions:

- Volatility-based stops.
- Volatility-adjusted sizing.
- Max drawdown guard.
- Event blackout placeholders.

Add SQLite paper journal:

- One daily research record per signal.
- Open/closed status.
- Realised PnL when closed.

### Step 8: Data Providers

Separate data provider concerns from strategy logic:

- yfinance remains prototype market source.
- FRED and RBA become macro providers.
- OANDA stays placeholder only.
- CSV provider exists only for tests/offline fixtures.

### Step 9: ML Layer

Only after calibration, backtest, and walk-forward validation are stable:

- Logistic regression or ridge first.
- Random forest optional.
- XGBoost only if deployment remains simple.
- No deep learning.
- No reinforcement learning.

## Target Architecture

The desired end-state structure is:

```text
alphafx/
  data/
    providers.py
    yfinance_provider.py
    fred_provider.py
    rba_provider.py
    csv_provider.py
  features/
    macro_features.py
    technical_features.py
  signals/
    scoring.py
    calibration.py
    factor_weights.py
  backtest/
    engine.py
    walk_forward.py
    metrics.py
    benchmarks.py
  risk/
    position_sizing.py
    stops.py
    regime_filter.py
  paper/
    journal.py
    trade_log.py
  agents/
    signal_agent.py
    risk_agent.py
    explanation_agent.py
  dashboard/
    app.py
  storage/
    sqlite.py
  config/
    settings.py
```

Do not migrate everything at once. Move modules only when a feature needs the boundary.

## Non-Goals For V2

- Live trading.
- Fully automated order execution.
- IBKR integration.
- OANDA trading integration.
- Options.
- Barrier products.
- Multi-currency support.
- Deep learning.
- Reinforcement learning.
