# AlphaFX Roadmap

This roadmap keeps the project aligned with the original positioning:

AlphaFX is an explainable macro-factor research platform for AUD/USD directional signals.

It is not an AI Forex Bot, automated money-making robot, or live trading system.

## Current Baseline: V1

Status: shipped.

V1 includes:

- Streamlit dashboard
- SQLite persistence
- yfinance collector for AUD/USD, DXY, and VIX
- Optional CSV uploads for AU2Y, US2Y, and iron ore
- Feature engine
- Score-based directional signal engine
- Backtest engine
- Risk suggestion engine
- AI-style explanation, contrarian, and judge agents
- Unit tests for core quant logic

Known V1 limitations:

- Signal probability is still score-mapped rather than historically calibrated.
- Backtest is useful but still too coarse for research-grade trade analysis.
- Holding-period logic needs true entry/exit trade records.
- Macro factors are incomplete unless data is manually supplied.
- No walk-forward validation yet.
- No factor weight stability or information coefficient diagnostics yet.
- The explanation, contrarian, and judge agents are template-based string
  builders, not real LLM calls.

## Next Version: V2 Professional Research Upgrade

Goal: upgrade AlphaFX from a macro-factor scoring demo into a verifiable AUD/USD quant research platform.

V2 should be implemented incrementally, keeping the app runnable after every step.

## V2.0 P0 Research Validity

Priority: fix signal reliability, calibration, and backtesting rigor before adding complex ML.

### P0.1 Signal Forward-Return Analysis

Add historical diagnostics for bullish, bearish, and neutral signals over 20, 40, and 60 trading days.

Outputs by signal class and horizon:

- Average forward return.
- Median forward return.
- Hit rate.
- Win/loss ratio.
- Max drawdown.
- Sharpe ratio.
- Profit factor.
- Sample size.

Acceptance criteria:

- The app can explain how each signal class historically behaved.
- Diagnostics are calculated from historical outcomes, not hardcoded expectations.
- Missing or small sample sizes are clearly flagged.

### P0.2 Calibrated Probability

Replace the current hardcoded score-to-probability mapping as the primary probability source.

Plan:

- Keep score mapping as fallback only.
- Calibrate probability from historical hit rate by signal class and horizon.
- Start with simple expanding-window or prior-window calibration.
- Avoid using future outcomes to calibrate the latest signal.

Acceptance criteria:

- Latest probability is labelled as calibrated or fallback.
- The calibration sample window and sample size are visible.
- Tests cover probability fallback and calibrated probability behavior.

### P0.3 Trade-Level Backtest

Upgrade backtesting from daily-position approximation to explicit trades.

Required trade fields:

- Signal date.
- Entry date.
- Exit date.
- Position.
- Holding period.
- Entry price.
- Exit price.
- Realised return.
- Transaction cost.
- Trade-level PnL.
- Status.

Rules:

- Avoid look-ahead bias.
- Today signal can only execute from the next trading day.
- Holding period must be simulated as actual entry-to-exit trades.

Acceptance criteria:

- Backtest exposes a trade list.
- Metrics are derived from trade-level PnL where appropriate.
- Tests cover next-day execution and holding-period exits.

### P0.4 Benchmark Comparison

Add simple transparent baselines:

- Buy and hold AUD/USD.
- Always flat.
- Random signal baseline with fixed seed.
- Simple moving-average crossover.
- Naive momentum baseline.

Acceptance criteria:

- Strategy performance is always shown next to benchmarks.
- Random baseline is deterministic for reproducibility.

### P0.5 Cost Model

Extend the cost assumptions beyond a single transaction-cost input:

- Transaction cost bps.
- Spread bps.
- Slippage bps.
- Rollover/swap placeholder.
- Broker cost config placeholder.

Acceptance criteria:

- Backtest UI shows cost assumptions.
- Costs are included in trade-level returns.
- The app labels rollover as a placeholder until real broker data exists.

## V2.1 P1 Validation And Diagnostics

Priority: add robustness checks before treating signals as research-grade.

### Walk-Forward Validation

Implement walk-forward backtesting:

- Train/calibration window: default 3 years.
- Test window: default 6 months.
- Roll forward through history.
- Recompute calibration or factor weights in each window.

Outputs:

- In-sample metrics.
- Out-of-sample metrics.
- Degradation ratio.
- Rolling Sharpe.
- Rolling drawdown.
- Yearly returns.

### Factor Weight Diagnostics

Keep this simple and transparent before any complex ML:

- Logistic regression.
- Ridge regression.
- Information coefficient ranking.
- Correlation with future 20/40/60 day AUD/USD returns.

Outputs per factor:

- Coefficient.
- Information coefficient.
- Hit contribution.
- Stability across time windows.
- Feature importance.

Do not add deep learning or reinforcement learning.

### Risk Management Upgrade

Replace fixed stop/take-profit assumptions with research-grade risk suggestions:

- ATR or volatility-based stop loss.
- Volatility-adjusted position sizing.
- Max risk per trade.
- Max portfolio exposure.
- Max drawdown guard.
- No-trade regime when volatility is extreme.
- Event blackout placeholder for FOMC, CPI, NFP, and RBA decisions.

### Paper Trade Journal

Add a SQLite-backed journal for daily research records.

Fields:

- Date.
- AUD/USD price.
- Signal.
- Score.
- Calibrated probability.
- Factor values.
- Factor contributions.
- Recommended position.
- Stop loss.
- Take profit.
- Explanation.
- Entry price.
- Exit price.
- Realised PnL.
- Status: open or closed.

## V2.2 P1 Dashboard Upgrade

Upgrade Streamlit from a signal display into a research dashboard.

Dashboard sections:

- Current signal with calibrated probability.
- Factor breakdown.
- Risk recommendation.
- Backtest equity and drawdown.
- Trade list.
- Monthly returns.
- Yearly returns.
- Signal diagnostics.
- Hit rate by horizon.
- Factor contribution chart.
- Walk-forward results.
- Rolling Sharpe.
- Rolling drawdown.

## V2.3 P1 Macro Auto-Data Foundation

Problem: V1 supports AU2Y, US2Y, and iron ore as CSV uploads, but users should not need to find, download, format, or upload macro CSV files.

Goal: make macro factor collection fully automatic. Do not keep CSV upload in the core workflow.

Data sources:

- US2Y: FRED `DGS2`, 2-year Treasury constant maturity rate.
- AU2Y: RBA Statistical Table F2, Australian Government 2-year bond yield.
- Iron ore: FRED `PIORECRUSDM`, global iron ore price in USD per metric ton.

V2.3 should update:

- Dependencies:
  - Add `pandas_datareader` only if it makes FRED access materially cleaner.
  - Otherwise use direct CSV/HTTP readers from pandas to keep deployment simple.
- Database:
  - Add a `macro_data` table with `date`, `symbol`, `value`, `source`, `frequency`, and `created_at`.
  - Store canonical symbols such as `US2Y`, `AU2Y`, and `IRON_ORE`.
  - Keep `market_data` for traded/market symbols from yfinance.
- Data layer:
  - Add macro download methods to `DataAgent`, or split into a dedicated `MacroDataAgent` if the class becomes too broad.
  - Download and normalize FRED/RBA series into the same schema.
  - Preserve source and frequency metadata.
  - Handle missing network/source failures gracefully with clear status messages.
- Feature layer:
  - Build `yield_spread = AU2Y - US2Y` automatically.
  - Forward-fill lower-frequency iron ore data onto the market date index.
  - Mark stale values if a macro series has not updated within an expected window.
  - Show whether each factor came from FRED, RBA, yfinance, is stale, or is unavailable.
- UI:
  - Replace `Download / Refresh` with a clearer action such as `Download market + macro data`.
  - Remove CSV upload controls for AU2Y, US2Y, and iron ore.
  - Add a data status panel that shows latest date, source, rows, and freshness for each input.
- Tests:
  - FRED CSV parser.
  - RBA F2 parser.
  - Macro data upsert/load.
  - Yield spread construction.
  - Monthly iron ore forward-fill.
  - Stale/missing data status.

V2.3 acceptance criteria:

- A user can deploy the app, press one refresh button, and get AUD/USD, DXY, VIX, US2Y, AU2Y, and iron ore factors without manually downloading CSVs.
- The app still runs if any macro source is unavailable.
- Factor View clearly distinguishes available, stale, and missing data.

## V2.4 P2 Data Provider Architecture

Abstract data sources behind provider interfaces so strategy logic does not depend directly on yfinance.

Suggested provider structure:

- `YFinanceProvider`
- `FREDProvider`
- `RBAProvider`
- `TreasuryProvider`
- `OANDAProvider` placeholder
- `CSVProvider` for offline tests only, not user-facing workflow

Acceptance criteria:

- Data collection code is separate from signal logic.
- Providers normalize into common market/macro dataframes.
- Tests can use offline fixtures without network access.

## V2.5 P2 Machine Learning

Status: shipped as a research comparison. The ML signal is leak-free
(walk-forward, point-in-time features) and surfaced alongside the rule signal,
but the rule signal stays primary/live and the ML tab carries a prominent
small-sample overfitting warning (~N/horizon independent observations).

Goal: add a transparent ML research layer that compares a model against the existing rule-based strategy.

V2.5 should update:

- Dependencies:
  - Add `xgboost` if install size and Streamlit Cloud compatibility are acceptable.
  - Keep a `scikit-learn` fallback model, such as `HistGradientBoostingClassifier` or `RandomForestClassifier`, so the app remains deployable if XGBoost fails.
- Data and features:
  - Add future return targets: `audusd_future_return_20d` and `target_up_20d`.
  - Create a clean ML dataset builder that drops rows with unavailable target or required features.
  - Use automatically downloaded macro factors from V2.0 when available.
  - Preserve optional factor behavior: any unavailable factor must be excluded explicitly, and the model must report which features were actually used.
- Model layer:
  - Add an `MLSignalAgent`.
  - Use time-series split or walk-forward validation, not random train/test split.
  - Output model probability, predicted class, feature importance, training sample count, and validation metrics.
  - Avoid overfitting by keeping the first model simple and comparing it to the rule-based baseline.
- Backtesting:
  - Add an ML strategy mode alongside the rule-based strategy.
  - Compare rule-based and ML equity curves, drawdowns, Sharpe, win rate, and trade count.
  - Make transaction costs and leverage apply consistently to both.
- UI:
  - Add an ML tab or extend Backtest with a strategy selector.
  - Show rule signal versus ML signal side by side.
  - Show feature importance and validation metrics.
  - Include a plain-English warning when sample size is small or validation quality is weak.
- Tests:
  - Target construction.
  - ML dataset row filtering.
  - Probability-to-signal mapping.
  - Walk-forward split does not leak future data.
  - ML backtest mode uses only prior data.

V2.5 acceptance criteria:

- The app shows rule-based and ML signals side by side.
- The app compares rule-based and ML backtests over the same dates, costs, and leverage.
- The model reports validation metrics, feature importance, and sample-size warnings.
- The AI report explains where the ML model agrees or disagrees with the rule model (template-level until V2.6 lands the LLM layer).

## V2.6 P2 LLM Explanation Layer

Goal: upgrade the explanation, contrarian, and judge agents from template-based
string builders into real Claude calls, without letting the LLM create signals.

Boundary (non-negotiable):

- Quant agents own `signal`, `score`, and `probability`. The LLM cannot change them.
- The LLM consumes only pre-computed numbers (an evidence pack), never raw market
  data, and never states a direction different from the quant signal.
- The judge may record an LLM dissent separately, but `final_signal` must equal the
  quant signal.

Plan:

- Add a thin `alphafx/llm/` layer: API client wrapper, evidence-pack builder,
  prompts, structured-output schemas, and LLM-backed narrator agents.
- Keep the existing template agents as the offline fallback when no API key is set
  or a call fails.
- Never call the LLM inside the backtest or walk-forward loops; only on the latest
  signal and on on-demand reports.
- Use the official `anthropic` SDK with the model id configured in `config.py`
  (default `claude-opus-4-8`).
- Persist prompt, model, response, and token usage for auditability and
  reproducibility.

Acceptance criteria:

- LLM explanations are grounded in the evidence pack and validated against the
  quant signal direction before display.
- The app runs and explains signals even with no API key (template fallback).
- The judge's structured output never overrides the quant signal.
- Every LLM call is logged with enough detail to reproduce it.

All V2 work should not add:

- Live trading.
- IBKR.
- Options.
- Barrier products.
- Automated order execution.

## V3 First Touch Model

Predict whether AUD/USD hits an upper target before a lower target.

Example:

- Current: `0.7000`
- Upper: `0.7210`
- Lower: `0.6790`
- Output: `P(Upper First)`

This version should connect more directly to stop-loss and take-profit research.

## V4 IBKR Paper Trading

Connect to Interactive Brokers paper trading only after backtesting is stable and the signal logic is auditable.

Required before V4:

- More robust backtest storage.
- Clear order preview UI.
- Manual confirmation flow.
- Strong separation between research signals and executable orders.

## V5 Live Trading

Manual approval required before every live order.

No fully automated live trading initially.

## V6 Barrier Option Research

Optional future module.

Do not mix this into the directional AUD/USD research app until the core model and paper trading layers are mature.
