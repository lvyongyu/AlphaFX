# AlphaFX Roadmap

This roadmap keeps the project aligned with the original principle:

Quant model generates the signal. AI explains, challenges, and summarizes it.

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

## Next Version: V2 Research Upgrade

Goal: make the app useful for a normal user who should not need to find, download, format, or upload macro CSV files, then add a transparent ML research layer that predicts whether the future 20-trading-day AUD/USD return is positive.

V2 is split into two phases:

- V2.0: Macro auto-data foundation.
- V2.1: Machine-learning comparison layer.

## V2.0 Macro Auto-Data Foundation

Problem: V1 supports AU2Y, US2Y, and iron ore as CSV uploads, but most users will not know where to download these files or how to format them. Keeping upload as a normal workflow makes the app feel unfinished and creates avoidable user burden. If those factors stay missing, the factor view, rule signal, and future ML model are weaker.

Goal: make macro factor collection fully automatic. Do not keep CSV upload in the core V2 UI.

Data sources:

- US2Y: FRED `DGS2`, 2-year Treasury constant maturity rate.
- AU2Y: RBA Statistical Table F2, Australian Government 2-year bond yield.
- Iron ore: FRED `PIORECRUSDM`, global iron ore price in USD per metric ton.

V2.0 should update:

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

V2.0 acceptance criteria:

- A user can deploy the app, press one refresh button, and get AUD/USD, DXY, VIX, US2Y, AU2Y, and iron ore factors without manually downloading CSVs.
- The app still runs if any macro source is unavailable.
- Factor View clearly distinguishes available, stale, and missing data.

## V2.1 Machine Learning

Goal: add a transparent ML research layer that compares a model against the existing rule-based strategy.

V2 should update:

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

V2.1 acceptance criteria:

- The app shows rule-based and ML signals side by side.
- The app compares rule-based and ML backtests over the same dates, costs, and leverage.
- The model reports validation metrics, feature importance, and sample-size warnings.
- The AI report explains where the ML model agrees or disagrees with the rule model.

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
