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

## Next Version: V2 Machine Learning

Goal: add a transparent ML research layer that predicts whether the future 20-trading-day AUD/USD return is positive, then compares that model against the existing rule-based strategy.

V2 should update:

- Dependencies:
  - Add `xgboost` if install size and Streamlit Cloud compatibility are acceptable.
  - Keep a `scikit-learn` fallback model, such as `HistGradientBoostingClassifier` or `RandomForestClassifier`, so the app remains deployable if XGBoost fails.
- Data and features:
  - Add future return targets: `audusd_future_return_20d` and `target_up_20d`.
  - Create a clean ML dataset builder that drops rows with unavailable target or required features.
  - Preserve optional factor behavior: yield and iron ore can be absent, but the model must report which features were actually used.
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

V2 should not add:

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
