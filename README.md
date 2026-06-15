# AlphaFX

AlphaFX is an explainable macro-factor research platform for AUD/USD directional signals.

It combines macro indicators, factor scoring, historical calibration, walk-forward backtesting, and AI-style explanations to support FX research and paper trading decisions.

AlphaFX is not an AI Forex Bot, an automated trading robot, or a profit-generation system.

## Scope

The project focuses only on AUD/USD until the research process is stable.

Core principles:

- Quant models generate signals.
- AI-style agents explain, challenge, and summarize signals.
- Probabilities should be calibrated from history where possible, not asserted as certainty.
- Backtests must be transparent about assumptions, costs, and limitations.
- Research and paper trading come before any broker integration.

## Risk Notice

This software is for research and education only.

- Not financial advice.
- Not a profit guarantee.
- No live trading in V1 or V2.
- Research and paper trading only.
- FX trading involves substantial risk, including leverage, spread, slippage, and regime-change risk.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run tests:

```bash
pytest
```

## Macro Data

The next planned data upgrade should make macro factors one-click by downloading data automatically:

- US2Y from FRED `DGS2`
- AU2Y from the RBA F2 capital market yields table
- Iron ore from FRED `PIORECRUSDM`

CSV upload is not planned for the core workflow. If a source is unavailable, the app should keep running and clearly mark that factor as unavailable or stale.

## LLM Explanations

The explanation, contrarian, and judge agents can run as real Claude calls
(`alphafx/llm/`). This is **opt-in and never sets the signal**: the quant model
owns signal, score, and probability; the LLM only explains, challenges, and
summarizes, and the judge's `final_signal` is forced to equal the quant signal.

- Enable by setting `ANTHROPIC_API_KEY` and turning on "Use LLM explanations"
  in the sidebar. Without a key, the app falls back to the deterministic
  template agents and runs fully offline.
- The model id is configured in `alphafx/config.py` (default `claude-opus-4-8`;
  override with `ALPHAFX_LLM_REPORT_MODEL` / `ALPHAFX_LLM_NARRATION_MODEL`, or
  force-disable with `ALPHAFX_LLM_DISABLED=1`).
- Every LLM call is logged to the `llm_calls` table for auditability, and the
  LLM is never called inside the backtest or walk-forward loops.

## ML Research Layer

A simple, leak-free ML model (`alphafx/ml/`) runs **alongside** the rule signal
for comparison — it is **not** a replacement. The rule signal stays primary/live.

- Targets are forward returns; features are the point-in-time engineered factors.
- Validation is walk-forward / time-series split only (no random split, no leak),
  and the ML backtest uses out-of-sample predictions only.
- Primary model is a regularized logistic regression; XGBoost is used if installed
  (optional), otherwise the app falls back to scikit-learn and still runs.
- Because the independent sample is small (~rows / horizon), ML overfits easily;
  the ML tab shows feature importance, per-fold validation metrics, a rule-vs-ML
  backtest, and a prominent small-sample warning.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned versions and [DESIGN.md](DESIGN.md) for the planned professional research-platform architecture.
