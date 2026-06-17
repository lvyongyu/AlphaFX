# AlphaFX

> An explainable macro-factor research platform for AUD/USD directional signals.

[![CI](https://github.com/lvyongyu/AlphaFX/actions/workflows/ci.yml/badge.svg)](https://github.com/lvyongyu/AlphaFX/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/stage-research%20%26%20paper%20trading-orange)

AlphaFX combines macro indicators, transparent factor scoring, historical
probability calibration, walk-forward backtesting, and AI-assisted explanations
to support FX **research** and **paper-trading** decisions. The quant layer owns
the signal; every other layer — ML, LLM, execution — is a comparison, an
explanation, or a downstream consumer that can never change it.

AlphaFX is **not** an AI forex bot, an automated trading robot, or a
profit-generation system.

## Table of Contents

- [What AlphaFX Is](#what-alphafx-is)
- [Core Principles](#core-principles)
- [Architecture](#architecture)
- [Project Status](#project-status)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Macro Data](#macro-data)
- [LLM Explanations](#llm-explanations)
- [ML Research Layer](#ml-research-layer)
- [Paper Trading](#paper-trading)
- [Project Layout](#project-layout)
- [Roadmap](#roadmap)
- [Risk Notice](#risk-notice)

## What AlphaFX Is

AlphaFX helps a researcher answer a focused set of questions about AUD/USD:

- What is the current directional signal, and how confident is it?
- Which macro factors support or challenge that signal?
- How did comparable signals behave historically?
- How robust is the signal out of sample?
- What are the main risks and failure modes?

The project deliberately covers **AUD/USD only** until the research process is
stable, and stays in research / paper-trading territory before any broker
integration.

## Core Principles

- **The quant layer owns the signal.** Quant models produce `signal`, `score`,
  and `probability`. ML and LLM layers may compare, explain, challenge, and
  summarize — never override.
- **No look-ahead.** Macro factors are publication-lagged (point-in-time),
  probability calibration is expanding-window, and ML uses out-of-sample
  predictions only.
- **Calibrate from history.** Probabilities come from historical hit rates where
  the sample supports it, falling back to a score map (clearly labelled) only
  when it does not.
- **Transparent backtests.** Assumptions, costs, trade records, and small-sample
  uncertainty are exposed, not hidden.
- **Research before execution.** Validation and paper logging come before any
  paper-broker connection, which comes before any live order.

## Architecture

The end-to-end flow — data → features → signal → calibration → research outputs
→ explanation — is documented with a diagram in
[docs/architecture.md](docs/architecture.md). Two boundaries are load-bearing:
the quant layer owns the signal, and no stage uses future data.

## Project Status

| Capability | Status |
| --- | --- |
| Streamlit research dashboard | ✅ Shipped |
| SQLite persistence | ✅ Shipped |
| Market data (yfinance: AUD/USD, DXY, VIX) | ✅ Shipped |
| Macro auto-data (FRED, RBA) | ✅ Shipped |
| Score-based directional signal | ✅ Shipped |
| Forward-return diagnostics & calibrated probability | ✅ Shipped |
| Trade-level backtest, benchmarks, cost model | ✅ Shipped |
| Walk-forward validation & factor IC diagnostics | ✅ Shipped |
| ML research comparison (leak-free) | ✅ Shipped |
| LLM explanation layer (opt-in) | ✅ Shipped |
| Local paper-trading simulation | ✅ Shipped |
| Broker-connected paper trading (demo accounts) | 🗺️ Planned — see [ROADMAP](ROADMAP.md) |
| Live trading (manual approval) | 🗺️ Planned — see [ROADMAP](ROADMAP.md) |

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run the test suite:

```bash
pytest
```

XGBoost is an optional ML extra and is **not** in the core requirements, so the
app deploys light and the ML layer falls back to scikit-learn without it. To
enable the XGBoost path:

```bash
pip install -r requirements-ml.txt
```

## Configuration

The app runs fully offline with no configuration. The following environment
variables (also loadable from a gitignored `.env`; see `.env.example`) are
optional:

| Variable | Purpose | Default |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | Enable real LLM explanations | unset (template fallback) |
| `ALPHAFX_LLM_REPORT_MODEL` | Model for judge/report calls | `claude-opus-4-8` |
| `ALPHAFX_LLM_NARRATION_MODEL` | Model for per-day explanation/contrarian calls | `claude-opus-4-8` |
| `ALPHAFX_LLM_DISABLED` | Force the LLM off even if a key is present | unset |

## Macro Data

Macro factors are downloaded automatically — no manual CSV preparation:

- **US2Y** from FRED `DGS2`
- **AU2Y** from the RBA F2 capital-market yields table
- **Iron ore** from FRED `PIORECRUSDM`

CSV upload is not part of the core workflow. If a source is unavailable, the app
keeps running and clearly marks that factor as unavailable or stale.

## LLM Explanations

The explanation, contrarian, and judge agents can run as real Claude calls
(`alphafx/llm/`). This is **opt-in and never sets the signal**: the quant model
owns signal, score, and probability; the LLM only explains, challenges, and
summarizes, and the judge's `final_signal` is forced to equal the quant signal.

- Enable by setting `ANTHROPIC_API_KEY` and turning on "Use LLM explanations" in
  the sidebar. Without a key, the app falls back to deterministic template
  agents and runs fully offline.
- The model id is configured in `alphafx/config.py` (default `claude-opus-4-8`).
- Every LLM call is logged to the `llm_calls` table for auditability, and the
  LLM is never called inside the backtest or walk-forward loops.

## ML Research Layer

A simple, leak-free ML model (`alphafx/ml/`) runs **alongside** the rule signal
for comparison — it is **not** a replacement. The rule signal stays primary/live.

- Targets are forward returns; features are the point-in-time engineered factors.
- Validation is walk-forward / time-series split only (no random split, no leak),
  and the ML backtest uses out-of-sample predictions only.
- Primary model is a regularized logistic regression; XGBoost is used if installed
  (optional, `requirements-ml.txt`), otherwise the app falls back to scikit-learn
  and still runs.
- Because the independent sample is small (~rows / horizon), ML overfits easily;
  the ML tab shows feature importance, per-fold validation metrics, a rule-vs-ML
  backtest, and a prominent small-sample warning.

## Paper Trading

A fully local paper-trading backend (`alphafx/trade/`) simulates fills with **no
broker, no network, and no credentials**. It shares the broker-agnostic
`OrderIntent` interface a real broker would use, so the execution backend can be
swapped later without touching signal logic.

```bash
python scripts/paper_trade.py            # run one local paper-trading step
python scripts/paper_trade.py --json     # machine-readable output
python scripts/paper_trade.py --export   # write data/ snapshots for committing
```

Position size scales with the risk-approved leverage (vol-aware, capped at 5x by
`RiskAgent`): traded notional = `base_units × size_factor × leverage`, so the
paper PnL reflects leverage directly.

### Leverage risk demo

`scripts/leverage_sim.py` replays a Monte-Carlo of synthetic AUD/USD paths
through the real `PaperBroker` at a ladder of leverages — deliberately above the
5x cap — to show how leverage amplifies losses for the *same* signal and stops:

```bash
python scripts/leverage_sim.py 1 5 20 50
```

```text
  lev  median_ret   p05_ret  med_maxDD  worst_trade  blow_up_rate
    1       -1.4%    -36.1%     -20.9%        -6.8%            0%
    5      -14.0%   -100.0%     -75.7%       -34.1%           33%
   20     -100.0%   -100.0%    -107.4%      -110.7%           79%
   50     -100.0%   -100.0%    -116.2%      -276.8%           91%
```

Leverage does not change the win rate — it multiplies the drawdown and the
left tail. Even at the 5x cap, a marginal signal is liquidated on ~1/3 of paths;
at 20x the account is wiped on ~80%. (Synthetic paths, no market-data network in
this environment — illustrative of the leverage effect, not a strategy backtest.)

A scheduled GitHub Actions workflow (`.github/workflows/daily.yml`) runs the
paper-trading step on weekdays and commits the resulting snapshots. Connecting to
broker **demo** accounts and, later, live execution behind manual approval are
tracked in the [Roadmap](ROADMAP.md).

## Project Layout

```text
alphafx/
  data/         # provider adapters: yfinance / FRED / RBA / CSV (offline tests)
  features.py   # point-in-time factor engineering (macro publication-lagged)
  signals.py    # score-based directional signal (quant layer — owns the signal)
  diagnostics.py# forward-return stats, calibration, factor IC / rolling IC
  backtest.py   # trade-level backtest, benchmarks, cost model, Sharpe CI
  walk_forward.py # rolling out-of-sample validation
  risk.py       # volatility-based stops / sizing / no-trade gate
  ml/           # leak-free ML research comparison (logreg / optional xgboost)
  llm/          # Claude client, evidence pack, prompts, schemas, narrator
  trade/        # OrderIntent + local PaperBroker (broker-agnostic interface)
  paper.py      # SQLite paper journal
  dashboard/    # Streamlit UI, split per tab
scripts/        # headless signal + paper-trading entry points
docs/           # architecture diagram and notes
tests/          # pytest suite
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned versions — including the
broker-connected paper-trading and live-trading execution track — and
[DESIGN.md](DESIGN.md) for the research-platform architecture.

## Risk Notice

This software is for research and education only.

- Not financial advice.
- Not a profit guarantee.
- Research and paper trading only; live trading is gated behind manual approval
  and is not enabled by default at any stage.
- FX trading involves substantial risk, including leverage, spread, slippage,
  and regime-change risk.
