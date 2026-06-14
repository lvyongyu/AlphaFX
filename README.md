# AlphaFX AUD/USD Quant Agent

AI-agent-style quantitative research app for AUD/USD directional signals over a 20 to 60 trading day horizon.

V1 focuses on:

- AUD/USD market data collection with `yfinance`
- SQLite persistence
- Feature engineering
- Score-based quant signals
- Backtesting
- Paper-trade risk suggestions
- AI-style natural-language explanation without requiring an LLM
- Streamlit dashboard

Not included in V1:

- Live trading
- IBKR integration
- Options or barrier products
- Structured FX products

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

V2 should make macro factors one-click by downloading data automatically:

- US2Y from FRED `DGS2`
- AU2Y from the RBA F2 capital market yields table
- Iron ore from FRED `PIORECRUSDM`

CSV upload is not planned for V2. If a source is unavailable, the app should keep running and clearly mark that factor as unavailable or stale.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned versions. The next branch of work is V2: first automate macro data collection, then add a machine-learning research layer that compares against the current rule-based signal engine without adding live trading.
