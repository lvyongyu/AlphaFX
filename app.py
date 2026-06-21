from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

st.set_page_config(page_title="AlphaFX AUD/USD Research Platform", page_icon="FX", layout="wide")

from alphafx.config import llm_enabled  # noqa: E402 - must follow set_page_config
from alphafx.dashboard import cache, context as ctxmod  # noqa: E402
from alphafx.dashboard.tabs import (  # noqa: E402
    ai_report,
    backtest,
    chart,
    diagnostics,
    factor,
    hero,
    journal,
    ml,
    portfolio,
    positions,
    validation,
)

with st.sidebar:
    st.header("Data")
    default_end = date.today()
    default_start = default_end - timedelta(days=365 * 5)
    start = st.date_input("Start date", default_start)
    end = st.date_input("End date", default_end)
    leverage = st.slider("Paper leverage", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
    st.caption("Research and paper trading only. No live order execution.")
    use_llm = st.toggle(
        "Use LLM explanations",
        value=llm_enabled(),
        help="When off, deterministic template explanations are used. The LLM never sets the signal.",
    )
    if use_llm and not llm_enabled():
        st.caption("Set ANTHROPIC_API_KEY to enable the LLM. Falling back to templates for now.")
    refresh = st.button("Download market + macro data", type="primary", use_container_width=True)

build_kwargs = dict(compute_features=cache.cached_features, compute_ml=cache.cached_ml_result)
if refresh:
    with st.spinner("Downloading market and macro data..."):
        ctx = ctxmod.build_context(start, end, leverage, use_llm, refresh=True, **build_kwargs)
else:
    ctx = ctxmod.build_context(start, end, leverage, use_llm, refresh=False, **build_kwargs)

if ctx.status == ctxmod.NO_DATA:
    st.info("Use the sidebar to download AUD/USD, DXY, VIX, and macro data.")
    st.stop()
if ctx.status == ctxmod.NO_SIGNAL:
    st.warning("Data loaded, but not enough history is available for a signal yet.")
    st.stop()

st.title("AlphaFX")
st.caption("Explainable macro-factor research for AUD/USD. Not financial advice. Not a profit guarantee.")

# A degraded run (dead factors / empty macro) must be impossible to miss — the
# model silently dropping to fewer factors is what hid the no-trade bug before.
for _warning in ctx.warnings:
    st.warning(_warning)

# The answer first — today's call, the factors behind it, plain English.
hero.render(ctx)

# Portfolio-level daily summary: today's call across every traded pair (AUD/EUR/CHF).
st.subheader("📋  Portfolio summary")
portfolio.render(ctx)

# Everything else is opt-in detail, collapsed by default.
with st.expander("📈  Price chart & data"):
    chart.render(ctx)
with st.expander("🧮  Factor detail"):
    factor.render(ctx)
with st.expander("📊  Backtest performance"):
    backtest.render(ctx)
with st.expander("🔬  Validation & ML"):
    diagnostics.render(ctx)
    validation.render(ctx)
    ml.render(ctx)
with st.expander("📄  Full AI report"):
    ai_report.render(ctx)
with st.expander("📌  Paper positions"):
    positions.render(ctx)
with st.expander("📓  Paper journal"):
    journal.render(ctx)
