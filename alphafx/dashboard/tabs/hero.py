from __future__ import annotations

import streamlit as st

from alphafx.dashboard.ui import fmt_pct
from alphafx.ml import ml_rule_agreement

from alphafx.dashboard.context import ResearchContext

_ARROW = {"bullish": "▲", "bearish": "▼", "neutral": "■"}
_BANNER = {"bullish": st.success, "bearish": st.error, "neutral": st.info}
_CHIP = {"bullish": "↑", "bearish": "↓", "neutral": "→", "not available": "–"}


def render(ctx: ResearchContext) -> None:
    """The always-visible hero: today's call, the factors behind it, plain English."""
    signal = ctx.latest_signal
    risk = ctx.risk
    sig = str(signal.get("signal", "neutral"))
    prob = signal.get("probability", 0.5)
    conf = signal.get("confidence", "Low")

    banner = _BANNER.get(sig, st.info)
    banner(f"{_ARROW.get(sig, '■')}  **AUD/USD — {sig.upper()}**   ·   {prob:.0%} probability   ·   {conf} confidence")

    cols = st.columns(4)
    cols[0].metric("Action", risk.action)
    cols[1].metric("Leverage", f"{risk.leverage:.1f}x")
    cols[2].metric("Stop loss", fmt_pct(risk.stop_loss))
    cols[3].metric("Take profit", fmt_pct(risk.take_profit))

    chips = "   ".join(
        f"{row['factor']} {_CHIP.get(row['stance'], '–')}" for _, row in ctx.factor_table.iterrows()
    )
    st.markdown("**Why:**  " + chips)

    st.write(ctx.judgement.get("explanation", ""))
    if ctx.contrarian.get("main_risk"):
        st.caption("⚠️ " + ctx.contrarian["main_risk"])
    st.caption(ml_rule_agreement(sig, ctx.ml_latest_signal))
    st.caption(risk.warning)
