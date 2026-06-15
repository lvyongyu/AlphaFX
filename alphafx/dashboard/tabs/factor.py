from __future__ import annotations

import streamlit as st

from alphafx.dashboard.ui import fmt_num

from alphafx.dashboard.context import ResearchContext


def render(ctx: ResearchContext) -> None:
    latest_signal = ctx.latest_signal
    factor_table = ctx.factor_table
    st.subheader("Factor Contributions")
    display = factor_table.copy()
    display["current_value"] = display["current_value"].map(fmt_num)
    display["change_20d"] = display["change_20d"].map(fmt_num)
    st.dataframe(display, hide_index=True, use_container_width=True)

    score_cols = ["aud_momentum_score", "dxy_score", "yield_score", "ironore_score", "vix_score"]
    chart = latest_signal[score_cols].rename(
        {
            "aud_momentum_score": "AUD momentum",
            "dxy_score": "DXY",
            "yield_score": "Yield spread",
            "ironore_score": "Iron ore",
            "vix_score": "VIX",
        }
    )
    st.bar_chart(chart.dropna())
