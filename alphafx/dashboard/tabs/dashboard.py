from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px

from alphafx.config import DEFAULT_SYMBOLS
from alphafx.dashboard.ui import metric_cards

from alphafx.dashboard.context import ResearchContext


def render(ctx: ResearchContext) -> None:
    data_agent = ctx.data_agent
    market_data = ctx.market_data
    macro_data = ctx.macro_data
    latest_signal = ctx.latest_signal
    risk = ctx.risk
    judgement = ctx.judgement
    st.subheader("Latest AUD/USD Signal")
    metric_cards(latest_signal, risk)
    st.write(judgement["explanation"])
    st.warning(risk.warning)

    aud = market_data[market_data["symbol"] == DEFAULT_SYMBOLS.audusd].copy()
    aud["date"] = pd.to_datetime(aud["date"])
    fig = px.line(aud, x="date", y="close", title="AUD/USD Close")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Data completeness"):
        st.dataframe(data_agent.completeness_report(market_data), use_container_width=True)
        st.dataframe(data_agent.macro_status_report(macro_data), use_container_width=True)
        st.caption(
            "Macro factors are revised series applied with a publication lag "
            "(yields ~1 business day, iron ore ~21), so each value enters features "
            "only as-of when it would have been available — not point-in-time vintage data."
        )
