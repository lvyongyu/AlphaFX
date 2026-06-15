from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from alphafx.config import DEFAULT_SYMBOLS

from alphafx.dashboard.context import ResearchContext


def render(ctx: ResearchContext) -> None:
    market_data = ctx.market_data
    macro_data = ctx.macro_data
    data_agent = ctx.data_agent

    aud = market_data[market_data["symbol"] == DEFAULT_SYMBOLS.audusd].copy()
    aud["date"] = pd.to_datetime(aud["date"])
    st.plotly_chart(px.line(aud, x="date", y="close", title="AUD/USD Close"), use_container_width=True)

    st.dataframe(data_agent.completeness_report(market_data), use_container_width=True)
    st.dataframe(data_agent.macro_status_report(macro_data), use_container_width=True)
    st.caption(
        "Macro factors are revised series applied with a publication lag "
        "(yields ~1 business day, iron ore ~21), so each value enters features "
        "only as-of when it would have been available — not point-in-time vintage data."
    )
