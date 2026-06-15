from __future__ import annotations

import streamlit as st

from alphafx.dashboard.context import ResearchContext


def render(ctx: ResearchContext) -> None:
    paper_journal_agent = ctx.paper_journal_agent
    st.subheader("Paper Trade Journal")
    journal = paper_journal_agent.load()
    if journal.empty:
        st.info("No paper journal records yet.")
    else:
        st.dataframe(journal, use_container_width=True, hide_index=True)
