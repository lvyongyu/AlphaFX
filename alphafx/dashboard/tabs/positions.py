from __future__ import annotations

import pandas as pd
import streamlit as st

from alphafx.config import DATA_DIR

from alphafx.dashboard.context import ResearchContext


def render(ctx: ResearchContext) -> None:
    st.caption(
        "Paper trading only — no live orders. Positions come from the headless job "
        "(`scripts/paper_trade.py --export`) or the daily GitHub Action, committed as CSV."
    )
    pos_csv = DATA_DIR / "paper_positions.csv"
    hist_csv = DATA_DIR / "signal_history.csv"

    if not pos_csv.exists() and not hist_csv.exists():
        st.info("No paper-trading snapshots yet. Run `python scripts/paper_trade.py --export` or trigger the daily job.")
        return

    if pos_csv.exists():
        positions = pd.read_csv(pos_csv)
        open_p = positions[positions["status"] == "open"]
        closed = positions[positions["status"] == "closed"]
        realised = float(closed["realised_pnl"].fillna(0.0).sum()) if not closed.empty else 0.0
        cols = st.columns(3)
        cols[0].metric("Open positions", len(open_p))
        cols[1].metric("Closed trades", len(closed))
        cols[2].metric("Realised PnL", f"{realised:.2f}")
        if not open_p.empty:
            st.subheader("Open")
            st.dataframe(open_p, use_container_width=True, hide_index=True)
        if not closed.empty:
            st.subheader("Closed")
            st.dataframe(closed, use_container_width=True, hide_index=True)

    if hist_csv.exists():
        hist = pd.read_csv(hist_csv)
        if not hist.empty:
            st.subheader("Daily signal history")
            st.dataframe(hist.tail(30), use_container_width=True, hide_index=True)
