from __future__ import annotations

import pandas as pd
import streamlit as st

from alphafx.config import DATA_DIR
from alphafx.instruments import LIVE_PORTFOLIO, get_instrument

from alphafx.dashboard.cache import cached_portfolio_summary
from alphafx.dashboard.context import ResearchContext


def render(ctx: ResearchContext) -> None:
    st.caption(
        "Live call for every pair in the trading portfolio (AUD/EUR/CHF). These are the "
        "pairs with a positive risk-adjusted edge and low mutual correlation; together they "
        "lift Sharpe vs AUD-only and roughly halve drawdown. Paper trading only — no live orders."
    )

    summary = cached_portfolio_summary(ctx.start, ctx.end, ctx.leverage)
    if summary.empty:
        st.info("No portfolio data yet — download market data from the sidebar.")
        return

    trading = summary[summary["Action"].astype(str).str.startswith(("BUY", "SELL"))]

    # Portfolio-level read: how many pairs want a trade today + the committed paper book.
    open_n, realised = _paper_book_totals()
    cols = st.columns(3)
    cols[0].metric("Pairs signalling a trade today", f"{len(trading)} / {len(summary)}")
    cols[1].metric("Open paper positions", open_n)
    cols[2].metric("Realised PnL (paper)", f"{realised:.2f}")

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config={"Confidence": st.column_config.NumberColumn(format="%.3f")},
    )

    open_book = _portfolio_open_positions()
    if not open_book.empty:
        st.subheader("Open paper positions (portfolio)")
        st.dataframe(open_book, use_container_width=True, hide_index=True)
    else:
        st.caption("No open paper positions. Run `python scripts/paper_trade.py --export` or the daily job.")


def _paper_book_totals() -> tuple[int, float]:
    pos_csv = DATA_DIR / "paper_positions.csv"
    if not pos_csv.exists():
        return 0, 0.0
    positions = pd.read_csv(pos_csv)
    if positions.empty:
        return 0, 0.0
    open_n = int((positions["status"] == "open").sum())
    closed = positions[positions["status"] == "closed"]
    realised = float(closed["realised_pnl"].fillna(0.0).sum()) if not closed.empty else 0.0
    return open_n, realised


def _portfolio_open_positions() -> pd.DataFrame:
    pos_csv = DATA_DIR / "paper_positions.csv"
    if not pos_csv.exists():
        return pd.DataFrame()
    positions = pd.read_csv(pos_csv)
    if positions.empty or "instrument" not in positions.columns:
        return pd.DataFrame()
    portfolio_ids = {get_instrument(n).oanda for n in LIVE_PORTFOLIO}
    open_book = positions[(positions["status"] == "open") & (positions["instrument"].isin(portfolio_ids))]
    keep = [c for c in ["instrument", "side", "units", "entry_price", "entry_date", "stop_loss_pct"] if c in open_book.columns]
    return open_book[keep].reset_index(drop=True)
