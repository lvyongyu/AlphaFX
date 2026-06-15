from __future__ import annotations

import pandas as pd
import streamlit as st

from alphafx.dashboard.context import ResearchContext


def render(ctx: ResearchContext) -> None:
    signal_agent = ctx.signal_agent
    walk_forward_agent = ctx.walk_forward_agent
    factor_diagnostics_agent = ctx.factor_diagnostics_agent
    market_data = ctx.market_data
    features = ctx.features
    latest_signal = ctx.latest_signal
    st.subheader("Walk-Forward Validation")
    wf = walk_forward_agent.run(market_data, features)
    if wf.empty:
        st.info("Not enough history for the default 3-year train / 6-month test walk-forward run.")
    else:
        st.dataframe(wf, use_container_width=True, hide_index=True)

    st.subheader("Factor Diagnostics")
    st.caption(
        "`ic_t_stat_adjusted` uses the independent (non-overlapping) sample size, so it does "
        "not overstate the information coefficient's significance the way the naive t-stat does."
    )
    factor_diag = factor_diagnostics_agent.analyze(features)
    if factor_diag.empty:
        st.info("Not enough complete feature history for factor diagnostics.")
    else:
        st.dataframe(factor_diag, use_container_width=True, hide_index=True)

    st.subheader("Factor Correlation")
    corr = factor_diagnostics_agent.factor_correlation(features)
    if corr.empty:
        st.info("Not enough factor history for a correlation matrix.")
    else:
        st.dataframe(corr.round(2), use_container_width=True)
        st.caption(
            "DXY and VIX both proxy the USD/risk axis and are usually correlated, so equal weight "
            "slightly double-counts that driver. We keep equal weight by design: orthogonalizing it "
            "away empirically tends to reduce IC and returns and is unstable."
        )

    st.subheader("Experimental: IC-weighted signal")
    ic_w = factor_diagnostics_agent.ic_weights(features)
    if not ic_w:
        st.info("IC weights unavailable (insufficient history).")
    else:
        ic_signals = signal_agent.generate_signals(features, weights=ic_w, persist=False)
        ic_latest = signal_agent.latest_signal(ic_signals)
        comparison = pd.DataFrame(
            [
                {"scheme": "equal weight (default / live)", "signal": latest_signal["signal"], "score": float(latest_signal["score"])},
                {"scheme": "IC-weighted (experimental)", "signal": ic_latest.get("signal"), "score": float(ic_latest.get("score", float("nan")))},
            ]
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.caption(
            "IC weights are in-sample and the supporting evidence comes from large equity cross-sections; "
            "on this small sample they can overfit. Validate via walk-forward before trusting. The default "
            "live signal stays equal-weight."
        )
