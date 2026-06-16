from __future__ import annotations

import streamlit as st

from alphafx.dashboard.context import ResearchContext


def render(ctx: ResearchContext) -> None:
    forward_diagnostics = ctx.forward_diagnostics
    latest_signal = ctx.latest_signal
    st.subheader("Signal Diagnostics")
    st.caption("Forward-return statistics are historical diagnostics, not a profit forecast.")
    st.caption(
        "`effective_sample_size` ≈ sample_size / horizon counts independent (non-overlapping) "
        "observations; `mean_t_stat_adjusted` discounts the naive t-stat for overlap."
    )
    if forward_diagnostics.empty:
        st.info("Not enough history for signal diagnostics yet.")
    else:
        st.dataframe(forward_diagnostics, use_container_width=True, hide_index=True)

    st.subheader("Rolling factor IC")
    st.caption(
        "Each factor's information coefficient per non-overlapping window. The signal hard-codes "
        "each factor's direction — an IC that flips sign across windows is a warning that the assumed "
        "direction is not stable over time."
    )
    rolling_ic = ctx.factor_diagnostics_agent.rolling_ic_table(ctx.features)
    if rolling_ic.empty:
        st.info("Not enough history for rolling factor IC yet.")
    else:
        pivot = rolling_ic.pivot_table(index="window_end", columns="factor", values="information_coefficient")
        st.line_chart(pivot)
        st.dataframe(rolling_ic, use_container_width=True, hide_index=True)

    st.subheader("Calibration")
    st.write(f"Latest probability source: `{latest_signal.get('probability_source', 'fallback_score_map')}`")
    st.write(f"Calibration sample size: `{int(latest_signal.get('calibration_sample_size', 0))}`")
