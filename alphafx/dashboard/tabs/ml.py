from __future__ import annotations

import pandas as pd
import streamlit as st

from alphafx.ml import ml_rule_agreement

from alphafx.dashboard.cache import cached_backtest
from alphafx.dashboard.context import ResearchContext


def render(ctx: ResearchContext) -> None:
    start = ctx.start
    end = ctx.end
    leverage = ctx.leverage
    market_data = ctx.market_data
    signals = ctx.signals
    latest_signal = ctx.latest_signal
    ml_result = ctx.ml_result
    ml_signals = ctx.ml_signals
    ml_latest_signal = ctx.ml_latest_signal
    st.subheader("ML (experimental)")
    st.caption(
        "A simple, leak-free ML model for comparison against the rule strategy — NOT a replacement. "
        "The rule signal stays primary/live. Trained with walk-forward validation on point-in-time features."
    )
    if ml_result.get("warning"):
        st.warning(ml_result["warning"])

    cols = st.columns(3)
    cols[0].metric("Rule signal (live)", str(latest_signal["signal"]).upper())
    cols[1].metric("ML signal (experimental)", str(ml_latest_signal).upper() if ml_latest_signal else "N/A")
    cols[2].metric("Independent obs (~N/horizon)", ml_result.get("effective_n", 0))
    st.write(ml_rule_agreement(latest_signal["signal"], ml_latest_signal))
    st.caption(f"Training rows: {ml_result.get('training_samples', 0)} · features used: {', '.join(ml_result.get('manifest', [])) or 'none'}")

    fold_metrics = ml_result.get("fold_metrics", pd.DataFrame())
    if not fold_metrics.empty:
        st.subheader("Walk-forward validation (per fold)")
        st.dataframe(fold_metrics, use_container_width=True, hide_index=True)

    importance = ml_result.get("feature_importance", pd.DataFrame())
    if not importance.empty:
        st.subheader("Feature importance")
        st.bar_chart(importance.set_index("feature")["importance"])

    if not ml_signals.empty:
        st.subheader("Rule vs ML backtest (same dates, costs, leverage)")
        rule_bt, rule_m = cached_backtest(market_data, signals, start, end, leverage)
        ml_bt, ml_m = cached_backtest(market_data, ml_signals, start, end, leverage)
        keys = ["total_return", "sharpe", "max_drawdown", "win_rate", "number_of_trades", "strategy_vs_random_percentile"]
        st.dataframe(
            pd.DataFrame(
                [
                    {"strategy": "rule (live)", **{k: rule_m.get(k) for k in keys}},
                    {"strategy": "ML (experimental)", **{k: ml_m.get(k) for k in keys}},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        if not rule_bt.empty and not ml_bt.empty:
            overlay = (
                rule_bt[["date", "equity"]].rename(columns={"equity": "rule"})
                .merge(ml_bt[["date", "equity"]].rename(columns={"equity": "ml"}), on="date", how="outer")
                .sort_values("date")
            )
            st.line_chart(overlay.set_index("date"))
        st.caption(
            "ML predictions are out-of-sample only (walk-forward), so this is an honest comparison — "
            "but on this sample size the ML edge is easily noise. Do not treat ML as the better signal by default."
        )
    else:
        st.info("No ML predictions yet — need more clean history to train.")
