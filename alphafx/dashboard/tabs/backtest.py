from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px

from alphafx.dashboard.ui import fmt_pct

from alphafx.dashboard.context import ResearchContext


def render(ctx: ResearchContext) -> None:
    start = ctx.start
    end = ctx.end
    leverage = ctx.leverage
    backtest_agent = ctx.backtest_agent
    market_data = ctx.market_data
    signals = ctx.signals
    latest_feature = ctx.latest_feature
    st.subheader("Backtest")
    col1, col2, col3, col4 = st.columns(4)
    bt_start = col1.date_input("Backtest start", start, key="bt_start")
    bt_end = col2.date_input("Backtest end", end, key="bt_end")
    holding = col3.number_input("Holding period", min_value=5, max_value=60, value=20, step=5)
    cost = col4.number_input("Transaction cost bps", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
    c1, c2, c3, c4 = st.columns(4)
    spread = c1.number_input("Spread bps", min_value=0.0, max_value=50.0, value=1.5, step=0.5)
    slippage = c2.number_input("Slippage bps", min_value=0.0, max_value=50.0, value=1.0, step=0.5)
    broker_swap = c3.number_input("Broker swap markup bps/day", min_value=0.0, max_value=10.0, value=0.3, step=0.1)
    spread_value = latest_feature.get("yield_spread")
    default_swap = float(spread_value) if spread_value is not None and not pd.isna(spread_value) else 0.0
    swap = c4.number_input(
        "AU−US carry % (annual)",
        min_value=-10.0,
        max_value=10.0,
        value=round(default_swap, 2),
        step=0.25,
        help="Signed overnight interest differential. Long AUD earns it when positive; short pays it.",
    )
    st.caption(
        f"Costs applied: spread {spread} bps + slippage {slippage} bps (round trip), "
        f"broker swap {broker_swap} bps/day, and a directional carry of {swap}%/yr."
    )
    bt_data, metrics = backtest_agent.run(
        market_data, signals, bt_start, bt_end, holding, leverage, cost, spread, slippage, broker_swap, swap
    )

    metric_cols = st.columns(6)
    for col, key in zip(
        metric_cols,
        ["total_return", "annualized_return", "sharpe", "max_drawdown", "win_rate", "profit_factor"],
    ):
        value = metrics[key]
        col.metric(key.replace("_", " ").title(), fmt_pct(value) if key != "sharpe" and key != "profit_factor" else f"{value:.2f}")

    if not bt_data.empty:
        st.plotly_chart(px.line(bt_data, x="date", y="equity", title="Equity Curve"), use_container_width=True)
        st.plotly_chart(px.area(bt_data, x="date", y="drawdown", title="Drawdown"), use_container_width=True)

        yearly = backtest_agent.yearly_returns(bt_data)
        if not yearly.empty:
            st.subheader("Yearly Returns")
            st.caption("Exposes regime concentration — an edge living in one year/regime shows up here.")
            st.bar_chart(yearly.set_index("year")["return"])

        roll = bt_data.assign(date=pd.to_datetime(bt_data["date"])).set_index("date")["strategy_return"]
        rolling_sharpe = (roll.rolling(63).mean() / roll.rolling(63).std()) * (252 ** 0.5)
        if rolling_sharpe.notna().any():
            st.subheader("Rolling Sharpe (63d)")
            st.line_chart(rolling_sharpe.dropna())

        n_trades = int(metrics.get("number_of_trades", 0))
        sharpe_trade = metrics.get("sharpe_trade", 0.0)
        ci_low = metrics.get("sharpe_trade_ci_low", 0.0)
        ci_high = metrics.get("sharpe_trade_ci_high", 0.0)
        t_stat = metrics.get("avg_trade_t_stat", 0.0)
        st.metric("Per-trade Sharpe (95% CI)", f"{sharpe_trade:.2f}  [{ci_low:.2f}, {ci_high:.2f}]")
        straddles_zero = ci_low <= 0.0 <= ci_high
        st.caption(
            f"Based on {n_trades} non-overlapping trades (independent observations). "
            f"Mean-trade t-stat {t_stat:.2f}. "
            + (
                "The CI straddles 0 — at this sample size the edge is **not** distinguishable from zero."
                if straddles_zero
                else "The CI excludes 0, but small samples still warrant caution."
            )
        )

        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)
        if hasattr(backtest_agent, "last_trades") and not backtest_agent.last_trades.empty:
            st.subheader("Trade List")
            st.dataframe(backtest_agent.last_trades, use_container_width=True, hide_index=True)
        benchmark_keys = [k for k in metrics if k.startswith("benchmark_")]
        st.subheader("Benchmarks")
        st.dataframe(
            pd.DataFrame(
                [{"benchmark": k.replace("benchmark_", "").replace("_return", "").replace("_", " "), "return": metrics[k]} for k in benchmark_keys]
            ),
            use_container_width=True,
            hide_index=True,
        )
        percentile = metrics.get("strategy_vs_random_percentile", 0.0)
        st.metric("Strategy vs random percentile", fmt_pct(percentile))
        st.caption(
            f"The strategy beats {percentile:.0%} of 200 random baselines "
            f"(random mean {fmt_pct(metrics.get('benchmark_random_mean'))}, "
            f"5–95% band {fmt_pct(metrics.get('benchmark_random_p05'))} to {fmt_pct(metrics.get('benchmark_random_p95'))}). "
            "A high percentile is evidence the edge is not luck; near 50% means it is indistinguishable from random."
        )
