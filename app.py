from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from alphafx.agents import (
    AIExplanationAgent,
    BacktestAgent,
    ContrarianAgent,
    DataAgent,
    FactorDiagnosticsAgent,
    FeatureAgent,
    JudgeAgent,
    PaperJournalAgent,
    QuantSignalAgent,
    RiskAgent,
    SignalDiagnosticsAgent,
    WalkForwardAgent,
)
from alphafx.config import DEFAULT_SYMBOLS


st.set_page_config(page_title="AlphaFX AUD/USD Research Platform", page_icon="FX", layout="wide")


def fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def fmt_num(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.4f}"


def metric_cards(signal: pd.Series, risk):
    cols = st.columns(7)
    values = [
        ("Signal", signal.get("signal", "neutral").upper()),
        ("Probability", fmt_pct(signal.get("probability", 0.5))),
        ("Confidence", signal.get("confidence", "Low")),
        ("Action", risk.action),
        ("Leverage", f"{risk.leverage:.1f}x"),
        ("Stop Loss", fmt_pct(risk.stop_loss)),
        ("Take Profit", fmt_pct(risk.take_profit)),
    ]
    for col, (label, value) in zip(cols, values):
        col.metric(label, value)


def latest_non_empty(features: pd.DataFrame, signals: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    signal_agent = QuantSignalAgent()
    latest_signal = signal_agent.latest_signal(signals)
    if latest_signal.empty:
        return pd.Series(dtype=object), pd.Series(dtype=object)
    latest_date = pd.to_datetime(latest_signal["date"])
    feature = features[pd.to_datetime(features["date"]) == latest_date]
    latest_feature = feature.iloc[-1] if not feature.empty else features.dropna(how="all").iloc[-1]
    return latest_feature, latest_signal


with st.sidebar:
    st.header("Data")
    default_end = date.today()
    default_start = default_end - timedelta(days=365 * 5)
    start = st.date_input("Start date", default_start)
    end = st.date_input("End date", default_end)
    leverage = st.slider("Paper leverage", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
    st.caption("Research and paper trading only. No live order execution.")
    refresh = st.button("Download market + macro data", type="primary", use_container_width=True)

data_agent = DataAgent()
feature_agent = FeatureAgent()
signal_agent = QuantSignalAgent()
diagnostics_agent = SignalDiagnosticsAgent()
risk_agent = RiskAgent()
backtest_agent = BacktestAgent()
walk_forward_agent = WalkForwardAgent()
factor_diagnostics_agent = FactorDiagnosticsAgent()
paper_journal_agent = PaperJournalAgent()
explain_agent = AIExplanationAgent()
contrarian_agent = ContrarianAgent()
judge_agent = JudgeAgent()

if refresh:
    with st.spinner("Downloading market and macro data..."):
        data_agent.download_market_data(start, end)
        data_agent.download_macro_data(start, end)

market_data = data_agent.load_market_data()
if market_data.empty:
    st.info("Use the sidebar to download AUD/USD, DXY, VIX, and macro data.")
    st.stop()

macro_data = data_agent.load_macro_data()
features = feature_agent.build_features(market_data, macro_data=macro_data)
raw_signals = signal_agent.generate_signals(features)
calibration = diagnostics_agent.calibration_frame(market_data, raw_signals, horizon=20, min_samples=20)
signals = signal_agent.generate_signals(features, calibration=calibration)
forward_diagnostics = diagnostics_agent.forward_return_diagnostics(market_data, raw_signals)
latest_feature, latest_signal = latest_non_empty(features, signals)

if latest_signal.empty:
    st.warning("Data loaded, but not enough history is available for a signal yet.")
    st.stop()

factor_table = feature_agent.factor_table(latest_feature, latest_signal)
risk = risk_agent.suggest(
    signal=latest_signal["signal"],
    probability=latest_signal["probability"],
    volatility=latest_feature.get("audusd_vol_20d"),
    user_leverage=leverage,
)
explanation = explain_agent.explain(latest_signal, factor_table)
contrarian = contrarian_agent.critique(latest_signal, factor_table)
judgement = judge_agent.judge(latest_signal, risk, explanation, contrarian)
aud_latest = market_data[market_data["symbol"] == DEFAULT_SYMBOLS.audusd].sort_values("date").iloc[-1]["close"]
paper_journal_agent.record_signal(latest_feature, latest_signal, factor_table, risk, judgement["explanation"], audusd_price=float(aud_latest))

st.title("AlphaFX")
st.caption("Explainable macro-factor research platform for AUD/USD directional signals. Not financial advice. Not a profit guarantee.")

tabs = st.tabs(["Dashboard", "Factor View", "Backtest", "Signal Diagnostics", "Validation", "Paper Journal", "AI Report"])

with tabs[0]:
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

with tabs[1]:
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

with tabs[2]:
    st.subheader("Backtest")
    col1, col2, col3, col4 = st.columns(4)
    bt_start = col1.date_input("Backtest start", start, key="bt_start")
    bt_end = col2.date_input("Backtest end", end, key="bt_end")
    holding = col3.number_input("Holding period", min_value=5, max_value=60, value=20, step=5)
    cost = col4.number_input("Transaction cost bps", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
    c1, c2, c3 = st.columns(3)
    spread = c1.number_input("Spread bps", min_value=0.0, max_value=50.0, value=1.0, step=0.5)
    slippage = c2.number_input("Slippage bps", min_value=0.0, max_value=50.0, value=1.0, step=0.5)
    rollover = c3.number_input("Rollover bps / day placeholder", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    bt_data, metrics = backtest_agent.run(market_data, signals, bt_start, bt_end, holding, leverage, cost, spread, slippage, rollover)

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

with tabs[3]:
    st.subheader("Signal Diagnostics")
    st.caption("Forward-return statistics are historical diagnostics, not a profit forecast.")
    if forward_diagnostics.empty:
        st.info("Not enough history for signal diagnostics yet.")
    else:
        st.dataframe(forward_diagnostics, use_container_width=True, hide_index=True)

    st.subheader("Calibration")
    st.write(f"Latest probability source: `{latest_signal.get('probability_source', 'fallback_score_map')}`")
    st.write(f"Calibration sample size: `{int(latest_signal.get('calibration_sample_size', 0))}`")

with tabs[4]:
    st.subheader("Walk-Forward Validation")
    wf = walk_forward_agent.run(market_data, features)
    if wf.empty:
        st.info("Not enough history for the default 3-year train / 6-month test walk-forward run.")
    else:
        st.dataframe(wf, use_container_width=True, hide_index=True)

    st.subheader("Factor Diagnostics")
    factor_diag = factor_diagnostics_agent.analyze(features)
    if factor_diag.empty:
        st.info("Not enough complete feature history for factor diagnostics.")
    else:
        st.dataframe(factor_diag, use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("Paper Trade Journal")
    journal = paper_journal_agent.load()
    if journal.empty:
        st.info("No paper journal records yet.")
    else:
        st.dataframe(journal, use_container_width=True, hide_index=True)

with tabs[6]:
    st.subheader("AI Report")
    bullish = factor_table[factor_table["stance"] == "bullish"]["factor"].tolist()
    bearish = factor_table[factor_table["stance"] == "bearish"]["factor"].tolist()

    st.markdown(
        f"""
### 1. Current Signal
{latest_signal["signal"].upper()} AUD/USD, probability {latest_signal["probability"]:.0%}, confidence {latest_signal["confidence"]}.

### 2. Key Bullish Drivers
{", ".join(bullish) if bullish else "No bullish drivers are currently active."}

### 3. Key Bearish Drivers
{", ".join(bearish) if bearish else "No bearish drivers are currently active."}

### 4. Contrarian View
{contrarian["main_risk"]}

{contrarian["alternative_scenario"]}

### 5. Risk Management
Suggested action: **{risk.action}**. Leverage: **{risk.leverage:.1f}x**. Stop loss: **{risk.stop_loss:.0%}**. Take profit: **{risk.take_profit:.0%}**.

### 6. What To Watch
{contrarian["watch"]}
"""
    )
