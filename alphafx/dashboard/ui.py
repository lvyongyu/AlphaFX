from __future__ import annotations

import pandas as pd
import streamlit as st


def fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def fmt_num(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.4f}"


def metric_cards(signal: pd.Series, risk) -> None:
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
    # strict: cols is sized to match `values`, so a length mismatch is a bug —
    # better a loud error than a metric silently missing from the dashboard.
    for col, (label, value) in zip(cols, values, strict=True):
        col.metric(label, value)
