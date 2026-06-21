from __future__ import annotations

import pandas as pd
import streamlit as st

from alphafx.backtest import BacktestAgent
from alphafx.features import FeatureAgent
from alphafx.ml import MLSignalAgent

from alphafx.dashboard.context import _empty_ml_result

# Streamlit reruns the whole script on every widget interaction. These caches
# key on input *content* (st.cache_data hashes DataFrames), so moving an
# unrelated widget no longer rebuilds features or retrains the ML model — the
# heavy work only re-runs when the underlying data actually changes.


@st.cache_data(show_spinner=False)
def cached_features(market_data: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
    return FeatureAgent().build_features(market_data, macro_data=macro_data)


@st.cache_data(show_spinner="Training ML model…")
def cached_ml_result(features: pd.DataFrame) -> dict:
    try:
        return MLSignalAgent().walk_forward_predict(features)
    except Exception as exc:  # noqa: BLE001 - keep the app running if ML is unavailable
        return _empty_ml_result(f"ML unavailable: {exc}")


@st.cache_data(show_spinner=False)
def cached_backtest(
    market_data: pd.DataFrame, signals: pd.DataFrame, start, end, leverage: float
) -> tuple[pd.DataFrame, dict]:
    return BacktestAgent().run(market_data, signals, start, end, leverage=leverage)


@st.cache_data(show_spinner="Computing portfolio summary…")
def cached_portfolio_summary(start, end, leverage: float) -> pd.DataFrame:
    """Latest live call for every pair in the trading portfolio, one row per pair.

    Runs the full signal pipeline per instrument (features computed per-pair; the
    heavy ML pass is skipped — the summary only needs the latest gated signal and
    risk action). Cached on (start, end, leverage) so it re-runs only when the
    window changes.
    """
    from alphafx.database import Database
    from alphafx.instruments import LIVE_PORTFOLIO, get_instrument
    from alphafx.dashboard.context import build_context, _empty_ml_result

    db = Database()
    skip_ml = lambda _features: _empty_ml_result("skipped for portfolio summary")  # noqa: E731
    rows: list[dict] = []
    for name in LIVE_PORTFOLIO:
        cfg = get_instrument(name)
        ctx = build_context(
            start, end, leverage, use_llm=False, refresh=False, db=db,
            instrument=name, compute_ml=skip_ml,
        )
        if ctx.status != "ok":
            rows.append({"Pair": cfg.label, "Date": "—", "Signal": "—", "Action": ctx.status,
                         "Confidence": None, "Factors": "—", "Price": None})
            continue
        ls = ctx.latest_signal
        live = int(sum(
            ctx.raw_signals[c].notna().sum() > 0
            for c in ctx.signal_agent.score_columns if c in ctx.raw_signals
        ))
        rows.append({
            "Pair": cfg.label,
            "Date": str(pd.to_datetime(ls["date"]).date()),
            "Signal": str(ls["signal"]).title(),
            "Action": ctx.risk.action,
            "Confidence": round(float(ls["probability"]), 3),
            "Factors": f"{live}/5",
            "Price": round(float(ctx.aud_latest), 5),
        })
    return pd.DataFrame(rows)
