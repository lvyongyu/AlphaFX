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
