from __future__ import annotations

import numpy as np
import pandas as pd

from alphafx.agents import BacktestAgent, FeatureAgent, QuantSignalAgent
from alphafx.config import DEFAULT_SYMBOLS
from alphafx.ml import MLSignalAgent, build_dataset, build_targets, ml_rule_agreement


def oscillating_market_data(periods: int = 400) -> pd.DataFrame:
    # Sine-wave prices so forward returns contain BOTH up and down classes.
    dates = pd.bdate_range("2022-01-01", periods=periods)
    t = np.arange(periods)
    aud = 0.65 + 0.05 * np.sin(t / 15.0)
    dxy = 100.0 - 5.0 * np.sin(t / 15.0)
    vix = 18.0 + 5.0 * np.cos(t / 15.0)
    rows = []
    for i, dt in enumerate(dates):
        rows.extend(
            [
                {"date": dt, "symbol": DEFAULT_SYMBOLS.audusd, "open": aud[i], "high": aud[i], "low": aud[i], "close": float(aud[i]), "source": "test"},
                {"date": dt, "symbol": DEFAULT_SYMBOLS.dxy, "open": dxy[i], "high": dxy[i], "low": dxy[i], "close": float(dxy[i]), "source": "test"},
                {"date": dt, "symbol": DEFAULT_SYMBOLS.vix, "open": vix[i], "high": vix[i], "low": vix[i], "close": float(vix[i]), "source": "test"},
            ]
        )
    return pd.DataFrame(rows)


# ---- C1: dataset builder ----

def test_target_matches_sign_and_drops_last_rows():
    features = FeatureAgent().build_features(oscillating_market_data())
    df = build_targets(features, horizon=20)
    valid = df.dropna(subset=["audusd_future_return_20d"])
    assert ((valid["audusd_future_return_20d"] > 0).astype(float) == valid["target_up_20d"]).all()
    # Last `horizon` rows have no future and so a NaN target.
    assert df["target_up_20d"].tail(20).isna().all()


def test_dataset_filters_unavailable_factors_into_manifest():
    # No macro data supplied -> yield_spread / ironore are all-NaN and excluded.
    features = FeatureAgent().build_features(oscillating_market_data())
    X, y, dates, manifest = build_dataset(features, horizon=20)
    assert "yield_spread" not in manifest
    assert "ironore_return_20d" not in manifest
    assert "audusd_return_20d" in manifest
    assert not X.isna().any().any()
    assert len(X) == len(y) == len(dates)


# ---- C2: MLSignalAgent ----

def test_probability_to_signal_mapping():
    agent = MLSignalAgent(upper=0.55, lower=0.45)
    assert agent.map_probability_to_signal(0.60) == "bullish"
    assert agent.map_probability_to_signal(0.40) == "bearish"
    assert agent.map_probability_to_signal(0.50) == "neutral"
    assert agent.map_probability_to_signal(float("nan")) == "neutral"


def test_time_series_split_has_no_leak():
    splits = MLSignalAgent.time_series_splits(120, 5)
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()  # every test row is strictly after training


def test_walk_forward_outputs_and_effective_n():
    features = FeatureAgent().build_features(oscillating_market_data())
    result = MLSignalAgent(horizon=20).walk_forward_predict(features)
    assert not result["predictions"].empty
    assert not result["fold_metrics"].empty
    assert result["training_samples"] > 0
    assert result["effective_n"] == result["training_samples"] // 20
    assert not result["feature_importance"].empty


# ---- C3: ML backtest comparison ----

def test_to_signals_shape_and_backtest_runs():
    features = FeatureAgent().build_features(oscillating_market_data())
    result = MLSignalAgent().walk_forward_predict(features)
    ml_signals = MLSignalAgent.to_signals(result["predictions"])
    assert list(ml_signals.columns) == ["date", "signal", "score", "probability"]
    market = oscillating_market_data()
    _, metrics = BacktestAgent().run(market, ml_signals, "2022-01-01", "2024-01-01")
    assert "total_return" in metrics


def test_ml_backtest_is_out_of_sample_only():
    features = FeatureAgent().build_features(oscillating_market_data())
    X, y, dates, _ = build_dataset(features, horizon=20)
    result = MLSignalAgent().walk_forward_predict(features)
    preds = result["predictions"]
    # OOS predictions never cover the initial training window — the earliest
    # prediction date is strictly after the dataset's first date.
    assert pd.to_datetime(preds["date"]).min() > pd.to_datetime(dates).min()


# ---- C5: rule vs ML narrative ----

def test_ml_rule_agreement_text():
    assert "agree" in ml_rule_agreement("bullish", "bullish").lower()
    disagree = ml_rule_agreement("bullish", "bearish").lower()
    assert "disagree" in disagree
    assert "primary" in disagree  # the rule stays primary
    assert "unavailable" in ml_rule_agreement("bullish", None).lower()
