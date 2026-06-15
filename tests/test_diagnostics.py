from __future__ import annotations

import pandas as pd

from alphafx.agents import FactorDiagnosticsAgent, FeatureAgent, QuantSignalAgent, SignalDiagnosticsAgent
from alphafx.config import DEFAULT_SYMBOLS

from factories import sample_market_data


def test_forward_return_diagnostics():
    market = sample_market_data()
    features = FeatureAgent().build_features(market)
    signals = QuantSignalAgent().generate_signals(features)
    diagnostics = SignalDiagnosticsAgent().forward_return_diagnostics(market, signals, horizons=(20,))
    bullish = diagnostics[diagnostics["signal"] == "bullish"].iloc[0]
    assert bullish["sample_size"] > 0
    assert bullish["hit_rate"] == 1.0


def test_forward_diagnostics_reports_effective_sample_size():
    market = sample_market_data()
    features = FeatureAgent().build_features(market)
    signals = QuantSignalAgent().generate_signals(features)
    diag = SignalDiagnosticsAgent().forward_return_diagnostics(market, signals, horizons=(20,))
    bullish = diag[diag["signal"] == "bullish"].iloc[0]
    # Independent count discounts the overlapping daily sample.
    assert bullish["effective_sample_size"] == bullish["sample_size"] // 20
    assert bullish["effective_sample_size"] < bullish["sample_size"]
    # Overlap-adjusted significance never exceeds the naive t-stat.
    assert abs(bullish["mean_t_stat_adjusted"]) <= abs(bullish["mean_t_stat"])


def test_factor_ic_significance_discounted_for_overlap():
    from alphafx.agents import FactorDiagnosticsAgent

    market = sample_market_data()
    features = FeatureAgent().build_features(market)
    diag = FactorDiagnosticsAgent().analyze(features, horizon=20)
    assert not diag.empty
    row = diag.iloc[0]
    assert row["effective_sample_size"] <= row["sample_size"]
    assert abs(row["ic_t_stat_adjusted"]) <= abs(row["ic_t_stat"])


def test_ic_weights_are_nonneg_and_keyed_by_score_column():
    features = FeatureAgent().build_features(sample_market_data())
    weights = FactorDiagnosticsAgent().ic_weights(features, horizon=20)
    assert weights  # available on this sample
    assert set(weights).issubset(set(QuantSignalAgent.score_columns))
    assert all(v >= 0 for v in weights.values())


def test_factor_correlation_matrix_exposes_dxy_vix_overlap():
    features = FeatureAgent().build_features(sample_market_data())
    corr = FactorDiagnosticsAgent().factor_correlation(features)
    assert not corr.empty
    # Symmetric square matrix with the labelled scoring factors, incl. DXY & VIX.
    assert "DXY trend" in corr.columns and "VIX" in corr.columns
    assert corr.shape[0] == corr.shape[1]
    assert abs(float(corr.loc["DXY trend", "VIX"]) - float(corr.loc["VIX", "DXY trend"])) < 1e-9


def test_calibration_frame_is_backward_looking():
    # Perturbing only the LAST trading days must not change the calibrated
    # probability for early dates — proving calibration_frame has no future leak.
    market = sample_market_data()
    features = FeatureAgent().build_features(market)
    raw = QuantSignalAgent().generate_signals(features)
    cal_a = SignalDiagnosticsAgent().calibration_frame(market, raw, horizon=20, min_samples=5)

    market2 = market.copy()
    aud_mask = market2["symbol"] == DEFAULT_SYMBOLS.audusd
    aud_dates = sorted(pd.to_datetime(market2.loc[aud_mask, "date"]).unique())
    tail = set(aud_dates[-10:])
    market2.loc[aud_mask & pd.to_datetime(market2["date"]).isin(tail), "close"] *= 1.5
    cal_b = SignalDiagnosticsAgent().calibration_frame(market2, raw, horizon=20, min_samples=5)

    merged = cal_a.merge(cal_b, on="date", suffixes=("_a", "_b"))
    cutoff = pd.Timestamp(sorted(tail)[0]) - pd.Timedelta(days=40)
    early = merged[pd.to_datetime(merged["date"]) < cutoff]
    assert not early.empty
    assert (early["calibrated_probability_a"] == early["calibrated_probability_b"]).all()


def test_walkforward_calibration_is_labelled_distinctly():
    market = sample_market_data()
    features = FeatureAgent().build_features(market)
    raw = QuantSignalAgent().generate_signals(features)
    wf_cal = SignalDiagnosticsAgent().make_walkforward_calibration(market, raw, horizon=20, min_samples=5)
    signals = QuantSignalAgent().generate_signals(features, calibration=wf_cal)
    active = signals[signals["probability_source"] == "walkforward_calibration"]
    assert not active.empty
    # The old full-sample method name is gone, so it cannot be misused live.
    assert not hasattr(SignalDiagnosticsAgent, "calibration_map")
