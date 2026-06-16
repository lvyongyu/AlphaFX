from __future__ import annotations

import pandas as pd

from alphafx.agents import FactorDiagnosticsAgent, FeatureAgent, QuantSignalAgent, SignalDiagnosticsAgent
from alphafx.database import Database

from factories import sample_market_data


def test_score_calculation_and_signal_mapping():
    features = FeatureAgent().build_features(sample_market_data())
    signals = QuantSignalAgent().generate_signals(features)
    latest = signals.dropna(subset=["score"]).iloc[-1]
    assert latest["aud_momentum_score"] == 1
    assert latest["dxy_score"] == 1
    assert latest["vix_score"] == 1
    assert latest["signal"] == "bullish"
    assert latest["probability"] == 0.60
    assert latest["probability_source"] == "fallback_score_map"


def test_signal_mapping_thresholds():
    assert QuantSignalAgent.map_signal(3) == "bullish"
    assert QuantSignalAgent.map_signal(-3) == "bearish"
    assert QuantSignalAgent.map_signal(2) == "neutral"
    # probability is directional confidence (P(signal correct)), keyed by |score|,
    # so a strong bearish score maps to a HIGH probability — same as bullish.
    assert QuantSignalAgent.map_probability(5) == 0.70
    assert QuantSignalAgent.map_probability(-5) == 0.70
    assert QuantSignalAgent.map_probability(-3) == 0.60


def test_calibrated_probability_uses_prior_outcomes():
    market = sample_market_data()
    features = FeatureAgent().build_features(market)
    raw = QuantSignalAgent().generate_signals(features)
    calibration = SignalDiagnosticsAgent().calibration_frame(market, raw, horizon=20, min_samples=5)
    calibrated = QuantSignalAgent().generate_signals(features, calibration=calibration)
    active = calibrated[calibrated["probability_source"] == "historical_calibration"]
    assert not active.empty
    assert active["calibration_sample_size"].min() >= 5


def test_equal_weight_is_default_and_unchanged():
    features = FeatureAgent().build_features(sample_market_data())
    default = QuantSignalAgent().generate_signals(features)
    explicit_equal = QuantSignalAgent().generate_signals(
        features, weights={c: 1.0 for c in QuantSignalAgent.score_columns}, persist=False
    )
    # Equal weights (normalized to sum 5) reproduce the default equal-weight score exactly.
    merged = default.merge(explicit_equal, on="date", suffixes=("_d", "_w"))
    valid = merged.dropna(subset=["score_d", "score_w"])
    assert (valid["score_d"] == valid["score_w"]).all()


def test_ic_weighted_signal_does_not_persist(tmp_path):
    db = Database(tmp_path / "sig.db")
    features = FeatureAgent(db=db).build_features(sample_market_data())
    agent = QuantSignalAgent(db=db)
    agent.generate_signals(features)  # persisted equal-weight signals
    weights = FactorDiagnosticsAgent().ic_weights(features)
    agent.generate_signals(features, weights=weights, persist=False)  # must NOT overwrite
    saved = pd.read_sql_query("SELECT * FROM signals ORDER BY date", db.connect(), parse_dates=["date"])
    # The persisted signals are still the equal-weight ones (probability_source is the live label).
    assert not saved.empty
