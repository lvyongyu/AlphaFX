from __future__ import annotations

import pandas as pd

from alphafx.agents import FeatureAgent

from factories import sample_market_data


def test_feature_calculation_core_returns():
    features = FeatureAgent().build_features(sample_market_data())
    latest = features.iloc[-1]
    assert latest["audusd_return_20d"] > 0
    assert latest["dxy_return_20d"] < 0
    assert latest["vix_change_20d"] < 0
    assert latest["audusd_vol_20d"] >= 0


def test_macro_alignment_applies_publication_lag():
    # A macro value observed on a date must not appear in features until the
    # publication lag has passed (no vintage/revised look-ahead).
    fa = FeatureAgent()
    index = pd.bdate_range("2024-01-01", periods=20)
    frame = pd.DataFrame({"date": ["2024-01-03", "2024-01-10"], "value": [1.0, 2.0]})
    aligned = fa._align_optional_series(index, frame, lag_days=3)
    assert aligned.loc[pd.Timestamp("2024-01-10")] == 1.0  # new obs not yet available
    assert aligned.loc[pd.Timestamp("2024-01-15")] == 2.0  # available after the 3-day lag

    # With no lag the value would leak onto its own observation date.
    no_lag = fa._align_optional_series(index, frame, lag_days=0)
    assert no_lag.loc[pd.Timestamp("2024-01-10")] == 2.0
