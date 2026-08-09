"""Three-factor model tests — offline, synthetic panels, no network.

The defended properties are the ones whose failure would be invisible in a
backtest: signal direction (a flipped sign still produces a plausible curve),
lookback lag, and the z-scoring that stops one factor's units from dominating
the blend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphafx.carry import gross_turnover, rank_weights
from alphafx.factors import (
    MOMENTUM_LOOKBACK_DAYS,
    REER_PUBLICATION_LAG_MONTHS,
    VALUE_LOOKBACK_DAYS,
    build_reer_panel,
    carry_signal,
    combine_signals,
    cross_sectional_zscore,
    factor_correlations,
    factor_portfolio,
    hold_between_rebalances,
    momentum_signal,
    reer_symbol,
    value_signal,
)


def _days(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


# ---- signal definitions and their direction ----

def test_carry_signal_is_the_level_not_the_change():
    # A constant, positive differential must score positive every day. The old
    # model used the 20-day CHANGE, which is zero here — the bug this replaces.
    index = _days("2024-01-01", 4)
    rates = pd.DataFrame({"AUD": [5.0] * 4, "JPY": [0.5] * 4, "USD": [2.0] * 4}, index=index)

    signal = carry_signal(rates, ["AUD", "JPY"])
    assert (signal["AUD"] == 3.0).all()
    assert (signal["JPY"] == -1.5).all()
    assert signal.diff().dropna().abs().sum().sum() == 0.0  # a change-based signal would be flat


def test_momentum_is_positive_after_a_rise_and_uses_only_the_past():
    n = MOMENTUM_LOOKBACK_DAYS + 5
    index = _days("2020-01-01", n)
    rising = pd.Series(np.linspace(1.0, 1.5, n), index=index)
    spot = pd.DataFrame({"AUD": rising})

    signal = momentum_signal(spot)
    assert signal["AUD"].iloc[:MOMENTUM_LOOKBACK_DAYS].isna().all()  # no value before warm-up
    assert signal["AUD"].iloc[-1] > 0                                # rose over the window


def test_value_is_positive_for_a_currency_that_got_cheaper_in_real_terms():
    n = VALUE_LOOKBACK_DAYS + 5
    index = _days("2016-01-01", n)
    # AUD's real exchange rate falls (cheap -> prefer long), JPY's rises.
    reer = pd.DataFrame(
        {"AUD": np.linspace(120.0, 80.0, n), "JPY": np.linspace(80.0, 120.0, n)}, index=index
    )

    signal = value_signal(reer, ["AUD", "JPY"])
    assert signal["AUD"].iloc[-1] > 0   # depreciated in real terms -> cheap -> long
    assert signal["JPY"].iloc[-1] < 0   # appreciated -> expensive -> short


def test_value_ignores_a_currency_with_no_reer():
    reer = pd.DataFrame({"AUD": [100.0] * 3}, index=_days("2024-01-01", 3))
    assert list(value_signal(reer, ["AUD", "NZD"]).columns) == ["AUD"]


# ---- publication lag on REER ----

def test_reer_panel_withholds_a_month_until_published():
    macro = pd.DataFrame([
        {"date": "2024-01-01", "symbol": reer_symbol("AUD"), "value": 100.0,
         "source": "TEST", "frequency": "monthly"}
    ])
    index = pd.to_datetime(["2024-01-20", "2024-02-05"])

    panel = build_reer_panel(macro, index)
    assert np.isnan(panel.loc["2024-01-20", "AUD"])   # inside the month it describes
    assert panel.loc["2024-02-05", "AUD"] == 100.0    # published the following month
    assert REER_PUBLICATION_LAG_MONTHS >= 1


# ---- z-scoring and blending ----

def test_zscore_makes_factors_with_different_units_comparable():
    # Carry in percentage points is ~100x the scale of a log return. Without
    # standardising, a raw average would be carry with rounding error.
    index = pd.to_datetime(["2024-01-02"])
    carry = pd.DataFrame([{"A": 5.0, "B": 0.0, "C": -5.0}], index=index)
    momentum = pd.DataFrame([{"A": -0.05, "B": 0.0, "C": 0.05}], index=index)

    blended = combine_signals({"carry": carry, "momentum": momentum})
    # Equal and opposite after scaling, so the blend cancels to zero.
    assert blended.loc[index[0]].abs().max() < 1e-12


def test_zscore_drops_rows_with_too_few_names_or_no_dispersion():
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    signal = pd.DataFrame(
        [{"A": 1.0, "B": 2.0, "C": 3.0}, {"A": 4.0, "B": 4.0, "C": 4.0}], index=index
    )
    scaled = cross_sectional_zscore(signal)
    assert scaled.loc[index[0]].notna().all()   # real dispersion survives
    assert scaled.loc[index[1]].isna().all()    # flat row cannot be ranked


def test_blend_uses_whichever_factors_are_warm():
    # Momentum has not warmed up on this date; the score must fall back to carry
    # rather than going NaN and standing the book down.
    index = pd.to_datetime(["2024-01-02"])
    carry = pd.DataFrame([{"A": 3.0, "B": 0.0, "C": -3.0}], index=index)
    momentum = pd.DataFrame([{"A": np.nan, "B": np.nan, "C": np.nan}], index=index)

    blended = combine_signals({"carry": carry, "momentum": momentum})
    assert blended.loc[index[0], "A"] > 0
    assert blended.loc[index[0], "C"] < 0


def test_zero_weighted_factor_is_excluded_entirely():
    index = pd.to_datetime(["2024-01-02"])
    carry = pd.DataFrame([{"A": 3.0, "B": 0.0, "C": -3.0}], index=index)
    value = pd.DataFrame([{"A": -9.0, "B": 0.0, "C": 9.0}], index=index)

    only_carry = combine_signals({"carry": carry, "value": value}, {"carry": 1.0, "value": 0.0})
    assert only_carry.loc[index[0], "A"] > 0   # value would have flipped the sign


# ---- portfolio wiring ----

def test_portfolio_is_dollar_neutral_and_lags_its_weights():
    index = _days("2024-01-01", 6)
    rates = pd.DataFrame(
        {"AUD": [5.0] * 6, "NZD": [4.0] * 6, "JPY": [0.0] * 6, "CHF": [-1.0] * 6, "USD": [2.0] * 6},
        index=index,
    )
    spot = pd.DataFrame({c: [1.0] * 6 for c in ["AUD", "NZD", "JPY", "CHF"]}, index=index)
    reer = pd.DataFrame({c: [100.0] * 6 for c in ["AUD", "NZD", "JPY", "CHF"]}, index=index)

    result, signals = factor_portfolio(spot, rates, reer, basket=1, cost_bps=0.0)

    assert set(signals) == {"carry", "momentum", "value"}
    assert result.weights.sum(axis=1).abs().max() < 1e-12   # dollar-neutral
    assert result.weights.iloc[0].abs().sum() == 0.0        # nothing held on day one
    # Only carry is warm here, so the book must be long the top yielder.
    assert result.weights["AUD"].iloc[-1] > 0
    assert result.weights["CHF"].iloc[-1] < 0


def test_portfolio_returns_empty_when_there_is_no_overlap():
    empty = pd.DataFrame()
    result, signals = factor_portfolio(empty, pd.DataFrame({"USD": []}), empty)
    assert result.returns.empty and signals == {}


def test_factor_correlations_are_symmetric_with_unit_diagonal():
    index = _days("2024-01-01", 30)
    rng = np.random.default_rng(0)
    streams = {
        "carry": pd.Series(rng.normal(size=30), index=index),
        "momentum": pd.Series(rng.normal(size=30), index=index),
    }
    corr = factor_correlations(streams)
    assert abs(corr.loc["carry", "carry"] - 1.0) < 1e-12
    assert abs(corr.loc["carry", "momentum"] - corr.loc["momentum", "carry"]) < 1e-12


def test_monthly_rebalance_holds_the_book_between_marks():
    # A signal that flips every day must still trade only at the month boundary.
    index = _days("2024-01-01", 40)
    flipping = pd.DataFrame(
        {"A": [1.0, -1.0] * 20, "B": [0.0] * 40, "C": [-1.0, 1.0] * 20}, index=index
    )
    weights = rank_weights(flipping, basket=1)

    daily = hold_between_rebalances(weights, freq=None)
    monthly = hold_between_rebalances(weights, freq="MS")

    assert gross_turnover(daily).mean() > gross_turnover(monthly).mean() * 5
    assert monthly.sum(axis=1).abs().max() < 1e-12   # still dollar-neutral
