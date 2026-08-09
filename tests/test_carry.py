"""Cross-sectional carry tests — fully offline, synthetic panels, no network.

The properties worth defending here are the ones that would silently manufacture
alpha if they broke: publication lag, weight lag, and dollar neutrality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphafx.carry import (
    PUBLICATION_LAG_MONTHS,
    build_rate_panel,
    build_spot_panel,
    cross_sectional_carry,
    decompose,
    excess_returns,
    performance,
    rank_weights,
    rate_symbol,
    single_pair_carry,
    yearly_breakdown,
)


def _macro(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": d, "symbol": s, "value": v, "source": "TEST", "frequency": "monthly"} for d, s, v in rows]
    )


def _days(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


# ---- no look-ahead ----

def test_rate_panel_withholds_a_month_until_it_is_published():
    # A rate describing January is not known during January. With a 2-month lag
    # it may only be used from March onwards.
    macro = _macro([("2024-01-01", rate_symbol("AUD"), 4.0)])
    index = pd.to_datetime(["2024-01-15", "2024-02-15", "2024-03-01", "2024-04-01"])

    panel = build_rate_panel(macro, index)

    assert np.isnan(panel.loc["2024-01-15", "AUD"])   # during the month itself
    assert np.isnan(panel.loc["2024-02-15", "AUD"])   # still inside the lag
    assert panel.loc["2024-03-01", "AUD"] == 4.0      # released
    assert panel.loc["2024-04-01", "AUD"] == 4.0      # and carried forward


def test_publication_lag_is_configurable_and_zero_lag_is_visible_immediately():
    macro = _macro([("2024-01-01", rate_symbol("AUD"), 4.0)])
    index = pd.to_datetime(["2024-01-01", "2024-01-15"])

    assert build_rate_panel(macro, index, lag_months=0).loc["2024-01-15", "AUD"] == 4.0
    assert PUBLICATION_LAG_MONTHS >= 1  # the default must never be "see it instantly"


def test_rate_panel_keeps_the_latest_value_per_month():
    macro = _macro([
        ("2024-01-01", rate_symbol("AUD"), 4.0),
        ("2024-02-01", rate_symbol("AUD"), 5.0),
    ])
    panel = build_rate_panel(macro, pd.to_datetime(["2024-03-15", "2024-04-15"]))
    assert panel.loc["2024-03-15", "AUD"] == 4.0   # only January is out yet
    assert panel.loc["2024-04-15", "AUD"] == 5.0   # February lands a month later


def test_empty_macro_returns_an_empty_panel_not_an_error():
    assert build_rate_panel(pd.DataFrame(), _days("2024-01-01", 3)).empty


# ---- quote conventions ----

def test_usd_base_pairs_are_inverted_to_usd_per_unit():
    # USD/JPY quotes JPY per USD, so 100 -> 0.01 USD per JPY. AUD/USD already
    # quotes USD per AUD and must be left alone.
    market = pd.DataFrame([
        {"date": "2024-01-01", "symbol": "USDJPY=X", "close": 100.0},
        {"date": "2024-01-01", "symbol": "AUDUSD=X", "close": 0.65},
    ])
    spot = build_spot_panel(market, currencies=["JPY", "AUD"])

    assert spot.loc["2024-01-01", "JPY"] == 0.01
    assert spot.loc["2024-01-01", "AUD"] == 0.65


def test_a_currency_with_no_price_rows_is_skipped_not_faked():
    market = pd.DataFrame([{"date": "2024-01-01", "symbol": "AUDUSD=X", "close": 0.65}])
    spot = build_spot_panel(market, currencies=["AUD", "NZD"])
    assert list(spot.columns) == ["AUD"]


# ---- portfolio construction ----

def test_rank_weights_are_dollar_neutral_and_pick_the_extremes():
    carry = pd.DataFrame(
        [{"AUD": 3.0, "NZD": 2.0, "EUR": 0.0, "JPY": -1.0}],
        index=pd.to_datetime(["2024-01-02"]),
    )
    weights = rank_weights(carry, basket=1)

    assert weights.loc["2024-01-02", "AUD"] == 1.0    # highest carry
    assert weights.loc["2024-01-02", "JPY"] == -1.0   # lowest carry
    assert weights.loc["2024-01-02", ["NZD", "EUR"]].tolist() == [0.0, 0.0]
    assert weights.sum(axis=1).abs().max() < 1e-12    # dollar-neutral


def test_rank_weights_stay_flat_when_the_universe_is_too_small():
    # Three currencies cannot fill two long and two short slots; a partially
    # invested book would not be dollar-neutral, so the day is skipped.
    carry = pd.DataFrame(
        [{"AUD": 3.0, "EUR": 1.0, "JPY": -1.0}], index=pd.to_datetime(["2024-01-02"])
    )
    assert rank_weights(carry, basket=2).abs().sum().sum() == 0.0


def test_rank_weights_ignore_currencies_with_no_rate_yet():
    carry = pd.DataFrame(
        [{"AUD": 3.0, "NZD": np.nan, "JPY": -1.0}], index=pd.to_datetime(["2024-01-02"])
    )
    weights = rank_weights(carry, basket=1)
    assert weights.loc["2024-01-02", "AUD"] == 1.0
    assert weights.loc["2024-01-02", "JPY"] == -1.0
    assert weights.loc["2024-01-02", "NZD"] == 0.0


def test_excess_return_is_spot_move_plus_the_rate_differential():
    index = pd.to_datetime(["2024-01-01", "2024-01-02"])
    spot = pd.DataFrame({"AUD": [1.0, 1.10]}, index=index)          # +10% spot
    rates = pd.DataFrame({"AUD": [5.04, 5.04], "USD": [2.52, 2.52]}, index=index)

    # 2.52 percentage points of differential over 252 days = 1bp/day.
    result = excess_returns(spot, rates)
    assert abs(result.loc["2024-01-02", "AUD"] - (0.10 + 0.0001)) < 1e-12


# ---- the lag that matters most ----

def test_todays_return_is_earned_by_yesterdays_weights():
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    # AUD carry is high from the start, JPY low, so the book is long AUD/short JPY.
    rates = pd.DataFrame(
        {"AUD": [5.0] * 3, "JPY": [-1.0] * 3, "USD": [2.0] * 3}, index=index
    )
    # AUD jumps on day 2. A book that used day-2's own signal would bank it;
    # ours must be positioned from day 1 to earn it.
    spot = pd.DataFrame({"AUD": [1.0, 1.0, 1.05], "JPY": [1.0, 1.0, 1.0]}, index=index)

    result = cross_sectional_carry(spot, rates, basket=1, cost_bps=0.0)

    assert result.returns.loc["2024-01-02"] == 0.0        # weights not yet on
    assert result.returns.loc["2024-01-03"] > 0.04        # earned with a day's lag
    assert result.weights.loc["2024-01-02", "AUD"] == 0.0


def test_costs_reduce_returns_and_scale_with_turnover():
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    rates = pd.DataFrame({"AUD": [5.0] * 3, "JPY": [-1.0] * 3, "USD": [2.0] * 3}, index=index)
    spot = pd.DataFrame({"AUD": [1.0, 1.01, 1.02], "JPY": [1.0, 1.0, 1.0]}, index=index)

    free = cross_sectional_carry(spot, rates, basket=1, cost_bps=0.0)
    charged = cross_sectional_carry(spot, rates, basket=1, cost_bps=10.0)

    assert charged.returns.sum() < free.returns.sum()
    assert free.turnover > 0.0


def test_single_pair_carry_flips_direction_one_day_after_the_differential():
    index = _days("2024-01-01", 5)
    # AUD yields above USD for three days, then below.
    rates = pd.DataFrame({"AUD": [5.0, 5.0, 5.0, 1.0, 1.0], "USD": [2.0] * 5}, index=index)
    spot = pd.DataFrame({"AUD": [1.0] * 5}, index=index)

    result = single_pair_carry(spot, rates, currency="AUD", cost_bps=0.0)
    held, carry = result.weights["AUD"], result.carry["AUD"]

    assert carry.loc[index[3]] == -1.0        # the signal flips on day 4
    assert held.loc[index[2]] == 1.0          # long while carry was positive
    assert held.loc[index[3]] == 1.0          # still yesterday's sign on the flip day
    assert held.loc[index[4]] == -1.0         # short only from the following day


# ---- reporting ----

def test_performance_reports_zero_for_an_empty_series():
    assert performance(pd.Series(dtype=float))["sharpe"] == 0.0


def test_performance_reports_zero_for_a_constant_series():
    # Zero variance means Sharpe is undefined, not infinite. Returning 0 keeps a
    # degenerate window from printing a spectacular ratio.
    assert performance(pd.Series([0.001] * 20, index=_days("2024-01-01", 20)))["sharpe"] == 0.0


def test_performance_sharpe_is_positive_for_a_rising_series():
    returns = pd.Series([0.002, 0.001] * 25, index=_days("2024-01-01", 50))
    stats = performance(returns)
    assert stats["sharpe"] > 0 and stats["return_pct"] > 0 and stats["hit_rate"] == 1.0


def test_t_stat_scales_with_the_length_of_the_record():
    # The same Sharpe over a longer record is stronger evidence. Two years of the
    # same pattern must produce a larger t than one.
    short = pd.Series([0.002, 0.001] * 126, index=_days("2024-01-01", 252))
    long = pd.Series([0.002, 0.001] * 252, index=_days("2024-01-01", 504))

    short_stats, long_stats = performance(short), performance(long)
    # Same pattern, so the Sharpes match bar the ddof=1 correction on sample size.
    assert abs(short_stats["sharpe"] / long_stats["sharpe"] - 1.0) < 0.01
    # Twice the record should be ~sqrt(2) more evidence.
    assert long_stats["t_stat"] > short_stats["t_stat"] * 1.3


def test_decompose_separates_interest_from_spot():
    index = _days("2024-01-01", 6)
    # AUD pays 2.52pp over USD (1bp/day) and its spot is flat; JPY is the mirror.
    # So the whole return must land in the interest line, not the spot line.
    rates = pd.DataFrame({"AUD": [5.04] * 6, "JPY": [-0.04] * 6, "USD": [2.52] * 6}, index=index)
    spot = pd.DataFrame({"AUD": [1.0] * 6, "JPY": [1.0] * 6}, index=index)

    result = cross_sectional_carry(spot, rates, basket=1, cost_bps=0.0)
    parts = decompose(result, spot, rates)

    assert parts["interest_pct"] > 0.0
    assert abs(parts["spot_pct"]) < 1e-9
    assert abs(parts["total_pct"] - parts["interest_pct"]) < 1e-6


def test_decompose_attributes_a_pure_spot_move_to_the_spot_line():
    index = _days("2024-01-01", 6)
    rates = pd.DataFrame({"AUD": [2.52] * 6, "JPY": [2.52] * 6, "USD": [2.52] * 6}, index=index)
    spot = pd.DataFrame({"AUD": [1.0, 1.0, 1.02, 1.02, 1.02, 1.02], "JPY": [1.0] * 6}, index=index)

    parts = decompose(cross_sectional_carry(spot, rates, basket=1, cost_bps=0.0), spot, rates)
    assert abs(parts["interest_pct"]) < 1e-9   # no differential to earn
    assert parts["spot_pct"] != 0.0


def test_yearly_breakdown_splits_by_calendar_year():
    index = pd.to_datetime(["2023-06-01", "2023-07-03", "2024-06-03", "2024-07-01"])
    table = yearly_breakdown(pd.Series([0.02, 0.01, -0.02, -0.01], index=index))
    assert list(table.index) == [2023, 2024]
    assert table.loc[2023, "return_pct"] > 0 > table.loc[2024, "return_pct"]
