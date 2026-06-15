from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphafx.agents import BacktestAgent, FeatureAgent, QuantSignalAgent

from factories import _flat_trade_data, sample_market_data


def test_long_return_calculation():
    returns = pd.Series([0.0, 0.01, 0.02])
    positions = pd.Series([1.0, 1.0, 1.0])
    strategy = positions * returns * 2.0
    assert strategy.iloc[-1] == 0.04


def test_short_return_calculation():
    returns = pd.Series([0.0, -0.01, -0.02])
    positions = pd.Series([-1.0, -1.0, -1.0])
    strategy = positions * returns * 2.0
    assert strategy.iloc[-1] == 0.04


def test_leverage_calculation_in_backtest():
    market = sample_market_data()
    features = FeatureAgent().build_features(market)
    signals = QuantSignalAgent().generate_signals(features)
    data_1x, metrics_1x = BacktestAgent().run(market, signals, "2024-02-01", "2024-04-30", leverage=1.0)
    data_2x, metrics_2x = BacktestAgent().run(market, signals, "2024-02-01", "2024-04-30", leverage=2.0)
    assert data_1x["strategy_return"].abs().sum() > 0
    assert data_2x["strategy_return"].abs().sum() > data_1x["strategy_return"].abs().sum()
    assert metrics_2x["total_return"] > metrics_1x["total_return"]


def test_drawdown_calculation():
    data = pd.DataFrame({"strategy_return": [0.0, 0.1, -0.1, -0.1]})
    data["equity"] = (1 + data["strategy_return"]).cumprod()
    data["drawdown"] = data["equity"] / data["equity"].cummax() - 1
    metrics = BacktestAgent.calculate_metrics(data.assign(trade=0, trade_return=np.nan))
    assert metrics["max_drawdown"] < 0


def test_sharpe_calculation():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    assert BacktestAgent.sharpe(returns) > 0
    assert BacktestAgent.sharpe(pd.Series([0.0, 0.0])) == 0.0


def test_trade_level_backtest_next_day_entry_and_costs():
    market = sample_market_data()
    features = FeatureAgent().build_features(market)
    signals = QuantSignalAgent().generate_signals(features)
    agent = BacktestAgent()
    _, metrics = agent.run(market, signals, "2024-02-01", "2024-07-01", holding_period=20, transaction_cost_bps=2, spread_bps=1, slippage_bps=1)
    trades = agent.last_trades
    assert not trades.empty
    first = trades.iloc[0]
    assert pd.to_datetime(first["entry_date"]) > pd.to_datetime(first["signal_date"])
    assert first["holding_period"] == 20
    assert first["cost"] > 0
    assert metrics["number_of_trades"] == len(trades)


def test_benchmark_random_is_deterministic():
    market = sample_market_data()
    features = FeatureAgent().build_features(market)
    signals = QuantSignalAgent().generate_signals(features)
    data_1, metrics_1 = BacktestAgent().run(market, signals, "2024-02-01", "2024-07-01")
    data_2, metrics_2 = BacktestAgent().run(market, signals, "2024-02-01", "2024-07-01")
    assert metrics_1["benchmark_random_return"] == metrics_2["benchmark_random_return"]
    assert len(data_1) == len(data_2)


def test_yearly_returns_compound_to_total():
    market = sample_market_data()
    features = FeatureAgent().build_features(market)
    signals = QuantSignalAgent().generate_signals(features)
    bt, metrics = BacktestAgent().run(market, signals, "2024-02-01", "2024-07-01")
    yearly = BacktestAgent.yearly_returns(bt)
    assert not yearly.empty
    compounded = float((1.0 + yearly["return"]).prod() - 1.0)
    assert compounded == pytest.approx(metrics["total_return"], rel=1e-9, abs=1e-9)


def test_random_benchmark_distribution_and_percentile():
    market = sample_market_data()
    features = FeatureAgent().build_features(market)
    signals = QuantSignalAgent().generate_signals(features)
    _, m1 = BacktestAgent().run(market, signals, "2024-02-01", "2024-07-01")
    _, m2 = BacktestAgent().run(market, signals, "2024-02-01", "2024-07-01")
    # Deterministic across runs (fixed seed sequence).
    assert m1["strategy_vs_random_percentile"] == m2["strategy_vs_random_percentile"]
    assert m1["benchmark_random_mean"] == m2["benchmark_random_mean"]
    # A valid probability, and the band is ordered.
    assert 0.0 <= m1["strategy_vs_random_percentile"] <= 1.0
    assert m1["benchmark_random_p05"] <= m1["benchmark_random_p95"]


def test_swap_carry_signs_by_direction():
    long_trades = BacktestAgent().build_trades(
        _flat_trade_data("bullish"), holding_period=10, leverage=1.0,
        transaction_cost_bps=0, spread_bps=0, slippage_bps=0, rollover_bps_per_day=0, swap_annual_pct=2.0,
    )
    short_trades = BacktestAgent().build_trades(
        _flat_trade_data("bearish"), holding_period=10, leverage=1.0,
        transaction_cost_bps=0, spread_bps=0, slippage_bps=0, rollover_bps_per_day=0, swap_annual_pct=2.0,
    )
    # Long AUD earns positive carry when the AU-US differential is positive; short pays it.
    assert long_trades.iloc[0]["carry"] > 0
    assert short_trades.iloc[0]["carry"] < 0
    assert long_trades.iloc[0]["realised_return"] > short_trades.iloc[0]["realised_return"]


def test_broker_swap_markup_reduces_return():
    base = BacktestAgent().build_trades(
        _flat_trade_data("bullish"), holding_period=10, leverage=1.0,
        transaction_cost_bps=0, spread_bps=0, slippage_bps=0, rollover_bps_per_day=0.0, swap_annual_pct=0.0,
    )
    costed = BacktestAgent().build_trades(
        _flat_trade_data("bullish"), holding_period=10, leverage=1.0,
        transaction_cost_bps=0, spread_bps=0, slippage_bps=0, rollover_bps_per_day=0.5, swap_annual_pct=0.0,
    )
    assert costed.iloc[0]["realised_return"] < base.iloc[0]["realised_return"]
