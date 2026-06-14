from __future__ import annotations

import numpy as np
import pandas as pd

from alphafx.agents import BacktestAgent, FeatureAgent, QuantSignalAgent
from alphafx.config import DEFAULT_SYMBOLS


def sample_market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=90)
    rows = []
    for i, dt in enumerate(dates):
        rows.extend(
            [
                {"date": dt, "symbol": DEFAULT_SYMBOLS.audusd, "open": 0.65, "high": 0.66, "low": 0.64, "close": 0.65 + i * 0.001, "source": "test"},
                {"date": dt, "symbol": DEFAULT_SYMBOLS.dxy, "open": 105, "high": 106, "low": 104, "close": 105 - i * 0.05, "source": "test"},
                {"date": dt, "symbol": DEFAULT_SYMBOLS.vix, "open": 15, "high": 16, "low": 14, "close": 18 - i * 0.02, "source": "test"},
            ]
        )
    return pd.DataFrame(rows)


def test_feature_calculation_core_returns():
    features = FeatureAgent().build_features(sample_market_data())
    latest = features.iloc[-1]
    assert latest["audusd_return_20d"] > 0
    assert latest["dxy_return_20d"] < 0
    assert latest["vix_change_20d"] < 0
    assert latest["audusd_vol_20d"] >= 0


def test_score_calculation_and_signal_mapping():
    features = FeatureAgent().build_features(sample_market_data())
    signals = QuantSignalAgent().generate_signals(features)
    latest = signals.dropna(subset=["score"]).iloc[-1]
    assert latest["aud_momentum_score"] == 1
    assert latest["dxy_score"] == 1
    assert latest["vix_score"] == 1
    assert latest["signal"] == "bullish"
    assert latest["probability"] == 0.60


def test_signal_mapping_thresholds():
    assert QuantSignalAgent.map_signal(3) == "bullish"
    assert QuantSignalAgent.map_signal(-3) == "bearish"
    assert QuantSignalAgent.map_signal(2) == "neutral"
    assert QuantSignalAgent.map_probability(5) == 0.70
    assert QuantSignalAgent.map_probability(-5) == 0.30


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

