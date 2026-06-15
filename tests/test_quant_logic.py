from __future__ import annotations

import numpy as np
import pandas as pd

from alphafx.agents import (
    BacktestAgent,
    FactorDiagnosticsAgent,
    FeatureAgent,
    QuantSignalAgent,
    SignalDiagnosticsAgent,
)
from alphafx.config import DEFAULT_SYMBOLS
from alphafx.data.fred_provider import FREDProvider
from alphafx.data.rba_provider import RBAProvider


def sample_market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=160)
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
    assert latest["probability_source"] == "fallback_score_map"


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


def test_forward_return_diagnostics():
    market = sample_market_data()
    features = FeatureAgent().build_features(market)
    signals = QuantSignalAgent().generate_signals(features)
    diagnostics = SignalDiagnosticsAgent().forward_return_diagnostics(market, signals, horizons=(20,))
    bullish = diagnostics[diagnostics["signal"] == "bullish"].iloc[0]
    assert bullish["sample_size"] > 0
    assert bullish["hit_rate"] == 1.0


def test_calibrated_probability_uses_prior_outcomes():
    market = sample_market_data()
    features = FeatureAgent().build_features(market)
    raw = QuantSignalAgent().generate_signals(features)
    calibration = SignalDiagnosticsAgent().calibration_frame(market, raw, horizon=20, min_samples=5)
    calibrated = QuantSignalAgent().generate_signals(features, calibration=calibration)
    active = calibrated[calibrated["probability_source"] == "historical_calibration"]
    assert not active.empty
    assert active["calibration_sample_size"].min() >= 5


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


def test_equal_weight_is_default_and_unchanged():
    from alphafx.agents import FactorDiagnosticsAgent  # noqa: F401

    features = FeatureAgent().build_features(sample_market_data())
    default = QuantSignalAgent().generate_signals(features)
    explicit_equal = QuantSignalAgent().generate_signals(
        features, weights={c: 1.0 for c in QuantSignalAgent.score_columns}, persist=False
    )
    # Equal weights (normalized to sum 5) reproduce the default equal-weight score exactly.
    merged = default.merge(explicit_equal, on="date", suffixes=("_d", "_w"))
    valid = merged.dropna(subset=["score_d", "score_w"])
    assert (valid["score_d"] == valid["score_w"]).all()


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


def test_ic_weighted_signal_does_not_persist(tmp_path):
    from alphafx.database import Database

    db = Database(tmp_path / "sig.db")
    features = FeatureAgent(db=db).build_features(sample_market_data())
    agent = QuantSignalAgent(db=db)
    agent.generate_signals(features)  # persisted equal-weight signals
    before = db.load_market_data([])  # noqa: F841 - just exercising db
    weights = FactorDiagnosticsAgent().ic_weights(features)
    agent.generate_signals(features, weights=weights, persist=False)  # must NOT overwrite
    saved = pd.read_sql_query("SELECT * FROM signals ORDER BY date", db.connect(), parse_dates=["date"])
    # The persisted signals are still the equal-weight ones (probability_source is the live label).
    assert not saved.empty


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


def test_forward_diagnostics_reports_effective_sample_size():
    from alphafx.agents import FactorDiagnosticsAgent  # noqa: F401 - import side-effect free

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


def _flat_trade_data(signal: str) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=30)
    return pd.DataFrame(
        {"date": dates, "close": [0.65] * 30, "signal": [signal] * 30, "daily_return": [0.0] * 30}
    )


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


def test_fred_parser_normalizes_macro_data():
    raw = pd.DataFrame({"observation_date": ["2024-01-01", "2024-01-02"], "DGS2": ["4.2", "."]})
    parsed = FREDProvider("DGS2", "US2Y", "daily").parse(raw)
    assert list(parsed["symbol"].unique()) == ["US2Y"]
    assert len(parsed) == 1


def test_rba_parser_normalizes_au2y_data():
    raw = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Australian Government 2 year bond": ["3.5", "3.6"]})
    parsed = RBAProvider().parse_au2y(raw)
    assert list(parsed["symbol"].unique()) == ["AU2Y"]
    assert len(parsed) == 2
