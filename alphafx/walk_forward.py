from __future__ import annotations

import pandas as pd

from .backtest import BacktestAgent
from .diagnostics import SignalDiagnosticsAgent
from .signals import QuantSignalAgent


class WalkForwardAgent:
    def run(
        self,
        market_data: pd.DataFrame,
        features: pd.DataFrame,
        train_days: int = 252 * 3,
        test_days: int = 126,
        holding_period: int = 20,
    ) -> pd.DataFrame:
        if features.empty:
            return pd.DataFrame()
        features = features.assign(date=pd.to_datetime(features["date"])).sort_values("date").reset_index(drop=True)
        rows: list[dict[str, object]] = []
        start_idx = train_days
        signal_agent = QuantSignalAgent()
        diagnostics = SignalDiagnosticsAgent()
        backtester = BacktestAgent()
        while start_idx < len(features) - holding_period:
            train = features.iloc[start_idx - train_days : start_idx].copy()
            test = features.iloc[start_idx : min(start_idx + test_days, len(features))].copy()
            if test.empty:
                break
            # Internal validation signals must not overwrite the live signals table.
            train_signals = signal_agent.generate_signals(train, persist=False)
            calibration = diagnostics.make_walkforward_calibration(market_data, train_signals, horizon=holding_period, min_samples=10)
            test_signals = signal_agent.generate_signals(test, calibration=calibration, persist=False)
            _, metrics = backtester.run(market_data, test_signals, test["date"].min(), test["date"].max(), holding_period=holding_period)
            _, in_metrics = backtester.run(market_data, train_signals, train["date"].min(), train["date"].max(), holding_period=holding_period)
            rows.append(
                {
                    "train_start": train["date"].min(),
                    "train_end": train["date"].max(),
                    "test_start": test["date"].min(),
                    "test_end": test["date"].max(),
                    "in_sample_sharpe": in_metrics["sharpe"],
                    "out_sample_sharpe": metrics["sharpe"],
                    # Only meaningful when in-sample Sharpe is positive; a negative
                    # denominator would flip the ratio's sign and invert its meaning.
                    "degradation_ratio": (
                        metrics["sharpe"] / in_metrics["sharpe"] if in_metrics["sharpe"] > 0 else float("nan")
                    ),
                    "out_sample_return": metrics["total_return"],
                    "out_sample_drawdown": metrics["max_drawdown"],
                    "trades": metrics["number_of_trades"],
                }
            )
            start_idx += test_days
        return pd.DataFrame(rows)
