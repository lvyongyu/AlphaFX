from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .config import DEFAULT_SYMBOLS


class BacktestAgent:
    def run(
        self,
        market_data: pd.DataFrame,
        signals: pd.DataFrame,
        start_date: date | str,
        end_date: date | str,
        holding_period: int = 20,
        leverage: float = 2.0,
        transaction_cost_bps: float = 2.0,
        spread_bps: float = 0.0,
        slippage_bps: float = 0.0,
        rollover_bps_per_day: float = 0.0,
        swap_annual_pct: float = 0.0,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        aud = (
            market_data[market_data["symbol"] == DEFAULT_SYMBOLS.audusd]
            .assign(date=lambda x: pd.to_datetime(x["date"]))
            .sort_values("date")[["date", "close"]]
        )
        sig_cols = [c for c in ["date", "signal", "score", "probability"] if c in signals.columns]
        sig = signals.assign(date=lambda x: pd.to_datetime(x["date"]))[sig_cols]
        data = aud.merge(sig, on="date", how="left").ffill()
        data = data[(data["date"] >= pd.to_datetime(start_date)) & (data["date"] <= pd.to_datetime(end_date))].copy()
        if data.empty:
            return data, self.empty_metrics()

        data["daily_return"] = data["close"].pct_change().fillna(0.0)
        trades = self.build_trades(
            data.reset_index(drop=True),
            holding_period=holding_period,
            leverage=leverage,
            transaction_cost_bps=transaction_cost_bps,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            rollover_bps_per_day=rollover_bps_per_day,
            swap_annual_pct=swap_annual_pct,
        )
        data = self.apply_trades_to_daily_frame(data.reset_index(drop=True), trades, leverage=leverage)
        data["equity"] = (1.0 + data["strategy_return"]).cumprod()
        data["drawdown"] = data["equity"] / data["equity"].cummax() - 1.0
        data["future_return"] = data["close"].shift(-holding_period) / data["close"] - 1.0
        metrics = self.calculate_metrics(data)
        metrics.update(self.trade_metrics(trades))
        metrics.update(self.benchmark_metrics(data, holding_period=holding_period))
        self.last_trades = trades
        return data, metrics

    def build_trades(
        self,
        data: pd.DataFrame,
        holding_period: int,
        leverage: float,
        transaction_cost_bps: float,
        spread_bps: float,
        slippage_bps: float,
        rollover_bps_per_day: float,
        swap_annual_pct: float = 0.0,
    ) -> pd.DataFrame:
        trades: list[dict[str, object]] = []
        i = 0
        round_trip_cost = ((transaction_cost_bps + spread_bps + slippage_bps) * 2.0) / 10000.0
        daily_carry_rate = (swap_annual_pct / 100.0) / 252.0
        while i < len(data) - 1:
            signal = str(data.iloc[i].get("signal", "neutral"))
            position = 1.0 if signal == "bullish" else -1.0 if signal == "bearish" else 0.0
            if position == 0.0:
                i += 1
                continue
            entry_idx = i + 1
            exit_idx = min(entry_idx + int(holding_period), len(data) - 1)
            if entry_idx >= exit_idx:
                break
            entry = data.iloc[entry_idx]
            exit_ = data.iloc[exit_idx]
            holding_days = exit_idx - entry_idx
            raw_return = (float(exit_["close"]) / float(entry["close"]) - 1.0) * position
            # Broker swap markup is always a cost; carry (AU-US rate differential)
            # is signed by position: a long AUD earns positive carry when AU > US.
            broker_swap = (rollover_bps_per_day * holding_days) / 10000.0
            carry = position * daily_carry_rate * holding_days * leverage
            cost = round_trip_cost + broker_swap
            trades.append(
                {
                    "signal_date": data.iloc[i]["date"],
                    "entry_date": entry["date"],
                    "exit_date": exit_["date"],
                    "signal": signal,
                    "score": data.iloc[i].get("score", np.nan),
                    "probability": data.iloc[i].get("probability", np.nan),
                    "position": position,
                    "holding_period": int(holding_days),
                    "entry_price": float(entry["close"]),
                    "exit_price": float(exit_["close"]),
                    "raw_return": float(raw_return),
                    "transaction_cost": float(round_trip_cost),
                    "rollover_cost": float(broker_swap),
                    "carry": float(carry),
                    "cost": float(cost),
                    "realised_return": float(raw_return * leverage - cost + carry),
                    "status": "closed",
                }
            )
            i = exit_idx + 1
        return pd.DataFrame(trades)

    def apply_trades_to_daily_frame(self, data: pd.DataFrame, trades: pd.DataFrame, leverage: float = 1.0) -> pd.DataFrame:
        out = data.copy()
        out["position"] = 0.0
        out["trade"] = 0.0
        if trades.empty:
            out["strategy_return"] = 0.0
            return out
        for _, trade in trades.iterrows():
            mask = (out["date"] >= trade["entry_date"]) & (out["date"] <= trade["exit_date"])
            out.loc[mask, "position"] = float(trade["position"])
            out.loc[out["date"] == trade["exit_date"], "trade"] = 1.0
        out["strategy_return"] = out["position"].shift(1).fillna(0.0) * out["daily_return"] * leverage
        for _, trade in trades.iterrows():
            # Net the explicit costs out and the signed carry in, on the exit day.
            net = float(trade.get("carry", 0.0)) - float(trade["cost"])
            out.loc[out["date"] == trade["exit_date"], "strategy_return"] += net
        return out

    @staticmethod
    def yearly_returns(data: pd.DataFrame) -> pd.DataFrame:
        """Compounded strategy return per calendar year — exposes regime concentration.

        The per-year compounded returns multiply back to the full-window total
        return, so a strategy whose edge lives in one regime/year is visible.
        """
        if data.empty or "strategy_return" not in data.columns:
            return pd.DataFrame(columns=["year", "return"])
        d = data.assign(date=pd.to_datetime(data["date"]))
        d["year"] = d["date"].dt.year
        out = (
            d.groupby("year")["strategy_return"].apply(lambda r: float((1.0 + r).prod() - 1.0)).reset_index()
        )
        return out.rename(columns={"strategy_return": "return"})

    @staticmethod
    def calculate_metrics(data: pd.DataFrame) -> dict[str, float]:
        if data.empty:
            return BacktestAgent.empty_metrics()
        total_return = data["equity"].iloc[-1] - 1.0
        days = max(len(data), 1)
        annualized_return = data["equity"].iloc[-1] ** (252 / days) - 1.0
        sharpe = BacktestAgent.sharpe(data["strategy_return"])
        max_drawdown = float(data["drawdown"].min())
        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "sharpe": float(sharpe),
            "max_drawdown": max_drawdown,
        }

    @staticmethod
    def trade_metrics(trades: pd.DataFrame) -> dict[str, float]:
        if trades.empty:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "number_of_trades": 0,
                "average_trade": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
            }
        returns = trades["realised_return"].dropna()
        metrics = {
            "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
            "profit_factor": BacktestAgent.profit_factor(returns),
            "number_of_trades": int(len(returns)),
            "average_trade": float(returns.mean()) if len(returns) else 0.0,
            "best_trade": float(returns.max()) if len(returns) else 0.0,
            "worst_trade": float(returns.min()) if len(returns) else 0.0,
        }
        metrics.update(BacktestAgent.trade_significance(returns))
        return metrics

    @staticmethod
    def trade_significance(returns: pd.Series) -> dict[str, float]:
        """Small-sample uncertainty on the trade-level edge.

        Trades are non-overlapping (built with i = exit_idx + 1), so trade
        returns are ~independent and `number_of_trades` IS the effective sample
        size. With AUD/USD's 20-day holding period a 5-year window yields only
        ~60 trades, so any Sharpe is noisy — these fields make that explicit:

        - `avg_trade_t_stat`: t-stat that the mean trade return is > 0.
        - `sharpe_trade`: per-trade Sharpe (mean/std, not annualized).
        - `sharpe_trade_ci_low/high`: 95% CI via Lo (2002) SE
          ≈ sqrt((1 + 0.5*SR^2) / n). A CI straddling 0 means the edge is not
          distinguishable from zero at this sample size.
        """
        returns = returns.dropna()
        n = int(len(returns))
        std = float(returns.std(ddof=1)) if n > 1 else 0.0
        if n < 2 or std == 0.0:
            return {
                "avg_trade_t_stat": 0.0,
                "sharpe_trade": 0.0,
                "sharpe_trade_ci_low": 0.0,
                "sharpe_trade_ci_high": 0.0,
            }
        mean = float(returns.mean())
        t_stat = mean / (std / np.sqrt(n))
        sharpe_trade = mean / std
        se = np.sqrt((1.0 + 0.5 * sharpe_trade**2) / n)
        return {
            "avg_trade_t_stat": float(t_stat),
            "sharpe_trade": float(sharpe_trade),
            "sharpe_trade_ci_low": float(sharpe_trade - 1.96 * se),
            "sharpe_trade_ci_high": float(sharpe_trade + 1.96 * se),
        }

    @staticmethod
    def benchmark_metrics(data: pd.DataFrame, holding_period: int = 20, random_seeds: int = 200) -> dict[str, float]:
        daily = data["daily_return"].fillna(0.0)
        close = data["close"]
        rng = np.random.default_rng(42)
        random_position = pd.Series(rng.choice([-1.0, 0.0, 1.0], len(data)), index=data.index).shift(1).fillna(0.0)
        sma_fast = close.rolling(20).mean()
        sma_slow = close.rolling(60).mean()
        sma_position = pd.Series(np.where(sma_fast > sma_slow, 1.0, -1.0), index=data.index).shift(1).fillna(0.0)
        momentum_position = pd.Series(np.where(close.pct_change(holding_period) > 0, 1.0, -1.0), index=data.index).shift(1).fillna(0.0)

        # A single random draw is reproducible but is not a significance test.
        # Build a distribution over many fixed seeds and report the strategy's
        # percentile within it (the fraction of random baselines it beats).
        random_returns = np.array(
            [
                float((1.0 + pd.Series(np.random.default_rng(s).choice([-1.0, 0.0, 1.0], len(data)), index=data.index).shift(1).fillna(0.0) * daily).prod() - 1.0)
                for s in range(int(random_seeds))
            ]
        )
        strategy_total = float(data["equity"].iloc[-1] - 1.0) if "equity" in data.columns and len(data) else 0.0
        percentile = float((random_returns < strategy_total).mean()) if len(random_returns) else 0.0
        return {
            "benchmark_buy_hold_return": float((1.0 + daily).prod() - 1.0),
            "benchmark_flat_return": 0.0,
            "benchmark_random_return": float((1.0 + random_position * daily).prod() - 1.0),
            "benchmark_random_mean": float(random_returns.mean()) if len(random_returns) else 0.0,
            "benchmark_random_p05": float(np.percentile(random_returns, 5)) if len(random_returns) else 0.0,
            "benchmark_random_p95": float(np.percentile(random_returns, 95)) if len(random_returns) else 0.0,
            "strategy_vs_random_percentile": percentile,
            "benchmark_sma_return": float((1.0 + sma_position * daily).prod() - 1.0),
            "benchmark_momentum_return": float((1.0 + momentum_position * daily).prod() - 1.0),
        }

    @staticmethod
    def sharpe(returns: pd.Series) -> float:
        returns = returns.dropna()
        if returns.empty or returns.std(ddof=0) == 0:
            return 0.0
        return float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252))

    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        returns = returns.dropna()
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        return float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else np.inf

    @staticmethod
    def max_drawdown_from_returns(returns: pd.Series) -> float:
        returns = returns.dropna()
        if returns.empty:
            return 0.0
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        return float(drawdown.min())

    @staticmethod
    def empty_metrics() -> dict[str, float]:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "number_of_trades": 0,
            "average_trade": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "avg_trade_t_stat": 0.0,
            "sharpe_trade": 0.0,
            "sharpe_trade_ci_low": 0.0,
            "sharpe_trade_ci_high": 0.0,
        }
