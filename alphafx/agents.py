from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_SYMBOLS
from .data.fred_provider import FREDProvider
from .data.rba_provider import RBAProvider
from .data.yfinance_provider import YFinanceProvider
from .database import Database


def _to_date_column(index: pd.Index) -> pd.Series:
    return pd.to_datetime(index).tz_localize(None).date


def _score_positive(value: float | None) -> int | float:
    if pd.isna(value):
        return np.nan
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _score_negative(value: float | None) -> int | float:
    if pd.isna(value):
        return np.nan
    if value < 0:
        return 1
    if value > 0:
        return -1
    return 0


class DataAgent:
    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()
        self.market_provider = YFinanceProvider()

    def download_market_data(
        self,
        start: date | str,
        end: date | str,
        symbols: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        symbols = symbols or {
            "audusd": DEFAULT_SYMBOLS.audusd,
            "dxy": DEFAULT_SYMBOLS.dxy,
            "vix": DEFAULT_SYMBOLS.vix,
        }
        frames: list[pd.DataFrame] = []
        for symbol in symbols.values():
            frame = self.market_provider.download(symbol, start, end)
            if not frame.empty:
                frames.append(frame)
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self.db.upsert_market_data(data)
        return data

    def download_macro_data(self, start: date | str, end: date | str | None = None) -> pd.DataFrame:
        providers = [
            FREDProvider("DGS2", "US2Y", "daily"),
            FREDProvider("PIORECRUSDM", "IRON_ORE", "monthly"),
        ]
        frames: list[pd.DataFrame] = []
        for provider in providers:
            try:
                frame = provider.download(start, end)
                if not frame.empty:
                    frames.append(frame)
            except Exception:
                continue
        try:
            au2y = RBAProvider().download_au2y(start, end)
            if not au2y.empty:
                frames.append(au2y)
        except Exception:
            pass
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self.db.upsert_macro_data(data)
        return data

    def load_market_data(self) -> pd.DataFrame:
        return self.db.load_market_data([DEFAULT_SYMBOLS.audusd, DEFAULT_SYMBOLS.dxy, DEFAULT_SYMBOLS.vix])

    def load_macro_data(self) -> pd.DataFrame:
        return self.db.load_macro_data(["US2Y", "AU2Y", "IRON_ORE"])

    def completeness_report(self, market_data: pd.DataFrame) -> pd.DataFrame:
        if market_data.empty:
            return pd.DataFrame(columns=["symbol", "first_date", "last_date", "rows", "missing_close"])
        return (
            market_data.groupby("symbol")
            .agg(
                first_date=("date", "min"),
                last_date=("date", "max"),
                rows=("close", "size"),
                missing_close=("close", lambda s: int(s.isna().sum())),
            )
            .reset_index()
        )

    def macro_status_report(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        if macro_data.empty:
            return pd.DataFrame(columns=["symbol", "source", "frequency", "latest_date", "rows", "status"])
        today = pd.Timestamp.today().normalize()
        report = (
            macro_data.groupby("symbol")
            .agg(
                source=("source", "last"),
                frequency=("frequency", "last"),
                latest_date=("date", "max"),
                rows=("value", "size"),
            )
            .reset_index()
        )
        report["age_days"] = (today - pd.to_datetime(report["latest_date"])).dt.days
        report["status"] = np.where(report["age_days"] > 45, "stale", "available")
        return report


class FeatureAgent:
    # Macro series are revised; using the revised value on its observation date is
    # look-ahead. Apply a conservative publication lag (in business days) so each
    # macro value only enters features once it would actually have been available.
    YIELD_PUBLICATION_LAG_DAYS = 1
    IRON_ORE_PUBLICATION_LAG_DAYS = 21

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()

    def build_features(
        self,
        market_data: pd.DataFrame,
        macro_data: pd.DataFrame | None = None,
        au2y: pd.DataFrame | None = None,
        us2y: pd.DataFrame | None = None,
        iron_ore: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if market_data.empty:
            return pd.DataFrame()
        wide = (
            market_data.assign(date=pd.to_datetime(market_data["date"]))
            .pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
            .sort_index()
            .ffill()
        )
        features = pd.DataFrame(index=wide.index)
        aud = wide.get(DEFAULT_SYMBOLS.audusd)
        dxy = wide.get(DEFAULT_SYMBOLS.dxy)
        vix = wide.get(DEFAULT_SYMBOLS.vix)

        features["audusd_return_20d"] = aud.pct_change(20, fill_method=None) if aud is not None else np.nan
        features["audusd_return_60d"] = aud.pct_change(60, fill_method=None) if aud is not None else np.nan
        features["audusd_vol_20d"] = aud.pct_change(fill_method=None).rolling(20).std() * np.sqrt(252) if aud is not None else np.nan
        features["dxy_return_20d"] = dxy.pct_change(20, fill_method=None) if dxy is not None else np.nan
        features["dxy_return_60d"] = dxy.pct_change(60, fill_method=None) if dxy is not None else np.nan
        features["vix_level"] = vix if vix is not None else np.nan
        features["vix_change_20d"] = vix.diff(20) if vix is not None else np.nan

        if macro_data is not None and not macro_data.empty:
            au2y = au2y if au2y is not None else self._macro_symbol_frame(macro_data, "AU2Y")
            us2y = us2y if us2y is not None else self._macro_symbol_frame(macro_data, "US2Y")
            iron_ore = iron_ore if iron_ore is not None else self._macro_symbol_frame(macro_data, "IRON_ORE")

        spread = self._build_yield_spread(features.index, au2y, us2y, lag_days=self.YIELD_PUBLICATION_LAG_DAYS)
        features["yield_spread"] = spread
        features["yield_spread_change_20d"] = spread.diff(20) if spread is not None else np.nan

        iron = self._align_optional_series(features.index, iron_ore, lag_days=self.IRON_ORE_PUBLICATION_LAG_DAYS)
        features["ironore_return_20d"] = iron.pct_change(20, fill_method=None) if iron is not None else np.nan

        features = features.reset_index().rename(columns={"index": "date"})
        self.db.save_features(features)
        return features

    def _macro_symbol_frame(self, macro_data: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
        subset = macro_data[macro_data["symbol"] == symbol].copy()
        if subset.empty:
            return None
        return subset.rename(columns={"value": "value"})[["date", "value"]]

    def factor_table(self, latest_feature: pd.Series, latest_signal: pd.Series) -> pd.DataFrame:
        rows = [
            ("AUD momentum", latest_feature.get("audusd_return_20d"), latest_feature.get("audusd_return_20d"), latest_signal.get("aud_momentum_score")),
            ("DXY trend", latest_feature.get("dxy_return_20d"), latest_feature.get("dxy_return_20d"), latest_signal.get("dxy_score")),
            ("Yield spread", latest_feature.get("yield_spread"), latest_feature.get("yield_spread_change_20d"), latest_signal.get("yield_score")),
            ("Iron ore trend", latest_feature.get("ironore_return_20d"), latest_feature.get("ironore_return_20d"), latest_signal.get("ironore_score")),
            ("VIX", latest_feature.get("vix_level"), latest_feature.get("vix_change_20d"), latest_signal.get("vix_score")),
        ]
        table = pd.DataFrame(rows, columns=["factor", "current_value", "change_20d", "contribution"])
        table["stance"] = table["contribution"].map({1: "bullish", -1: "bearish", 0: "neutral"}).fillna("not available")
        return table

    def _build_yield_spread(
        self,
        index: pd.DatetimeIndex,
        au2y: pd.DataFrame | None,
        us2y: pd.DataFrame | None,
        lag_days: int = 0,
    ) -> pd.Series | None:
        au = self._align_optional_series(index, au2y, lag_days=lag_days)
        us = self._align_optional_series(index, us2y, lag_days=lag_days)
        if au is None or us is None:
            return None
        return au - us

    def _align_optional_series(
        self, index: pd.DatetimeIndex, frame: pd.DataFrame | None, lag_days: int = 0
    ) -> pd.Series | None:
        if frame is None or frame.empty:
            return None
        df = frame.copy()
        df.columns = [str(c).lower().strip() for c in df.columns]
        if "date" not in df.columns:
            return None
        value_column = next((c for c in ["close", "yield", "price", "value"] if c in df.columns), None)
        if value_column is None:
            numeric = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
            value_column = numeric[0] if numeric else None
        if value_column is None:
            return None
        series = pd.Series(df[value_column].values, index=pd.to_datetime(df["date"])).sort_index()
        if lag_days:
            # An observation only becomes available `lag_days` business days later.
            series.index = series.index + pd.offsets.BusinessDay(lag_days)
        return series.reindex(index).ffill()


class QuantSignalAgent:
    probability_map = {5: 0.70, 4: 0.65, 3: 0.60, 2: 0.55, 1: 0.52, 0: 0.50, -1: 0.48, -2: 0.45, -3: 0.40, -4: 0.35, -5: 0.30}

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()

    def generate_signals(self, features: pd.DataFrame, calibration: pd.DataFrame | dict[str, float] | None = None) -> pd.DataFrame:
        # `calibration` accepts two shapes with very different look-ahead profiles:
        #   - DataFrame  -> expanding-window, per-date calibrated probability
        #                   (SignalDiagnosticsAgent.calibration_frame). Safe for
        #                   the live/latest signal. Labelled "historical_calibration".
        #   - dict       -> a single full-sample hit rate per signal class. This is
        #                   IN-SAMPLE and only valid on walk-forward TRAIN data
        #                   (make_walkforward_calibration). Labelled
        #                   "walkforward_calibration" so it can never be mistaken
        #                   for a backward-looking live probability.
        if features.empty:
            return pd.DataFrame()
        out = features[["date"]].copy()
        out["aud_momentum_score"] = features["audusd_return_20d"].apply(_score_positive)
        out["dxy_score"] = features["dxy_return_20d"].apply(_score_negative)
        out["yield_score"] = features["yield_spread_change_20d"].apply(_score_positive)
        out["ironore_score"] = features["ironore_return_20d"].apply(_score_positive)
        out["vix_score"] = features["vix_change_20d"].apply(_score_negative)
        score_cols = ["aud_momentum_score", "dxy_score", "yield_score", "ironore_score", "vix_score"]
        out["score"] = out[score_cols].sum(axis=1, min_count=1)
        out["signal"] = out["score"].apply(self.map_signal)
        out["fallback_probability"] = out["score"].apply(self.map_probability)
        out["probability"] = out["fallback_probability"]
        out["probability_source"] = "fallback_score_map"
        out["calibration_sample_size"] = 0
        if isinstance(calibration, dict) and calibration:
            mapped = out["signal"].map(calibration)
            calibrated = mapped.notna()
            out.loc[calibrated, "probability"] = mapped[calibrated]
            # In-sample walk-forward calibration only — never a live probability.
            out.loc[calibrated, "probability_source"] = "walkforward_calibration"
            out.loc[calibrated, "calibration_sample_size"] = -1
        elif isinstance(calibration, pd.DataFrame) and not calibration.empty:
            out = out.drop(columns=["calibration_sample_size"]).merge(
                calibration[["date", "calibrated_probability", "calibration_sample_size"]], on="date", how="left"
            )
            out["calibration_sample_size"] = out["calibration_sample_size"].fillna(0).astype(int)
            calibrated = out["calibrated_probability"].notna()
            out.loc[calibrated, "probability"] = out.loc[calibrated, "calibrated_probability"]
            out.loc[calibrated, "probability_source"] = "historical_calibration"
            out = out.drop(columns=["calibrated_probability"])
        out["confidence"] = out.apply(lambda row: self.map_confidence(row["score"], row[score_cols].notna().sum()), axis=1)
        self.db.save_signals(out)
        return out

    def latest_signal(self, signals: pd.DataFrame) -> pd.Series:
        valid = signals.dropna(subset=["score"])
        return valid.iloc[-1] if not valid.empty else pd.Series(dtype=object)

    @classmethod
    def map_signal(cls, score: float) -> str:
        if pd.isna(score):
            return "neutral"
        if score >= 3:
            return "bullish"
        if score <= -3:
            return "bearish"
        return "neutral"

    @classmethod
    def map_probability(cls, score: float) -> float:
        if pd.isna(score):
            return 0.50
        rounded = int(max(-5, min(5, round(score))))
        return cls.probability_map[rounded]

    @staticmethod
    def map_confidence(score: float, available_factors: int) -> str:
        if pd.isna(score) or available_factors < 3:
            return "Low"
        if abs(score) >= 4 and available_factors >= 4:
            return "High"
        if abs(score) >= 2:
            return "Medium"
        return "Low"


class SignalDiagnosticsAgent:
    def forward_return_diagnostics(
        self,
        market_data: pd.DataFrame,
        signals: pd.DataFrame,
        horizons: tuple[int, ...] = (20, 40, 60),
    ) -> pd.DataFrame:
        data = self._signal_price_frame(market_data, signals)
        if data.empty:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        for horizon in horizons:
            frame = data.copy()
            frame["forward_return"] = frame["close"].shift(-horizon) / frame["close"] - 1.0
            frame = frame.dropna(subset=["signal", "forward_return"])
            for signal, group in frame.groupby("signal"):
                returns = group["forward_return"].dropna()
                directional = self.directional_outcome(signal, returns)
                strategy = self.strategy_returns(signal, returns)
                # Daily-overlapping h-day forward returns are autocorrelated, so
                # the raw count overstates significance. Discount to independent
                # observations (~ N / horizon) and report an overlap-adjusted
                # t-stat alongside the naive one.
                effective_n = int(len(returns) // max(horizon, 1))
                t_naive, t_adjusted = self.mean_t_stats(strategy, effective_n)
                rows.append(
                    {
                        "signal": signal,
                        "horizon": horizon,
                        "sample_size": int(len(returns)),
                        "effective_sample_size": effective_n,
                        "average_forward_return": float(returns.mean()) if len(returns) else 0.0,
                        "median_forward_return": float(returns.median()) if len(returns) else 0.0,
                        "hit_rate": float(directional.mean()) if len(directional) else 0.0,
                        "win_loss_ratio": self.win_loss_ratio(returns),
                        "max_drawdown": BacktestAgent.max_drawdown_from_returns(strategy),
                        "sharpe": BacktestAgent.sharpe(strategy),
                        "profit_factor": BacktestAgent.profit_factor(strategy),
                        "mean_t_stat": t_naive,
                        "mean_t_stat_adjusted": t_adjusted,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def mean_t_stats(returns: pd.Series, effective_n: int) -> tuple[float, float]:
        """t-stat of the mean: naive (every obs independent) vs overlap-adjusted.

        The adjusted t-stat uses the independent count instead of the raw count,
        so it never overstates significance more than the naive one.
        """
        returns = returns.dropna()
        n = len(returns)
        std = float(returns.std(ddof=1)) if n > 1 else 0.0
        if n < 2 or std == 0.0:
            return 0.0, 0.0
        mean = float(returns.mean())
        t_naive = mean / (std / np.sqrt(n))
        eff = max(1, min(effective_n, n))
        t_adjusted = mean / (std / np.sqrt(eff))
        return float(t_naive), float(t_adjusted)

    def calibration_frame(
        self,
        market_data: pd.DataFrame,
        signals: pd.DataFrame,
        horizon: int = 20,
        min_samples: int = 20,
    ) -> pd.DataFrame:
        data = self._signal_price_frame(market_data, signals)
        if data.empty:
            return pd.DataFrame(columns=["date", "calibrated_probability", "calibration_sample_size"])
        data["forward_return"] = data["close"].shift(-horizon) / data["close"] - 1.0
        probabilities: list[dict[str, object]] = []
        for idx, row in data.iterrows():
            if row["signal"] == "neutral":
                continue
            known = data.iloc[: max(0, idx - horizon + 1)].dropna(subset=["forward_return", "signal"])
            known = known[known["signal"] == row["signal"]]
            if len(known) < min_samples:
                continue
            hits = self.directional_outcome(str(row["signal"]), known["forward_return"])
            probabilities.append(
                {
                    "date": row["date"],
                    "calibrated_probability": float(hits.mean()),
                    "calibration_sample_size": int(len(hits)),
                }
            )
        return pd.DataFrame(probabilities)

    def make_walkforward_calibration(
        self,
        market_data: pd.DataFrame,
        signals: pd.DataFrame,
        horizon: int = 20,
        min_samples: int = 10,
    ) -> dict[str, float]:
        """Full-sample hit rate per signal class — WALK-FORWARD TRAIN DATA ONLY.

        This computes the hit rate over the entire `signals` frame it is given,
        so it is in-sample by construction. It is only valid when `signals` is a
        walk-forward training slice (no future relative to the test window). Never
        use it to set the probability of the latest/live signal — use the
        expanding-window `calibration_frame` for that.
        """
        diagnostics = self.forward_return_diagnostics(market_data, signals, horizons=(horizon,))
        if diagnostics.empty:
            return {}
        diagnostics = diagnostics[diagnostics["signal"].isin(["bullish", "bearish"])]
        eligible = diagnostics[diagnostics["sample_size"] >= min_samples]
        return {str(row["signal"]): float(row["hit_rate"]) for _, row in eligible.iterrows()}

    def _signal_price_frame(self, market_data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        aud = (
            market_data[market_data["symbol"] == DEFAULT_SYMBOLS.audusd]
            .assign(date=lambda x: pd.to_datetime(x["date"]))
            .sort_values("date")[["date", "close"]]
            .reset_index(drop=True)
        )
        if aud.empty or signals.empty:
            return pd.DataFrame()
        sig_cols = ["date", "signal", "score", "probability"]
        sig = signals.assign(date=lambda x: pd.to_datetime(x["date"]))[[c for c in sig_cols if c in signals.columns]]
        return aud.merge(sig, on="date", how="left").ffill().reset_index(drop=True)

    def strategy_returns(self, signal: str, returns: pd.Series) -> pd.Series:
        if signal == "bullish":
            return returns
        if signal == "bearish":
            return -returns
        return pd.Series(np.zeros(len(returns)), index=returns.index)

    def directional_outcome(self, signal: str, returns: pd.Series) -> pd.Series:
        if signal == "bullish":
            return returns > 0
        if signal == "bearish":
            return returns < 0
        return returns.abs() <= 0.005

    def win_loss_ratio(self, returns: pd.Series) -> float:
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if wins.empty or losses.empty:
            return 0.0
        return float(abs(wins.mean() / losses.mean()))


@dataclass
class RiskSuggestion:
    action: str
    position_size: str
    leverage: float
    stop_loss: float
    take_profit: float
    warning: str
    max_risk_per_trade: float = 0.01
    regime: str = "normal"


class RiskAgent:
    def suggest(
        self,
        signal: str,
        probability: float,
        volatility: float | None,
        max_drawdown: float | None = None,
        user_leverage: float = 2.0,
        max_risk_per_trade: float = 0.01,
    ) -> RiskSuggestion:
        leverage = min(max(float(user_leverage), 1.0), 5.0)
        high_vol = pd.notna(volatility) and volatility > 0.18
        extreme_vol = pd.notna(volatility) and volatility > 0.25
        if high_vol:
            leverage = min(leverage, 2.0)
        if extreme_vol:
            action = "NO TRADE"
            leverage = 1.0
        elif signal == "bullish" and probability >= 0.60:
            action = "BUY AUD/USD"
        elif signal == "bearish" and probability <= 0.40:
            action = "SELL AUD/USD"
        else:
            action = "NO TRADE"
            leverage = 1.0
        warning_parts = ["Paper trading only. No live order execution is implemented."]
        if extreme_vol:
            warning_parts.append("Extreme volatility regime: no-trade guard is active.")
        if high_vol:
            warning_parts.append("Volatility is elevated, so leverage is capped.")
        if max_drawdown is not None and pd.notna(max_drawdown) and max_drawdown < -0.15:
            warning_parts.append("Backtest drawdown is material; reduce size or avoid the trade.")
        vol_stop = max(0.01, min(0.06, float(volatility) / 5.0)) if pd.notna(volatility) else 0.02
        return RiskSuggestion(
            action=action,
            position_size="Small" if high_vol or action == "NO TRADE" else "Standard",
            leverage=leverage,
            stop_loss=vol_stop,
            take_profit=vol_stop * 2.0,
            warning=" ".join(warning_parts),
            max_risk_per_trade=max_risk_per_trade,
            regime="extreme_vol_no_trade" if extreme_vol else "elevated_vol" if high_vol else "normal",
        )


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
        return {
            "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
            "profit_factor": BacktestAgent.profit_factor(returns),
            "number_of_trades": int(len(returns)),
            "average_trade": float(returns.mean()) if len(returns) else 0.0,
            "best_trade": float(returns.max()) if len(returns) else 0.0,
            "worst_trade": float(returns.min()) if len(returns) else 0.0,
        }

    @staticmethod
    def benchmark_metrics(data: pd.DataFrame, holding_period: int = 20) -> dict[str, float]:
        daily = data["daily_return"].fillna(0.0)
        close = data["close"]
        rng = np.random.default_rng(42)
        random_position = pd.Series(rng.choice([-1.0, 0.0, 1.0], len(data)), index=data.index).shift(1).fillna(0.0)
        sma_fast = close.rolling(20).mean()
        sma_slow = close.rolling(60).mean()
        sma_position = pd.Series(np.where(sma_fast > sma_slow, 1.0, -1.0), index=data.index).shift(1).fillna(0.0)
        momentum_position = pd.Series(np.where(close.pct_change(holding_period) > 0, 1.0, -1.0), index=data.index).shift(1).fillna(0.0)
        return {
            "benchmark_buy_hold_return": float((1.0 + daily).prod() - 1.0),
            "benchmark_flat_return": 0.0,
            "benchmark_random_return": float((1.0 + random_position * daily).prod() - 1.0),
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
        }


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
            train_signals = signal_agent.generate_signals(train)
            calibration = diagnostics.make_walkforward_calibration(market_data, train_signals, horizon=holding_period, min_samples=10)
            test_signals = signal_agent.generate_signals(test, calibration=calibration)
            bt, metrics = backtester.run(market_data, test_signals, test["date"].min(), test["date"].max(), holding_period=holding_period)
            in_bt, in_metrics = backtester.run(market_data, train_signals, train["date"].min(), train["date"].max(), holding_period=holding_period)
            rows.append(
                {
                    "train_start": train["date"].min(),
                    "train_end": train["date"].max(),
                    "test_start": test["date"].min(),
                    "test_end": test["date"].max(),
                    "in_sample_sharpe": in_metrics["sharpe"],
                    "out_sample_sharpe": metrics["sharpe"],
                    "degradation_ratio": metrics["sharpe"] / in_metrics["sharpe"] if in_metrics["sharpe"] else 0.0,
                    "out_sample_return": metrics["total_return"],
                    "out_sample_drawdown": metrics["max_drawdown"],
                    "trades": metrics["number_of_trades"],
                }
            )
            start_idx += test_days
        return pd.DataFrame(rows)


class FactorDiagnosticsAgent:
    feature_columns = [
        "audusd_return_20d",
        "audusd_return_60d",
        "audusd_vol_20d",
        "dxy_return_20d",
        "dxy_return_60d",
        "vix_level",
        "vix_change_20d",
        "yield_spread",
        "yield_spread_change_20d",
        "ironore_return_20d",
    ]

    def analyze(self, features: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
        if features.empty:
            return pd.DataFrame()
        df = features.assign(date=pd.to_datetime(features["date"])).sort_values("date").copy()
        df["future_return"] = df["audusd_return_20d"].shift(-horizon)
        rows: list[dict[str, object]] = []
        for column in [c for c in self.feature_columns if c in df.columns]:
            pair = df[[column, "future_return"]].dropna()
            if len(pair) < 10:
                continue
            corr = float(pair[column].corr(pair["future_return"]))
            sign = np.sign(pair[column])
            target = np.sign(pair["future_return"])
            n = int(len(pair))
            # Overlapping h-day forward returns inflate IC significance; discount
            # to independent observations (~ N / horizon) for the adjusted t-stat.
            effective_n = max(1, n // max(horizon, 1))
            ic_t = self._ic_t_stat(corr, n)
            ic_t_adjusted = self._ic_t_stat(corr, effective_n)
            rows.append(
                {
                    "factor": column,
                    "sample_size": n,
                    "effective_sample_size": effective_n,
                    "information_coefficient": corr,
                    "coefficient_proxy": corr,
                    "ic_t_stat": ic_t,
                    "ic_t_stat_adjusted": ic_t_adjusted,
                    "hit_contribution": float((sign == target).mean()),
                    "stability": self._rolling_stability(pair[column], pair["future_return"]),
                    "feature_importance": abs(corr),
                }
            )
        return pd.DataFrame(rows).sort_values("feature_importance", ascending=False) if rows else pd.DataFrame()

    @staticmethod
    def _ic_t_stat(ic: float, n: int) -> float:
        if n is None or n < 3 or pd.isna(ic):
            return 0.0
        denom = 1.0 - ic * ic
        if denom <= 1e-9:
            return 0.0
        return float(ic * np.sqrt(n - 2) / np.sqrt(denom))

    def _rolling_stability(self, factor: pd.Series, target: pd.Series, window: int = 126) -> float:
        if len(factor) < window * 2:
            return 0.0
        corrs = []
        for start in range(0, len(factor) - window + 1, window):
            corr = factor.iloc[start : start + window].corr(target.iloc[start : start + window])
            if pd.notna(corr):
                corrs.append(np.sign(corr))
        return float(np.mean(corrs)) if corrs else 0.0


class PaperJournalAgent:
    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()

    def record_signal(
        self,
        latest_feature: pd.Series,
        latest_signal: pd.Series,
        factor_table: pd.DataFrame,
        risk: RiskSuggestion,
        explanation: str,
        audusd_price: float | None = None,
    ) -> None:
        if latest_signal.empty:
            return
        date_value = pd.to_datetime(latest_signal["date"]).strftime("%Y-%m-%d")
        row = {
            "date": date_value,
            "audusd_price": audusd_price,
            "signal": latest_signal.get("signal"),
            "score": latest_signal.get("score"),
            "calibrated_probability": latest_signal.get("probability"),
            "factor_values": json.dumps(latest_feature.drop(labels=["date"], errors="ignore").to_dict(), default=str),
            "factor_contributions": factor_table.to_json(orient="records"),
            "recommended_position": risk.action,
            "stop_loss": risk.stop_loss,
            "take_profit": risk.take_profit,
            "explanation": explanation,
            "entry_price": None,
            "exit_price": None,
            "realised_pnl": None,
            "status": "open" if risk.action != "NO TRADE" else "no_trade",
        }
        self.db.upsert_paper_journal(row)

    def load(self) -> pd.DataFrame:
        return self.db.load_paper_journal()


class AIExplanationAgent:
    def explain(self, signal: pd.Series, factors: pd.DataFrame) -> str:
        if signal.empty:
            return "No signal is available yet. Download data and generate features first."
        bullish = factors[factors["stance"] == "bullish"]["factor"].tolist()
        bearish = factors[factors["stance"] == "bearish"]["factor"].tolist()
        missing = factors[factors["stance"] == "not available"]["factor"].tolist()
        text = [
            f"The quant model is {signal['signal']} with a {signal['probability']:.0%} probability and {signal['confidence'].lower()} confidence.",
            f"The raw score is {signal['score']:.0f}, based on available factor contributions.",
        ]
        if bullish:
            text.append("Bullish support comes from " + ", ".join(bullish) + ".")
        if bearish:
            text.append("Bearish pressure comes from " + ", ".join(bearish) + ".")
        if missing:
            text.append("Missing factors are reported as unavailable: " + ", ".join(missing) + ".")
        return " ".join(text)


class ContrarianAgent:
    def critique(self, signal: pd.Series, factors: pd.DataFrame) -> dict[str, str]:
        if signal.empty:
            return {"main_risk": "No signal to critique.", "alternative_scenario": "", "watch": ""}
        bearish = factors[factors["stance"] == "bearish"]["factor"].tolist()
        bullish = factors[factors["stance"] == "bullish"]["factor"].tolist()
        if signal["signal"] == "bullish":
            risk = "The bullish case could fail if USD strength returns or risk sentiment deteriorates."
            opposing = bearish or ["DXY, VIX, yields, or commodity trends"]
            scenario = "A reversal in " + ", ".join(opposing) + " would weaken the long AUD/USD setup."
        elif signal["signal"] == "bearish":
            risk = "The bearish case could fail if global risk appetite improves or commodities strengthen."
            opposing = bullish or ["AUD momentum, yield spread, or iron ore"]
            scenario = "Improvement in " + ", ".join(opposing) + " would challenge the short AUD/USD setup."
        else:
            risk = "Neutral signals can hide regime changes because factor disagreement is high."
            scenario = "A cleaner trend in DXY, VIX, yields, or iron ore would move the model away from neutral."
        return {
            "main_risk": risk,
            "alternative_scenario": scenario,
            "watch": "Watch the next 20 trading days of DXY, VIX, AU-US yield spread, iron ore, and AUD/USD momentum.",
        }


class JudgeAgent:
    def judge(self, signal: pd.Series, risk: RiskSuggestion, explanation: str, contrarian: dict[str, str]) -> dict[str, Any]:
        if signal.empty:
            return {
                "final_signal": "neutral",
                "final_confidence": "Low",
                "trade": "NO TRADE",
                "explanation": "No complete signal is available.",
            }
        return {
            "final_signal": signal["signal"],
            "final_confidence": signal["confidence"],
            "trade": risk.action,
            "explanation": f"{explanation} Contrarian view: {contrarian['main_risk']}",
        }
