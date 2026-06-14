from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .config import DEFAULT_SYMBOLS
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
            raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            frame = pd.DataFrame(
                {
                    "date": _to_date_column(raw.index),
                    "symbol": symbol,
                    "open": raw.get("Open"),
                    "high": raw.get("High"),
                    "low": raw.get("Low"),
                    "close": raw.get("Close"),
                    "source": "yfinance",
                }
            ).dropna(subset=["close"])
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self.db.upsert_market_data(data)
        return data

    def load_market_data(self) -> pd.DataFrame:
        return self.db.load_market_data([DEFAULT_SYMBOLS.audusd, DEFAULT_SYMBOLS.dxy, DEFAULT_SYMBOLS.vix])

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


class FeatureAgent:
    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()

    def build_features(
        self,
        market_data: pd.DataFrame,
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

        spread = self._build_yield_spread(features.index, au2y, us2y)
        features["yield_spread"] = spread
        features["yield_spread_change_20d"] = spread.diff(20) if spread is not None else np.nan

        iron = self._align_optional_series(features.index, iron_ore)
        features["ironore_return_20d"] = iron.pct_change(20, fill_method=None) if iron is not None else np.nan

        features = features.reset_index().rename(columns={"index": "date"})
        self.db.save_features(features)
        return features

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
    ) -> pd.Series | None:
        au = self._align_optional_series(index, au2y)
        us = self._align_optional_series(index, us2y)
        if au is None or us is None:
            return None
        return au - us

    def _align_optional_series(self, index: pd.DatetimeIndex, frame: pd.DataFrame | None) -> pd.Series | None:
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
        return series.reindex(index).ffill()


class QuantSignalAgent:
    probability_map = {5: 0.70, 4: 0.65, 3: 0.60, 2: 0.55, 1: 0.52, 0: 0.50, -1: 0.48, -2: 0.45, -3: 0.40, -4: 0.35, -5: 0.30}

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
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
        out["probability"] = out["score"].apply(self.map_probability)
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


@dataclass
class RiskSuggestion:
    action: str
    position_size: str
    leverage: float
    stop_loss: float
    take_profit: float
    warning: str


class RiskAgent:
    def suggest(
        self,
        signal: str,
        probability: float,
        volatility: float | None,
        max_drawdown: float | None = None,
        user_leverage: float = 2.0,
    ) -> RiskSuggestion:
        leverage = min(max(float(user_leverage), 1.0), 5.0)
        high_vol = pd.notna(volatility) and volatility > 0.18
        if high_vol:
            leverage = min(leverage, 2.0)
        if signal == "bullish" and probability >= 0.60:
            action = "BUY AUD/USD"
        elif signal == "bearish" and probability <= 0.40:
            action = "SELL AUD/USD"
        else:
            action = "NO TRADE"
            leverage = 1.0
        warning_parts = ["Paper trading only. No live order execution is implemented."]
        if high_vol:
            warning_parts.append("Volatility is elevated, so leverage is capped.")
        if max_drawdown is not None and pd.notna(max_drawdown) and max_drawdown < -0.15:
            warning_parts.append("Backtest drawdown is material; reduce size or avoid the trade.")
        return RiskSuggestion(
            action=action,
            position_size="Small" if high_vol or action == "NO TRADE" else "Standard",
            leverage=leverage,
            stop_loss=0.02,
            take_profit=0.04,
            warning=" ".join(warning_parts),
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
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        aud = (
            market_data[market_data["symbol"] == DEFAULT_SYMBOLS.audusd]
            .assign(date=lambda x: pd.to_datetime(x["date"]))
            .sort_values("date")[["date", "close"]]
        )
        sig = signals.assign(date=lambda x: pd.to_datetime(x["date"]))[["date", "signal"]]
        data = aud.merge(sig, on="date", how="left").ffill()
        data = data[(data["date"] >= pd.to_datetime(start_date)) & (data["date"] <= pd.to_datetime(end_date))].copy()
        if data.empty:
            return data, self.empty_metrics()

        data["daily_return"] = data["close"].pct_change().fillna(0.0)
        data["position"] = data["signal"].map({"bullish": 1.0, "bearish": -1.0}).fillna(0.0)
        data["position"] = data["position"].shift(1).fillna(0.0)
        data["trade"] = data["position"].diff().abs().fillna(data["position"].abs())
        cost = data["trade"] * (transaction_cost_bps / 10000.0)
        data["strategy_return"] = data["position"] * data["daily_return"] * leverage - cost
        data["equity"] = (1.0 + data["strategy_return"]).cumprod()
        data["drawdown"] = data["equity"] / data["equity"].cummax() - 1.0
        data["future_return"] = data["close"].shift(-holding_period) / data["close"] - 1.0
        data["trade_return"] = data["position"] * data["future_return"] * leverage
        metrics = self.calculate_metrics(data)
        return data, metrics

    @staticmethod
    def calculate_metrics(data: pd.DataFrame) -> dict[str, float]:
        if data.empty:
            return BacktestAgent.empty_metrics()
        total_return = data["equity"].iloc[-1] - 1.0
        days = max(len(data), 1)
        annualized_return = data["equity"].iloc[-1] ** (252 / days) - 1.0
        sharpe = BacktestAgent.sharpe(data["strategy_return"])
        max_drawdown = float(data["drawdown"].min())
        trades = data.loc[data["trade"] > 0, "trade_return"].dropna()
        wins = trades[trades > 0]
        losses = trades[trades < 0]
        profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else np.inf
        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "sharpe": float(sharpe),
            "max_drawdown": max_drawdown,
            "win_rate": float((trades > 0).mean()) if len(trades) else 0.0,
            "profit_factor": profit_factor,
            "number_of_trades": int(len(trades)),
            "average_trade": float(trades.mean()) if len(trades) else 0.0,
            "best_trade": float(trades.max()) if len(trades) else 0.0,
            "worst_trade": float(trades.min()) if len(trades) else 0.0,
        }

    @staticmethod
    def sharpe(returns: pd.Series) -> float:
        returns = returns.dropna()
        if returns.empty or returns.std(ddof=0) == 0:
            return 0.0
        return float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252))

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
