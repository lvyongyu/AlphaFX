from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestAgent
from .config import DEFAULT_SYMBOLS


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
        # The target pair is whatever FX symbol the context loaded that isn't the
        # shared DXY/VIX macro context. A per-instrument context holds exactly one
        # such symbol, so this stays correct for any pair without threading the
        # symbol through every caller; AUD/USD is the fallback if it's ambiguous.
        target = [s for s in market_data["symbol"].unique() if s not in (DEFAULT_SYMBOLS.dxy, DEFAULT_SYMBOLS.vix)]
        target_symbol = target[0] if len(target) == 1 else DEFAULT_SYMBOLS.audusd
        aud = (
            market_data[market_data["symbol"] == target_symbol]
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

    # The raw feature behind each ±1 scoring component, with a display label.
    # DXY and VIX both proxy the USD/risk axis and are typically correlated —
    # the correlation matrix makes that visible (the double-count is monitored,
    # not "fixed" by orthogonalization, which tends to hurt IC/returns).
    scoring_features = {
        "aud_momentum_score": ("audusd_return_20d", "AUD momentum"),
        "dxy_score": ("dxy_return_20d", "DXY trend"),
        "yield_score": ("yield_spread_change_20d", "Yield spread"),
        "ironore_score": ("ironore_return_20d", "Iron ore"),
        "vix_score": ("vix_change_20d", "VIX"),
    }

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
                    "information_ratio": self._information_ratio(pair[column], pair["future_return"]),
                    "hit_contribution": float((sign == target).mean()),
                    "stability": self._rolling_stability(pair[column], pair["future_return"]),
                    "feature_importance": abs(corr),
                }
            )
        return pd.DataFrame(rows).sort_values("feature_importance", ascending=False) if rows else pd.DataFrame()

    def rolling_ic_table(
        self, features: pd.DataFrame, horizon: int = 20, window: int = 126
    ) -> pd.DataFrame:
        """Per-window IC for each scoring factor — exposes whether a factor's
        direction is stable over time or only worked in one regime.

        The signal hard-codes each factor's sign (e.g. "DXY up -> AUD down"). A
        factor whose IC flips sign across windows is a warning that its assumed
        direction is not reliable. Windows are non-overlapping blocks of `window`
        rows; each row's `information_coefficient` is the correlation of the
        point-in-time factor with the forward `horizon`-day AUD/USD return inside
        that block. Long format (factor, window, window_end, IC) so the UI can
        pivot it into a line/heatmap.
        """
        empty = pd.DataFrame(columns=["factor", "window", "window_end", "information_coefficient"])
        if features.empty:
            return empty
        df = features.assign(date=pd.to_datetime(features["date"])).sort_values("date").reset_index(drop=True)
        df["future_return"] = df["audusd_return_20d"].shift(-horizon)
        cols = {raw: label for _score_col, (raw, label) in self.scoring_features.items() if raw in df.columns}
        rows: list[dict[str, object]] = []
        for raw, label in cols.items():
            pair = df[[raw, "future_return", "date"]].dropna().reset_index(drop=True)
            for window_no, start in enumerate(range(0, len(pair) - window + 1, window), start=1):
                block = pair.iloc[start : start + window]
                corr = block[raw].corr(block["future_return"])
                if pd.notna(corr):
                    rows.append(
                        {
                            "factor": label,
                            "window": window_no,
                            "window_end": block["date"].iloc[-1],
                            "information_coefficient": float(corr),
                        }
                    )
        return pd.DataFrame(rows) if rows else empty

    def factor_correlation(self, features: pd.DataFrame) -> pd.DataFrame:
        """Correlation matrix among the five scoring features.

        Surfaces the DXY/VIX overlap (USD/risk axis) so the double-count is
        visible and monitored. We deliberately do NOT orthogonalize it away —
        empirically that tends to reduce IC and returns and is unstable.
        """
        if features.empty:
            return pd.DataFrame()
        cols = {raw: label for raw, label in (v for v in self.scoring_features.values()) if raw in features.columns}
        present = [raw for raw in cols]
        if len(present) < 2:
            return pd.DataFrame()
        corr = features[present].corr()
        corr = corr.rename(index=cols, columns=cols)
        return corr

    def ic_weights(self, features: pd.DataFrame, horizon: int = 20) -> dict[str, float]:
        """Experimental |IC|-based weights keyed by score column.

        In-sample by construction — only use inside walk-forward, and treat with
        caution: the +12.4% IC-weighting result in the literature comes from large
        equity cross-sections, while this model has only ~N/horizon independent
        observations, so IC weighting can overfit. Returns {} when unavailable
        (caller then falls back to equal weight).
        """
        diag = self.analyze(features, horizon=horizon)
        if diag.empty:
            return {}
        ic_by_feature = dict(zip(diag["factor"], diag["information_coefficient"].abs()))
        weights = {
            score_col: float(ic_by_feature.get(raw, 0.0))
            for score_col, (raw, _label) in self.scoring_features.items()
            if raw in ic_by_feature
        }
        return weights if any(v > 0 for v in weights.values()) else {}

    @staticmethod
    def _rolling_ics(factor: pd.Series, target: pd.Series, window: int = 126) -> list[float]:
        ics: list[float] = []
        for start in range(0, len(factor) - window + 1, window):
            corr = factor.iloc[start : start + window].corr(target.iloc[start : start + window])
            if pd.notna(corr):
                ics.append(float(corr))
        return ics

    def _information_ratio(self, factor: pd.Series, target: pd.Series, window: int = 126) -> float:
        ics = self._rolling_ics(factor, target, window)
        if len(ics) < 2:
            return 0.0
        std = float(np.std(ics, ddof=1))
        return float(np.mean(ics) / std) if std > 0 else 0.0

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
