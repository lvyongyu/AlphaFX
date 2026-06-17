from __future__ import annotations

import numpy as np
import pandas as pd

from .database import Database


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

class QuantSignalAgent:
    # `probability` is the confidence that the SIGNAL DIRECTION is correct
    # (P(signal right)), NOT P(up). Keyed by |score| so a strong bearish signal
    # maps to a HIGH probability, exactly like the calibrated hit rate from
    # SignalDiagnosticsAgent. Keeping both producers on one convention lets the
    # fallback and calibrated probabilities be used interchangeably and lets
    # RiskAgent gate both directions on a single directional-confidence threshold.
    probability_map = {0: 0.50, 1: 0.52, 2: 0.55, 3: 0.60, 4: 0.65, 5: 0.70}

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()

    score_columns = ["aud_momentum_score", "dxy_score", "yield_score", "ironore_score", "vix_score"]

    def generate_signals(
        self,
        features: pd.DataFrame,
        calibration: pd.DataFrame | dict[str, float] | None = None,
        weights: dict[str, float] | None = None,
        persist: bool = True,
    ) -> pd.DataFrame:
        # `weights` is an OPTIONAL, experimental factor weighting. Default (None)
        # keeps the equal-weight rule signal unchanged — equal weight is robust and
        # hard to beat, and orthogonalization tends to hurt IC/returns, so the
        # default is deliberately left alone (see project memory). When provided,
        # weights are normalized to sum to len(factors) so the -5..+5 scale and the
        # +/-3 thresholds still hold; this path is for walk-forward-validated
        # comparison only, never silently the default.
        #
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
        score_cols = self.score_columns
        if weights:
            w = pd.Series(weights, dtype=float).reindex(score_cols).fillna(0.0)
            total = float(w.sum())
            if total > 0:
                w = w * (len(score_cols) / total)  # keep the -5..+5 scale
            out["score"] = out[score_cols].mul(w, axis=1).sum(axis=1, min_count=1)
        else:
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
        if persist:
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
        magnitude = int(min(5, abs(round(score))))
        return cls.probability_map[magnitude]

    @staticmethod
    def map_confidence(score: float, available_factors: int) -> str:
        if pd.isna(score) or available_factors < 3:
            return "Low"
        if abs(score) >= 4 and available_factors >= 4:
            return "High"
        if abs(score) >= 2:
            return "Medium"
        return "Low"
