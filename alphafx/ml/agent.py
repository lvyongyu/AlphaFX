from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .dataset import build_dataset


def _make_model() -> tuple[str, Any]:
    """Deliberately low-capacity model for a small sample.

    XGBoost if available, else a regularized logistic regression (simple,
    interpretable, hard to overfit). The app stays deployable without xgboost.
    """
    try:  # pragma: no cover - depends on optional install
        from xgboost import XGBClassifier

        return "xgboost", XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8, eval_metric="logloss"
        )
    except Exception:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return "logreg", Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, C=0.5))]
        )


def ml_rule_agreement(rule_signal: str | None, ml_signal: str | None) -> str:
    if not ml_signal:
        return "ML signal unavailable."
    if not rule_signal:
        return "Rule signal unavailable."
    if rule_signal == ml_signal:
        return f"Rule and ML agree: both {rule_signal}."
    return f"Rule and ML disagree: rule is {rule_signal}, ML is {ml_signal}. The rule signal remains primary."


class MLSignalAgent:
    """Simple, leak-free ML comparison signal — research only, never the live signal."""

    def __init__(
        self,
        horizon: int = 20,
        upper: float = 0.55,
        lower: float = 0.45,
        n_splits: int = 5,
        min_effective_n: int = 30,
    ) -> None:
        self.horizon = horizon
        self.upper = upper
        self.lower = lower
        self.n_splits = n_splits
        self.min_effective_n = min_effective_n

    def map_probability_to_signal(self, p: float | None) -> str:
        if p is None or pd.isna(p):
            return "neutral"
        if p >= self.upper:
            return "bullish"
        if p <= self.lower:
            return "bearish"
        return "neutral"

    def effective_independent_n(self, n_obs: int) -> int:
        return int(n_obs // max(self.horizon, 1))

    @staticmethod
    def time_series_splits(n_samples: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_splits)
        return [(tr, te) for tr, te in tscv.split(np.arange(n_samples))]

    @staticmethod
    def _proba_up(model: Any, X: pd.DataFrame) -> np.ndarray:
        proba = model.predict_proba(X)
        classes = list(getattr(model, "classes_", [0.0, 1.0]))
        if 1.0 in classes:
            return proba[:, classes.index(1.0)]
        return np.full(len(X), 1.0 if (classes and classes[0] == 1.0) else 0.0)

    @staticmethod
    def _feature_importance(name: str, model: Any, manifest: list[str]) -> pd.DataFrame:
        if name == "xgboost":  # pragma: no cover - optional
            imp = np.asarray(model.feature_importances_, dtype=float)
        else:
            clf = model.named_steps["clf"]
            imp = np.abs(np.asarray(clf.coef_[0], dtype=float))
        return (
            pd.DataFrame({"feature": manifest, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def walk_forward_predict(self, features: pd.DataFrame) -> dict[str, Any]:
        """Leak-free out-of-sample probabilities via expanding time-series CV."""
        X, y, dates, manifest = build_dataset(features, horizon=self.horizon)
        result: dict[str, Any] = {
            "predictions": pd.DataFrame(columns=["date", "ml_probability", "ml_signal"]),
            "fold_metrics": pd.DataFrame(),
            "feature_importance": pd.DataFrame(),
            "training_samples": int(len(X)),
            "effective_n": self.effective_independent_n(len(X)),
            "manifest": manifest,
            "warning": None,
        }
        if len(X) < max(self.n_splits + 1, 40) or not manifest:
            result["warning"] = "Not enough clean data to train an ML model."
            return result

        from sklearn.metrics import accuracy_score, roc_auc_score

        preds: list[dict[str, Any]] = []
        fold_rows: list[dict[str, Any]] = []
        for fi, (tr, te) in enumerate(self.time_series_splits(len(X), self.n_splits)):
            X_tr, y_tr = X.iloc[tr], y.iloc[tr]
            X_te, y_te = X.iloc[te], y.iloc[te]
            if y_tr.nunique() < 2:
                p_up = np.full(len(te), float(y_tr.mean()))
            else:
                _, model = _make_model()
                model.fit(X_tr, y_tr)
                p_up = self._proba_up(model, X_te)
            for j, idx in enumerate(te):
                prob = float(p_up[j])
                preds.append(
                    {"date": dates.iloc[idx], "ml_probability": prob, "ml_signal": self.map_probability_to_signal(prob)}
                )
            y_hat = (p_up >= 0.5).astype(float)
            acc = float(accuracy_score(y_te, y_hat)) if len(y_te) else float("nan")
            try:
                auc = float(roc_auc_score(y_te, p_up)) if y_te.nunique() == 2 else float("nan")
            except Exception:
                auc = float("nan")
            fold_rows.append(
                {
                    "fold": fi + 1,
                    "train_size": int(len(tr)),
                    "test_size": int(len(te)),
                    "accuracy": acc,
                    "auc": auc,
                    "hit_rate": float((y_hat == y_te.values).mean()) if len(y_te) else float("nan"),
                }
            )

        result["predictions"] = pd.DataFrame(preds)
        result["fold_metrics"] = pd.DataFrame(fold_rows)
        if y.nunique() >= 2:
            name, model = _make_model()
            model.fit(X, y)
            result["feature_importance"] = self._feature_importance(name, model, manifest)
        if result["effective_n"] < self.min_effective_n:
            result["warning"] = (
                f"Only ~{result['effective_n']} independent observations "
                f"({result['training_samples']} rows / {self.horizon}d). ML overfits easily here — "
                "treat as a research comparison, not a better signal than the rule."
            )
        return result

    @staticmethod
    def to_signals(predictions: pd.DataFrame) -> pd.DataFrame:
        """Shape ML predictions like QuantSignalAgent output so BacktestAgent can run them.

        Predictions are out-of-sample only, so the resulting backtest is OOS by
        construction. The score is cosmetic — BacktestAgent positions off `signal`.
        """
        cols = ["date", "signal", "score", "probability"]
        if predictions is None or predictions.empty:
            return pd.DataFrame(columns=cols)
        out = predictions.rename(columns={"ml_signal": "signal", "ml_probability": "probability"}).copy()
        out["score"] = out["signal"].map({"bullish": 3.0, "bearish": -3.0, "neutral": 0.0})
        return out[cols]
