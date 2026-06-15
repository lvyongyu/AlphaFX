from __future__ import annotations

import numpy as np
import pandas as pd

# Engineered features used as model inputs. The forward-return label is built
# separately and is never included here (no target leakage). Macro factors are
# already publication-lagged by FeatureAgent (A3), so the dataset is point-in-time.
FEATURE_COLUMNS = [
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


def build_targets(features: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Add the forward-return target and its up/down label.

    audusd_future_return_20d at t is the return from t to t+horizon (the trailing
    20d return as-of t+horizon). The last `horizon` rows have no future and get a
    NaN target so they are dropped from training.
    """
    df = features.assign(date=pd.to_datetime(features["date"])).sort_values("date").reset_index(drop=True)
    df["audusd_future_return_20d"] = df["audusd_return_20d"].shift(-horizon)
    df["target_up_20d"] = np.where(df["audusd_future_return_20d"] > 0, 1.0, 0.0)
    df.loc[df["audusd_future_return_20d"].isna(), "target_up_20d"] = np.nan
    return df


def build_dataset(
    features: pd.DataFrame,
    horizon: int = 20,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Return (X, y, dates, manifest).

    The manifest lists only the features actually used — any factor that is
    entirely unavailable (e.g. macro not downloaded) is excluded explicitly
    rather than dropping every row. Rows with an unavailable target or any used
    feature missing are removed.
    """
    if features is None or features.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]"), []
    df = build_targets(features, horizon=horizon)
    candidate = feature_columns or FEATURE_COLUMNS
    manifest = [c for c in candidate if c in df.columns and df[c].notna().any()]
    if not manifest:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]"), []
    clean = df.dropna(subset=manifest + ["target_up_20d"]).reset_index(drop=True)
    X = clean[manifest].reset_index(drop=True)
    y = clean["target_up_20d"].astype(float).reset_index(drop=True)
    dates = pd.to_datetime(clean["date"]).reset_index(drop=True)
    return X, y, dates, manifest
