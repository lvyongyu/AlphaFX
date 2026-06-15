from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_SYMBOLS
from .database import Database


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
