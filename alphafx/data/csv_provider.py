from __future__ import annotations

import pandas as pd


class CSVProvider:
    source = "CSV"

    def parse_macro(self, frame: pd.DataFrame, symbol: str, frequency: str = "unknown") -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        df = frame.copy()
        df.columns = [str(c).lower().strip() for c in df.columns]
        if "date" not in df.columns:
            return pd.DataFrame()
        value_col = next((c for c in ["value", "close", "yield", "price"] if c in df.columns), None)
        if value_col is None:
            numeric = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
            value_col = numeric[0] if numeric else None
        if value_col is None:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "date": pd.to_datetime(df["date"]),
                "symbol": symbol,
                "value": pd.to_numeric(df[value_col], errors="coerce"),
                "source": self.source,
                "frequency": frequency,
            }
        ).dropna(subset=["date", "value"])
