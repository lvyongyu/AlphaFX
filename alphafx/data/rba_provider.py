from __future__ import annotations

import pandas as pd


class RBAProvider:
    source = "RBA"
    f2_url = "https://www.rba.gov.au/statistics/tables/csv/f2-data.csv"

    def download_au2y(self, start: object, end: object | None = None) -> pd.DataFrame:
        raw = pd.read_csv(self.f2_url, skiprows=10)
        return self.parse_au2y(raw, start=start, end=end)

    def parse_au2y(self, raw: pd.DataFrame, start: object | None = None, end: object | None = None) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        df.columns = [str(c).strip() for c in df.columns]
        date_col = next((c for c in df.columns if c.lower() in {"date", "series id"}), df.columns[0])
        candidates = [c for c in df.columns if "2" in c and ("year" in c.lower() or "yr" in c.lower()) and "government" in c.lower()]
        value_col = candidates[0] if candidates else self._first_numeric_column(df, exclude={date_col})
        if value_col is None:
            return pd.DataFrame()
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df[date_col], errors="coerce"),
                "symbol": "AU2Y",
                "value": pd.to_numeric(df[value_col], errors="coerce"),
                "source": self.source,
                "frequency": "daily",
            }
        ).dropna(subset=["date", "value"])
        if start is not None:
            out = out[out["date"] >= pd.to_datetime(start)]
        if end is not None:
            out = out[out["date"] <= pd.to_datetime(end)]
        return out

    def _first_numeric_column(self, df: pd.DataFrame, exclude: set[str]) -> str | None:
        for column in df.columns:
            if column in exclude:
                continue
            values = pd.to_numeric(df[column], errors="coerce")
            if values.notna().sum() > 0:
                return column
        return None

