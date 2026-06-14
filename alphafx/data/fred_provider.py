from __future__ import annotations

import pandas as pd


class FREDProvider:
    source = "FRED"
    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(self, series_id: str, symbol: str, frequency: str) -> None:
        self.series_id = series_id
        self.symbol = symbol
        self.frequency = frequency

    def download(self, start: object, end: object | None = None) -> pd.DataFrame:
        url = f"{self.base_url}?id={self.series_id}"
        raw = pd.read_csv(url)
        return self.parse(raw, start=start, end=end)

    def parse(self, raw: pd.DataFrame, start: object | None = None, end: object | None = None) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        df.columns = [str(c).strip() for c in df.columns]
        value_column = self.series_id if self.series_id in df.columns else df.columns[-1]
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["observation_date"]),
                "symbol": self.symbol,
                "value": pd.to_numeric(df[value_column], errors="coerce"),
                "source": self.source,
                "frequency": self.frequency,
            }
        ).dropna(subset=["value"])
        if start is not None:
            out = out[out["date"] >= pd.to_datetime(start)]
        if end is not None:
            out = out[out["date"] <= pd.to_datetime(end)]
        return out

