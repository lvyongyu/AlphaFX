from __future__ import annotations

import pandas as pd
import yfinance as yf


class YFinanceProvider:
    source = "yfinance"

    def download(self, symbol: str, start: object, end: object) -> pd.DataFrame:
        raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(raw.index).tz_localize(None).date,
                "symbol": symbol,
                "open": raw.get("Open"),
                "high": raw.get("High"),
                "low": raw.get("Low"),
                "close": raw.get("Close"),
                "source": self.source,
            }
        ).dropna(subset=["close"])

