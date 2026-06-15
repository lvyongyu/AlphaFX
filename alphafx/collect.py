from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .config import DEFAULT_SYMBOLS
from .data.fred_provider import FREDProvider
from .data.rba_provider import RBAProvider
from .data.yfinance_provider import YFinanceProvider
from .database import Database


class DataAgent:
    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()
        self.market_provider = YFinanceProvider()

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
            frame = self.market_provider.download(symbol, start, end)
            if not frame.empty:
                frames.append(frame)
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self.db.upsert_market_data(data)
        return data

    def download_macro_data(self, start: date | str, end: date | str | None = None) -> pd.DataFrame:
        providers = [
            FREDProvider("DGS2", "US2Y", "daily"),
            FREDProvider("PIORECRUSDM", "IRON_ORE", "monthly"),
        ]
        frames: list[pd.DataFrame] = []
        for provider in providers:
            try:
                frame = provider.download(start, end)
                if not frame.empty:
                    frames.append(frame)
            except Exception:
                continue
        try:
            au2y = RBAProvider().download_au2y(start, end)
            if not au2y.empty:
                frames.append(au2y)
        except Exception:
            pass
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self.db.upsert_macro_data(data)
        return data

    def load_market_data(self) -> pd.DataFrame:
        return self.db.load_market_data([DEFAULT_SYMBOLS.audusd, DEFAULT_SYMBOLS.dxy, DEFAULT_SYMBOLS.vix])

    def load_macro_data(self) -> pd.DataFrame:
        return self.db.load_macro_data(["US2Y", "AU2Y", "IRON_ORE"])

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

    def macro_status_report(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        if macro_data.empty:
            return pd.DataFrame(columns=["symbol", "source", "frequency", "latest_date", "rows", "status"])
        today = pd.Timestamp.today().normalize()
        report = (
            macro_data.groupby("symbol")
            .agg(
                source=("source", "last"),
                frequency=("frequency", "last"),
                latest_date=("date", "max"),
                rows=("value", "size"),
            )
            .reset_index()
        )
        report["age_days"] = (today - pd.to_datetime(report["latest_date"])).dt.days
        report["status"] = np.where(report["age_days"] > 45, "stale", "available")
        return report
