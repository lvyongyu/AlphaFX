from __future__ import annotations

from typing import Protocol

import pandas as pd


class MarketDataProvider(Protocol):
    def download(self, symbol: str, start: object, end: object) -> pd.DataFrame:
        """Return normalized OHLC rows with date, symbol, open, high, low, close, source."""


class MacroDataProvider(Protocol):
    def download(self, start: object, end: object | None = None) -> pd.DataFrame:
        """Return normalized macro rows with date, symbol, value, source, frequency."""

