from __future__ import annotations

import pandas as pd

from alphafx.config import DEFAULT_SYMBOLS


def sample_market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=160)
    rows = []
    for i, dt in enumerate(dates):
        rows.extend(
            [
                {"date": dt, "symbol": DEFAULT_SYMBOLS.audusd, "open": 0.65, "high": 0.66, "low": 0.64, "close": 0.65 + i * 0.001, "source": "test"},
                {"date": dt, "symbol": DEFAULT_SYMBOLS.dxy, "open": 105, "high": 106, "low": 104, "close": 105 - i * 0.05, "source": "test"},
                {"date": dt, "symbol": DEFAULT_SYMBOLS.vix, "open": 15, "high": 16, "low": 14, "close": 18 - i * 0.02, "source": "test"},
            ]
        )
    return pd.DataFrame(rows)


def _flat_trade_data(signal: str) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=30)
    return pd.DataFrame(
        {"date": dates, "close": [0.65] * 30, "signal": [signal] * 30, "daily_return": [0.0] * 30}
    )
