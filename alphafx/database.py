from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    source TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, symbol)
);

CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    audusd_return_20d REAL,
    audusd_return_60d REAL,
    audusd_vol_20d REAL,
    dxy_return_20d REAL,
    dxy_return_60d REAL,
    vix_level REAL,
    vix_change_20d REAL,
    yield_spread REAL,
    yield_spread_change_20d REAL,
    ironore_return_20d REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    signal TEXT,
    probability REAL,
    confidence TEXT,
    score REAL,
    aud_momentum_score REAL,
    dxy_score REAL,
    yield_score REAL,
    ironore_score REAL,
    vix_score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT,
    start_date TEXT,
    end_date TEXT,
    holding_period INTEGER,
    leverage REAL,
    transaction_cost_bps REAL,
    total_return REAL,
    annualized_return REAL,
    sharpe REAL,
    max_drawdown REAL,
    win_rate REAL,
    profit_factor REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS backtest_daily_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    date TEXT,
    signal TEXT,
    position REAL,
    daily_return REAL,
    strategy_return REAL,
    equity REAL,
    drawdown REAL,
    FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
);
"""


class Database:
    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def upsert_market_data(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        rows = frame[["date", "symbol", "open", "high", "low", "close", "source"]].to_records(index=False)
        sql = """
        INSERT INTO market_data (date, symbol, open, high, low, close, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, symbol) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            source = excluded.source
        """
        with self.connect() as conn:
            conn.executemany(sql, rows)

    def load_market_data(self, symbols: Iterable[str]) -> pd.DataFrame:
        symbols = list(symbols)
        if not symbols:
            return pd.DataFrame()
        placeholders = ",".join("?" for _ in symbols)
        with self.connect() as conn:
            return pd.read_sql_query(
                f"SELECT * FROM market_data WHERE symbol IN ({placeholders}) ORDER BY date",
                conn,
                params=symbols,
                parse_dates=["date"],
            )

    def save_features(self, features: pd.DataFrame) -> None:
        if features.empty:
            return
        columns = [
            "date",
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
        sql = f"""
        INSERT INTO features ({", ".join(columns)})
        VALUES ({", ".join("?" for _ in columns)})
        ON CONFLICT(date) DO UPDATE SET
            audusd_return_20d = excluded.audusd_return_20d,
            audusd_return_60d = excluded.audusd_return_60d,
            audusd_vol_20d = excluded.audusd_vol_20d,
            dxy_return_20d = excluded.dxy_return_20d,
            dxy_return_60d = excluded.dxy_return_60d,
            vix_level = excluded.vix_level,
            vix_change_20d = excluded.vix_change_20d,
            yield_spread = excluded.yield_spread,
            yield_spread_change_20d = excluded.yield_spread_change_20d,
            ironore_return_20d = excluded.ironore_return_20d
        """
        out = features.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        with self.connect() as conn:
            conn.executemany(sql, out[columns].where(pd.notna(out[columns]), None).to_records(index=False))

    def save_signals(self, signals: pd.DataFrame) -> None:
        if signals.empty:
            return
        columns = [
            "date",
            "signal",
            "probability",
            "confidence",
            "score",
            "aud_momentum_score",
            "dxy_score",
            "yield_score",
            "ironore_score",
            "vix_score",
        ]
        sql = f"""
        INSERT INTO signals ({", ".join(columns)})
        VALUES ({", ".join("?" for _ in columns)})
        ON CONFLICT(date) DO UPDATE SET
            signal = excluded.signal,
            probability = excluded.probability,
            confidence = excluded.confidence,
            score = excluded.score,
            aud_momentum_score = excluded.aud_momentum_score,
            dxy_score = excluded.dxy_score,
            yield_score = excluded.yield_score,
            ironore_score = excluded.ironore_score,
            vix_score = excluded.vix_score
        """
        out = signals.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        with self.connect() as conn:
            conn.executemany(sql, out[columns].where(pd.notna(out[columns]), None).to_records(index=False))

