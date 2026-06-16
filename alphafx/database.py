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

CREATE TABLE IF NOT EXISTS macro_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    value REAL,
    source TEXT,
    frequency TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, symbol)
);

CREATE TABLE IF NOT EXISTS paper_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    audusd_price REAL,
    signal TEXT,
    score REAL,
    calibrated_probability REAL,
    factor_values TEXT,
    factor_contributions TEXT,
    recommended_position TEXT,
    stop_loss REAL,
    take_profit REAL,
    explanation TEXT,
    entry_price REAL,
    exit_price REAL,
    realised_pnl REAL,
    status TEXT,
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

CREATE TABLE IF NOT EXISTS paper_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument TEXT NOT NULL,
    side TEXT NOT NULL,
    units INTEGER NOT NULL,
    entry_price REAL,
    entry_date TEXT,
    stop_loss_pct REAL,
    take_profit_pct REAL,
    exit_price REAL,
    exit_date TEXT,
    exit_reason TEXT,
    realised_pnl REAL,
    status TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    role TEXT,
    model TEXT,
    prompt_hash TEXT,
    system_prompt TEXT,
    user_payload TEXT,
    response_text TEXT,
    structured_output TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class Database:
    def __init__(self, path: Path | None = None) -> None:
        # Resolve the default at call time (not at def time) so tests can redirect
        # the default DB to a temp path by monkeypatching alphafx.database.DB_PATH.
        self.path = Path(path) if path is not None else DB_PATH
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

    def upsert_macro_data(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        columns = ["date", "symbol", "value", "source", "frequency"]
        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        rows = out[columns].where(pd.notna(out[columns]), None).to_records(index=False)
        sql = """
        INSERT INTO macro_data (date, symbol, value, source, frequency)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(date, symbol) DO UPDATE SET
            value = excluded.value,
            source = excluded.source,
            frequency = excluded.frequency
        """
        with self.connect() as conn:
            conn.executemany(sql, rows)

    def load_macro_data(self, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        params: list[str] = []
        where = ""
        if symbols:
            params = list(symbols)
            placeholders = ",".join("?" for _ in params)
            where = f"WHERE symbol IN ({placeholders})"
        with self.connect() as conn:
            return pd.read_sql_query(
                f"SELECT * FROM macro_data {where} ORDER BY date",
                conn,
                params=params,
                parse_dates=["date"],
            )

    def upsert_paper_journal(self, row: dict[str, object]) -> None:
        columns = [
            "date",
            "audusd_price",
            "signal",
            "score",
            "calibrated_probability",
            "factor_values",
            "factor_contributions",
            "recommended_position",
            "stop_loss",
            "take_profit",
            "explanation",
            "entry_price",
            "exit_price",
            "realised_pnl",
            "status",
        ]
        values = [row.get(column) for column in columns]
        sql = f"""
        INSERT INTO paper_journal ({", ".join(columns)})
        VALUES ({", ".join("?" for _ in columns)})
        ON CONFLICT(date) DO UPDATE SET
            audusd_price = excluded.audusd_price,
            signal = excluded.signal,
            score = excluded.score,
            calibrated_probability = excluded.calibrated_probability,
            factor_values = excluded.factor_values,
            factor_contributions = excluded.factor_contributions,
            recommended_position = excluded.recommended_position,
            stop_loss = excluded.stop_loss,
            take_profit = excluded.take_profit,
            explanation = excluded.explanation,
            entry_price = excluded.entry_price,
            exit_price = excluded.exit_price,
            realised_pnl = excluded.realised_pnl,
            status = excluded.status
        """
        with self.connect() as conn:
            conn.execute(sql, values)

    def load_paper_journal(self) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query("SELECT * FROM paper_journal ORDER BY date", conn, parse_dates=["date"])

    def log_llm_call(self, row: dict[str, object]) -> None:
        """Persist one LLM call so any explanation can be reproduced/reviewed."""
        columns = [
            "date",
            "role",
            "model",
            "prompt_hash",
            "system_prompt",
            "user_payload",
            "response_text",
            "structured_output",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
        ]
        values = [row.get(column) for column in columns]
        sql = f"""
        INSERT INTO llm_calls ({", ".join(columns)})
        VALUES ({", ".join("?" for _ in columns)})
        """
        with self.connect() as conn:
            conn.execute(sql, values)

    def load_llm_calls(self, limit: int = 50) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM llm_calls ORDER BY id DESC LIMIT ?", conn, params=[int(limit)]
            )

    def insert_paper_position(self, row: dict[str, object]) -> int:
        columns = [
            "instrument",
            "side",
            "units",
            "entry_price",
            "entry_date",
            "stop_loss_pct",
            "take_profit_pct",
            "status",
        ]
        values = [row.get(c) for c in columns]
        sql = f"INSERT INTO paper_positions ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})"
        with self.connect() as conn:
            cur = conn.execute(sql, values)
            return int(cur.lastrowid)

    def close_paper_position(
        self, position_id: int, exit_price: float, exit_date: str, exit_reason: str, realised_pnl: float
    ) -> None:
        sql = """
        UPDATE paper_positions
        SET exit_price = ?, exit_date = ?, exit_reason = ?, realised_pnl = ?, status = 'closed'
        WHERE id = ?
        """
        with self.connect() as conn:
            conn.execute(sql, [exit_price, exit_date, exit_reason, realised_pnl, int(position_id)])

    def load_open_positions(self, instrument: str | None = None) -> pd.DataFrame:
        where = "WHERE status = 'open'"
        params: list[object] = []
        if instrument:
            where += " AND instrument = ?"
            params.append(instrument)
        with self.connect() as conn:
            return pd.read_sql_query(f"SELECT * FROM paper_positions {where} ORDER BY id", conn, params=params)

    def load_paper_positions(self) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query("SELECT * FROM paper_positions ORDER BY id", conn)

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
