from __future__ import annotations

import json

import pandas as pd

from .database import Database
from .risk import RiskSuggestion


class PaperJournalAgent:
    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()

    def record_signal(
        self,
        latest_feature: pd.Series,
        latest_signal: pd.Series,
        factor_table: pd.DataFrame,
        risk: RiskSuggestion,
        explanation: str,
        audusd_price: float | None = None,
        instrument: str = "AUDUSD",
    ) -> None:
        if latest_signal.empty:
            return
        date_value = pd.to_datetime(latest_signal["date"]).strftime("%Y-%m-%d")
        row = {
            "date": date_value,
            "instrument": instrument,
            "price": audusd_price,
            "signal": latest_signal.get("signal"),
            "score": latest_signal.get("score"),
            "calibrated_probability": latest_signal.get("probability"),
            "factor_values": json.dumps(latest_feature.drop(labels=["date"], errors="ignore").to_dict(), default=str),
            "factor_contributions": factor_table.to_json(orient="records"),
            "recommended_position": risk.action,
            "stop_loss": risk.stop_loss,
            "take_profit": risk.take_profit,
            "explanation": explanation,
            "entry_price": None,
            "exit_price": None,
            "realised_pnl": None,
            "status": "open" if risk.action != "NO TRADE" else "no_trade",
        }
        self.db.upsert_paper_journal(row)

    def load(self) -> pd.DataFrame:
        return self.db.load_paper_journal()
