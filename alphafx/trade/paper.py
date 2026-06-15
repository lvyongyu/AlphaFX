from __future__ import annotations

from typing import Any

from ..database import Database
from .order import OrderIntent


class PaperBroker:
    """Fully local paper-trading backend — no broker, no network, no credentials.

    Simulates market fills at a given reference price into the paper_positions
    table, tracks open positions, auto-closes on stop-loss / take-profit, and
    records realised PnL. Same OrderIntent interface a real broker would use, so
    the execution backend can be swapped later without touching signal logic.
    """

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()

    def place(self, intent: OrderIntent, price: float, when: str) -> dict[str, Any]:
        if not intent.is_trade:
            return {"status": "skipped", "reason": "NO TRADE"}
        if not self.db.load_open_positions(intent.instrument).empty:
            return {"status": "skipped", "reason": "already in a position"}
        position_id = self.db.insert_paper_position(
            {
                "instrument": intent.instrument,
                "side": intent.side,
                "units": int(intent.units),
                "entry_price": float(price),
                "entry_date": str(when),
                "stop_loss_pct": intent.stop_loss_pct,
                "take_profit_pct": intent.take_profit_pct,
                "status": "open",
            }
        )
        return {"status": "opened", "id": position_id, "units": int(intent.units), "entry_price": float(price)}

    def update(self, price: float, when: str) -> list[dict[str, Any]]:
        """Mark open positions to market; auto-close any whose stop/TP is hit."""
        closed: list[dict[str, Any]] = []
        for pos in self.db.load_open_positions().itertuples():
            reason = self._exit_reason(pos, price)
            if reason is None:
                continue
            pnl = float(pos.units) * (float(price) - float(pos.entry_price))
            self.db.close_paper_position(int(pos.id), float(price), str(when), reason, pnl)
            closed.append({"id": int(pos.id), "reason": reason, "realised_pnl": pnl, "exit_price": float(price)})
        return closed

    @staticmethod
    def _exit_reason(pos: Any, price: float) -> str | None:
        entry = float(pos.entry_price)
        long = int(pos.units) > 0
        sl = pos.stop_loss_pct
        tp = pos.take_profit_pct
        if long:
            if tp is not None and price >= entry * (1.0 + float(tp)):
                return "take_profit"
            if sl is not None and price <= entry * (1.0 - float(sl)):
                return "stop_loss"
        else:
            if tp is not None and price <= entry * (1.0 - float(tp)):
                return "take_profit"
            if sl is not None and price >= entry * (1.0 + float(sl)):
                return "stop_loss"
        return None

    def unrealised(self, price: float) -> float:
        positions = self.db.load_open_positions()
        if positions.empty:
            return 0.0
        return float((positions["units"] * (float(price) - positions["entry_price"])).sum())

    def realised(self) -> float:
        positions = self.db.load_paper_positions()
        closed = positions[positions["status"] == "closed"]
        return float(closed["realised_pnl"].fillna(0.0).sum()) if not closed.empty else 0.0
