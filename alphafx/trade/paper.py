from __future__ import annotations

from typing import Any

import numpy as np

from ..database import Database
from .order import OrderIntent


class PaperBroker:
    """Fully local paper-trading backend — no broker, no network, no credentials.

    Simulates market fills at a given reference price into the paper_positions
    table, tracks open positions, auto-closes on the triple-barrier exit, and
    records realised PnL. Same OrderIntent interface a real broker would use, so
    the execution backend can be swapped later without touching signal logic.

    Exit = triple-barrier (Lopez de Prado): a TIME barrier at the signal horizon
    (MAX_HOLDING_DAYS) is the primary exit, plus optional horizontal stop-loss /
    take-profit barriers. The signal's edge is validated at the 20-day horizon, so
    holding past it (the old SL/TP-only engine held a median 45 / max 236 days)
    just gave the edge back — verified by the 5-year replay.
    """

    # Vertical barrier: exit no matter what after this many business days, matching
    # the 20-day horizon the signal is calibrated on.
    MAX_HOLDING_DAYS = 20

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

    def update(
        self, price: float, when: str, max_holding_days: int | None = None, instrument: str | None = None
    ) -> list[dict[str, Any]]:
        """Mark open positions to market; auto-close on the triple barrier.

        Time barrier (primary): close once a position has been held for
        `max_holding_days` business days (defaults to MAX_HOLDING_DAYS). Horizontal
        barriers (stop-loss / take-profit) close earlier if hit.

        `price` marks the given `instrument` only — in a multi-instrument book each
        pair must be marked with its OWN price, so pass `instrument` to avoid
        closing another pair's position against the wrong price. Default (None)
        marks every open position, correct for a single-instrument book.
        """
        max_holding_days = self.MAX_HOLDING_DAYS if max_holding_days is None else max_holding_days
        closed: list[dict[str, Any]] = []
        for pos in self.db.load_open_positions(instrument).itertuples():
            reason = self._exit_reason(pos, price)
            if reason is None and max_holding_days is not None:
                if self._holding_days(pos.entry_date, when) >= max_holding_days:
                    reason = "time_barrier"
            if reason is None:
                continue
            pnl = float(pos.units) * (float(price) - float(pos.entry_price))
            self.db.close_paper_position(int(pos.id), float(price), str(when), reason, pnl)
            closed.append({"id": int(pos.id), "reason": reason, "realised_pnl": pnl, "exit_price": float(price)})
        return closed

    @staticmethod
    def _holding_days(entry_date: Any, when: Any) -> int:
        """Business days held, from entry to the current mark date."""
        try:
            entry = np.datetime64(str(entry_date)[:10])
            now = np.datetime64(str(when)[:10])
            return int(np.busday_count(entry, now))
        except Exception:  # noqa: BLE001 - malformed dates should never block an exit check
            return 0

    @staticmethod
    def _exit_reason(pos: Any, price: float) -> str | None:
        entry = float(pos.entry_price)
        long = int(pos.units) > 0
        # A NULL barrier reloads from SQLite as NaN, not None — treat both as "no barrier".
        sl = pos.stop_loss_pct
        tp = pos.take_profit_pct
        sl = None if sl is None or (isinstance(sl, float) and np.isnan(sl)) else float(sl)
        tp = None if tp is None or (isinstance(tp, float) and np.isnan(tp)) else float(tp)
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

    def unrealised(self, price: float, instrument: str | None = None) -> float:
        positions = self.db.load_open_positions(instrument)
        if positions.empty:
            return 0.0
        return float((positions["units"] * (float(price) - positions["entry_price"])).sum())

    def realised(self) -> float:
        positions = self.db.load_paper_positions()
        closed = positions[positions["status"] == "closed"]
        return float(closed["realised_pnl"].fillna(0.0).sum()) if not closed.empty else 0.0
