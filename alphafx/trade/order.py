from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INSTRUMENT = "AUD_USD"


@dataclass
class OrderIntent:
    instrument: str
    side: str  # "buy" | "sell" | "none"
    units: int  # signed; 0 = no trade
    order_type: str  # "MARKET"
    stop_loss_pct: float | None
    take_profit_pct: float | None
    rationale: str

    @property
    def is_trade(self) -> bool:
        return self.units != 0


def build_order_intent(
    signal: Any, risk: Any, base_units: int = 1000, instrument: str = INSTRUMENT
) -> OrderIntent:
    """Translate the deterministic risk decision into a broker order intent.

    The RiskAgent — not the LLM or the raw signal — decides whether to trade
    (it returns NO TRADE on no edge or extreme volatility). This only formats
    that decision into an order; it never overrides the gate. NO TRADE -> units 0.
    `instrument` is the broker pair id (e.g. "AUD_USD", "EUR_USD").
    """
    action = str(getattr(risk, "action", "NO TRADE")).upper()
    size_factor = 0.5 if getattr(risk, "position_size", "Standard") == "Small" else 1.0
    units = int(round(base_units * size_factor))
    if action.startswith("BUY"):
        side, signed = "buy", units
    elif action.startswith("SELL"):
        side, signed = "sell", -units
    else:
        side, signed = "none", 0
    return OrderIntent(
        instrument=instrument,
        side=side,
        units=signed,
        order_type="MARKET",
        stop_loss_pct=float(getattr(risk, "stop_loss", 0.0)) if signed and getattr(risk, "stop_loss", None) is not None else None,
        take_profit_pct=float(getattr(risk, "take_profit", 0.0)) if signed and getattr(risk, "take_profit", None) is not None else None,
        rationale=(
            f"{getattr(risk, 'action', 'NO TRADE')}; "
            f"signal={signal.get('signal')}, prob={signal.get('probability')}"
        ),
    )
