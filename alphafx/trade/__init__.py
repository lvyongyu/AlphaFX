"""Execution layer for AlphaFX — paper trading first, real brokers later.

`order.build_order_intent` turns the deterministic risk decision into a
broker-agnostic OrderIntent. `paper.PaperBroker` is a fully local fill
simulator (no broker/network). A real broker (e.g. OANDA practice) can be added
later behind the same OrderIntent interface without touching signal logic.
"""

from .order import OrderIntent, build_order_intent
from .paper import PaperBroker

__all__ = ["OrderIntent", "build_order_intent", "PaperBroker"]
