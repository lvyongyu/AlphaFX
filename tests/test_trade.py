from __future__ import annotations

import pandas as pd

from alphafx.database import Database
from alphafx.risk import RiskSuggestion
from alphafx.trade.order import build_order_intent
from alphafx.trade.paper import PaperBroker


def _signal(sig: str = "bullish", prob: float = 0.6) -> pd.Series:
    return pd.Series({"signal": sig, "probability": prob, "date": "2024-06-01"})


def _risk(action: str, size: str = "Standard") -> RiskSuggestion:
    return RiskSuggestion(
        action=action, position_size=size, leverage=2.0, stop_loss=0.02, take_profit=0.04, warning=""
    )


# ---- order intent gates on the deterministic risk decision ----

def test_build_order_intent_buy_sell_no_trade():
    buy = build_order_intent(_signal("bullish"), _risk("BUY AUD/USD"), base_units=1000)
    assert buy.side == "buy" and buy.units == 1000 and buy.is_trade

    sell = build_order_intent(_signal("bearish"), _risk("SELL AUD/USD"), base_units=1000)
    assert sell.side == "sell" and sell.units == -1000

    flat = build_order_intent(_signal("neutral"), _risk("NO TRADE"), base_units=1000)
    assert flat.units == 0 and not flat.is_trade  # risk gate -> no order


def test_small_position_size_halves_units():
    small = build_order_intent(_signal("bullish"), _risk("BUY AUD/USD", size="Small"), base_units=1000)
    assert small.units == 500


# ---- paper broker lifecycle ----

def test_paper_open_then_take_profit(tmp_path):
    db = Database(tmp_path / "paper.db")
    broker = PaperBroker(db)
    intent = build_order_intent(_signal("bullish"), _risk("BUY AUD/USD"), base_units=1000)

    opened = broker.place(intent, price=0.65, when="2024-06-01")
    assert opened["status"] == "opened"
    assert len(db.load_open_positions()) == 1

    # 0.66 is below the +4% take-profit (0.676): nothing closes.
    assert broker.update(0.66, "2024-06-02") == []
    # 0.68 is above 0.676: take-profit fires, realised PnL is positive for a long.
    closed = broker.update(0.68, "2024-06-03")
    assert len(closed) == 1 and closed[0]["reason"] == "take_profit"
    assert db.load_open_positions().empty
    assert broker.realised() > 0


def test_paper_no_trade_opens_nothing(tmp_path):
    db = Database(tmp_path / "paper.db")
    broker = PaperBroker(db)
    intent = build_order_intent(_signal("neutral"), _risk("NO TRADE"), base_units=1000)
    result = broker.place(intent, price=0.65, when="2024-06-01")
    assert result["status"] == "skipped"
    assert db.load_open_positions().empty


def test_paper_short_stop_loss(tmp_path):
    db = Database(tmp_path / "paper.db")
    broker = PaperBroker(db)
    intent = build_order_intent(_signal("bearish"), _risk("SELL AUD/USD"), base_units=1000)
    broker.place(intent, price=0.65, when="2024-06-01")
    # Short stop is +2% (0.663); price rising to 0.67 triggers stop_loss, PnL negative.
    closed = broker.update(0.67, "2024-06-02")
    assert len(closed) == 1 and closed[0]["reason"] == "stop_loss"
    assert broker.realised() < 0
