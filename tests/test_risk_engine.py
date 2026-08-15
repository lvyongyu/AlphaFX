"""risk_engine tests: fully offline, no credentials, no IG quota consumed.

Every row marked A.2 in docs/risk-engine-checklist.md maps to at least one test
here. The engine takes plain data, so nothing needs a network or a fake HTTP
transport — that is the point of keeping it I/O-free.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from alphafx.database import Database
from alphafx.execution import risk_engine as re_mod
from alphafx.execution.risk_engine import (
    DRAWDOWN_BREAKER,
    MAX_SIZE,
    MAX_TOTAL_LOTS,
    MONTHLY_LOSS_BREAKER,
    Decision,
    MarketContext,
    RiskEngine,
    TradeRequest,
    underlying_of,
)

MINI = "CS.D.AUDUSD.MINI.IP"
CFD = "CS.D.AUDUSD.CFD.IP"
NOW = datetime(2026, 8, 13, 3, 0, tzinfo=timezone.utc)  # a Thursday
CALENDAR = (("2026-08-13T04:30:00Z", "RBA decision"),)
CALENDAR_THROUGH = "2026-12-31T23:59:59Z"


def engine(tmp_path, calendar=(), through=CALENDAR_THROUGH) -> RiskEngine:
    return RiskEngine(Database(tmp_path / "exec.db"), calendar=calendar, calendar_through=through)


def request(**overrides) -> TradeRequest:
    base = {
        "epic": MINI,
        "direction": "BUY",
        "signal_date": "2026-08-13",
        "stop_distance_pct": 0.05,
        "risk_action": "BUY AUD/USD",
    }
    return TradeRequest(**{**base, **overrides})


def market(**overrides) -> MarketContext:
    base = {"bid": 0.6500, "offer": 0.6502, "market_status": "TRADEABLE"}
    return MarketContext(**{**base, **overrides})


def position(epic: str, size: float = 1.0, direction: str = "BUY") -> dict:
    return {"market": {"epic": epic}, "position": {"size": size, "direction": direction}}


def evaluate(tmp_path, req=None, mkt=None, now=NOW, **engine_kwargs) -> Decision:
    return engine(tmp_path, **engine_kwargs).evaluate(req or request(), mkt or market(), now=now)


# ---- the master gate (row 25) ----

def test_automated_execution_is_off_by_default():
    # The single most important constant in the module: re-enabling the daily
    # cron alone must not be able to start trading.
    assert re_mod.EXECUTION_ENABLED is False


def test_a_clean_request_passes_every_risk_check_but_is_still_refused(tmp_path):
    decision = evaluate(tmp_path)

    assert decision.risk_checks_passed is True  # nothing about the trade is wrong
    assert decision.allowed is False  # but the master gate is shut
    assert any("EXECUTION_ENABLED" in reason for reason in decision.reasons)


def test_allowed_follows_the_master_gate_when_it_is_open(tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "EXECUTION_ENABLED", True)
    decision = evaluate(tmp_path)

    assert decision.allowed is True
    assert decision.reasons == []
    assert decision.size > 0 and decision.stop_distance_points > 0


# ---- upstream suggestion (row 11) ----

def test_no_trade_from_the_risk_agent_is_obeyed_unconditionally(tmp_path, monkeypatch):
    # Even with the master gate open and everything else clean.
    monkeypatch.setattr(re_mod, "EXECUTION_ENABLED", True)
    decision = evaluate(tmp_path, req=request(risk_action="NO TRADE"))

    assert decision.allowed is False
    assert decision.size == 0.0
    assert any("NO TRADE" in reason for reason in decision.reasons)


# ---- sizing (rows 1-4) ----

def test_size_is_back_solved_from_the_stop_at_one_percent(tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "EXECUTION_ENABLED", True)
    # 1% of the fixed 10,000 AUD notional is 100 AUD. A 5% stop loses 500 AUD
    # per lot, so 0.2 lots risks exactly 100.
    decision = evaluate(tmp_path, req=request(stop_distance_pct=0.05))

    assert decision.size == pytest.approx(0.2)
    assert decision.size * re_mod.CONTRACT_UNITS * 0.05 == pytest.approx(
        re_mod.NOTIONAL_AUD * re_mod.MAX_RISK_PER_TRADE
    )


def test_sizing_uses_a_fixed_notional_not_the_broker_balance():
    # The Demo account holds ~9 figures; sizing off it would make the 1% rule
    # meaningless and ship a limit that had never bound anything. The rule is
    # structural, not a convention: the broker's balance is not an input to
    # sizing, because MarketContext has no field to carry it.
    assert re_mod.NOTIONAL_AUD == 10_000.0
    assert "balance" not in MarketContext.__dataclass_fields__
    assert "balance" not in TradeRequest.__dataclass_fields__


def test_stop_distance_is_converted_to_points(tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "EXECUTION_ENABLED", True)
    decision = evaluate(tmp_path, req=request(stop_distance_pct=0.05))
    # mid 0.6501 * 5% = 0.0325 in price, which is 325 pips.
    assert decision.stop_distance_points == pytest.approx(325.1, abs=0.5)


@pytest.mark.parametrize("stop", [None, 0.0, -0.02])
def test_a_missing_stop_is_refused(tmp_path, stop):
    decision = evaluate(tmp_path, req=request(stop_distance_pct=stop))
    assert decision.risk_checks_passed is False
    assert decision.size == 0.0


def test_a_wide_stop_refuses_rather_than_rounding_up_to_the_minimum(tmp_path):
    # A 12% stop (the RiskAgent's widest) prices the position at 0.08 lots,
    # under the 0.1 minimum. Rounding up would breach the 1% risk limit.
    decision = evaluate(tmp_path, req=request(stop_distance_pct=0.12))

    assert decision.risk_checks_passed is False
    assert decision.size == 0.0
    assert any("refusing rather than rounding up" in reason for reason in decision.reasons)


def test_the_brokers_own_minimum_wins_when_it_is_stricter(tmp_path):
    # 0.2 lots clears the built-in 0.1 floor but not a broker minimum of 0.5.
    decision = evaluate(tmp_path, mkt=market(min_deal_size=0.5))
    assert decision.risk_checks_passed is False
    assert any("below the 0.5 minimum" in reason for reason in decision.reasons)


def test_size_can_never_exceed_the_hard_lot_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "EXECUTION_ENABLED", True)
    # A stop tight enough to price a huge position must hit the cap, not pass it.
    decision = evaluate(tmp_path, req=request(stop_distance_pct=0.001))
    assert decision.size <= MAX_SIZE
    assert decision.risk_checks_passed is False
    assert any(f"exceeds the {MAX_SIZE} lot cap" in reason for reason in decision.reasons)


# ---- positions (rows 5-7) ----

def test_underlying_ignores_the_contract_type():
    assert underlying_of(MINI) == underlying_of(CFD) == "AUDUSD"
    assert underlying_of("CS.D.EURUSD.MINI.IP") == "EURUSD"
    assert underlying_of("weird-epic") == "weird-epic"  # unparseable falls back to itself


def test_a_manual_standard_contract_position_blocks_the_mini(tmp_path):
    # The account really does hold a manual CS.D.AUDUSD.CFD.IP long while the
    # script trades the MINI. Matching by epic would not see it.
    decision = evaluate(tmp_path, mkt=market(positions=(position(CFD),)))

    assert decision.risk_checks_passed is False
    assert any(CFD in reason for reason in decision.reasons)


def test_an_unrelated_instrument_does_not_block(tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "EXECUTION_ENABLED", True)
    decision = evaluate(tmp_path, mkt=market(positions=(position("CS.D.EURUSD.MINI.IP", size=0.2),)))
    assert decision.allowed is True


def test_total_exposure_is_capped_across_instruments(tmp_path):
    # Three correlated USD pairs at the cap: per-instrument limits alone would
    # permit one bet with three tickets.
    held = tuple(
        position(f"CS.D.{pair}.MINI.IP", size=MAX_TOTAL_LOTS / 3)
        for pair in ("EURUSD", "GBPUSD", "USDCHF")
    )
    decision = evaluate(tmp_path, mkt=market(positions=held))

    assert decision.risk_checks_passed is False
    assert any("total exposure" in reason for reason in decision.reasons)


# ---- market conditions (rows 17, 19) ----

@pytest.mark.parametrize("status", ["CLOSED", "EDITS_ONLY", "OFFLINE"])
def test_a_non_tradeable_market_is_refused(tmp_path, status):
    decision = evaluate(tmp_path, mkt=market(market_status=status))
    assert decision.risk_checks_passed is False
    assert any(status in reason for reason in decision.reasons)


def test_a_wide_spread_relative_to_the_stop_is_refused(tmp_path):
    # 100 pips of spread against a 325 pip stop starts the trade a third of the
    # way to being stopped out.
    decision = evaluate(tmp_path, mkt=market(bid=0.6500, offer=0.6600))

    assert decision.risk_checks_passed is False
    assert any("spread" in reason for reason in decision.reasons)


# ---- signal freshness (row 24) ----

def test_a_stale_signal_is_refused(tmp_path):
    decision = evaluate(tmp_path, req=request(signal_date="2026-08-07"))
    assert decision.risk_checks_passed is False
    assert any("business days old" in reason for reason in decision.reasons)


def test_a_weekend_gap_does_not_make_a_friday_signal_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "EXECUTION_ENABLED", True)
    monday = datetime(2026, 8, 17, 3, 0, tzinfo=timezone.utc)
    decision = evaluate(tmp_path, req=request(signal_date="2026-08-14"), now=monday)  # Friday
    assert decision.allowed is True


def test_an_unreadable_signal_date_is_refused(tmp_path):
    decision = evaluate(tmp_path, req=request(signal_date="not-a-date"))
    assert decision.risk_checks_passed is False
    assert any("unreadable" in reason for reason in decision.reasons)


# ---- event blackout (row 18) ----

def test_an_unpopulated_calendar_fails_closed(tmp_path):
    # The shipped calendar is empty on purpose. A blackout that silently lapses
    # is worse than one that refuses.
    decision = evaluate(tmp_path, through=None)
    assert decision.risk_checks_passed is False
    assert any("economic calendar is not populated" in reason for reason in decision.reasons)


def test_the_shipped_calendar_is_empty_and_therefore_refuses():
    assert re_mod.ECONOMIC_CALENDAR == ()
    assert re_mod.CALENDAR_THROUGH is None


def test_an_expired_calendar_fails_closed(tmp_path):
    decision = evaluate(tmp_path, calendar=CALENDAR, through="2026-08-01T00:00:00Z")
    assert decision.risk_checks_passed is False
    assert any("only covers through" in reason for reason in decision.reasons)


def test_trades_inside_the_event_window_are_refused(tmp_path):
    # NOW is 03:00, the RBA decision is 04:30 — 1.5h out, inside the 2h window.
    decision = evaluate(tmp_path, calendar=CALENDAR)
    assert decision.risk_checks_passed is False
    assert any("RBA decision" in reason for reason in decision.reasons)


def test_trades_outside_the_event_window_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "EXECUTION_ENABLED", True)
    clear = NOW - timedelta(hours=3)  # 00:00, 4.5h before the release
    decision = evaluate(tmp_path, req=request(signal_date="2026-08-12"), calendar=CALENDAR, now=clear)
    assert decision.allowed is True


# ---- circuit breakers (rows 13-16) ----

def test_the_monthly_loss_breaker_trips_and_persists(tmp_path):
    db = Database(tmp_path / "exec.db")
    eng = RiskEngine(db, calendar=(), calendar_through=CALENDAR_THROUGH)
    eng.record_balance("2026-08-01", 10_000.0)
    eng.record_balance("2026-08-13", 9_400.0)  # -6% on the month

    assert eng.check_breakers(NOW) == [MONTHLY_LOSS_BREAKER]
    # A brand-new engine on the same DB still sees it: the daily job is a fresh
    # process every run, so an in-memory breaker would not exist at all.
    assert RiskEngine(Database(tmp_path / "exec.db")).active_breakers(NOW) == [MONTHLY_LOSS_BREAKER]

    decision = RiskEngine(db, calendar=(), calendar_through=CALENDAR_THROUGH).evaluate(
        request(), market(), now=NOW
    )
    assert decision.risk_checks_passed is False
    assert any(MONTHLY_LOSS_BREAKER in reason for reason in decision.reasons)


def test_the_monthly_breaker_stops_binding_next_month(tmp_path):
    db = Database(tmp_path / "exec.db")
    eng = RiskEngine(db)
    eng.record_balance("2026-08-01", 10_000.0)
    eng.record_balance("2026-08-13", 9_400.0)
    eng.check_breakers(NOW)

    assert eng.active_breakers(NOW) == [MONTHLY_LOSS_BREAKER]
    # September: the rule is "no new positions this month", so its own scope ends.
    assert eng.active_breakers(datetime(2026, 9, 1, tzinfo=timezone.utc)) == []


def test_the_drawdown_breaker_needs_a_human_to_clear_it(tmp_path):
    db = Database(tmp_path / "exec.db")
    eng = RiskEngine(db)
    eng.record_balance("2026-06-01", 10_000.0)
    eng.record_balance("2026-08-13", 8_000.0)  # -20% from the peak

    assert DRAWDOWN_BREAKER in eng.check_breakers(NOW)
    # Unlike the monthly one it has no scope, so time alone never clears it.
    assert DRAWDOWN_BREAKER in eng.active_breakers(datetime(2027, 1, 1, tzinfo=timezone.utc))

    # Nor does the balance recovering — a recovery is not evidence that whatever
    # caused the loss is fixed.
    eng.record_balance("2026-08-20", 10_500.0)
    eng.check_breakers(NOW)
    assert DRAWDOWN_BREAKER in eng.active_breakers(NOW)

    assert eng.clear_breaker(DRAWDOWN_BREAKER, cleared_by="operator", now=NOW) == 1
    assert eng.active_breakers(NOW) == []


@pytest.mark.parametrize("who", ["", "auto", "system"])
def test_a_breaker_cannot_be_cleared_automatically(tmp_path, who):
    with pytest.raises(ValueError, match="human identifier"):
        RiskEngine(Database(tmp_path / "exec.db")).clear_breaker(DRAWDOWN_BREAKER, cleared_by=who)


def test_a_breaker_does_not_trip_twice(tmp_path):
    eng = RiskEngine(Database(tmp_path / "exec.db"))
    eng.record_balance("2026-08-01", 10_000.0)
    eng.record_balance("2026-08-13", 9_400.0)

    assert eng.check_breakers(NOW) == [MONTHLY_LOSS_BREAKER]
    assert eng.check_breakers(NOW) == []  # already tripped, not re-recorded
    assert len(eng.breaker_history()) == 1


def test_breaker_trips_are_auditable(tmp_path):
    eng = RiskEngine(Database(tmp_path / "exec.db"))
    eng.record_balance("2026-08-01", 10_000.0)
    eng.record_balance("2026-08-13", 9_400.0)
    eng.check_breakers(NOW)
    eng.clear_breaker(MONTHLY_LOSS_BREAKER, cleared_by="operator", now=NOW)

    row = eng.breaker_history().iloc[0]
    assert row["breaker"] == MONTHLY_LOSS_BREAKER
    assert row["tripped_at"] == NOW.isoformat()
    assert row["tripped_value"] == pytest.approx(-0.06)
    assert row["threshold"] == pytest.approx(-0.05)
    assert "9400.0" in row["detail"]
    assert row["cleared_by"] == "operator"


def test_breakers_are_computed_from_realised_cash_not_open_position_marks(tmp_path):
    # Row 16: the manual CFD long is not the script's, and letting its
    # mark-to-market move a breaker would let a human trade halt the system.
    # Recording CASH balance excludes every open position's unrealised PnL.
    eng = RiskEngine(Database(tmp_path / "exec.db"))
    eng.record_balance("2026-08-01", 10_000.0)
    eng.record_balance("2026-08-13", 9_900.0)  # -1% realised

    assert eng.check_breakers(NOW) == []
    assert eng.active_breakers(NOW) == []


def test_no_breaker_trips_without_at_least_two_observations(tmp_path):
    eng = RiskEngine(Database(tmp_path / "exec.db"))
    eng.record_balance("2026-08-13", 1.0)
    assert eng.check_breakers(NOW) == []


# ---- reporting ----

def test_every_failing_check_is_reported_not_just_the_first(tmp_path):
    decision = evaluate(
        tmp_path,
        req=request(signal_date="2026-07-01", stop_distance_pct=0.12),
        mkt=market(market_status="CLOSED", positions=(position(CFD),)),
    )
    # Stale signal, closed market, unsizeable stop, and a clashing position.
    assert len(decision.reasons) >= 4
    assert decision.summary().startswith("REFUSE:")
