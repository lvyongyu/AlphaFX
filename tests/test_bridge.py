"""Bridge tests: fully offline, no credentials, no IG quota consumed.

The IG client is a fake, so every request the bridge would make is inspectable
and nothing reaches the network.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from alphafx.database import Database
from alphafx.execution import bridge as bridge_mod
from alphafx.execution import risk_engine as re_mod
from alphafx.execution.bridge import (
    DRY_RUN,
    LIVE,
    IG_EPICS,
    SignalBridge,
    direction_of,
    export_execution_log,
    load_signal_file,
)
from alphafx.execution.ig_client import DEFAULT_EPIC, IGError
from alphafx.execution.risk_engine import RiskEngine

NOW = datetime(2026, 8, 13, 3, 0, tzinfo=timezone.utc)  # a Thursday
CFD = "CS.D.AUDUSD.CFD.IP"
CALENDAR_THROUGH = "2026-12-31T23:59:59Z"


class FakeClient:
    """Stand-in for IGClient. Records every call; places nothing anywhere."""

    def __init__(self, snapshot=None, positions=(), balance=10_000.0, opens=None):
        self.snapshot = snapshot or {"bid": 0.6500, "offer": 0.6502, "marketStatus": "TRADEABLE"}
        self.positions = list(positions)
        self.balance = balance
        self.opens: list[dict] = []
        self.open_result = opens or {"dealReference": "REF1", "dealStatus": "ACCEPTED"}
        self.logged_in = False

    def login(self):
        self.logged_in = True

    def get_positions(self):
        return self.positions

    def get_account(self):
        return {"balance": {"balance": self.balance}}

    def get_market(self, epic=DEFAULT_EPIC):
        if isinstance(self.snapshot, Exception):
            raise self.snapshot
        return self.snapshot

    def open_position(self, direction, size, stop_distance, epic=DEFAULT_EPIC, **kwargs):
        self.opens.append({"direction": direction, "size": size,
                           "stop_distance": stop_distance, "epic": epic})
        return self.open_result


def leg(**overrides) -> dict:
    base = {
        "instrument": "AUDUSD",
        "status": "ok",
        "date": "2026-08-13",
        "price": 0.6501,
        "signal": "bullish",
        "action": "BUY AUD/USD",
        "stop_loss": 0.05,
        "take_profit": None,
    }
    return {**base, **overrides}


def payload(*legs) -> dict:
    return {"status": "ok", "date": "2026-08-13", "legs": list(legs or [leg()])}


def bridge(tmp_path, client=None, **engine_kwargs) -> SignalBridge:
    db = Database(tmp_path / "bridge.db")
    engine = RiskEngine(db, calendar=(), calendar_through=CALENDAR_THROUGH, **engine_kwargs)
    return SignalBridge(client=client or FakeClient(), engine=engine, db=db)


# ---- the submission guard ----

def test_dry_run_is_the_default_and_sends_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "EXECUTION_ENABLED", True)  # even with the gate open
    client = FakeClient()
    report = bridge(tmp_path, client).run(payload(), now=NOW)

    assert report.mode == DRY_RUN
    assert client.opens == []  # nothing was placed
    order = report.orders[0]
    assert order.decision.allowed is True  # it WOULD have gone
    assert order.submitted is False


def test_live_is_inert_while_the_master_gate_is_shut(tmp_path):
    client = FakeClient()
    report = bridge(tmp_path, client).run(payload(), live=True, now=NOW)

    assert report.mode == LIVE
    assert client.opens == []
    order = report.orders[0]
    assert order.decision.risk_checks_passed is True  # the trade itself was fine
    assert order.decision.allowed is False  # the gate stopped it
    assert order.submitted is False


def test_live_submits_only_when_both_the_flag_and_the_gate_agree(tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "EXECUTION_ENABLED", True)
    client = FakeClient()
    report = bridge(tmp_path, client).run(payload(), live=True, now=NOW)

    order = report.orders[0]
    assert order.submitted is True
    assert order.deal_status == "ACCEPTED"
    sent = client.opens[0]
    assert sent["direction"] == "BUY"
    assert sent["epic"] == DEFAULT_EPIC
    assert sent["size"] == order.decision.size
    assert sent["stop_distance"] == order.decision.stop_distance_points
    assert sent["stop_distance"] > 0  # a stop always goes with the order


def test_a_rejected_deal_is_not_recorded_as_a_fill(tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "EXECUTION_ENABLED", True)
    client = FakeClient(opens={"dealReference": "REF9", "dealStatus": "REJECTED"})
    report = bridge(tmp_path, client).run(payload(), live=True, now=NOW)

    assert report.orders[0].deal_status == "REJECTED"


def test_a_broker_error_on_submit_is_captured_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "EXECUTION_ENABLED", True)

    class Exploding(FakeClient):
        def open_position(self, *a, **k):
            raise IGError("market closed")

    report = bridge(tmp_path, Exploding()).run(payload(), live=True, now=NOW)
    order = report.orders[0]
    assert order.submitted is False
    assert "market closed" in order.deal_status


# ---- refusals that never reach the engine ----

def test_no_trade_legs_are_refused_but_still_logged(tmp_path):
    br = bridge(tmp_path)
    report = br.run(payload(leg(action="NO TRADE", signal="neutral")), now=NOW)

    order = report.orders[0]
    assert order.decision is None
    assert "NO TRADE" in order.reasons[0]
    # Logged anyway: the dry-run record has to show the day was considered.
    assert len(br.db.load_execution_log()) == 1


def test_a_leg_without_a_stop_is_refused_not_defaulted(tmp_path):
    report = bridge(tmp_path).run(payload(leg(stop_loss=None)), now=NOW)
    assert "no stop_loss" in report.orders[0].reasons[0]


def test_an_instrument_with_no_epic_mapping_is_refused(tmp_path):
    report = bridge(tmp_path).run(payload(leg(instrument="EURUSD")), now=NOW)
    assert "no IG epic mapping" in report.orders[0].reasons[0]


def test_only_audusd_is_wired_to_an_epic():
    # An epic typo is an order on the wrong instrument, so the map is explicit
    # rather than derived from the instrument name.
    assert IG_EPICS == {"AUDUSD": DEFAULT_EPIC}


def test_a_failed_leg_in_the_signal_file_is_refused(tmp_path):
    report = bridge(tmp_path).run(payload(leg(status="no_data")), now=NOW)
    assert "status is 'no_data'" in report.orders[0].reasons[0]


def test_an_unreadable_quote_refuses_that_leg_only(tmp_path):
    client = FakeClient(snapshot=IGError("quote unavailable"))
    report = bridge(tmp_path, client).run(payload(), now=NOW)
    assert "could not read the market" in report.orders[0].reasons[0]


@pytest.mark.parametrize("action,expected", [
    ("BUY AUD/USD", "BUY"), ("SELL AUD/USD", "SELL"), ("NO TRADE", None), (None, None), ("", None),
])
def test_direction_parsing(action, expected):
    assert direction_of(action) == expected


# ---- the signal file ----

def test_a_pre_portfolio_signal_file_is_refused(tmp_path):
    # The committed data/latest_signal.json is still the old single-instrument
    # shape. Guessing at it would be guessing at what to trade.
    legacy = {"status": "ok", "date": "2026-06-19", "signal": "neutral", "action": "NO TRADE"}
    report = bridge(tmp_path).run(legacy, now=NOW)

    assert report.status == "bad_signal_file"
    assert "paper_trade.py --export" in report.note


def test_no_broker_session_is_reported_not_crashed(tmp_path):
    db = Database(tmp_path / "bridge.db")
    report = SignalBridge(client=None, db=db).run(payload(), now=NOW)
    assert report.status == "no_broker"


def test_load_signal_file_reads_json(tmp_path):
    path = tmp_path / "latest_signal.json"
    path.write_text(json.dumps(payload()), encoding="utf-8")
    assert load_signal_file(path)["legs"][0]["instrument"] == "AUDUSD"


# ---- engine integration ----

def test_the_bridge_adds_no_rules_of_its_own(tmp_path):
    # Every refusal on a leg that reaches the engine must come from the engine.
    br = bridge(tmp_path)
    report = br.run(payload(leg(date="2026-07-01")), now=NOW)  # stale signal

    order = report.orders[0]
    assert order.decision is not None
    assert order.reasons == list(order.decision.reasons)
    assert any("business days old" in reason for reason in order.reasons)


def test_the_manual_cfd_position_blocks_the_mini_order(tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "EXECUTION_ENABLED", True)
    client = FakeClient(positions=[{"market": {"epic": CFD}, "position": {"size": 1.0}}])
    report = bridge(tmp_path, client).run(payload(), live=True, now=NOW)

    assert report.orders[0].decision.allowed is False
    assert client.opens == []


def test_positions_are_read_once_per_run_not_once_per_leg(tmp_path):
    calls = {"n": 0}

    class Counting(FakeClient):
        def get_positions(self):
            calls["n"] += 1
            return self.positions

    bridge(tmp_path, Counting()).run(payload(leg(), leg(instrument="EURUSD")), now=NOW)
    assert calls["n"] == 1


# ---- breakers ----

def test_the_run_records_the_cash_balance_and_trips_breakers(tmp_path):
    db = Database(tmp_path / "bridge.db")
    engine = RiskEngine(db, calendar=(), calendar_through=CALENDAR_THROUGH)
    engine.record_balance("2026-08-01", 10_000.0)

    client = FakeClient(balance=9_400.0)  # -6% on the month
    report = SignalBridge(client=client, engine=engine, db=db).run(payload(), now=NOW)

    assert report.breakers_tripped == [re_mod.MONTHLY_LOSS_BREAKER]
    assert any("circuit breaker" in reason for reason in report.orders[0].reasons)


def test_the_balance_recorded_is_cash_not_equity(tmp_path):
    # Row 16: equity would fold in the manual CFD position's mark-to-market,
    # letting a human trade move a circuit breaker.
    db = Database(tmp_path / "bridge.db")
    engine = RiskEngine(db, calendar=(), calendar_through=CALENDAR_THROUGH)

    class WithOpenPnL(FakeClient):
        def get_account(self):
            return {"balance": {"balance": 10_000.0, "profitLoss": -5_000.0}}

    SignalBridge(client=WithOpenPnL(), engine=engine, db=db).run(payload(), now=NOW)
    balances = db.load_execution_balances()
    assert float(balances.iloc[-1]["balance"]) == 10_000.0  # profitLoss ignored


# ---- the dry-run record ----

def test_every_leg_of_every_run_is_logged(tmp_path):
    br = bridge(tmp_path)
    br.run(payload(leg(), leg(instrument="EURUSD")), now=NOW)

    rows = br.db.load_execution_log()
    assert len(rows) == 2
    assert set(rows["mode"]) == {DRY_RUN}
    assert set(rows["instrument"]) == {"AUDUSD", "EURUSD"}


def test_the_log_keeps_the_order_that_would_have_been_sent(tmp_path):
    # A refusal that records nothing is useless for comparing against the paper
    # book; the size and stop have to survive the refusal.
    br = bridge(tmp_path)
    br.run(payload(), now=NOW)

    row = br.db.load_execution_log().iloc[0]
    assert row["direction"] == "BUY"
    assert row["size"] > 0
    assert row["stop_distance_points"] > 0
    assert row["risk_checks_passed"] == 1
    assert row["allowed"] == 0
    assert "EXECUTION_ENABLED" in row["reasons"]
    assert row["submitted"] == 0


def test_the_log_is_append_only_across_runs(tmp_path):
    br = bridge(tmp_path)
    br.run(payload(), now=NOW)
    br.run(payload(), now=NOW)
    assert len(br.db.load_execution_log()) == 2


# ---- the committable mirror ----

def test_export_writes_the_log_to_csv(tmp_path):
    br = bridge(tmp_path)
    br.run(payload(leg(), leg(instrument="EURUSD")), now=NOW)

    path = export_execution_log(br.db, tmp_path)
    rows = pd.read_csv(path)
    assert len(rows) == 2
    assert set(rows["instrument"]) == {"AUDUSD", "EURUSD"}
    assert "id" not in rows.columns  # the local row id means nothing across runs


def test_export_appends_across_runs_with_a_fresh_database(tmp_path):
    # This is the CI case: every run starts from an empty SQLite file, so the CSV
    # is the only thing that accumulates.
    later = datetime(2026, 8, 14, 3, 0, tzinfo=timezone.utc)
    for index, when in enumerate((NOW, later)):
        db = Database(tmp_path / f"run{index}.db")  # a brand-new DB each time
        engine = RiskEngine(db, calendar=(), calendar_through=CALENDAR_THROUGH)
        SignalBridge(client=FakeClient(), engine=engine, db=db).run(payload(), now=when)
        export_execution_log(db, tmp_path)

    rows = pd.read_csv(tmp_path / "execution_log.csv")
    assert len(rows) == 2  # the first run survived the second run's empty DB
    assert sorted(rows["run_at"]) == [NOW.isoformat(), later.isoformat()]


def test_export_is_idempotent(tmp_path):
    br = bridge(tmp_path)
    br.run(payload(), now=NOW)

    export_execution_log(br.db, tmp_path)
    export_execution_log(br.db, tmp_path)
    assert len(pd.read_csv(tmp_path / "execution_log.csv")) == 1


def test_export_from_an_empty_database_never_truncates(tmp_path):
    br = bridge(tmp_path)
    br.run(payload(), now=NOW)
    export_execution_log(br.db, tmp_path)

    empty = Database(tmp_path / "empty.db")
    export_execution_log(empty, tmp_path)
    assert len(pd.read_csv(tmp_path / "execution_log.csv")) == 1


def test_bridge_module_declares_no_thresholds():
    # The bridge is plumbing. Any number that decides something belongs in
    # risk_engine, where it is tested as a rule — so the module has no numeric
    # constants of its own at all.
    numbers = {
        name: value
        for name, value in vars(bridge_mod).items()
        if not name.startswith("_")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    assert numbers == {}
