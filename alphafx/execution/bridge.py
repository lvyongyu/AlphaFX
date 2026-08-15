"""Signal file -> validated IG order. Dry-run by default; decides nothing itself.

The bridge is plumbing between three things that already exist: the signal
snapshot written by `scripts/paper_trade.py --export`, the deterministic gate in
`risk_engine`, and the REST transport in `ig_client`. It contributes no rules of
its own — every refusal in its output comes from the engine.

What makes a dry-run worth running: every leg of every run is appended to
`execution_log`, refusals included, with the size and stop that WOULD have been
sent. That log is the parallel record to compare against the paper book over the
same dates, which is the whole point of step A.3.

Submission is guarded three deep, and today the first guard alone stops
everything:

  1. `risk_engine.EXECUTION_ENABLED` is False, so `Decision.allowed` is never
     True and `--live` is inert by construction.
  2. `--live` must be passed explicitly; the default is dry-run.
  3. `ig_client.open_position` re-checks the stop and the size cap.

Missing inputs fail closed. A leg with no stop, an instrument with no IG epic,
a signal file in the old pre-portfolio shape — each is refused with a reason,
never defaulted into something plausible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..database import Database
from .ig_client import DEFAULT_EPIC, IGError
from .risk_engine import Decision, MarketContext, RiskEngine, TradeRequest

# Only AUD/USD is wired to an IG epic. Everything else in the research portfolio
# is refused rather than mapped by guesswork — an epic typo is an order on the
# wrong instrument. [Add entries only against IG's own epic list.]
IG_EPICS: dict[str, str] = {"AUDUSD": DEFAULT_EPIC}

DRY_RUN = "dry-run"
LIVE = "live"
NO_TRADE = "NO TRADE"


@dataclass
class PlannedOrder:
    """What one instrument resolved to on one run."""

    instrument: str
    epic: str | None = None
    direction: str | None = None
    signal_date: str | None = None
    decision: Decision | None = None
    refusal: str | None = None  # set when the leg never reached the engine
    submitted: bool = False
    deal_reference: str | None = None
    deal_status: str | None = None

    @property
    def reasons(self) -> list[str]:
        if self.refusal:
            return [self.refusal]
        return list(self.decision.reasons) if self.decision else []

    @property
    def size(self) -> float:
        return self.decision.size if self.decision else 0.0

    @property
    def stop_distance_points(self) -> float:
        return self.decision.stop_distance_points if self.decision else 0.0

    def as_row(self, run_at: str, mode: str) -> dict[str, Any]:
        return {
            "run_at": run_at,
            "signal_date": self.signal_date,
            "instrument": self.instrument,
            "epic": self.epic,
            "mode": mode,
            "direction": self.direction,
            "size": self.size,
            "stop_distance_points": self.stop_distance_points,
            "risk_checks_passed": int(bool(self.decision and self.decision.risk_checks_passed)),
            "allowed": int(bool(self.decision and self.decision.allowed)),
            "reasons": "; ".join(self.reasons),
            "submitted": int(self.submitted),
            "deal_reference": self.deal_reference,
            "deal_status": self.deal_status,
        }


@dataclass
class BridgeReport:
    status: str
    mode: str
    run_at: str
    orders: list[PlannedOrder] = field(default_factory=list)
    breakers_tripped: list[str] = field(default_factory=list)
    note: str | None = None


def load_signal_file(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export_execution_log(db: Database, out_dir: str | Path = "data") -> str:
    """Mirror the log to a committable CSV, and return the path written.

    A CI run starts from a fresh checkout, so the SQLite file is empty every time
    and the dry-run record would never accumulate — which would defeat the whole
    point of running it daily. The CSV is the durable copy.

    Rows are keyed by (run_at, instrument): re-exporting is idempotent, and an
    export from an empty DB adds nothing rather than truncating what is already
    there. History is only ever appended to.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "execution_log.csv"

    fresh = db.load_execution_log(limit=10_000)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
    if combined.empty:
        return str(path)
    combined = (
        combined.drop(columns=["id"], errors="ignore")
        .drop_duplicates(subset=["run_at", "instrument"], keep="last")
        .sort_values(["run_at", "instrument"])
    )
    combined.to_csv(path, index=False)
    return str(path)


def direction_of(action: str | None) -> str | None:
    """"BUY AUD/USD" -> "BUY". Anything that is not clearly a side returns None."""
    first = str(action or "").strip().split(" ")[0].upper()
    return first if first in ("BUY", "SELL") else None


class SignalBridge:
    """Turns a signal snapshot into IG orders, or into logged refusals."""

    def __init__(
        self,
        client: Any = None,
        engine: RiskEngine | None = None,
        db: Database | None = None,
    ) -> None:
        self.db = db or Database()
        self.engine = engine or RiskEngine(self.db)
        self.client = client

    def run(
        self,
        payload: dict[str, Any],
        live: bool = False,
        now: datetime | None = None,
    ) -> BridgeReport:
        now = now or datetime.now(timezone.utc)
        run_at = now.isoformat()
        mode = LIVE if live else DRY_RUN

        if self.client is None:
            return BridgeReport("no_broker", mode, run_at, note="No IG session: cannot read the market.")
        legs = payload.get("legs")
        if not isinstance(legs, list):
            return BridgeReport(
                "bad_signal_file", mode, run_at,
                note="Signal file has no `legs`: it predates the portfolio export. "
                     "Re-run scripts/paper_trade.py --export.",
            )

        try:
            self.client.login()
            positions = tuple(self.client.get_positions())
            tripped = self._refresh_breakers(now)
        except IGError as exc:
            return BridgeReport("broker_error", mode, run_at, note=str(exc))

        orders = [self._plan(leg, positions, now) for leg in legs]
        for order in orders:
            if live and order.decision and order.decision.allowed:
                self._submit(order)
            self.db.log_execution_attempt(order.as_row(run_at, mode))
        return BridgeReport("ok", mode, run_at, orders=orders, breakers_tripped=tripped)

    # ---- per leg ----

    def _plan(self, leg: dict[str, Any], positions: tuple[dict, ...], now: datetime) -> PlannedOrder:
        instrument = str(leg.get("instrument") or "?")
        signal_date = str(leg.get("date") or "")
        planned = PlannedOrder(instrument=instrument, signal_date=signal_date)

        if leg.get("status") != "ok":
            planned.refusal = f"signal leg status is {leg.get('status')!r}"
            return planned

        epic = IG_EPICS.get(instrument)
        if epic is None:
            planned.refusal = f"{instrument} has no IG epic mapping — execution is AUD/USD only"
            return planned
        planned.epic = epic

        action = leg.get("action")
        direction = direction_of(action)
        if direction is None:
            # NO TRADE is the common case and is not an error; it is still logged
            # so the dry-run record shows the day was considered.
            planned.refusal = f"RiskAgent action is {action!r}"
            return planned
        planned.direction = direction

        stop = leg.get("stop_loss")
        if stop is None:
            planned.refusal = (
                "signal leg carries no stop_loss — sizing is derived from the stop, "
                "so there is nothing to size off"
            )
            return planned

        try:
            snapshot = self.client.get_market(epic)
        except IGError as exc:
            planned.refusal = f"could not read the market: {exc}"
            return planned

        request = TradeRequest(
            epic=epic,
            direction=direction,
            signal_date=signal_date[:10],
            stop_distance_pct=float(stop),
            risk_action=str(action),
        )
        market = MarketContext(
            bid=float(snapshot["bid"]),
            offer=float(snapshot["offer"]),
            market_status=str(snapshot["marketStatus"]),
            positions=positions,
        )
        planned.decision = self.engine.evaluate(request, market, now=now)
        return planned

    def _submit(self, order: PlannedOrder) -> None:
        """Only reachable when Decision.allowed is True, which needs EXECUTION_ENABLED."""
        try:
            confirmation = self.client.open_position(
                direction=order.direction,
                size=order.size,
                stop_distance=order.stop_distance_points,
                epic=order.epic,
            )
        except IGError as exc:
            order.deal_status = f"ERROR: {exc}"
            return
        order.submitted = True
        order.deal_reference = confirmation.get("dealReference")
        # A dealReference is an acknowledgement; dealStatus is the fill.
        order.deal_status = confirmation.get("dealStatus")

    # ---- breakers ----

    def _refresh_breakers(self, now: datetime) -> list[str]:
        """Record today's realised balance, then let the engine trip on it.

        Cash `balance`, never `balance + profitLoss`: equity would fold in the
        mark-to-market of the manual CFD position, letting a human trade halt or
        un-halt the system.
        """
        account = self.client.get_account()
        balance = (account.get("balance") or {}).get("balance")
        if balance is None:
            return []
        self.engine.record_balance(now.date().isoformat(), float(balance))
        return self.engine.check_breakers(now)
