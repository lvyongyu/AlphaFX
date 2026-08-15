"""Hard pre-trade checks. Deterministic by construction — never intelligent.

Every rule here is a comparison against a constant. Nothing in this module
reasons, ranks, or weighs: given the same inputs it always returns the same
decision, and it is fully testable offline with no credentials and no quota.
`docs/risk-engine-checklist.md` is the spec, row by row.

This engine sits ABOVE `ig_client` and BELOW nothing — it is the gate. Several
of its checks are repeated inside `ig_client.open_position`; that duplication is
deliberate, so bypassing the gate still hits a floor.

Two invariants govern edits:

  1. **Only tighten.** Making a limit stricter is routine. Loosening one — or
     adding a bypass flag — is a strategy change and needs the same evidence bar
     as a signal change.
  2. **No LLM, ever.** Enforced mechanically by
     `test_decision_path_never_imports_the_llm_or_review_layer`.

The engine performs no I/O of its own beyond its own SQLite state: the caller
fetches the quote, positions and balance from `IGClient` and hands them over as
plain data. That is what keeps the whole check surface unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from ..database import Database
from .ig_client import DEFAULT_EPIC, MAX_SIZE

# --- master gate -------------------------------------------------------------
# Automated execution stays off until the signal-quality gate opens (roadmap
# step B.4: positive out-of-sample walk-forward including spread costs). This
# constant is the code-side half of that switch, so re-enabling the daily.yml
# cron alone cannot start trading. [DO NOT flip] without the walk-forward
# evidence and an explicit decision recorded in the commit.
EXECUTION_ENABLED = False

# --- sizing ------------------------------------------------------------------
# Size off a FIXED notional, never the IG balance. The Demo account holds ~9
# figures, so 1% of its real balance is not a constraint and the rule would ship
# having never bound anything. 10,000 AUD is roughly the scale a first real
# account would run at, which is the scale the rule needs to be tested at.
NOTIONAL_AUD = 10_000.0
MAX_RISK_PER_TRADE = 0.01
CONTRACT_UNITS = 10_000.0  # 1 mini lot = 10,000 AUD
SIZE_STEP = 0.01
# Conservative floor. IG publishes the true minimum in the market snapshot
# (`dealingRules.minDealSize`); when the caller supplies it, the stricter of the
# two wins. This constant only applies when IG's figure is unavailable.
MIN_SIZE = 0.1
POINT = 0.0001  # AUD/USD: one point is one pip

# --- exposure ----------------------------------------------------------------
MAX_TOTAL_LOTS = 3.0  # across every instrument, manual positions included

# --- circuit breakers --------------------------------------------------------
MONTHLY_LOSS_LIMIT = 0.05
MAX_DRAWDOWN = 0.15
MONTHLY_LOSS_BREAKER = "monthly_loss"
DRAWDOWN_BREAKER = "account_drawdown"

# --- market conditions -------------------------------------------------------
TRADEABLE = "TRADEABLE"
# A spread this large relative to the stop means the trade starts a meaningful
# fraction of the way to being stopped out. Catches thin liquidity and stale
# quotes — the case a backtest never shows.
MAX_SPREAD_FRACTION_OF_STOP = 0.25

# --- freshness ---------------------------------------------------------------
# A stalled cron must not replay yesterday's call into today's market.
MAX_SIGNAL_AGE_BDAYS = 1

# --- event blackout ----------------------------------------------------------
BLACKOUT_HOURS = 2
# (UTC ISO timestamp, label) for RBA / FOMC / CPI / NFP releases.
#
# [CALENDAR_FROM, CALENDAR_THROUGH] is the window this list is COMPLETE over —
# not the span of its entries. The blackout check fails CLOSED outside it,
# because a calendar that quietly runs out is a guard that lapses without anyone
# noticing, and a month with no entries is indistinguishable from a month with
# no events. Entries beyond CALENDAR_THROUGH are inert until it is moved
# forward; they sit here so extending coverage is a one-line change once the
# missing series is confirmed.
#
# Local times converted to UTC at the offset in force on each date (US DST ends
# 2026-11-01, AU DST starts 2026-10-04):
#   RBA decision    14:30 Sydney on the second day of the two-day meeting
#   FOMC statement  14:00 US Eastern on the second day
#   CPI / payrolls  08:30 US Eastern
#
# Sources: RBA 2026 Monetary Policy Board dates (media release mr-25-02); the
# Fed's published 2026 FOMC calendar; the BLS Employment Situation schedule.
# `test_calendar_coverage_is_complete_for_every_covered_month` fails if
# CALENDAR_THROUGH is moved past a month with no CPI or payrolls entry.
ECONOMIC_CALENDAR: tuple[tuple[str, str], ...] = (
    ("2026-09-04T12:30:00Z", "US non-farm payrolls"),
    ("2026-09-11T12:30:00Z", "US CPI"),
    ("2026-09-16T18:00:00Z", "FOMC decision"),
    ("2026-09-29T04:30:00Z", "RBA decision"),
    # --- beyond the covered window ---------------------------------------
    # Verified, but the CPI release dates for Oct/Nov/Dec 2026 are not, and a
    # month missing its CPI blackout is worse than a month that refuses
    # outright. Add them and move CALENDAR_THROUGH together.
    ("2026-10-02T12:30:00Z", "US non-farm payrolls"),
    ("2026-10-28T18:00:00Z", "FOMC decision"),
    ("2026-11-03T03:30:00Z", "RBA decision"),
    ("2026-11-06T13:30:00Z", "US non-farm payrolls"),
    ("2026-12-04T13:30:00Z", "US non-farm payrolls"),
    ("2026-12-08T03:30:00Z", "RBA decision"),
    ("2026-12-09T19:00:00Z", "FOMC decision"),
)
CALENDAR_FROM: str | None = "2026-08-15T00:00:00Z"
CALENDAR_THROUGH: str | None = "2026-09-30T23:59:59Z"

NO_TRADE = "NO TRADE"


@dataclass(frozen=True)
class TradeRequest:
    """What the caller wants to do, in the engine's own units."""

    epic: str
    direction: str  # "BUY" or "SELL"
    signal_date: str  # the date the signal is for, YYYY-MM-DD
    stop_distance_pct: float  # RiskAgent's disaster stop, as a fraction of price
    risk_action: str  # RiskAgent.action verbatim, e.g. "BUY AUD/USD" or "NO TRADE"


@dataclass(frozen=True)
class MarketContext:
    """The broker's view, fetched by the caller and passed in as plain data."""

    bid: float
    offer: float
    market_status: str
    positions: tuple[dict, ...] = ()
    min_deal_size: float | None = None

    @property
    def mid(self) -> float:
        return (float(self.bid) + float(self.offer)) / 2.0

    @property
    def spread_points(self) -> float:
        return (float(self.offer) - float(self.bid)) / POINT


@dataclass
class Decision:
    """The engine's verdict. Every failed check is listed, not just the first.

    `risk_checks_passed` and `allowed` differ only by the master gate, so a
    dry-run can report "this would have traded, but automated execution is off"
    instead of collapsing both cases into one refusal.
    """

    risk_checks_passed: bool
    allowed: bool
    size: float = 0.0
    stop_distance_points: float = 0.0
    reasons: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.allowed:
            return f"ALLOW {self.size} lots, stop {self.stop_distance_points} pts"
        return "REFUSE: " + "; ".join(self.reasons)


def underlying_of(epic: str) -> str:
    """The instrument an epic trades, ignoring the contract type.

    `CS.D.AUDUSD.MINI.IP` and `CS.D.AUDUSD.CFD.IP` are the same underlying in
    two wrappers. Matching positions by epic would let the engine open a MINI
    short while a standard-contract long sits on the account unseen, so it
    matches by underlying instead — strictly the more conservative reading.
    """
    parts = epic.split(".")
    return parts[2] if len(parts) >= 3 else epic


class RiskEngine:
    """Runs every pre-trade check and owns the persisted breaker state."""

    def __init__(
        self,
        db: Database | None = None,
        calendar: tuple[tuple[str, str], ...] | None = None,
        calendar_through: str | None = None,
        calendar_from: str | None = None,
    ) -> None:
        self.db = db or Database()
        # Passing `calendar` replaces the whole window, so a caller can never end
        # up with an injected list judged against the shipped coverage claim.
        self.calendar = ECONOMIC_CALENDAR if calendar is None else calendar
        if calendar is None:
            self.calendar_from = CALENDAR_FROM
            self.calendar_through = CALENDAR_THROUGH
        else:
            self.calendar_from = calendar_from
            self.calendar_through = calendar_through

    # ---- the gate ----

    def evaluate(
        self,
        request: TradeRequest,
        market: MarketContext,
        now: datetime | None = None,
    ) -> Decision:
        now = now or datetime.now(timezone.utc)
        reasons: list[str] = []
        passed: list[str] = []

        def record(ok: bool, name: str, reason: str) -> None:
            if ok:
                passed.append(name)
            else:
                reasons.append(reason)

        # Upstream suggestion. The engine may only ever refuse more than the
        # RiskAgent, never less, so NO TRADE ends the matter.
        record(
            request.risk_action != NO_TRADE,
            "risk_agent_action",
            "RiskAgent returned NO TRADE",
        )
        record(
            request.direction in ("BUY", "SELL"),
            "direction",
            f"direction must be BUY or SELL (got {request.direction!r})",
        )

        # Signal freshness.
        age = _business_days_between(request.signal_date, now)
        if age is None:
            reasons.append(f"signal date {request.signal_date!r} is unreadable")
        else:
            record(
                age <= MAX_SIGNAL_AGE_BDAYS,
                "signal_freshness",
                f"signal dated {request.signal_date} is {age} business days old "
                f"(limit {MAX_SIGNAL_AGE_BDAYS})",
            )

        # Market conditions.
        record(
            market.market_status == TRADEABLE,
            "market_status",
            f"market status is {market.market_status}, not {TRADEABLE}",
        )

        # Stop and sizing. Both are reported even when an earlier check failed,
        # so a dry-run log shows what the order WOULD have been.
        stop_points, size, sizing_error = self._size(request, market)
        record(sizing_error is None, "sizing", sizing_error or "")

        record(
            stop_points > 0,
            "stop_present",
            "stop distance is missing or non-positive",
        )
        record(
            stop_points <= 0 or market.spread_points <= MAX_SPREAD_FRACTION_OF_STOP * stop_points,
            "spread",
            f"spread {market.spread_points:.1f} pts exceeds "
            f"{MAX_SPREAD_FRACTION_OF_STOP:.0%} of the {stop_points:.1f} pt stop",
        )

        # Positions.
        clash = self._same_underlying(request.epic, market.positions)
        record(
            not clash,
            "one_position_per_underlying",
            f"already holding {clash} on the same underlying" if clash else "",
        )
        exposure = _total_lots(market.positions)
        record(
            exposure + size <= MAX_TOTAL_LOTS,
            "total_exposure",
            f"total exposure {exposure + size:.2f} lots would exceed the "
            f"{MAX_TOTAL_LOTS} lot cap",
        )

        # Circuit breakers (already-tripped state only — tripping is check_breakers').
        active = self.active_breakers(now)
        record(
            not active,
            "circuit_breakers",
            "circuit breaker active: " + ", ".join(active),
        )

        # Event blackout.
        blackout = self._blackout(now)
        record(blackout is None, "event_blackout", blackout or "")

        risk_ok = not reasons
        if not EXECUTION_ENABLED:
            reasons.append(
                "automated execution is disabled (EXECUTION_ENABLED is False until "
                "the signal-quality gate opens)"
            )
        return Decision(
            risk_checks_passed=risk_ok,
            allowed=risk_ok and EXECUTION_ENABLED,
            size=size if risk_ok else 0.0,
            stop_distance_points=stop_points if risk_ok else 0.0,
            reasons=reasons,
            passed=passed,
        )

    # ---- sizing ----

    def _size(self, request: TradeRequest, market: MarketContext) -> tuple[float, float, str | None]:
        """(stop in points, size in lots, error). Size is back-solved from the stop."""
        stop_fraction = float(request.stop_distance_pct or 0.0)
        if stop_fraction <= 0:
            return 0.0, 0.0, "stop_distance_pct is missing or non-positive"

        stop_points = round(market.mid * stop_fraction / POINT, 1)
        risk_budget = NOTIONAL_AUD * MAX_RISK_PER_TRADE
        loss_per_lot = CONTRACT_UNITS * stop_fraction
        # Floor to the contract step: rounding UP would breach the 1% limit.
        size = float(np.floor(risk_budget / loss_per_lot / SIZE_STEP) * SIZE_STEP)
        size = round(size, 2)

        floor = MIN_SIZE if market.min_deal_size is None else max(MIN_SIZE, float(market.min_deal_size))
        if size < floor:
            # Refuse rather than round up. A stop wide enough to price the
            # position below the broker minimum means this trade cannot be taken
            # at 1% risk, and taking it anyway would breach the limit.
            return stop_points, 0.0, (
                f"a {stop_fraction:.1%} stop prices the position at {size} lots, "
                f"below the {floor} minimum — refusing rather than rounding up"
            )
        if size > MAX_SIZE:
            return stop_points, 0.0, f"size {size} exceeds the {MAX_SIZE} lot cap"
        return stop_points, size, None

    def _same_underlying(self, epic: str, positions: tuple[dict, ...]) -> str | None:
        target = underlying_of(epic)
        for position in positions:
            held_epic = (position.get("market") or {}).get("epic", "")
            if held_epic and underlying_of(held_epic) == target:
                return held_epic
        return None

    # ---- event blackout ----

    def _blackout(self, now: datetime) -> str | None:
        """None when clear, otherwise the reason. Fails closed on a stale calendar."""
        if self.calendar_through is None:
            return (
                "economic calendar is not populated — the event blackout cannot be "
                "verified, so the trade is refused (see ECONOMIC_CALENDAR)"
            )
        through = _parse_utc(self.calendar_through)
        if through is None or now > through:
            return (
                f"economic calendar only covers through {self.calendar_through}; "
                "refusing rather than trading past an expired blackout list"
            )
        start = _parse_utc(self.calendar_from) if self.calendar_from else None
        if start is not None and now < start:
            return (
                f"economic calendar coverage starts at {self.calendar_from}; "
                "a month with no entries is indistinguishable from a month with "
                "no events, so this is refused rather than assumed clear"
            )
        window = timedelta(hours=BLACKOUT_HOURS)
        for stamp, label in self.calendar:
            when = _parse_utc(stamp)
            if when is not None and abs(now - when) <= window:
                return f"within {BLACKOUT_HOURS}h of {label} at {stamp}"
        return None

    # ---- circuit breakers ----

    def record_balance(self, when: str, balance: float) -> None:
        """Store the realised CASH balance — deliberately not equity.

        Cash excludes the mark-to-market of every open position, which is exactly
        what checklist row 16 asks for: the manual `CS.D.AUDUSD.CFD.IP` long is
        not the script's, and letting its unrealised swing move a breaker would
        let a human trade halt (or un-halt) the system.
        """
        self.db.record_execution_balance(when, balance)

    def check_breakers(self, now: datetime | None = None) -> list[str]:
        """Evaluate the loss limits against recorded balances and trip on breach.

        Called by the caller after refreshing the balance; `evaluate` only reads
        already-tripped state, so a check can never be skipped by not calling it
        and then be silently absent from the decision.
        """
        now = now or datetime.now(timezone.utc)
        balances = self.db.load_execution_balances()
        if len(balances) < 2:
            return []

        current = float(balances.iloc[-1]["balance"])
        tripped: list[str] = []

        peak = float(balances["balance"].max())
        if peak > 0 and (current / peak - 1.0) <= -MAX_DRAWDOWN:
            tripped.append(
                self._trip(
                    DRAWDOWN_BREAKER,
                    scope=None,  # never expires; only a human clears it
                    now=now,
                    value=current / peak - 1.0,
                    threshold=-MAX_DRAWDOWN,
                    detail=f"balance {current} against a peak of {peak}",
                )
            )

        month = str(balances.iloc[-1]["date"])[:7]
        in_month = balances[balances["date"].str[:7] == month]
        if len(in_month) >= 2:
            start = float(in_month.iloc[0]["balance"])
            if start > 0 and (current / start - 1.0) <= -MONTHLY_LOSS_LIMIT:
                tripped.append(
                    self._trip(
                        MONTHLY_LOSS_BREAKER,
                        scope=month,
                        now=now,
                        value=current / start - 1.0,
                        threshold=-MONTHLY_LOSS_LIMIT,
                        detail=f"balance {current} against {start} at the start of {month}",
                    )
                )
        return [name for name in tripped if name]

    def _trip(self, breaker: str, scope: str | None, now: datetime,
              value: float, threshold: float, detail: str) -> str:
        active = self.db.load_execution_breakers(active_only=True)
        if not active.empty:
            already = active[active["breaker"] == breaker]
            if scope is None and not already.empty:
                return ""  # already tripped and not yet cleared
            if scope is not None and (already["scope"] == scope).any():
                return ""
        self.db.trip_execution_breaker(
            {
                "breaker": breaker,
                "scope": scope,
                "tripped_at": now.isoformat(),
                "tripped_value": value,
                "threshold": threshold,
                "detail": detail,
            }
        )
        return breaker

    def active_breakers(self, now: datetime | None = None) -> list[str]:
        """Breakers currently blocking new positions.

        A monthly breaker is scoped to the month it tripped in and stops binding
        once that month is over — that is the rule's own definition, not an
        auto-clear. The drawdown breaker has no scope and binds until a human
        clears it.
        """
        now = now or datetime.now(timezone.utc)
        current_month = now.strftime("%Y-%m")
        rows = self.db.load_execution_breakers(active_only=True)
        if rows.empty:
            return []
        blocking = []
        for _, row in rows.iterrows():
            scope = row["scope"]
            if scope and scope != current_month:
                continue
            blocking.append(str(row["breaker"]))
        return sorted(set(blocking))

    def clear_breaker(self, breaker: str, cleared_by: str, now: datetime | None = None) -> int:
        """Manual reset. There is no automatic path back from a tripped breaker —
        an equity recovery is not evidence that whatever caused the loss is fixed."""
        now = now or datetime.now(timezone.utc)
        if not cleared_by or cleared_by.strip().lower() in ("", "auto", "automatic", "system"):
            raise ValueError("clear_breaker needs a human identifier in cleared_by")
        return self.db.clear_execution_breaker(breaker, now.isoformat(), cleared_by)

    def breaker_history(self):
        """Full audit trail: what tripped, on what number, when, and who cleared it."""
        return self.db.load_execution_breakers()


def _total_lots(positions: tuple[dict, ...]) -> float:
    """Open size across every instrument, manual positions included.

    Unlike the breakers, exposure counts the manual position: it is real risk on
    the account whoever opened it, and counting it is the conservative direction.
    """
    total = 0.0
    for position in positions:
        held = position.get("position") or {}
        try:
            total += abs(float(held.get("size") or 0.0))
        except (TypeError, ValueError):
            continue
    return total


def _business_days_between(signal_date: str, now: datetime) -> int | None:
    try:
        start = np.datetime64(str(signal_date)[:10], "D")
    except ValueError:
        return None
    return int(np.busday_count(start, np.datetime64(now.date(), "D")))


def _parse_utc(stamp: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


__all__ = [
    "RiskEngine",
    "TradeRequest",
    "MarketContext",
    "Decision",
    "underlying_of",
    "DEFAULT_EPIC",
    "EXECUTION_ENABLED",
]
