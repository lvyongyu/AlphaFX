# risk_engine Checklist

The pre-trade checks `alphafx/execution/risk_engine.py` enforces, and the ones
still outstanding. Companion to [execution-flow.md](execution-flow.md), which
shows where the engine sits in the refusal chain.

**Step A.2 has landed.** The engine is implemented and covered by
`tests/test_risk_engine.py` (40 tests, offline). Two rows remain open and are
marked as such: the economic calendar has no data in it yet (row 18, which
therefore refuses every trade), and the cross-pair correlation cap (row 8) is
deferred.

The check list was cross-referenced against the risk-analyst dimensions used by
[TradingAgents](https://github.com/TauricResearch/TradingAgents) — a multi-agent
LLM trading framework whose risk stage covers a broader surface than AlphaFX's
`RiskAgent` did. The dimensions were worth borrowing; **the implementation was
not**. There, a language model reads the numbers and judges. Here every rule is a
deterministic comparison against a constant, testable offline with no API key and
no ambiguity about what it will do at 3am. A rule an LLM evaluates is not a risk
limit, it is a suggestion.

Two rules govern edits:

1. **Only tighten.** A row may be made stricter at any time. Loosening one — or
   adding a bypass flag — is a strategy change and needs the same evidence bar as
   a signal change.
2. **Deterministic and offline-testable.** Every row must be decidable from
   arguments and constants. If a check needs a judgement call, it is not ready to
   be a check.

The engine does no I/O beyond its own SQLite breaker state. The caller fetches
the quote, positions and balance from `IGClient` and passes them in as plain
data, which is what keeps the whole check surface unit-testable.

## Status legend

- **built** — enforced today, with a test.
- **open** — implemented but not yet usable, or deliberately deferred; the row
  says which.
- **n/a** — the TradingAgents dimension does not transfer; the row records why.

## Position and sizing

| # | Check | Status | Where |
| --- | --- | --- | --- |
| 1 | Order carries a server-side stop | built ×2 | `risk_engine` refuses a missing/non-positive stop; `ig_client.open_position` refuses again. The duplicate is the point. |
| 2 | Size ≤ 1.0 lot | built ×2 | `MAX_SIZE`, imported by `risk_engine` from `ig_client` rather than redefined, so the two can never drift apart. |
| 3 | Size ≥ the broker minimum, and refuse rather than round | built | A stop wide enough to price the position below the minimum refuses. Rounding up would breach #4. The RiskAgent's widest stop (12%) really does hit this. |
| 4 | Risk per trade ≤ 1% of a **fixed notional** | built | Size back-solved from the stop against `NOTIONAL_AUD = 10,000`, **not** the IG balance: this Demo account holds ~9 figures, so 1% of the real balance would not constrain anything and the rule would ship untested. Structural, not conventional — `MarketContext` has no field that could carry a balance. |
| 5 | One open position per instrument | built | Enforced against live IG positions, by underlying (#6). `PaperBroker.place()` enforces the same rule on the paper book. |
| 6 | Match existing positions by underlying, not only by epic | built | `underlying_of()` reduces `CS.D.AUDUSD.MINI.IP` and `CS.D.AUDUSD.CFD.IP` to `AUDUSD`, so the manual standard-contract long blocks a MINI order. Epic matching would not have seen it. |
| 7 | Total open exposure cap across instruments | built | `MAX_TOTAL_LOTS = 3.0`, counting every open position including the manual one — real risk on the account is real whoever opened it. |
| 8 | Cap concurrent same-direction USD exposure | **open — deferred** | Needs a correlation input the engine does not have. #7 is the crude proxy until then. Recorded rather than pretended. |

## Volatility and regime

| # | Check | Status | Where |
| --- | --- | --- | --- |
| 9 | Extreme volatility → no trade | built | `RiskAgent`, 20d vol > 0.25. Reaches the engine as `risk_action = "NO TRADE"` (#11). |
| 10 | Elevated volatility → leverage capped | built | `RiskAgent`, vol > 0.18 caps leverage at 2. |
| 11 | `RiskAgent` NO TRADE is obeyed unconditionally | built | The engine may only refuse more than the upstream suggestion, never less. Tested with every other check clean and the master gate open. |
| 12 | Stop distance is volatility-scaled, not fixed | built | `RiskAgent` disaster stop, 2.5× horizon vol clamped to 4–12%; the engine converts it to IG points. |

## Account-level circuit breakers

| # | Check | Status | Where |
| --- | --- | --- | --- |
| 13 | Monthly loss ≥ 5% → no new positions this month | built | State persists in SQLite (`execution_breakers`). The daily job is a fresh process every run, so an in-memory breaker would reset before it could fire — a test asserts a brand-new engine on the same DB still sees the trip. |
| 14 | Equity drawdown ≥ 15% from peak → full stop | built | High-water mark from `execution_equity`. **Manual reset only**: `clear_breaker` rejects "auto"/"system"/blank, and a balance recovery does not clear it — a recovery is not evidence that whatever caused the loss is fixed. |
| 15 | Breaker state is auditable | built | Every trip records the value, the threshold, the timestamp, a human-readable detail, and who cleared it. `breaker_history()` returns the lot. |
| 16 | Breaker computed from **realised** account PnL | built | `record_balance()` takes IG's cash `balance`, deliberately not equity. Cash excludes every open position's mark-to-market, so the manual `CS.D.AUDUSD.CFD.IP` long cannot halt — or un-halt — the system. |

## Market and event conditions

| # | Check | Status | Where |
| --- | --- | --- | --- |
| 17 | `marketStatus == TRADEABLE` | built ×2 | Checked before sending, so a weekend fails on our side with a clear reason instead of as an IG rejection. |
| 18 | Event blackout ±2h around RBA / FOMC / CPI / NFP | built, **calendar empty** | The check works and **fails closed**: `ECONOMIC_CALENDAR` ships empty with `CALENDAR_THROUGH = None`, so every trade is refused until it is filled from the official schedules. A schedule written from memory would look authoritative and be wrong, and a blackout that silently lapses when a feed goes stale is worse than one that refuses. Filling it is a data task, not a code task. |
| 19 | Spread sanity check before sending | built | Refuses when the spread exceeds 25% of the stop distance. Catches thin liquidity and stale quotes — the case a backtest never shows. |
| 20 | Liquidity / market-impact assessment | **n/a** | A TradingAgents dimension that does not transfer. At ≤1.0 lot on a major pair, impact is not measurable; #19 is the useful residue. |
| 21 | Demo host is pinned | built | `BASE_URL` is a module constant, not config, with a test. |

## Signal-quality gate

| # | Check | Status | Where |
| --- | --- | --- | --- |
| 22 | Probability must come from realised history | built | `RiskAgent.EVIDENCE_SOURCES`; a fallback prior can never open a position. |
| 23 | Probability ≥ `MIN_CONFIDENCE` (0.52) | built | `RiskAgent`. |
| 24 | Signal freshness | built | Refuses a signal more than one **business** day old, so a stalled cron cannot replay yesterday's call — and a Friday signal is still fresh on Monday. |
| 25 | Master gate: automated execution stays off | built ×2 | `daily.yml`'s cron is disabled, and `EXECUTION_ENABLED = False` carries the same switch in code, so re-enabling the workflow alone cannot start trading. `Decision` reports `risk_checks_passed` separately from `allowed`, so a dry-run can say "this would have traded, but the gate is shut" instead of collapsing both cases into one refusal. |

## Deliberately not adopted from TradingAgents

- **A risk agent that reasons.** Their risk stage is an LLM reading reports and
  deciding. Every row above is a comparison against a constant. No LLM output may
  enter this module — enforced by
  `test_decision_path_never_imports_the_llm_or_review_layer`.
- **A portfolio manager that can approve or override.** There is no override path
  here. Checks refuse; nothing grants exceptions.
- **Lessons-learned fed back into sizing.** Post-hoc narrative adjusting the next
  trade's risk is exactly the loop `alphafx/review.py` is built to avoid.

## The three open questions, answered

1. **On a signal flip, close or hold?** — **Hold, for now.** execution-flow.md
   argues for *close, do not reverse*, and the argument is sound, but every
   backtest number on record (walk-forward, the multi-window runs, the −23% over
   ten years) describes hold-to-barrier. "Close on flip" is a different strategy
   with no evidence behind it, so `risk_engine` does **not** implement it. Add an
   early-exit mode to the backtest, compare over the same windows, and wire it in
   only if it survives. Shipping it now would mean running an unvalidated
   strategy under the banner of risk management.
2. **Match positions by epic or by underlying?** — **Underlying.** Strictly the
   more conservative reading, and the only one that can see the manual
   standard-contract long. Row 6.
3. **What resets a tripped breaker?** — **A human, by name.** `clear_breaker`
   requires an identifier and rejects "auto"/"system"/blank. Time clears only the
   monthly limit, and only because "no new positions *this month*" is that rule's
   own definition rather than an auto-clear. Row 14.

## What A.2 does not cover

- **Placing anything.** The engine decides; `bridge.py` and
  `scripts/execute_demo.py` (step A.3) are what would act on the decision, and
  they are not written.
- **Closing or managing an open position.** Every check here is pre-trade. Exit
  logic still lives in `PaperBroker`'s time barrier.
- **The calendar data** for row 18.

## Verifying

```bash
.venv/bin/python -m pytest tests/test_risk_engine.py -v   # 40 tests, offline, no credentials, no quota
.venv/bin/python -m pytest tests/test_execution.py -v     # 24 tests, the ig_client floor beneath it
```
