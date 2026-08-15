# risk_engine Checklist (step A.2 spec)

The pre-trade checks `alphafx/execution/risk_engine.py` must implement, and which
of them already exist elsewhere. Companion to
[execution-flow.md](execution-flow.md), which shows where the engine sits in the
refusal chain.

The check list is cross-referenced against the risk-analyst dimensions used by
[TradingAgents](https://github.com/TauricResearch/TradingAgents) — a multi-agent
LLM trading framework whose risk stage covers a broader surface than AlphaFX's
current `RiskAgent`. The dimensions are worth borrowing; **the implementation is
not**. There, a language model reads the numbers and judges. Here every rule is a
deterministic comparison against a constant, testable offline with no API key and
no ambiguity about what it will do at 3am. A rule an LLM evaluates is not a risk
limit, it is a suggestion.

Two rules that govern the whole table:

1. **Only tighten.** A row may be made stricter at any time. Loosening one — or
   adding a bypass flag — is a strategy change and needs the same evidence bar as
   a signal change.
2. **Deterministic and offline-testable.** Every row must be decidable from
   arguments and constants. If a check needs a judgement call, it is not ready to
   be a check.

## Status legend

- **built** — enforced today, with a test.
- **A.2** — this engine has to implement it.
- **later** — a real requirement, deliberately deferred; the row records why.
- **n/a** — the TradingAgents dimension does not transfer; the row records why.

## Position and sizing

| # | Check | Status | Where / what A.2 must do |
| --- | --- | --- | --- |
| 1 | Order carries a server-side stop | built + **A.2** | `ig_client.open_position` raises without `stop_distance`. A.2 repeats it as the upper gate — the duplicate is the point. |
| 2 | Size ≤ 1.0 lot | built + **A.2** | `MAX_SIZE` in `ig_client`. The **only** limit that actually binds on this Demo account (see #4). |
| 3 | Size ≥ the broker minimum, and refuse rather than round | **A.2** | A stop wide enough to price the position below the minimum must refuse, not silently shrink. Rounding up would breach #4. |
| 4 | Risk per trade ≤ 1% of a **fixed notional** | **A.2** | Size back-solved from the stop distance. Size off a hardcoded 10,000 AUD, **not** the IG balance: this Demo account holds ~9 figures, so 1% of the real balance is no constraint at all and the rule would ship untested. |
| 5 | One open position per instrument | built (paper) + **A.2** | `PaperBroker.place()` vetoes a second. A.2 needs the same rule against live IG positions, which raises #6. |
| 6 | Match existing positions by underlying, not only by epic | **A.2** | Open design question in execution-flow.md. The account holds a manual `CS.D.AUDUSD.CFD.IP` long while the code trades `...MINI.IP`; epic matching cannot see it and would open a MINI short against a standard long. |
| 7 | Total open exposure cap across instruments | **A.2** | The live portfolio is 3 correlated USD pairs. Per-instrument caps alone permit 3 simultaneous same-direction USD bets, which is one bet with three tickets. |
| 8 | Cap concurrent same-direction USD exposure | **later** | Needs a correlation input the engine does not have yet. Until then #7 is the crude proxy. Record the decision rather than pretend it is covered. |

## Volatility and regime

| # | Check | Status | Where / what A.2 must do |
| --- | --- | --- | --- |
| 9 | Extreme volatility → no trade | built | `RiskAgent`, 20d vol > 0.25. |
| 10 | Elevated volatility → leverage capped | built | `RiskAgent`, vol > 0.18 caps leverage at 2. |
| 11 | `RiskAgent` NO TRADE is obeyed unconditionally | **A.2** | The engine may only ever refuse more than the upstream suggestion, never less. Worth an explicit test: feed a NO TRADE and assert refusal regardless of every other input. |
| 12 | Stop distance is volatility-scaled, not fixed | built | `RiskAgent` disaster stop, 2.5× horizon vol clamped to 4–12%. |

## Account-level circuit breakers

| # | Check | Status | Where / what A.2 must do |
| --- | --- | --- | --- |
| 13 | Monthly loss ≥ 5% → no new positions this month | **A.2** | State must persist in SQLite. An in-memory breaker resets on every cron invocation, which means it does not exist. |
| 14 | Equity drawdown ≥ 15% from peak → full stop | **A.2** | Same persistence requirement. Needs a stored high-water mark, and must require a manual reset — an automatic one is not a circuit breaker. |
| 15 | Breaker state is auditable | **A.2** | Persist when a breaker tripped, on what number, and when it cleared. A breaker nobody can review after the fact cannot be trusted before the fact. |
| 16 | Breaker computed from **realised** account PnL | **A.2** | Exclude the manual `CS.D.AUDUSD.CFD.IP` position — it is not the script's, and letting it move the breaker means a human trade can halt or unhalt the system. |

## Market and event conditions

| # | Check | Status | Where / what A.2 must do |
| --- | --- | --- | --- |
| 17 | `marketStatus == TRADEABLE` | built | IG rejects otherwise; `ig_client` surfaces it. A.2 should check before sending, to fail on our side with a clear reason. |
| 18 | Event blackout ±2h around RBA / FOMC / CPI / NFP | **A.2** | Needs a calendar source. A **hardcoded schedule is acceptable and preferred** for the first version: a blackout that fails closed on a stale calendar is safer than one that silently lapses when a feed breaks. |
| 19 | Spread sanity check before sending | **A.2** | Refuse when the live spread exceeds a constant, or is a large fraction of the stop distance. Catches thin liquidity and stale quotes — the case a backtest never shows. |
| 20 | Liquidity / market-impact assessment | **n/a** | A TradingAgents dimension that does not transfer. At ≤1.0 lot on a major pair, impact is not measurable; #19 is the useful residue. |
| 21 | Demo host is pinned | built | `BASE_URL` is a module constant, not config, with a test. |

## Signal-quality gate

| # | Check | Status | Where / what A.2 must do |
| --- | --- | --- | --- |
| 22 | Probability must come from realised history | built | `RiskAgent.EVIDENCE_SOURCES`; a fallback prior can never open a position. |
| 23 | Probability ≥ `MIN_CONFIDENCE` (0.52) | built | `RiskAgent`. |
| 24 | Signal freshness | **A.2** | `data/latest_signal.json` carries a date. Refuse to act on a signal older than one trading day — a stalled cron job must not replay yesterday's call into today's market. |
| 25 | Master gate: automated execution stays off | built (workflow) + **A.2** | `daily.yml` cron is disabled. A.2 should carry the same switch in code, defaulting to refuse, so a re-enabled workflow alone cannot start trading. |

## Deliberately not adopted from TradingAgents

- **A risk agent that reasons.** Their risk stage is an LLM reading reports and
  deciding. Every row above is a comparison against a constant. No LLM output may
  enter this module — enforced by
  `test_decision_path_never_imports_the_llm_or_review_layer`.
- **A portfolio manager that can approve or override.** There is no override path
  here. Checks refuse; nothing grants exceptions.
- **Lessons-learned fed back into sizing.** Post-hoc narrative adjusting the next
  trade's risk is exactly the loop `alphafx/review.py` is built to avoid.

## Open questions A.2 has to answer

These are decisions, not code, and they are cheaper to settle before writing the
module than after:

1. **On a signal flip, close or hold?** execution-flow.md argues *close, do not
   reverse* — but every backtest number on record describes hold-to-barrier, so
   "close on flip" is an unvalidated strategy. Add an early-exit mode to the
   backtest and compare before wiring it in.
2. **Match positions by epic or by underlying?** (#6) Determines whether the
   manual standard-contract long is visible to the engine.
3. **What resets a tripped breaker?** (#14) Manual only is the safe answer; the
   mechanism still has to be specified.

## Verifying

Once A.2 lands, this file is the checklist for its test file: every **A.2** row
above should map to at least one test in `tests/test_execution.py`, and every
**built** row should already have one.

```bash
.venv/bin/python -m pytest tests/test_execution.py -v   # offline, no credentials, no quota
```
