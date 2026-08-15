"""Execution layer: the only part of AlphaFX that talks to a broker.

Everything above this package (signals, risk suggestions, paper trading) is
research and never places an order. This package is deliberately kept thin and
deterministic:

    ig_client    REST transport only — no strategy, no sizing decisions
    risk_engine  hard, non-negotiable pre-trade checks — deterministic, never
                 intelligent; see docs/risk-engine-checklist.md
    bridge       signal JSON -> validated IG order; plumbing only, no rules of
                 its own, and dry-run by default

`risk_engine.EXECUTION_ENABLED` is False: nothing in AlphaFX places an order
until the signal-quality gate opens, and that switch lives in code so that
re-enabling the daily.yml cron alone cannot start trading. It also makes
`scripts/execute_demo.py --live` inert by construction rather than by care.

The broker endpoint is locked to IG's Demo environment; see `ig_client.BASE_URL`.
"""
