"""Execution layer: the only part of AlphaFX that talks to a broker.

Everything above this package (signals, risk suggestions, paper trading) is
research and never places an order. This package is deliberately kept thin and
deterministic:

    ig_client    REST transport only — no strategy, no sizing decisions
    risk_engine  hard, non-negotiable pre-trade checks (step A.2)
    bridge       signal JSON -> validated IG order (step A.3)

The broker endpoint is locked to IG's Demo environment; see `ig_client.BASE_URL`.
"""
