from __future__ import annotations

# Backward-compatible facade. The original ~1000-line module was split into
# focused modules; importing from `alphafx.agents` keeps working unchanged.
from .backtest import BacktestAgent
from .collect import DataAgent
from .diagnostics import FactorDiagnosticsAgent, SignalDiagnosticsAgent
from .explain import AIExplanationAgent, ContrarianAgent, JudgeAgent
from .features import FeatureAgent
from .paper import PaperJournalAgent
from .risk import RiskAgent, RiskSuggestion
from .signals import QuantSignalAgent
from .walk_forward import WalkForwardAgent

__all__ = [
    "DataAgent",
    "FeatureAgent",
    "QuantSignalAgent",
    "SignalDiagnosticsAgent",
    "FactorDiagnosticsAgent",
    "RiskSuggestion",
    "RiskAgent",
    "BacktestAgent",
    "WalkForwardAgent",
    "PaperJournalAgent",
    "AIExplanationAgent",
    "ContrarianAgent",
    "JudgeAgent",
]
