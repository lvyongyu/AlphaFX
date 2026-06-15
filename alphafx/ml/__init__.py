"""ML research layer for AlphaFX.

A simple, leak-free ML signal for comparison against the rule-based strategy —
NOT a replacement for it. The rule signal stays primary/live. Trained with
time-series (walk-forward) validation on point-in-time features; given the small
independent sample (~N/horizon), it overfits easily and is surfaced with a
prominent warning.
"""

from .agent import MLSignalAgent, ml_rule_agreement
from .dataset import FEATURE_COLUMNS, build_dataset, build_targets

__all__ = [
    "MLSignalAgent",
    "ml_rule_agreement",
    "build_dataset",
    "build_targets",
    "FEATURE_COLUMNS",
]
