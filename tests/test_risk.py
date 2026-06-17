from __future__ import annotations

from alphafx.risk import RiskAgent
from alphafx.signals import QuantSignalAgent


# `probability` is directional confidence (P(signal correct)). The calibrated
# probability (a hit rate) and the fallback score-map now share this convention,
# so RiskAgent must gate both directions on the same >= threshold.


def test_calibrated_bearish_high_confidence_trades():
    # A confident bearish signal has a HIGH probability (e.g. a 0.70 hit rate).
    risk = RiskAgent().suggest(signal="bearish", probability=0.70, volatility=0.10)
    assert risk.action == "SELL AUD/USD"


def test_calibrated_bullish_high_confidence_trades():
    risk = RiskAgent().suggest(signal="bullish", probability=0.70, volatility=0.10)
    assert risk.action == "BUY AUD/USD"


def test_low_confidence_is_no_trade_both_directions():
    assert RiskAgent().suggest("bearish", 0.55, 0.10).action == "NO TRADE"
    assert RiskAgent().suggest("bullish", 0.55, 0.10).action == "NO TRADE"


def test_fallback_strong_bearish_score_trades():
    # The fallback map yields the same directional-confidence convention:
    # score -3 -> 0.60, which clears the trade threshold.
    prob = QuantSignalAgent.map_probability(-3)
    assert prob == 0.60
    assert RiskAgent().suggest("bearish", prob, 0.10).action == "SELL AUD/USD"


def test_extreme_volatility_blocks_otherwise_valid_trade():
    risk = RiskAgent().suggest("bullish", 0.70, volatility=0.30)
    assert risk.action == "NO TRADE"
    assert risk.regime == "extreme_vol_no_trade"
