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
    # Below the 0.52 confidence floor -> no trade either way.
    assert RiskAgent().suggest("bearish", 0.50, 0.10).action == "NO TRADE"
    assert RiskAgent().suggest("bullish", 0.50, 0.10).action == "NO TRADE"


def test_confidence_floor_is_052():
    assert RiskAgent().suggest("bullish", 0.52, 0.10).action == "BUY AUD/USD"
    assert RiskAgent().suggest("bullish", 0.519, 0.10).action == "NO TRADE"


def test_fallback_prior_never_trades_without_calibration_evidence():
    # The fallback score-map probability has NO realised track record, so even a
    # high fallback value must not open a trade (this was the circular-gate bug:
    # the only signals that traded were the ones with no evidence behind them).
    prob = QuantSignalAgent.map_probability(-3)
    assert prob == 0.60
    no_evidence = RiskAgent().suggest("bearish", prob, 0.10, probability_source="fallback_score_map")
    assert no_evidence.action == "NO TRADE"
    # The same probability, but backed by realised history, IS allowed to trade.
    with_evidence = RiskAgent().suggest("bearish", prob, 0.10, probability_source="historical_calibration")
    assert with_evidence.action == "SELL AUD/USD"


def test_extreme_volatility_blocks_otherwise_valid_trade():
    risk = RiskAgent().suggest("bullish", 0.70, volatility=0.30)
    assert risk.action == "NO TRADE"
    assert risk.regime == "extreme_vol_no_trade"
