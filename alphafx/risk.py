from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RiskSuggestion:
    action: str
    position_size: str
    leverage: float
    stop_loss: float
    take_profit: float
    warning: str
    max_risk_per_trade: float = 0.01
    regime: str = "normal"

class RiskAgent:
    # Confidence the SIGNAL DIRECTION is correct, required to trade. The calibrated
    # probability is NOT a reliable monotonic confidence dial (a threshold sweep
    # shows performance peaks at the LOW end and 0.53-0.54 is a dead zone), and at
    # ~30 trades nothing is statistically significant. So the real gate is the
    # evidence requirement below; this floor is set to the lowest principled level
    # — "calibrated hit rate at least slightly better than a coin flip" — which
    # also sits in the stable positive plateau (0.50-0.52). Not peak-fitted.
    MIN_CONFIDENCE = 0.52
    # A probability is only trade-worthy if it is backed by realised history.
    # The fallback score-map probability is a fixed prior with NO track record,
    # so it must never clear the gate on its own (that was the circular bug:
    # the only signals that "traded" were the ones with no evidence behind them).
    EVIDENCE_SOURCES = ("historical_calibration", "walkforward_calibration")

    def suggest(
        self,
        signal: str,
        probability: float,
        volatility: float | None,
        max_drawdown: float | None = None,
        user_leverage: float = 2.0,
        max_risk_per_trade: float = 0.01,
        probability_source: str = "historical_calibration",
        min_confidence: float | None = None,
    ) -> RiskSuggestion:
        min_confidence = self.MIN_CONFIDENCE if min_confidence is None else float(min_confidence)
        leverage = min(max(float(user_leverage), 1.0), 5.0)
        high_vol = pd.notna(volatility) and volatility > 0.18
        extreme_vol = pd.notna(volatility) and volatility > 0.25
        # Only a probability backed by realised outcomes may open a trade; a bare
        # fallback prior cannot, no matter how high its fixed value.
        has_evidence = probability_source in self.EVIDENCE_SOURCES
        if high_vol:
            leverage = min(leverage, 2.0)
        if extreme_vol:
            action = "NO TRADE"
            leverage = 1.0
        # `probability` is directional confidence (P(signal correct)), so the
        # same >= threshold gates both directions — a confident bearish signal
        # has a HIGH probability, not a low one.
        elif has_evidence and signal == "bullish" and probability >= min_confidence:
            action = "BUY AUD/USD"
        elif has_evidence and signal == "bearish" and probability >= min_confidence:
            action = "SELL AUD/USD"
        else:
            action = "NO TRADE"
            leverage = 1.0
        warning_parts = ["Paper trading only. No live order execution is implemented."]
        if extreme_vol:
            warning_parts.append("Extreme volatility regime: no-trade guard is active.")
        if not extreme_vol and not has_evidence and signal in ("bullish", "bearish"):
            warning_parts.append(
                "No calibrated track record for this signal yet (fallback prior only): no trade."
            )
        if high_vol:
            warning_parts.append("Volatility is elevated, so leverage is capped.")
        if max_drawdown is not None and pd.notna(max_drawdown) and max_drawdown < -0.15:
            warning_parts.append("Backtest drawdown is material; reduce size or avoid the trade.")
        vol_stop = max(0.01, min(0.06, float(volatility) / 5.0)) if pd.notna(volatility) else 0.02
        return RiskSuggestion(
            action=action,
            position_size="Small" if high_vol or action == "NO TRADE" else "Standard",
            leverage=leverage,
            stop_loss=vol_stop,
            take_profit=vol_stop * 2.0,
            warning=" ".join(warning_parts),
            max_risk_per_trade=max_risk_per_trade,
            regime="extreme_vol_no_trade" if extreme_vol else "elevated_vol" if high_vol else "normal",
        )
