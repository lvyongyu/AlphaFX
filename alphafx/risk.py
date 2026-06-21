from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RiskSuggestion:
    action: str
    position_size: str
    leverage: float
    stop_loss: float
    take_profit: float | None
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
    # The signal's edge is validated at a ~20-day horizon; the paper engine's
    # primary exit is a time barrier at this horizon (PaperBroker.MAX_HOLDING_DAYS).
    HORIZON_DAYS = 20

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
        # Triple-barrier exit (Lopez de Prado): the PRIMARY exit is the time barrier
        # at the ~20-day signal horizon. The stop-loss is a WIDE, volatility-scaled
        # DISASTER stop (~2.5x the horizon vol) — wide enough that it rarely fires in
        # normal noise (so it never cuts winners or the validated edge), but it bounds
        # tail/gap risk. There is NO take-profit: capping winners while losers run to
        # the time barrier flips the reward:risk the wrong way (verified by 5y replay).
        horizon_vol = float(volatility) * (self.HORIZON_DAYS / 252.0) ** 0.5 if pd.notna(volatility) else 0.04
        disaster_stop = max(0.04, min(0.12, 2.5 * horizon_vol))
        return RiskSuggestion(
            action=action,
            position_size="Small" if high_vol or action == "NO TRADE" else "Standard",
            leverage=leverage,
            stop_loss=disaster_stop,
            take_profit=None,
            warning=" ".join(warning_parts),
            max_risk_per_trade=max_risk_per_trade,
            regime="extreme_vol_no_trade" if extreme_vol else "elevated_vol" if high_vol else "normal",
        )
