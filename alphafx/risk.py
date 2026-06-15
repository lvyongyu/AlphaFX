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
    def suggest(
        self,
        signal: str,
        probability: float,
        volatility: float | None,
        max_drawdown: float | None = None,
        user_leverage: float = 2.0,
        max_risk_per_trade: float = 0.01,
    ) -> RiskSuggestion:
        leverage = min(max(float(user_leverage), 1.0), 5.0)
        high_vol = pd.notna(volatility) and volatility > 0.18
        extreme_vol = pd.notna(volatility) and volatility > 0.25
        if high_vol:
            leverage = min(leverage, 2.0)
        if extreme_vol:
            action = "NO TRADE"
            leverage = 1.0
        elif signal == "bullish" and probability >= 0.60:
            action = "BUY AUD/USD"
        elif signal == "bearish" and probability <= 0.40:
            action = "SELL AUD/USD"
        else:
            action = "NO TRADE"
            leverage = 1.0
        warning_parts = ["Paper trading only. No live order execution is implemented."]
        if extreme_vol:
            warning_parts.append("Extreme volatility regime: no-trade guard is active.")
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
