from __future__ import annotations

from typing import Any

import pandas as pd

from .risk import RiskSuggestion


class AIExplanationAgent:
    def explain(self, signal: pd.Series, factors: pd.DataFrame) -> str:
        if signal.empty:
            return "No signal is available yet. Download data and generate features first."
        bullish = factors[factors["stance"] == "bullish"]["factor"].tolist()
        bearish = factors[factors["stance"] == "bearish"]["factor"].tolist()
        missing = factors[factors["stance"] == "not available"]["factor"].tolist()
        text = [
            f"The quant model is {signal['signal']} with a {signal['probability']:.0%} probability and {signal['confidence'].lower()} confidence.",
            f"The raw score is {signal['score']:.0f}, based on available factor contributions.",
        ]
        if bullish:
            text.append("Bullish support comes from " + ", ".join(bullish) + ".")
        if bearish:
            text.append("Bearish pressure comes from " + ", ".join(bearish) + ".")
        if missing:
            text.append("Missing factors are reported as unavailable: " + ", ".join(missing) + ".")
        return " ".join(text)

class ContrarianAgent:
    def critique(self, signal: pd.Series, factors: pd.DataFrame) -> dict[str, str]:
        if signal.empty:
            return {"main_risk": "No signal to critique.", "alternative_scenario": "", "watch": ""}
        bearish = factors[factors["stance"] == "bearish"]["factor"].tolist()
        bullish = factors[factors["stance"] == "bullish"]["factor"].tolist()
        if signal["signal"] == "bullish":
            risk = "The bullish case could fail if USD strength returns or risk sentiment deteriorates."
            opposing = bearish or ["DXY, VIX, yields, or commodity trends"]
            scenario = "A reversal in " + ", ".join(opposing) + " would weaken the long AUD/USD setup."
        elif signal["signal"] == "bearish":
            risk = "The bearish case could fail if global risk appetite improves or commodities strengthen."
            opposing = bullish or ["AUD momentum, yield spread, or iron ore"]
            scenario = "Improvement in " + ", ".join(opposing) + " would challenge the short AUD/USD setup."
        else:
            risk = "Neutral signals can hide regime changes because factor disagreement is high."
            scenario = "A cleaner trend in DXY, VIX, yields, or iron ore would move the model away from neutral."
        return {
            "main_risk": risk,
            "alternative_scenario": scenario,
            "watch": "Watch the next 20 trading days of DXY, VIX, AU-US yield spread, iron ore, and AUD/USD momentum.",
        }

class JudgeAgent:
    def judge(self, signal: pd.Series, risk: RiskSuggestion, explanation: str, contrarian: dict[str, str]) -> dict[str, Any]:
        if signal.empty:
            return {
                "final_signal": "neutral",
                "final_confidence": "Low",
                "trade": "NO TRADE",
                "explanation": "No complete signal is available.",
            }
        return {
            "final_signal": signal["signal"],
            "final_confidence": signal["confidence"],
            "trade": risk.action,
            "explanation": f"{explanation} Contrarian view: {contrarian['main_risk']}",
        }
