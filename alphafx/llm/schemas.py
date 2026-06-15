from __future__ import annotations

# Structured-output schemas for the LLM layer. The judge schema includes
# final_signal, but the application enforces final_signal == quant signal in
# code regardless of what the model returns. llm_dissent is the only place the
# model may express disagreement.

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "final_signal": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "final_confidence": {"type": "string", "enum": ["Low", "Medium", "High"]},
        "trade": {"type": "string"},
        "llm_dissent": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["final_signal", "final_confidence", "trade", "llm_dissent", "explanation"],
    "additionalProperties": False,
}

CONTRARIAN_SCHEMA = {
    "type": "object",
    "properties": {
        "main_risk": {"type": "string"},
        "alternative_scenario": {"type": "string"},
        "watch": {"type": "string"},
    },
    "required": ["main_risk", "alternative_scenario", "watch"],
    "additionalProperties": False,
}
