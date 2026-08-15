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

# One turn in the red-team debate. Note what is NOT here: no direction, no
# verdict, no confidence. There is deliberately no field a debater could use to
# express a signal of its own, so a multi-round debate cannot drift into
# re-rating the quant call no matter how many rounds it runs.
DEBATE_TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "claim": {"type": "string"},
        "evidence_used": {"type": "string"},
        "weakest_link": {"type": "string"},
    },
    "required": ["claim", "evidence_used", "weakest_link"],
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
