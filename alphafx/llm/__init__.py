"""LLM explanation layer for AlphaFX.

The LLM explains, challenges, and summarizes the quant signal but never sets
it. Quant agents own signal/score/probability; the judge's final_signal is
forced to equal the quant signal in code. Falls back to the template agents
when no API key is set or a call fails.
"""

from .client import LLMClient, LLMError, LLMResponse
from .evidence import build_evidence_pack
from .narrator import LLMContrarianAgent, LLMExplanationAgent, LLMJudgeAgent

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "build_evidence_pack",
    "LLMExplanationAgent",
    "LLMContrarianAgent",
    "LLMJudgeAgent",
]
