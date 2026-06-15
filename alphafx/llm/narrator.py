from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..config import LLMConfig, default_llm_config, llm_enabled
from .client import LLMClient, LLMError, LLMResponse
from .evidence import build_evidence_pack
from .prompts import CONTRARIAN_SYSTEM, EXPLANATION_SYSTEM, JUDGE_SYSTEM, user_payload
from .schemas import CONTRARIAN_SCHEMA, JUDGE_SCHEMA

_OPPOSITE = {"bullish": "bearish", "bearish": "bullish"}


def _contradicts(text: str, signal: str | None) -> bool:
    """Direction-consistency guard: reject text that flips the quant direction.

    Trips only when the OPPOSITE direction word appears and the actual signal
    word does not — the egregious "LLM said the other way" case. Allows text
    that mentions both (normal discussion).
    """
    if not text or signal not in _OPPOSITE:
        return False
    low = text.lower()
    return _OPPOSITE[signal] in low and signal not in low


class _Narrator:
    role = "explanation"

    def __init__(
        self,
        fallback: Any,
        client: LLMClient | None = None,
        config: LLMConfig | None = None,
        db: Any = None,
    ) -> None:
        self.fallback = fallback
        self.config = config or default_llm_config()
        if client is not None:
            self.client: LLMClient | None = client
        elif llm_enabled():
            self.client = LLMClient(self.config)
        else:
            self.client = None
        self.db = db

    def _log(self, system: str, user: str, resp: LLMResponse | None, when: Any = None) -> None:
        if self.db is None or resp is None:
            return
        try:
            self.db.log_llm_call(
                {
                    "date": str(when) if when is not None else None,
                    "role": self.role,
                    "model": resp.model,
                    "prompt_hash": resp.prompt_hash,
                    "system_prompt": system,
                    "user_payload": user,
                    "response_text": resp.text,
                    "structured_output": json.dumps(resp.structured) if resp.structured is not None else None,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cache_read_tokens": resp.cache_read_tokens,
                }
            )
        except Exception:  # noqa: BLE001 - audit logging must never break the app
            pass


class LLMExplanationAgent(_Narrator):
    role = "explanation"

    def explain(self, signal: pd.Series, factors: pd.DataFrame) -> str:
        if signal is None or signal.empty or self.client is None:
            return self.fallback.explain(signal, factors)
        evidence = build_evidence_pack(signal, factors)
        user = user_payload(evidence)
        try:
            resp = self.client.call(
                EXPLANATION_SYSTEM, user, self.config.narration_model, self.config.narration_max_tokens
            )
        except LLMError:
            return self.fallback.explain(signal, factors)
        self._log(EXPLANATION_SYSTEM, user, resp, when=signal.get("date"))
        if not resp.text or _contradicts(resp.text, signal.get("signal")):
            return self.fallback.explain(signal, factors)
        return resp.text


class LLMContrarianAgent(_Narrator):
    role = "contrarian"

    def critique(self, signal: pd.Series, factors: pd.DataFrame) -> dict[str, str]:
        if signal is None or signal.empty or self.client is None:
            return self.fallback.critique(signal, factors)
        evidence = build_evidence_pack(signal, factors)
        user = user_payload(evidence)
        try:
            resp = self.client.call(
                CONTRARIAN_SYSTEM,
                user,
                self.config.narration_model,
                self.config.narration_max_tokens,
                schema=CONTRARIAN_SCHEMA,
            )
        except LLMError:
            return self.fallback.critique(signal, factors)
        self._log(CONTRARIAN_SYSTEM, user, resp, when=signal.get("date"))
        out = resp.structured
        if not out:
            return self.fallback.critique(signal, factors)
        template = self.fallback.critique(signal, factors)
        return {
            "main_risk": out.get("main_risk") or template["main_risk"],
            "alternative_scenario": out.get("alternative_scenario") or template["alternative_scenario"],
            "watch": out.get("watch") or template["watch"],
        }


class LLMJudgeAgent(_Narrator):
    role = "judge"

    def judge(
        self,
        signal: pd.Series,
        risk: Any,
        explanation: str,
        contrarian: dict[str, str],
    ) -> dict[str, Any]:
        if signal is None or signal.empty or self.client is None:
            return self.fallback.judge(signal, risk, explanation, contrarian)
        evidence = build_evidence_pack(signal, factors=None, risk=risk)
        user = user_payload(evidence, extra={"Explanation": explanation, "Contrarian view": contrarian})
        try:
            resp = self.client.call(
                JUDGE_SYSTEM,
                user,
                self.config.report_model,
                self.config.report_max_tokens,
                schema=JUDGE_SCHEMA,
                use_thinking=True,
            )
        except LLMError:
            return self.fallback.judge(signal, risk, explanation, contrarian)
        self._log(JUDGE_SYSTEM, user, resp, when=signal.get("date"))
        out = resp.structured or {}

        # ENFORCED IN CODE: the LLM can never override the quant signal, the
        # quant confidence, or the deterministic risk-engine trade action.
        quant_signal = signal.get("signal")
        dissent = str(out.get("llm_dissent", "") or "").strip()
        narrative = str(out.get("explanation", "") or "").strip() or explanation
        extra = f" LLM dissent: {dissent}" if dissent and dissent.lower() != "none" else ""
        return {
            "final_signal": quant_signal,
            "final_confidence": signal.get("confidence"),
            "trade": risk.action,
            "explanation": f"{narrative}{extra}".strip(),
            "llm_dissent": dissent,
        }
