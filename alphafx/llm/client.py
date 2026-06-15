from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from ..config import LLMConfig, default_llm_config


class LLMError(RuntimeError):
    """Raised on any LLM transport/parse failure so callers can fall back."""


@dataclass
class LLMResponse:
    text: str
    structured: dict[str, Any] | None
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    prompt_hash: str


class LLMClient:
    """Thin Anthropic wrapper: caching, error surface, optional structured output.

    Transport only — it holds no prompt content. The fixed system prompt is
    cached; the volatile evidence pack goes last in the user turn. `client` is
    injectable so tests run with no network.
    """

    def __init__(self, config: LLMConfig | None = None, client: Any = None) -> None:
        self.config = config or default_llm_config()
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on install
            raise LLMError("anthropic SDK is not installed") from exc
        self._client = anthropic.Anthropic()
        return self._client

    def call(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
        schema: dict[str, Any] | None = None,
        use_thinking: bool = False,
    ) -> LLMResponse:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
        }
        if use_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}

        try:
            resp = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize to one clean error for callers
            raise LLMError(str(exc)) from exc

        text = "".join(
            getattr(block, "text", "") for block in resp.content if getattr(block, "type", None) == "text"
        )
        structured: dict[str, Any] | None = None
        if schema is not None and text:
            try:
                structured = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMError(f"structured output was not valid JSON: {exc}") from exc

        usage = getattr(resp, "usage", None)
        prompt_hash = hashlib.sha256(f"{system}\x00{user}".encode("utf-8")).hexdigest()[:16]
        return LLMResponse(
            text=text,
            structured=structured,
            model=getattr(resp, "model", model),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            prompt_hash=prompt_hash,
        )
