from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "alphafx.db"


@dataclass(frozen=True)
class Symbols:
    audusd: str = "AUDUSD=X"
    dxy: str = "DX-Y.NYB"
    vix: str = "^VIX"


DEFAULT_SYMBOLS = Symbols()


@dataclass(frozen=True)
class LLMConfig:
    """Config for the LLM explanation layer.

    The LLM only explains, challenges, and summarizes the quant signal — it
    never sets the signal. `report_model` runs the heavier judge/report calls;
    `narration_model` runs the lighter per-day explanation and contrarian calls
    and may be pointed at a cheaper tier (e.g. claude-haiku-4-5) by env var.
    """

    report_model: str = "claude-opus-4-8"
    narration_model: str = "claude-opus-4-8"
    narration_max_tokens: int = 1024
    report_max_tokens: int = 2048


def default_llm_config() -> LLMConfig:
    return LLMConfig(
        report_model=os.environ.get("ALPHAFX_LLM_REPORT_MODEL", "claude-opus-4-8"),
        narration_model=os.environ.get("ALPHAFX_LLM_NARRATION_MODEL", "claude-opus-4-8"),
    )


def llm_enabled() -> bool:
    """LLM is opt-in: on only when an API key is present and not force-disabled.

    With no key (or ALPHAFX_LLM_DISABLED set), the app falls back to the
    template agents and still runs fully offline.
    """
    if os.environ.get("ALPHAFX_LLM_DISABLED", "").strip().lower() in {"1", "true", "yes"}:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))

