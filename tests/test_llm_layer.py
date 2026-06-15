from __future__ import annotations

import pandas as pd

from alphafx.agents import (
    AIExplanationAgent,
    ContrarianAgent,
    FeatureAgent,
    JudgeAgent,
    QuantSignalAgent,
    RiskAgent,
)
from alphafx.config import DEFAULT_SYMBOLS
from alphafx.database import Database
from alphafx.llm.client import LLMResponse
from alphafx.llm.evidence import build_evidence_pack
from alphafx.llm.narrator import LLMContrarianAgent, LLMExplanationAgent, LLMJudgeAgent


def sample_market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=160)
    rows = []
    for i, dt in enumerate(dates):
        rows.extend(
            [
                {"date": dt, "symbol": DEFAULT_SYMBOLS.audusd, "open": 0.65, "high": 0.66, "low": 0.64, "close": 0.65 + i * 0.001, "source": "test"},
                {"date": dt, "symbol": DEFAULT_SYMBOLS.dxy, "open": 105, "high": 106, "low": 104, "close": 105 - i * 0.05, "source": "test"},
                {"date": dt, "symbol": DEFAULT_SYMBOLS.vix, "open": 15, "high": 16, "low": 14, "close": 18 - i * 0.02, "source": "test"},
            ]
        )
    return pd.DataFrame(rows)


def build_signal_and_factors():
    market = sample_market_data()
    features = FeatureAgent().build_features(market)
    signals = QuantSignalAgent().generate_signals(features)
    latest_signal = QuantSignalAgent().latest_signal(signals)
    latest_feature = features.iloc[-1]
    factors = FeatureAgent().factor_table(latest_feature, latest_signal)
    return latest_signal, factors


class FakeClient:
    """Injectable stand-in for LLMClient — no network."""

    def __init__(self, text: str = "", structured: dict | None = None) -> None:
        self.text = text
        self.structured = structured
        self.calls: list[tuple] = []

    def call(self, system, user, model, max_tokens, schema=None, use_thinking=False):
        self.calls.append((system, user, model, schema))
        return LLMResponse(
            text=self.text,
            structured=self.structured,
            model=model,
            input_tokens=10,
            output_tokens=20,
            cache_read_tokens=0,
            prompt_hash="deadbeefcafe0000",
        )


def test_evidence_pack_has_no_fabricated_data():
    signal, factors = build_signal_and_factors()
    pack = build_evidence_pack(signal, factors)

    assert pack["signal"] == "bullish"
    assert pack["probability"] == float(signal["probability"])
    # Exactly the five known factors, none invented.
    names = {f["factor"] for f in pack["factors"]}
    assert names == {"AUD momentum", "DXY trend", "Yield spread", "Iron ore trend", "VIX"}
    # Macro data was not supplied, so some factors are unavailable — marked, not dropped.
    unavailable = [f for f in pack["factors"] if f["stance"] == "not available"]
    assert unavailable, "missing factors must be kept and marked, not silently dropped"
    assert all(f["current_value"] is None for f in unavailable)


def test_explanation_falls_back_when_disabled(monkeypatch):
    monkeypatch.setenv("ALPHAFX_LLM_DISABLED", "1")
    signal, factors = build_signal_and_factors()
    template = AIExplanationAgent()
    agent = LLMExplanationAgent(fallback=template)  # client=None + disabled -> no client
    assert agent.client is None
    assert agent.explain(signal, factors) == template.explain(signal, factors)


def test_direction_guard_rejects_contradicting_text():
    signal, factors = build_signal_and_factors()
    assert signal["signal"] == "bullish"
    template = AIExplanationAgent()
    fake = FakeClient(text="The setup is clearly bearish; AUD/USD should fall from here.")
    agent = LLMExplanationAgent(fallback=template, client=fake)
    result = agent.explain(signal, factors)
    # Contradicting LLM text is discarded in favor of the template.
    assert result == template.explain(signal, factors)


def test_judge_never_overrides_quant_signal():
    signal, factors = build_signal_and_factors()
    risk = RiskAgent().suggest(signal=signal["signal"], probability=signal["probability"], volatility=0.1)
    structured = {
        "final_signal": "bearish",  # LLM tries to flip it
        "final_confidence": "High",
        "trade": "SELL AUD/USD",
        "llm_dissent": "I read the macro backdrop as bearish.",
        "explanation": "Summary of the setup.",
    }
    fake = FakeClient(structured=structured)
    judge = LLMJudgeAgent(fallback=JudgeAgent(), client=fake)
    out = judge.judge(signal, risk, "explanation text", {"main_risk": "x", "alternative_scenario": "y", "watch": "z"})

    assert out["final_signal"] == signal["signal"] == "bullish"  # enforced, not the LLM's "bearish"
    assert out["trade"] == risk.action  # deterministic risk engine, not the LLM's "SELL"
    assert "I read the macro backdrop as bearish." in out["llm_dissent"]


def test_llm_call_is_audited(tmp_path):
    db = Database(tmp_path / "audit.db")
    signal, factors = build_signal_and_factors()
    template = AIExplanationAgent()
    fake = FakeClient(text="The quant model is bullish; AUD momentum and a falling DXY support it.")
    agent = LLMExplanationAgent(fallback=template, client=fake, db=db)
    agent.explain(signal, factors)

    calls = db.load_llm_calls()
    assert len(calls) == 1
    row = calls.iloc[0]
    assert row["role"] == "explanation"
    assert row["input_tokens"] == 10
    assert row["output_tokens"] == 20
    assert row["prompt_hash"] == "deadbeefcafe0000"


def test_contrarian_uses_structured_output():
    signal, factors = build_signal_and_factors()
    structured = {"main_risk": "USD strength returns", "alternative_scenario": "DXY rebounds", "watch": "FOMC"}
    fake = FakeClient(structured=structured)
    agent = LLMContrarianAgent(fallback=ContrarianAgent(), client=fake)
    out = agent.critique(signal, factors)
    assert out["main_risk"] == "USD strength returns"
    assert out["watch"] == "FOMC"
