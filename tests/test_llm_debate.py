from __future__ import annotations

import pytest

from alphafx.agents import ContrarianAgent, FeatureAgent, QuantSignalAgent
from alphafx.database import Database
from alphafx.llm.client import LLMError, LLMResponse
from alphafx.llm.debate import PROPONENT, SKEPTIC, LLMDebateAgent
from alphafx.llm.schemas import DEBATE_TURN_SCHEMA
from factories import sample_market_data


def build_signal_and_factors():
    features = FeatureAgent().build_features(sample_market_data())
    signals = QuantSignalAgent().generate_signals(features)
    latest_signal = QuantSignalAgent().latest_signal(signals)
    factors = FeatureAgent().factor_table(features.iloc[-1], latest_signal)
    return latest_signal, factors


def turn_payload(tag: str) -> dict:
    return {
        "claim": f"claim-{tag}",
        "evidence_used": f"evidence-{tag}",
        "weakest_link": f"weakness-{tag}",
    }


SYNTHESIS = {
    "main_risk": "Calibration rests on a thin sample.",
    "alternative_scenario": "DXY rebounds and the score decays.",
    "watch": "Next 20 sessions of DXY and the calibration sample count.",
}


class ScriptedClient:
    """Injectable stand-in for LLMClient — no network.

    `responses` is consumed one entry per call; an entry that is an exception is
    raised instead of returned, which is how the mid-debate failure path is
    exercised.
    """

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def call(self, system, user, model, max_tokens, schema=None, use_thinking=False):
        self.calls.append({"system": system, "user": user, "model": model, "schema": schema})
        item = self.responses.pop(0) if self.responses else {}
        if isinstance(item, Exception):
            raise item
        return LLMResponse(
            text="",
            structured=item,
            model=model,
            input_tokens=10,
            output_tokens=20,
            cache_read_tokens=0,
            prompt_hash="deadbeefcafe0000",
        )


def test_debate_falls_back_to_template_when_disabled(monkeypatch):
    monkeypatch.setenv("ALPHAFX_LLM_DISABLED", "1")
    signal, factors = build_signal_and_factors()
    template = ContrarianAgent()
    agent = LLMDebateAgent(fallback=template)

    assert agent.client is None
    result = agent.debate(signal, factors)
    assert result.source == "template"
    assert result.turns == []
    assert result.as_contrarian() == template.critique(signal, factors)


def test_debate_runs_both_sides_every_round_then_synthesises():
    signal, factors = build_signal_and_factors()
    client = ScriptedClient([turn_payload(str(i)) for i in range(4)] + [SYNTHESIS])
    agent = LLMDebateAgent(fallback=ContrarianAgent(), client=client)

    result = agent.debate(signal, factors, rounds=2)

    assert len(client.calls) == 5  # 2 * rounds + 1 synthesis
    assert [t.role for t in result.turns] == [PROPONENT, SKEPTIC, PROPONENT, SKEPTIC]
    assert [t.round for t in result.turns] == [1, 1, 2, 2]
    assert result.rounds == 2
    assert result.source == "llm"
    assert result.as_contrarian() == SYNTHESIS


def test_each_turn_sees_the_previous_transcript():
    signal, factors = build_signal_and_factors()
    client = ScriptedClient([turn_payload("a"), turn_payload("b"), SYNTHESIS])
    LLMDebateAgent(fallback=ContrarianAgent(), client=client).debate(signal, factors, rounds=1)

    opening, rebuttal, synthesis = client.calls
    assert "No rebuttal exists yet" in opening["user"]
    assert "claim-a" in rebuttal["user"]  # the skeptic must answer the proponent
    assert "weakness-a" in rebuttal["user"]
    assert "claim-b" in synthesis["user"]


def test_debate_has_no_field_for_a_direction():
    # Structural boundary: no matter how many rounds run, the schema gives a
    # debater nowhere to record a signal, verdict, or confidence of its own.
    forbidden = {"signal", "direction", "final_signal", "verdict", "recommendation", "confidence"}
    assert set(DEBATE_TURN_SCHEMA["properties"]) == {"claim", "evidence_used", "weakest_link"}
    assert not forbidden & set(DEBATE_TURN_SCHEMA["properties"])
    assert DEBATE_TURN_SCHEMA["additionalProperties"] is False


def test_result_signal_is_the_quant_signal_not_the_models():
    signal, factors = build_signal_and_factors()
    assert signal["signal"] == "bullish"
    # A model that smuggles a direction in anyway is ignored.
    smuggled = turn_payload("x") | {"signal": "bearish", "final_signal": "bearish"}
    client = ScriptedClient([smuggled, turn_payload("y"), SYNTHESIS])
    result = LLMDebateAgent(fallback=ContrarianAgent(), client=client).debate(signal, factors, rounds=1)

    assert result.signal == "bullish"
    assert not hasattr(result.turns[0], "signal")


@pytest.mark.parametrize("requested,expected", [(0, 1), (1, 1), (99, LLMDebateAgent.MAX_ROUNDS)])
def test_rounds_are_clamped(requested, expected):
    signal, factors = build_signal_and_factors()
    client = ScriptedClient([turn_payload(str(i)) for i in range(2 * LLMDebateAgent.MAX_ROUNDS)] + [SYNTHESIS])
    result = LLMDebateAgent(fallback=ContrarianAgent(), client=client).debate(
        signal, factors, rounds=requested
    )
    assert result.rounds == expected


def test_failure_midway_keeps_the_partial_transcript():
    signal, factors = build_signal_and_factors()
    template = ContrarianAgent()
    client = ScriptedClient([turn_payload("a"), turn_payload("b"), LLMError("boom")])
    result = LLMDebateAgent(fallback=template, client=client).debate(signal, factors, rounds=2)

    assert len(result.turns) == 2  # round 1 survived, round 2 stopped at the error
    assert result.rounds == 1
    # The synthesis never ran, so the fields degrade to the template critique.
    assert result.as_contrarian() == template.critique(signal, factors)


def test_debate_calls_are_audited(tmp_path):
    db = Database(tmp_path / "audit.db")
    signal, factors = build_signal_and_factors()
    client = ScriptedClient([turn_payload("a"), turn_payload("b"), SYNTHESIS])
    LLMDebateAgent(fallback=ContrarianAgent(), client=client, db=db).debate(signal, factors, rounds=1)

    roles = list(db.load_llm_calls()["role"])
    assert sorted(roles) == ["debate_proponent", "debate_skeptic", "debate_synthesis"]
