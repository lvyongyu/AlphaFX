"""Multi-round red-team debate over a pre-computed signal.

An upgrade of the single-shot contrarian pass: a proponent and a skeptic argue
across N rounds, each seeing the other's previous turn, and a synthesis step
reports what survived. Adversarial pressure surfaces assumptions a single
critique tends to miss.

The boundary is unchanged and is enforced structurally, not by prompt wording:
`DEBATE_TURN_SCHEMA` has no direction/verdict field, and `DebateResult.signal`
is copied from the quant layer in code. There is nothing here for the model to
overturn, however many rounds it argues.

Cost: `2 * rounds + 1` LLM calls per debate (5 at the default 2 rounds), so this
is an on-demand review tool, not something to run on every signal. It never runs
inside the backtest, walk-forward, or ML loops.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .client import LLMError
from .evidence import build_evidence_pack
from .narrator import _Narrator
from .prompts import (
    DEBATE_PROPONENT_SYSTEM,
    DEBATE_SKEPTIC_SYSTEM,
    DEBATE_SYNTHESIS_SYSTEM,
    user_payload,
)
from .schemas import CONTRARIAN_SCHEMA, DEBATE_TURN_SCHEMA

PROPONENT = "proponent"
SKEPTIC = "skeptic"


@dataclass
class DebateTurn:
    role: str
    round: int
    claim: str
    evidence_used: str
    weakest_link: str


@dataclass
class DebateResult:
    """Transcript plus the synthesis, in the shape the contrarian slot expects.

    `signal` is an echo of the quant signal written by this module, never by the
    model — it exists so a stored transcript says which call it argued about.
    """

    signal: str | None
    rounds: int = 0
    turns: list[DebateTurn] = field(default_factory=list)
    main_risk: str = ""
    alternative_scenario: str = ""
    watch: str = ""
    source: str = "template"

    def as_contrarian(self) -> dict[str, str]:
        """Drop-in replacement for ContrarianAgent.critique's return value."""
        return {
            "main_risk": self.main_risk,
            "alternative_scenario": self.alternative_scenario,
            "watch": self.watch,
        }

    def transcript(self) -> str:
        return _render(self.turns)


def _render(turns: list[DebateTurn]) -> str:
    return "\n\n".join(
        f"[round {turn.round}] {turn.role}: {turn.claim}\n"
        f"  evidence cited: {turn.evidence_used}\n"
        f"  own weakest point: {turn.weakest_link}"
        for turn in turns
    )


class LLMDebateAgent(_Narrator):
    """Runs the proponent/skeptic debate and synthesises it.

    `fallback` is the template ContrarianAgent: with no API key, on any transport
    error, or on a malformed turn, the synthesis fields degrade to the template
    critique so callers always get a usable dict.
    """

    role = "debate"
    DEFAULT_ROUNDS = 2
    MAX_ROUNDS = 4

    def debate(
        self,
        signal: pd.Series,
        factors: pd.DataFrame,
        rounds: int | None = None,
    ) -> DebateResult:
        rounds = self.DEFAULT_ROUNDS if rounds is None else int(rounds)
        rounds = max(1, min(rounds, self.MAX_ROUNDS))

        template = self.fallback.critique(signal, factors)
        quant_signal = None if signal is None or signal.empty else signal.get("signal")
        result = DebateResult(
            signal=quant_signal,
            main_risk=template["main_risk"],
            alternative_scenario=template["alternative_scenario"],
            watch=template["watch"],
        )
        if signal is None or signal.empty or self.client is None:
            return result

        evidence = build_evidence_pack(signal, factors)
        when = signal.get("date")
        turns: list[DebateTurn] = []
        for index in range(rounds):
            for role, system in ((PROPONENT, DEBATE_PROPONENT_SYSTEM), (SKEPTIC, DEBATE_SKEPTIC_SYSTEM)):
                turn = self._turn(role, system, evidence, turns, index + 1, when)
                if turn is None:
                    # Stop rather than let the next debater rebut a hole in the
                    # transcript. Whatever was argued so far still stands.
                    return self._finish(result, turns, evidence, when)
                turns.append(turn)
        return self._finish(result, turns, evidence, when)

    def _turn(
        self,
        role: str,
        system: str,
        evidence: dict[str, Any],
        turns: list[DebateTurn],
        round_no: int,
        when: Any,
    ) -> DebateTurn | None:
        user = user_payload(evidence, extra=_debate_context(turns, round_no))
        try:
            resp = self.client.call(
                system,
                user,
                self.config.narration_model,
                self.config.narration_max_tokens,
                schema=DEBATE_TURN_SCHEMA,
            )
        except LLMError:
            return None
        self._log(system, user, resp, when=when, role=f"debate_{role}")
        out = resp.structured
        if not out or not str(out.get("claim") or "").strip():
            return None
        return DebateTurn(
            role=role,
            round=round_no,
            claim=str(out.get("claim") or "").strip(),
            evidence_used=str(out.get("evidence_used") or "").strip(),
            weakest_link=str(out.get("weakest_link") or "").strip(),
        )

    def _finish(
        self,
        result: DebateResult,
        turns: list[DebateTurn],
        evidence: dict[str, Any],
        when: Any,
    ) -> DebateResult:
        result.turns = turns
        result.rounds = max((turn.round for turn in turns), default=0)
        if not turns:
            return result
        result.source = "llm"

        user = user_payload(evidence, extra={"Debate transcript": _render(turns)})
        try:
            resp = self.client.call(
                DEBATE_SYNTHESIS_SYSTEM,
                user,
                self.config.report_model,
                self.config.report_max_tokens,
                schema=CONTRARIAN_SCHEMA,
            )
        except LLMError:
            return result  # keep the transcript, keep the template synthesis
        self._log(DEBATE_SYNTHESIS_SYSTEM, user, resp, when=when, role="debate_synthesis")
        out = resp.structured or {}
        result.main_risk = out.get("main_risk") or result.main_risk
        result.alternative_scenario = out.get("alternative_scenario") or result.alternative_scenario
        result.watch = out.get("watch") or result.watch
        return result


def _debate_context(turns: list[DebateTurn], round_no: int) -> dict[str, Any]:
    if not turns:
        return {"Debate state": f"Round {round_no}, opening argument. No rebuttal exists yet."}
    return {
        "Debate state": f"Round {round_no}. Rebut the arguments below before adding new points.",
        "Debate transcript so far": _render(turns),
    }
