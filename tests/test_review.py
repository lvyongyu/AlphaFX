from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from alphafx.config import DEFAULT_SYMBOLS
from alphafx.database import Database
from alphafx.review import DecisionReviewAgent

HORIZON = 5
DATES = pd.bdate_range("2024-01-01", periods=40)


def seeded_db(tmp_path) -> Database:
    """A DB with a rising AUD/USD series and four logged decisions."""
    db = Database(tmp_path / "review.db")
    db.upsert_market_data(
        pd.DataFrame(
            [
                # Date as a string, matching what the providers hand the DB.
                {"date": dt.strftime("%Y-%m-%d"), "symbol": DEFAULT_SYMBOLS.audusd, "open": 0.65,
                 "high": 0.66, "low": 0.64, "close": 0.65 + i * 0.001, "source": "test"}
                for i, dt in enumerate(DATES)
            ]
        )
    )
    decisions = [
        # index, signal, action  -> price rises, so bullish is right and bearish is wrong
        (0, "bullish", "BUY AUD/USD"),
        (1, "bearish", "SELL AUD/USD"),
        (2, "bullish", "NO TRADE"),
        (38, "bullish", "BUY AUD/USD"),  # horizon has not elapsed -> pending
    ]
    for index, signal, action in decisions:
        db.upsert_paper_journal(
            {
                "date": DATES[index].strftime("%Y-%m-%d"),
                "instrument": "AUDUSD",
                "price": 0.65 + index * 0.001,
                "signal": signal,
                "score": 40.0,
                "calibrated_probability": 0.55,
                "recommended_position": action,
                "stop_loss": 0.05,
                "explanation": f"decision {index}",
                "status": "open" if action != "NO TRADE" else "no_trade",
            }
        )
    return db


def test_decisions_attach_the_realised_horizon_outcome(tmp_path):
    frame = DecisionReviewAgent(seeded_db(tmp_path)).decisions(horizon_days=HORIZON)

    assert len(frame) == 4
    by_date = frame.set_index(frame["date"].dt.strftime("%Y-%m-%d"))
    assert by_date.loc[DATES[0].strftime("%Y-%m-%d"), "outcome"] == "win"  # bullish into a rally
    assert by_date.loc[DATES[1].strftime("%Y-%m-%d"), "outcome"] == "loss"  # bearish into a rally
    # The signed return is the raw forward return flipped for a bearish call.
    bearish = by_date.loc[DATES[1].strftime("%Y-%m-%d")]
    assert bearish["signal_return"] == -bearish["forward_return"]


def test_unresolved_decisions_are_pending_not_wins(tmp_path):
    agent = DecisionReviewAgent(seeded_db(tmp_path))
    frame = agent.decisions(horizon_days=HORIZON)

    tail = frame[frame["date"] == DATES[38]].iloc[0]
    assert tail["outcome"] == "pending"
    assert pd.isna(tail["forward_return"])

    summary = agent.summary(frame)
    assert summary["decisions"] == 4
    assert summary["pending"] == 1
    assert summary["resolved"] == 3  # the pending row is excluded from every rate


def test_summary_separates_the_gate_from_all_signals(tmp_path):
    agent = DecisionReviewAgent(seeded_db(tmp_path))
    summary = agent.summary(agent.decisions(horizon_days=HORIZON))

    # 3 resolved directional decisions (1 win, 1 loss, 1 win-but-not-traded).
    assert summary["hit_rate_all_signals"] == 2 / 3
    # Only 2 of them cleared the gate, one of which won.
    assert summary["traded"] == 2
    assert summary["hit_rate_traded"] == 0.5


def test_lessons_are_recorded_with_provenance(tmp_path):
    agent = DecisionReviewAgent(seeded_db(tmp_path))
    agent.record_lesson(DATES[0].strftime("%Y-%m-%d"), "Sized up on a fallback prior.")
    agent.record_lesson(DATES[1].strftime("%Y-%m-%d"), "Iron ore was stale.", author="llm:test-model")

    notes = agent.lessons("AUDUSD")
    assert list(notes["author"]) == ["human", "llm:test-model"]
    assert "Iron ore was stale." in list(notes["lesson"])


def test_review_pack_renders_window_summary_and_boundary(tmp_path):
    agent = DecisionReviewAgent(seeded_db(tmp_path))
    agent.record_lesson(DATES[0].strftime("%Y-%m-%d"), "Watch the calibration sample size.")
    pack = agent.build_review_pack(weeks=1, end=DATES[5], horizon_days=HORIZON)

    assert "# AlphaFX review pack" in pack
    assert "## AUDUSD" in pack
    assert "| date | signal | prob | action | outcome | signal return |" in pack
    assert "hit rate, signals the gate traded" in pack
    assert "Watch the calibration sample size." in pack
    assert "Archive only." in pack


def test_empty_database_still_produces_a_pack(tmp_path):
    agent = DecisionReviewAgent(Database(tmp_path / "empty.db"))
    assert agent.decisions().empty
    assert "No decisions recorded in this window." in agent.build_review_pack()


# --- boundary guard ----------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "alphafx"
FORBIDDEN_PREFIXES = ("alphafx.llm", "alphafx.review")
# Exempt: the narration layer itself, the archive itself, and the dashboard
# orchestrator, whose whole job is to compose quant output with narration for
# display. Everything else under alphafx/ is decision path by default — a new
# module is guarded automatically without anyone remembering to list it.
EXEMPT_PARTS = ("llm", "dashboard")
EXEMPT_FILES = ("review.py",)


def _absolute_imports(path: Path, root: Path | None = None) -> set[str]:
    """Every module a file imports, with relative imports resolved to absolute."""
    module = ".".join(path.relative_to(root or PACKAGE_ROOT.parent).with_suffix("").parts)
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = module.split(".")[: -node.level]
                found.add(".".join(base + ([node.module] if node.module else [])))
            elif node.module:
                found.add(node.module)
    return found


def test_decision_path_never_imports_the_llm_or_review_layer():
    """The core rule, checked mechanically: an LLM narrative or a post-hoc lesson
    can never reach the code that produces a signal, a risk decision, or an order.

    TradingAgents feeds its lessons-learned back into the next decision. This is
    the test that stops AlphaFX from drifting into the same loop.
    """
    offenders: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT)
        if set(relative.parts) & set(EXEMPT_PARTS) or relative.name in EXEMPT_FILES:
            continue
        bad = {name for name in _absolute_imports(path) if name.startswith(FORBIDDEN_PREFIXES)}
        offenders += [f"alphafx/{relative} imports {name}" for name in sorted(bad)]

    assert not offenders, "decision-path modules must not import the narration or review layer: " + "; ".join(offenders)


def test_the_boundary_guard_actually_catches_a_violation(tmp_path):
    # A guard that cannot fail is not a guard. This proves the AST walk sees both
    # absolute and relative imports of the forbidden packages.
    sample = tmp_path / "alphafx" / "trade" / "sample.py"
    sample.parent.mkdir(parents=True)
    sample.write_text("from ..llm import LLMDebateAgent\nimport alphafx.review\n")

    found = _absolute_imports(sample, root=tmp_path)
    assert {"alphafx.llm", "alphafx.review"} <= found
    assert all(name.startswith(FORBIDDEN_PREFIXES) for name in found)
