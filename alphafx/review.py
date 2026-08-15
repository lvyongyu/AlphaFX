"""Decision log review and lesson archive — ARCHIVE ONLY, never a feedback loop.

Every daily run already writes one row per (date, instrument) to `paper_journal`,
including the days the risk gate said NO TRADE. That table is the decision log.
This module reads it back, attaches the realised forward return at the signal's
own horizon, and renders review material a human can sit down with once a week.

BOUNDARY (the reason this module exists as its own file): TradingAgents stores
"lessons learned" and feeds them back into the next decision. AlphaFX does not
and must not — that would put post-hoc narrative, LLM-written or human-written,
back into the decision path, which is the one thing the whole design forbids.
Lessons here are write-only with respect to trading: they are recorded for a
human to read during review and are never consulted by the signal, risk, or
execution code. `tests/test_review.py` enforces that with an import-graph check,
so the rule survives a future refactor that forgets this docstring.

Post-hoc outcomes are also why nothing here may inform a parameter change: the
project rule is that strategy changes need an out-of-sample comparison and a
commit trail, not a bad week.
"""
from __future__ import annotations

import pandas as pd

from .database import Database
from .instruments import get_instrument

# Matches RiskAgent.HORIZON_DAYS and PaperBroker.MAX_HOLDING_DAYS: the signal's
# edge is validated at ~20 trading days, so that is the horizon a decision is
# judged over. Scoring it at any other horizon reviews a strategy nobody ran.
OUTCOME_HORIZON_DAYS = 20

NO_TRADE = "NO TRADE"
_DIRECTION = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}


def _signal_return(row: pd.Series) -> float:
    """Forward return signed by the signal's direction: >0 means the call was right."""
    direction = _DIRECTION.get(row.get("signal"), 0.0)
    return float(row["forward_return"]) * direction


def _outcome(value: float) -> str:
    if pd.isna(value):
        return "pending"
    if value > 0:
        return "win"
    if value < 0:
        return "loss"
    return "flat"


class DecisionReviewAgent:
    """Builds weekly review material from the decision log. Reads, never decides."""

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()

    def decisions(
        self,
        instrument: str = "AUDUSD",
        horizon_days: int = OUTCOME_HORIZON_DAYS,
    ) -> pd.DataFrame:
        """Journal rows for one instrument with the realised horizon outcome attached.

        Rows whose horizon has not elapsed yet keep `forward_return` NaN and an
        outcome of "pending" — an unresolved decision is never counted as a win
        or a loss, which is how a review pack stays honest about small samples.
        """
        journal = self.db.load_paper_journal()
        columns = [
            "date", "instrument", "price", "signal", "score", "calibrated_probability",
            "recommended_position", "stop_loss", "explanation", "forward_return",
            "signal_return", "outcome", "traded",
        ]
        if journal.empty or "instrument" not in journal.columns:
            return pd.DataFrame(columns=columns)
        rows = journal[journal["instrument"] == instrument].copy()
        if rows.empty:
            return pd.DataFrame(columns=columns)

        rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
        rows = rows.sort_values("date").reset_index(drop=True)
        rows["forward_return"] = rows["date"].map(self._forward_returns(instrument, horizon_days))
        rows["signal_return"] = rows.apply(_signal_return, axis=1)
        rows["outcome"] = rows["signal_return"].map(_outcome)
        rows["traded"] = rows["recommended_position"].fillna(NO_TRADE) != NO_TRADE
        return rows[columns]

    def _forward_returns(self, instrument: str, horizon_days: int) -> pd.Series:
        """date -> return over the next `horizon_days` trading sessions (NaN at the tail)."""
        cfg = get_instrument(instrument)
        prices = self.db.load_market_data([cfg.fx_symbol])
        if prices.empty:
            return pd.Series(dtype=float)
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
        close = prices.sort_values("date").drop_duplicates("date").set_index("date")["close"]
        return close.shift(-horizon_days) / close - 1.0

    def summary(self, decisions: pd.DataFrame) -> dict[str, object]:
        """Headline numbers for the pack. Resolved-only; pending rows are excluded."""
        resolved = decisions[decisions["outcome"] != "pending"] if not decisions.empty else decisions
        directional = resolved[resolved["signal"].isin(["bullish", "bearish"])] if not resolved.empty else resolved
        traded = directional[directional["traded"]] if not directional.empty else directional

        def hit_rate(frame: pd.DataFrame) -> float | None:
            return float((frame["signal_return"] > 0).mean()) if len(frame) else None

        return {
            "decisions": int(len(decisions)),
            "pending": int((decisions["outcome"] == "pending").sum()) if not decisions.empty else 0,
            "resolved": int(len(resolved)),
            "traded": int(len(traded)),
            # The gate's whole job is to let the better subset through. If the
            # traded hit rate is not above the all-signals hit rate, it is not
            # earning its keep — that comparison is the point of the pack.
            "hit_rate_all_signals": hit_rate(directional),
            "hit_rate_traded": hit_rate(traded),
            "mean_signal_return_all": float(directional["signal_return"].mean()) if len(directional) else None,
            "mean_signal_return_traded": float(traded["signal_return"].mean()) if len(traded) else None,
        }

    # --- lesson archive (write-only with respect to the decision path) --------

    def record_lesson(
        self,
        decision_date: str,
        lesson: str,
        author: str = "human",
        instrument: str = "AUDUSD",
    ) -> None:
        """Archive one review note. Nothing reads this back into a trading decision.

        `author` records provenance ("human", or "llm:<model>" when a review note
        came out of an AI post-mortem) so a later reader can tell which notes are
        a machine's narrative and which are the operator's own.
        """
        self.db.insert_decision_lesson(
            {
                "decision_date": str(decision_date),
                "instrument": instrument,
                "author": author,
                "lesson": lesson,
            }
        )

    def lessons(self, instrument: str | None = None) -> pd.DataFrame:
        return self.db.load_decision_lessons(instrument)

    # --- rendering -----------------------------------------------------------

    def build_review_pack(
        self,
        instruments: list[str] | None = None,
        weeks: int = 1,
        end: object | None = None,
        horizon_days: int = OUTCOME_HORIZON_DAYS,
    ) -> str:
        """Render the review window as markdown for a human to read."""
        instruments = instruments or ["AUDUSD"]
        end_ts = pd.Timestamp(end).normalize() if end is not None else pd.Timestamp.today().normalize()
        start_ts = end_ts - pd.Timedelta(days=7 * int(weeks))

        parts = [
            f"# AlphaFX review pack: {start_ts.date()} to {end_ts.date()}",
            "",
            f"Outcome horizon: {horizon_days} trading days. Decisions whose horizon has not "
            "elapsed are shown as pending and excluded from every rate below.",
        ]
        for name in instruments:
            everything = self.decisions(name, horizon_days=horizon_days)
            window = everything
            if not everything.empty:
                window = everything[(everything["date"] >= start_ts) & (everything["date"] <= end_ts)]
            parts += ["", f"## {name}", ""]
            parts += self._render_window(window)
            parts += ["", "### Cumulative (all decisions on record)", ""]
            parts += self._render_summary(self.summary(everything))
            parts += self._render_lessons(name)

        parts += [
            "",
            "---",
            "",
            "Archive only. Nothing in this pack is read back by the signal, risk, or "
            "execution code, and a bad window is not a reason to change a parameter — "
            "strategy changes need an out-of-sample comparison and a commit trail.",
        ]
        return "\n".join(parts)

    def _render_window(self, window: pd.DataFrame) -> list[str]:
        if window.empty:
            return ["No decisions recorded in this window."]
        lines = [
            "| date | signal | prob | action | outcome | signal return |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for _, row in window.iterrows():
            prob = row["calibrated_probability"]
            ret = row["signal_return"]
            lines.append(
                f"| {row['date'].date()} | {row['signal']} | "
                f"{'n/a' if pd.isna(prob) else f'{float(prob):.0%}'} | "
                f"{row['recommended_position']} | {row['outcome']} | "
                f"{'n/a' if pd.isna(ret) else f'{float(ret):+.2%}'} |"
            )
        return lines

    def _render_summary(self, summary: dict[str, object]) -> list[str]:
        def pct(key: str) -> str:
            value = summary.get(key)
            return "n/a" if value is None else f"{float(value):.1%}"

        return [
            f"- decisions: {summary['decisions']} ({summary['resolved']} resolved, "
            f"{summary['pending']} pending, {summary['traded']} traded)",
            f"- hit rate, all directional signals: {pct('hit_rate_all_signals')}",
            f"- hit rate, signals the gate traded: {pct('hit_rate_traded')}",
            f"- mean signal return, all: {pct('mean_signal_return_all')}",
            f"- mean signal return, traded: {pct('mean_signal_return_traded')}",
        ]

    def _render_lessons(self, instrument: str) -> list[str]:
        notes = self.lessons(instrument)
        if notes.empty:
            return []
        lines = ["", "### Lesson archive", ""]
        for _, note in notes.iterrows():
            lines.append(f"- {note['decision_date']} ({note['author']}): {note['lesson']}")
        return lines
