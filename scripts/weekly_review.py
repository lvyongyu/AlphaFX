#!/usr/bin/env python3
"""Generate the weekly review pack from the decision log.

Every daily run writes one row per (date, instrument) to `paper_journal`,
NO-TRADE days included. This reads that log back, attaches the realised return
at the signal's own horizon, and prints markdown to read during review — or to
hand to an AI post-mortem.

Archive only: nothing this produces is read back by the signal, risk, or
execution code, and a bad window is not a reason to retune anything.

    python scripts/weekly_review.py                        # last week, live portfolio
    python scripts/weekly_review.py --weeks 4 --out data/review.md
    python scripts/weekly_review.py --lesson "Traded on a stale iron-ore print." \
        --lesson-date 2026-08-14
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `alphafx` importable

from alphafx.instruments import LIVE_PORTFOLIO  # noqa: E402
from alphafx.review import OUTCOME_HORIZON_DAYS, DecisionReviewAgent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaFX weekly review pack")
    parser.add_argument("--weeks", type=int, default=1, help="window length in weeks (default 1)")
    parser.add_argument("--end", default=None, help="window end date YYYY-MM-DD (default today)")
    parser.add_argument("--instruments", default=",".join(LIVE_PORTFOLIO),
                        help=f"comma-separated (default {','.join(LIVE_PORTFOLIO)})")
    parser.add_argument("--horizon", type=int, default=OUTCOME_HORIZON_DAYS,
                        help=f"outcome horizon in trading days (default {OUTCOME_HORIZON_DAYS})")
    parser.add_argument("--out", default=None, help="write markdown to this path instead of stdout")
    parser.add_argument("--lesson", default=None, help="append a review note to the archive and exit")
    parser.add_argument("--lesson-date", default=None, help="the decision date the note is about")
    parser.add_argument("--lesson-instrument", default="AUDUSD")
    parser.add_argument("--author", default="human", help='note provenance, e.g. "human" or "llm:<model>"')
    args = parser.parse_args()

    agent = DecisionReviewAgent()

    if args.lesson:
        if not args.lesson_date:
            parser.error("--lesson needs --lesson-date so the note is anchored to a decision")
        agent.record_lesson(args.lesson_date, args.lesson, args.author, args.lesson_instrument)
        print(f"recorded lesson for {args.lesson_instrument} {args.lesson_date} ({args.author})")
        return

    instruments = [s.strip().upper() for s in args.instruments.split(",") if s.strip()]
    pack = agent.build_review_pack(
        instruments=instruments, weeks=args.weeks, end=args.end, horizon_days=args.horizon
    )
    if args.out:
        Path(args.out).write_text(pack, encoding="utf-8")
        print(f"wrote {args.out}")
        return
    print(pack)


if __name__ == "__main__":
    main()
