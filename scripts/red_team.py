#!/usr/bin/env python3
"""Headless red-team debate over the latest signal.

Runs a proponent/skeptic debate across N rounds and prints what survived. This
is a review tool, run on demand — it costs `2 * rounds + 1` LLM calls and it
changes nothing: the quant signal and the risk action are printed as computed,
and the debate can only add commentary beside them.

Needs ANTHROPIC_API_KEY (via .env). Without one it degrades to the template
critique and says so.

Examples:
    python scripts/red_team.py                  # 2 rounds against the latest signal
    python scripts/red_team.py --rounds 3
    python scripts/red_team.py --no-refresh --json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `alphafx` importable

from alphafx.agents import ContrarianAgent  # noqa: E402
from alphafx.config import load_local_env  # noqa: E402
from alphafx.dashboard.context import build_context  # noqa: E402
from alphafx.database import Database  # noqa: E402
from alphafx.llm import LLMDebateAgent  # noqa: E402


def run(years: int, rounds: int, refresh: bool) -> dict:
    end = date.today()
    start = end - timedelta(days=365 * years)
    db = Database()
    # use_llm=False: the explanation/contrarian/judge agents are not needed here
    # and would triple the API bill for a run whose whole output is the debate.
    ctx = build_context(start, end, leverage=2.0, use_llm=False, refresh=refresh, db=db)
    if ctx.status != "ok":
        return {"status": ctx.status}

    result = LLMDebateAgent(fallback=ContrarianAgent(), db=db).debate(
        ctx.latest_signal, ctx.factor_table, rounds=rounds
    )
    return {
        "status": "ok",
        "date": str(ctx.latest_signal["date"]),
        "signal": ctx.latest_signal["signal"],
        "probability": round(float(ctx.latest_signal["probability"]), 4),
        "probability_source": ctx.latest_signal.get("probability_source"),
        "action": ctx.risk.action,
        "debate": asdict(result),
    }


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="AlphaFX red-team debate")
    parser.add_argument("--years", type=int, default=5, help="history window in years (default 5)")
    parser.add_argument(
        "--rounds",
        type=int,
        default=LLMDebateAgent.DEFAULT_ROUNDS,
        help=f"debate rounds, clamped to 1..{LLMDebateAgent.MAX_ROUNDS}",
    )
    parser.add_argument("--no-refresh", action="store_true", help="use cached DB instead of downloading")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    result = run(args.years, args.rounds, refresh=not args.no_refresh)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    if result["status"] != "ok":
        print(f"No signal ({result['status']}). Try without --no-refresh to download data.")
        return

    debate = result["debate"]
    print(f"{result['date']}  {result['signal'].upper()}  prob={result['probability']:.0%}  "
          f"action={result['action']}")
    print(f"debate source: {debate['source']}  rounds: {debate['rounds']}")
    if debate["source"] == "template":
        print("(no API key or the call failed — showing the offline template critique)")
    for turn in debate["turns"]:
        print(f"\n[round {turn['round']}] {turn['role'].upper()}")
        print(f"  {turn['claim']}")
        print(f"  evidence cited: {turn['evidence_used']}")
        print(f"  own weakest point: {turn['weakest_link']}")
    print("\n--- what survived ---")
    print(f"main risk:            {debate['main_risk']}")
    print(f"alternative scenario: {debate['alternative_scenario']}")
    print(f"watch:                {debate['watch']}")
    print("\nThe quant signal above is unchanged. This debate is commentary only.")


if __name__ == "__main__":
    main()
