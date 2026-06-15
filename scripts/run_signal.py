#!/usr/bin/env python3
"""Headless AlphaFX signal runner — no Streamlit.

Produces the latest AUD/USD signal (and optional LLM analysis) for cron jobs,
pipelines, or downstream systems. The quant model owns the signal; --llm only
adds explanation/critique on top.

Examples:
    python scripts/run_signal.py                 # download + print latest signal
    python scripts/run_signal.py --json          # machine-readable JSON
    python scripts/run_signal.py --llm           # include LLM analysis (needs ANTHROPIC_API_KEY)
    python scripts/run_signal.py --no-refresh     # use the cached DB, skip download
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `alphafx` importable

from alphafx.config import load_local_env  # noqa: E402
from alphafx.dashboard.context import build_context  # noqa: E402


def run(years: int, leverage: float, use_llm: bool, refresh: bool) -> dict:
    end = date.today()
    start = end - timedelta(days=365 * years)
    ctx = build_context(start, end, leverage, use_llm=use_llm, refresh=refresh)
    if ctx.status != "ok":
        return {"status": ctx.status}

    s = ctx.latest_signal
    result = {
        "status": "ok",
        "date": str(s["date"]),
        "signal": s["signal"],
        "probability": round(float(s["probability"]), 4),
        "probability_source": s.get("probability_source"),
        "confidence": s["confidence"],
        "action": ctx.risk.action,
        "leverage": ctx.risk.leverage,
        "stop_loss": ctx.risk.stop_loss,
        "take_profit": ctx.risk.take_profit,
        "factors": [
            {"factor": r["factor"], "stance": r["stance"]} for _, r in ctx.factor_table.iterrows()
        ],
        "ml_signal": ctx.ml_latest_signal,
    }
    if use_llm:
        result["explanation"] = ctx.judgement.get("explanation")
        result["contrarian"] = ctx.contrarian.get("main_risk")
        result["llm_dissent"] = ctx.judgement.get("llm_dissent")
    return result


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="AlphaFX headless signal runner")
    parser.add_argument("--years", type=int, default=5, help="history window in years (default 5)")
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--llm", action="store_true", help="include LLM analysis (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--no-refresh", action="store_true", help="use cached DB instead of downloading")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    result = run(args.years, args.leverage, use_llm=args.llm, refresh=not args.no_refresh)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    if result["status"] != "ok":
        print(f"No signal ({result['status']}). Try without --no-refresh to download data.")
        return

    print(f"{result['date']}  {result['signal'].upper()}  prob={result['probability']:.0%}  "
          f"conf={result['confidence']}  action={result['action']}")
    chips = "  ".join(f"{f['factor']}:{f['stance']}" for f in result["factors"])
    print("factors:", chips)
    if args.llm and result.get("explanation"):
        print("\n" + result["explanation"])


if __name__ == "__main__":
    main()
