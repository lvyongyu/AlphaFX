#!/usr/bin/env python3
"""Run the latest signal through the execution gate. Dry-run by default.

Reads `data/latest_signal.json` (written by `scripts/paper_trade.py --export`),
puts every leg through `risk_engine`, and records what WOULD have been sent to
IG Demo. That record is the point: it accumulates alongside the paper book so the
two can be compared over the same dates.

Nothing is submitted unless BOTH `--live` is passed AND
`risk_engine.EXECUTION_ENABLED` is True. It is False today, so `--live` is inert
by construction — the script says so rather than pretending it armed something.

    python scripts/execute_demo.py                  # dry-run against IG Demo
    python scripts/execute_demo.py --json
    python scripts/execute_demo.py --log            # show recent attempts and exit
    python scripts/execute_demo.py --live           # refused while the gate is shut

Reads from IG (session, quote, positions, balance); places nothing in dry-run.
No history endpoint is touched, so no weekly quota is consumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `alphafx` importable

from alphafx.config import load_local_env  # noqa: E402
from alphafx.database import Database  # noqa: E402
from alphafx.execution.bridge import SignalBridge, load_signal_file  # noqa: E402
from alphafx.execution.ig_client import IGClient, IGError  # noqa: E402
from alphafx.execution.risk_engine import EXECUTION_ENABLED  # noqa: E402

DEFAULT_SIGNAL_FILE = "data/latest_signal.json"


def report_to_dict(report) -> dict:
    return {
        "status": report.status,
        "mode": report.mode,
        "run_at": report.run_at,
        "execution_enabled": EXECUTION_ENABLED,
        "note": report.note,
        "breakers_tripped": report.breakers_tripped,
        "orders": [
            {
                "instrument": order.instrument,
                "epic": order.epic,
                "direction": order.direction,
                "size": order.size,
                "stop_distance_points": order.stop_distance_points,
                "risk_checks_passed": bool(order.decision and order.decision.risk_checks_passed),
                "allowed": bool(order.decision and order.decision.allowed),
                "reasons": order.reasons,
                "submitted": order.submitted,
                "deal_status": order.deal_status,
            }
            for order in report.orders
        ],
    }


def print_report(payload: dict) -> None:
    print(f"=== execute_demo [{payload['mode']}]  {payload['run_at']} ===")
    if payload["status"] != "ok":
        print(f"  {payload['status']}: {payload['note']}")
        return
    if payload["breakers_tripped"]:
        print(f"  CIRCUIT BREAKER TRIPPED THIS RUN: {', '.join(payload['breakers_tripped'])}")
    for order in payload["orders"]:
        head = f"  {order['instrument']:7s}"
        if order["allowed"]:
            print(f"{head} SEND {order['direction']} {order['size']} lots  "
                  f"stop {order['stop_distance_points']} pts  "
                  f"submitted={order['submitted']} status={order['deal_status']}")
            continue
        would = ""
        if order["risk_checks_passed"]:
            would = (f" (would have sent {order['direction']} {order['size']} lots, "
                     f"stop {order['stop_distance_points']} pts)")
        print(f"{head} REFUSE{would}")
        for reason in order["reasons"]:
            print(f"            - {reason}")
    if not payload["execution_enabled"]:
        print("\nEXECUTION_ENABLED is False: no order can be sent regardless of --live.")
        print("It opens only after the signal-quality gate (roadmap step B.4).")


def show_log(limit: int, as_json: bool) -> None:
    rows = Database().load_execution_log(limit)
    if as_json:
        print(rows.to_json(orient="records", indent=2))
        return
    if rows.empty:
        print("No execution attempts recorded yet.")
        return
    print(rows[["run_at", "instrument", "mode", "direction", "size", "allowed", "reasons"]]
          .to_string(index=False))


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="AlphaFX execution gate (IG Demo)")
    parser.add_argument("--signal-file", default=DEFAULT_SIGNAL_FILE)
    parser.add_argument("--live", action="store_true",
                        help="submit allowed orders (inert while EXECUTION_ENABLED is False)")
    parser.add_argument("--log", action="store_true", help="print recent attempts and exit")
    parser.add_argument("--limit", type=int, default=20, help="rows for --log (default 20)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.log:
        show_log(args.limit, args.json)
        return

    if args.live and not EXECUTION_ENABLED:
        # Say it before doing anything, so a --live run is never mistaken for an
        # armed one that simply found nothing to trade.
        print("--live has no effect: EXECUTION_ENABLED is False. Running as a dry-run.\n")

    path = Path(args.signal_file)
    if not path.exists():
        print(f"No signal file at {path}. Run: python scripts/paper_trade.py --export")
        return

    db = Database()
    try:
        client = IGClient()
    except IGError as exc:
        print(f"Cannot build an IG session: {exc}")
        return

    report = SignalBridge(client=client, db=db).run(load_signal_file(path), live=args.live)
    payload = report_to_dict(report)

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return
    print_report(payload)


if __name__ == "__main__":
    main()
