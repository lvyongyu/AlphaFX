#!/usr/bin/env python3
"""Headless paper trading — local fill simulation, no broker.

Each run: refresh data, mark/close open paper positions to the latest price,
then open a new paper position if the risk engine says BUY/SELL (NO TRADE opens
nothing). Fully local; a real broker can replace PaperBroker later.

    python scripts/paper_trade.py               # run one paper-trading step
    python scripts/paper_trade.py --no-refresh  # use cached data
    python scripts/paper_trade.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphafx.config import load_local_env  # noqa: E402
from alphafx.dashboard.context import build_context  # noqa: E402
from alphafx.database import Database  # noqa: E402
from alphafx.trade.order import build_order_intent  # noqa: E402
from alphafx.trade.paper import PaperBroker  # noqa: E402


def export_snapshot(result: dict, db: Database, out_dir: str = "data") -> list[str]:
    """Write diff-friendly text snapshots for committing to GitHub."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    (out / "latest_signal.json").write_text(json.dumps(result, indent=2, default=str))
    written.append(str(out / "latest_signal.json"))

    db.load_paper_positions().to_csv(out / "paper_positions.csv", index=False)
    written.append(str(out / "paper_positions.csv"))

    # Append one row per day to a growing history (idempotent on date).
    row = {k: result.get(k) for k in ["date", "signal", "action", "price", "realised_pnl", "unrealised_pnl"]}
    hist_path = out / "signal_history.csv"
    hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    if hist.empty or str(row["date"]) not in hist["date"].astype(str).values:
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        hist.to_csv(hist_path, index=False)
    written.append(str(hist_path))
    return written


def step(years: int, leverage: float, base_units: int, refresh: bool) -> dict:
    db = Database()
    end = date.today()
    start = end - timedelta(days=365 * years)
    ctx = build_context(start, end, leverage, use_llm=False, refresh=refresh, db=db)
    if ctx.status != "ok":
        return {"status": ctx.status}

    price = float(ctx.aud_latest)
    when = str(ctx.latest_signal["date"])
    broker = PaperBroker(db)

    closed = broker.update(price, when)
    # The risk engine owns the effective leverage: it is vol-aware and capped
    # (see RiskAgent), so requesting 20x still trades at the risk-approved size.
    effective_leverage = float(ctx.risk.leverage)
    intent = build_order_intent(ctx.latest_signal, ctx.risk, base_units=base_units, leverage=effective_leverage)
    opened = broker.place(intent, price, when)
    open_positions = broker.db.load_open_positions().to_dict(orient="records")

    return {
        "status": "ok",
        "date": when,
        "price": price,
        "signal": ctx.latest_signal["signal"],
        "action": ctx.risk.action,
        "requested_leverage": leverage,
        "effective_leverage": effective_leverage,
        "intent": {"side": intent.side, "units": intent.units},
        "opened": opened,
        "closed": closed,
        "open_positions": open_positions,
        "realised_pnl": broker.realised(),
        "unrealised_pnl": broker.unrealised(price),
    }


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="AlphaFX headless paper trading")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--base-units", type=int, default=1000, help="paper position size in AUD units")
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--export", action="store_true", help="write data/ snapshots (json+csv) for committing")
    args = parser.parse_args()

    result = step(args.years, args.leverage, args.base_units, refresh=not args.no_refresh)

    if args.export and result.get("status") == "ok":
        written = export_snapshot(result, Database())
        if not args.json:
            print("exported:", ", ".join(written))

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return
    if result["status"] != "ok":
        print(f"No signal ({result['status']}).")
        return
    lev_note = ""
    if result.get("requested_leverage") and result["requested_leverage"] != result.get("effective_leverage"):
        lev_note = f"  (requested {result['requested_leverage']:g}x, risk-capped to {result['effective_leverage']:g}x)"
    print(f"{result['date']}  price={result['price']:.5f}  {result['signal'].upper()}  "
          f"action={result['action']}  leverage={result.get('effective_leverage', 1):g}x{lev_note}")
    print(f"opened: {result['opened']}")
    if result["closed"]:
        print(f"closed: {result['closed']}")
    print(f"open positions: {len(result['open_positions'])}  "
          f"realised PnL: {result['realised_pnl']:.2f}  unrealised: {result['unrealised_pnl']:.2f}")


if __name__ == "__main__":
    main()
