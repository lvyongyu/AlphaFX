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

from alphafx.collect import DataAgent  # noqa: E402
from alphafx.config import DEFAULT_SYMBOLS, load_local_env  # noqa: E402
from alphafx.dashboard.context import build_context  # noqa: E402
from alphafx.database import Database  # noqa: E402
from alphafx.instruments import get_instrument  # noqa: E402
from alphafx.trade.order import build_order_intent  # noqa: E402
from alphafx.trade.paper import PaperBroker  # noqa: E402

# Live portfolio: the pairs with a genuine positive risk-adjusted edge (5y Sharpe
# AUD +0.18 / EUR +0.12 / CHF +0.30). Weakly correlated (~0.2), so this trio lifts
# Sharpe vs AUD-only and roughly halves drawdown. See alphafx-multi-instrument memo.
DEFAULT_PORTFOLIO = ["AUDUSD", "EURUSD", "USDCHF"]


def export_snapshot(result: dict, db: Database, out_dir: str = "data") -> list[str]:
    """Write diff-friendly text snapshots for committing to GitHub."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    (out / "latest_signal.json").write_text(json.dumps(result, indent=2, default=str))
    written.append(str(out / "latest_signal.json"))

    db.load_paper_positions().to_csv(out / "paper_positions.csv", index=False)
    written.append(str(out / "paper_positions.csv"))

    # Append one row per (date, instrument) to a growing history (idempotent).
    rows = [
        {"date": leg["date"], "instrument": leg["instrument"], "signal": leg["signal"],
         "action": leg["action"], "price": leg["price"]}
        for leg in result.get("legs", []) if leg.get("status") == "ok"
    ]
    hist_path = out / "signal_history.csv"
    hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    for row in rows:
        dup = (not hist.empty and "instrument" in hist.columns
               and ((hist["date"].astype(str) == str(row["date"]))
                    & (hist["instrument"].astype(str) == row["instrument"])).any())
        if not dup:
            hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(hist_path, index=False)
    written.append(str(hist_path))
    return written


def _refresh_all(data_agent: DataAgent, portfolio: list[str], start, end) -> None:
    """One download for the whole portfolio: every pair's price + shared DXY/VIX + macro."""
    symbols = {"dxy": DEFAULT_SYMBOLS.dxy, "vix": DEFAULT_SYMBOLS.vix}
    for name in portfolio:
        cfg = get_instrument(name)
        symbols[cfg.name] = cfg.fx_symbol
    data_agent.download_market_data(start, end, symbols=symbols)
    data_agent.download_macro_data(start, end)


def step(years: int, leverage: float, base_units: int, refresh: bool, portfolio: list[str] | None = None) -> dict:
    portfolio = portfolio or DEFAULT_PORTFOLIO
    db = Database()
    end = date.today()
    start = end - timedelta(days=365 * years)
    if refresh:
        _refresh_all(DataAgent(db=db), portfolio, start, end)

    broker = PaperBroker(db)
    legs: list[dict] = []
    unrealised_total = 0.0
    for name in portfolio:
        cfg = get_instrument(name)
        # Data already refreshed once above; each context just loads + computes.
        ctx = build_context(start, end, leverage, use_llm=False, refresh=False, db=db, instrument=name)
        if ctx.status != "ok":
            legs.append({"instrument": name, "status": ctx.status})
            continue
        price = float(ctx.aud_latest)
        when = str(ctx.latest_signal["date"])
        closed = broker.update(price, when, instrument=cfg.oanda)
        intent = build_order_intent(ctx.latest_signal, ctx.risk, base_units=base_units, instrument=cfg.oanda)
        opened = broker.place(intent, price, when)
        leg_unrealised = broker.unrealised(price, instrument=cfg.oanda)
        unrealised_total += leg_unrealised
        legs.append({
            "instrument": name,
            "status": "ok",
            "date": when,
            "price": price,
            "signal": ctx.latest_signal["signal"],
            "probability": float(ctx.latest_signal["probability"]),
            "probability_source": ctx.latest_signal.get("probability_source"),
            "action": ctx.risk.action,
            "warnings": list(ctx.warnings),
            "intent": {"side": intent.side, "units": intent.units},
            "opened": opened,
            "closed": closed,
            "unrealised_pnl": leg_unrealised,
        })

    ok_legs = [leg for leg in legs if leg.get("status") == "ok"]
    if not ok_legs:
        return {"status": legs[0].get("status", "no_data") if legs else "no_data", "legs": legs}
    return {
        "status": "ok",
        "date": max(leg["date"] for leg in ok_legs),
        "portfolio": portfolio,
        "legs": legs,
        "open_positions": broker.db.load_open_positions().to_dict(orient="records"),
        "realised_pnl": broker.realised(),
        "unrealised_pnl": unrealised_total,
    }


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="AlphaFX headless paper trading")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--base-units", type=int, default=1000, help="paper position size in base-currency units")
    parser.add_argument("--instruments", default=",".join(DEFAULT_PORTFOLIO),
                        help="comma-separated portfolio (default AUDUSD,EURUSD,USDCHF)")
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--export", action="store_true", help="write data/ snapshots (json+csv) for committing")
    args = parser.parse_args()

    portfolio = [s.strip().upper() for s in args.instruments.split(",") if s.strip()]
    result = step(args.years, args.leverage, args.base_units, refresh=not args.no_refresh, portfolio=portfolio)

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
    print(f"=== Paper portfolio {', '.join(result['portfolio'])}  ({result['date']}) ===")
    for leg in result["legs"]:
        if leg.get("status") != "ok":
            print(f"  {leg['instrument']:7s}  ({leg.get('status')})")
            continue
        for w in leg.get("warnings", []):
            print(f"  WARNING [{leg['instrument']}]: {w}")
        opened = leg["opened"].get("status")
        closed = f" closed={len(leg['closed'])}" if leg["closed"] else ""
        print(f"  {leg['instrument']:7s} price={leg['price']:.5f}  {leg['signal'].upper():8s}  "
              f"{leg['action']:12s} opened={opened}{closed}")
    print(f"open positions: {len(result['open_positions'])}  "
          f"realised PnL: {result['realised_pnl']:.2f}  unrealised: {result['unrealised_pnl']:.2f}")


if __name__ == "__main__":
    main()
