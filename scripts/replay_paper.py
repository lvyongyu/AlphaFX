#!/usr/bin/env python3
"""Replay headless paper trading day-by-day over a past window.

The live `paper_trade.py` only runs ONE step (today). This replays the SAME
engine (RiskAgent gate -> build_order_intent -> PaperBroker) across every
trading day in the window, so we can see the record the daily job would have
produced and sanity-check the fixes.

Look-ahead safety: the signal frame's per-date probability uses expanding-window
calibration and a point-in-time walk-forward adaptive sign, so each day only ever
sees information available on that day. Replaying the precomputed frame is causal.

A fresh temp DB holds the replay positions, so the live paper_positions table is
never touched.

    python scripts/replay_paper.py                  # past 365 days
    python scripts/replay_paper.py --days 90        # past 3 months
    python scripts/replay_paper.py --refresh        # re-download data first
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphafx.config import DEFAULT_SYMBOLS, load_local_env  # noqa: E402
from alphafx.dashboard.context import build_context  # noqa: E402
from alphafx.database import Database  # noqa: E402
from alphafx.risk import RiskAgent  # noqa: E402
from alphafx.trade.order import build_order_intent  # noqa: E402
from alphafx.trade.paper import PaperBroker  # noqa: E402


def replay(days: int, leverage: float, base_units: int, years: int, refresh: bool) -> dict:
    db = Database()
    end = date.today()
    start = end - timedelta(days=365 * years)
    ctx = build_context(start, end, leverage, use_llm=False, refresh=refresh, db=db)
    if ctx.status != "ok":
        return {"status": ctx.status}

    aud = (
        ctx.market_data[ctx.market_data["symbol"] == DEFAULT_SYMBOLS.audusd]
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .sort_values("date")[["date", "close"]]
    )
    sig = ctx.signals.dropna(subset=["score"]).assign(date=lambda x: pd.to_datetime(x["date"]))
    vol = ctx.features.assign(date=lambda x: pd.to_datetime(x["date"]))[["date", "audusd_vol_20d"]]
    frame = sig.merge(vol, on="date", how="left").merge(aud, on="date", how="inner").sort_values("date")

    window_start = pd.to_datetime(end) - pd.Timedelta(days=days)
    frame = frame[frame["date"] >= window_start].reset_index(drop=True)
    if frame.empty:
        return {"status": "no_rows"}

    # Fresh, isolated paper DB for the replay.
    tmp = Path(tempfile.mkdtemp()) / "replay.db"
    broker = PaperBroker(Database(tmp))
    risk_agent = RiskAgent()

    history: list[dict] = []
    for row in frame.itertuples():
        price = float(row.close)
        when = str(row.date.date())
        closed = broker.update(price, when)
        rk = risk_agent.suggest(
            signal=row.signal,
            probability=float(row.probability),
            volatility=row.audusd_vol_20d,
            user_leverage=leverage,
            probability_source=getattr(row, "probability_source", "fallback_score_map"),
        )
        intent = build_order_intent(
            {"signal": row.signal, "probability": float(row.probability)}, rk, base_units=base_units
        )
        opened = broker.place(intent, price, when)
        realised = broker.realised()
        unrealised = broker.unrealised(price)
        history.append(
            {
                "date": when,
                "signal": row.signal,
                "probability": round(float(row.probability), 4),
                "action": rk.action,
                "price": price,
                "opened": opened.get("status"),
                "closed": len(closed),
                "open_positions": len(broker.db.load_open_positions()),
                "realised_pnl": round(realised, 2),
                "unrealised_pnl": round(unrealised, 2),
                "equity_pnl": round(realised + unrealised, 2),
            }
        )

    positions = broker.db.load_paper_positions()
    last_price = float(frame.iloc[-1]["close"])
    return {
        "status": "ok",
        "history": pd.DataFrame(history),
        "positions": positions,
        "last_price": last_price,
        "window_start": str(frame.iloc[0]["date"].date()),
        "window_end": str(frame.iloc[-1]["date"].date()),
        "base_units": base_units,
    }


def summarize(result: dict) -> dict:
    pos = result["positions"]
    closed = pos[pos["status"] == "closed"] if not pos.empty else pos
    open_pos = pos[pos["status"] == "open"] if not pos.empty else pos
    wins = closed[closed["realised_pnl"] > 0] if not closed.empty else closed
    realised = float(closed["realised_pnl"].fillna(0).sum()) if not closed.empty else 0.0
    hist = result["history"]
    return {
        "window": f"{result['window_start']} -> {result['window_end']}",
        "trading_days": len(hist),
        "trades_closed": int(len(closed)),
        "trades_open": int(len(open_pos)),
        "win_rate": round(float(len(wins) / len(closed)), 3) if len(closed) else 0.0,
        "realised_pnl": round(realised, 2),
        "final_equity_pnl": float(hist.iloc[-1]["equity_pnl"]) if not hist.empty else 0.0,
        "avg_pnl_per_trade": round(realised / len(closed), 2) if len(closed) else 0.0,
        "notional_per_trade": round(result["base_units"] * result["last_price"], 2),
    }


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="Replay AlphaFX paper trading over a past window")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--years", type=int, default=5, help="history loaded for signal calibration")
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--base-units", type=int, default=1000)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    result = replay(args.days, args.leverage, args.base_units, args.years, args.refresh)
    if result["status"] != "ok":
        print(f"Replay produced no result: {result['status']}")
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.days}d"
    hist_path = out / f"paper_replay_{tag}_history.csv"
    trades_path = out / f"paper_replay_{tag}_trades.csv"
    result["history"].to_csv(hist_path, index=False)
    result["positions"].to_csv(trades_path, index=False)

    s = summarize(result)
    print(f"=== Paper-trade replay  {s['window']} ===")
    print(f"trading days     : {s['trading_days']}")
    print(f"trades closed    : {s['trades_closed']}  (open at end: {s['trades_open']})")
    print(f"win rate         : {s['win_rate']}")
    print(f"realised PnL     : {s['realised_pnl']}  (USD, {args.base_units} AUD units/trade)")
    print(f"final equity PnL : {s['final_equity_pnl']}  (realised + open MtM)")
    print(f"avg PnL / trade  : {s['avg_pnl_per_trade']}")
    print(f"notional / trade : ~{s['notional_per_trade']} USD")
    print(f"\nwrote: {hist_path}\nwrote: {trades_path}")


if __name__ == "__main__":
    main()
