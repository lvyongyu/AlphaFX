#!/usr/bin/env python3
"""Per-instrument validation — the go/no-go gate for the multi-instrument thesis.

For every configured instrument this runs the SAME engine used live (build_context
with walk-forward adaptive signs + expanding calibration) and reports, over a
multi-year window:

  - live factors         how many of the 5 score factors actually have data
                         (non-AUD pairs run on the 3 generic factors today)
  - own-momentum IC      raw aligned IC of the pair's 20d momentum vs its own
                         20d forward return (a quick read on whether there's any
                         linear signal at all; the engine learns the SIGN itself)
  - 5y replay            trades / win / realised PnL / equity PnL through the real
                         PaperBroker (triple-barrier), via scripts.replay_paper

It does NOT assume AUD's factors carry over — each pair stands on its own data.
A pair is flagged GO only if the 5y replay is net-positive AND it actually trades;
everything else is NO-GO (reported honestly, not tuned into looking good).

    python scripts/validate_instruments.py            # all 7, 5-year window
    python scripts/validate_instruments.py --years 5
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphafx.config import load_local_env  # noqa: E402
from alphafx.dashboard.context import build_context  # noqa: E402
from alphafx.database import Database  # noqa: E402
from alphafx.instruments import INSTRUMENTS  # noqa: E402
from scripts.replay_paper import replay, summarize  # noqa: E402


def own_momentum_ic(ctx, cfg, horizon: int = 20) -> tuple[float, int]:
    """Aligned IC of own 20d momentum vs own forward 20d return (sign-agnostic read)."""
    px = (
        ctx.market_data[ctx.market_data["symbol"] == cfg.fx_symbol]
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .sort_values("date")[["date", "close"]]
    )
    px["fwd"] = px["close"].shift(-horizon) / px["close"] - 1.0
    feat = ctx.features.assign(date=lambda x: pd.to_datetime(x["date"]))[["date", "audusd_return_20d"]]
    m = feat.merge(px[["date", "fwd"]], on="date", how="inner").dropna()
    if len(m) < 60:
        return float("nan"), len(m)
    return float(m["audusd_return_20d"].corr(m["fwd"])), len(m)


def validate(years: int) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=365 * years)
    db = Database()
    rows = []
    for name, cfg in INSTRUMENTS.items():
        ctx = build_context(start, end, 2.0, use_llm=False, refresh=False, db=db, instrument=name)
        if ctx.status != "ok":
            rows.append({"instrument": name, "status": ctx.status})
            continue
        live = int(sum(ctx.raw_signals[c].notna().sum() > 0 for c in ctx.signal_agent.score_columns if c in ctx.raw_signals))
        ic, n = own_momentum_ic(ctx, cfg)
        rep = replay(days=365 * years, leverage=2.0, base_units=1000, years=years, refresh=False, instrument=name)
        s = summarize(rep) if rep.get("status") == "ok" else {}
        realised = s.get("realised_pnl", float("nan"))
        rows.append(
            {
                "instrument": name,
                "label": cfg.label,
                "live_factors": live,
                "own_mom_IC": round(ic, 3) if ic == ic else None,
                "trades": s.get("trades_closed", 0),
                "win": s.get("win_rate", float("nan")),
                "realised_pnl": realised,
                "equity_pnl": s.get("final_equity_pnl", float("nan")),
                "verdict": "GO" if (isinstance(realised, (int, float)) and realised > 0 and s.get("trades_closed", 0) >= 5) else "NO-GO",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    load_local_env()
    ap = argparse.ArgumentParser(description="Validate each instrument for the multi-instrument portfolio")
    ap.add_argument("--years", type=int, default=5)
    args = ap.parse_args()

    table = validate(args.years)
    print(f"\n=== Per-instrument validation ({args.years}y window) ===")
    print(table.to_string(index=False))
    gos = table[table.get("verdict") == "GO"]["instrument"].tolist() if "verdict" in table else []
    print(f"\nGO ({len(gos)}): {', '.join(gos) or '(none)'}")
    print("NO-GO pairs are reported as-is — a weak/absent edge is a real finding, not tuned away.")
    out = Path("data") / f"instrument_validation_{args.years}y.csv"
    table.to_csv(out, index=False)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
