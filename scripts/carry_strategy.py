#!/usr/bin/env python3
"""Carry sleeve for AUD/USD + robustness and two-sleeve combination.

Carry — the LEVEL of the rate differential (AU2Y - US2Y) — is the strongest signal
in the project (IC +0.10/+0.19/+0.28 at 20/60/120d), but it cannot be harvested as a
1-of-5 vote in the 20d tactical model (that made results worse). This builds it as
its OWN sleeve and stress-tests it honestly:

  position = sign of today's rate differential  (ZERO look-ahead — the differential
  is observable today; nothing is fitted, the sign is fixed by carry theory). An
  optional deadband holds the position until the differential crosses +/-band, cutting
  whipsaw/turnover.

The decisive test is the YEARLY breakdown: if the edge is one Fed-cycle regime it will
show up as a single big year, not a persistent one. Also reports the two-sleeve combo
(carry + the existing 20d tactical model) — the right use of breadth: two weakly
correlated alpha sources, not 7 correlated pairs.

    python scripts/carry_strategy.py            # 10y
    python scripts/carry_strategy.py --years 5
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
from alphafx.instruments import get_instrument  # noqa: E402
from alphafx.risk import RiskAgent  # noqa: E402

ANNUAL = 252.0
COST_BPS = 2.0  # per unit of turnover, matching BacktestAgent's transaction_cost_bps


def perf(daily: pd.Series) -> dict:
    r = daily.dropna()
    if r.empty or r.std() == 0:
        return {"sharpe": 0.0, "ret%": 0.0, "maxDD%": 0.0}
    eq = (1 + r).cumprod()
    return {
        "sharpe": float(r.mean() / r.std() * np.sqrt(ANNUAL)),
        "ret%": float((eq.iloc[-1] - 1) * 100),
        "maxDD%": float((eq / eq.cummax() - 1).min() * 100),
    }


def carry_returns(spread: pd.Series, ret: pd.Series, band: float = 0.0) -> tuple[pd.Series, float]:
    """Daily net return of the carry sleeve. Position holds through the deadband."""
    pos = np.zeros(len(spread))
    cur = 0.0
    s = spread.values
    for i in range(len(s)):
        if np.isnan(s[i]):
            pos[i] = cur
            continue
        if s[i] > band:
            cur = 1.0
        elif s[i] < -band:
            cur = -1.0
        # within the band: hold `cur`
        pos[i] = cur
    pos = pd.Series(pos, index=spread.index)
    turnover = float(pos.diff().abs().sum() / len(pos))
    net = pos.shift(1).fillna(0.0) * ret - (COST_BPS / 1e4) * pos.diff().abs().fillna(0.0)
    return net, turnover


def tactical_returns(ctx, cfg, start, end) -> pd.Series:
    """The existing 20d gated tactical model's daily strategy return (return space)."""
    sig = ctx.signals.copy()
    floor = RiskAgent.MIN_CONFIDENCE
    ok = (sig.get("probability_source") == "historical_calibration") & (sig.get("probability") >= floor)
    sig.loc[~ok.fillna(False), "signal"] = "neutral"
    data, _ = ctx.backtest_agent.run(ctx.market_data, sig, start, end, holding_period=20, leverage=1.0,
                                     target_symbol=cfg.fx_symbol)
    if data.empty:
        return pd.Series(dtype=float)
    return pd.Series(data["strategy_return"].values, index=pd.to_datetime(data["date"]))


def main() -> None:
    load_local_env()
    ap = argparse.ArgumentParser(description="AUD/USD carry sleeve + robustness")
    ap.add_argument("--years", type=int, default=10)
    args = ap.parse_args()

    end = date.today()
    start = end - timedelta(days=365 * args.years)
    cfg = get_instrument("AUDUSD")
    ctx = build_context(start, end, 2.0, use_llm=False, refresh=False, db=Database(), instrument="AUDUSD")

    md = ctx.market_data
    px = (md[md["symbol"] == cfg.fx_symbol].assign(date=lambda x: pd.to_datetime(x["date"]))
          .sort_values("date")[["date", "close"]])
    px["ret"] = px["close"].pct_change()
    feat = ctx.features.assign(date=lambda x: pd.to_datetime(x["date"]))[["date", "yield_spread"]]
    df = feat.merge(px, on="date", how="inner").dropna(subset=["yield_spread", "ret"]).set_index("date").sort_index()

    print(f"\n=== AUD/USD carry sleeve ({args.years}y, gross, {COST_BPS}bps costs) ===")
    print("spec                         sharpe   ret%   maxDD%  turnover/day")
    specs = {"raw sign (band 0)": 0.0, "deadband 0.10%": 0.0010, "deadband 0.25%": 0.0025}
    best = None
    for name, band in specs.items():
        net, tn = carry_returns(df["yield_spread"], df["ret"], band=band)
        p = perf(net)
        print(f"  {name:26s} {p['sharpe']:+.2f}   {p['ret%']:+5.1f}  {p['maxDD%']:6.1f}   {tn:.4f}")
        if best is None:
            best = (name, net)

    # Robustness: yearly breakdown of the raw-sign carry sleeve (the decisive test).
    name, net = best
    print(f"\n=== YEARLY breakdown — {name} (is the edge persistent or one regime?) ===")
    yearly = pd.DataFrame(
        [{"year": y, **perf(net[net.index.year == y])} for y in sorted(set(net.index.year))]
    ).set_index("year")
    print(yearly[["sharpe", "ret%", "maxDD%"]].round(2).to_string())
    pos_years = int((yearly["ret%"] > 0).sum())
    print(f"positive years: {pos_years} / {len(yearly)}")

    # Two-sleeve combination: carry + the existing 20d tactical model.
    tact = tactical_returns(ctx, cfg, start, end)
    combo = pd.DataFrame({"carry": net, "tactical": tact}).dropna()
    if not combo.empty:
        port = combo.mean(axis=1)
        corr = float(combo["carry"].corr(combo["tactical"]))
        print(f"\n=== Two-sleeve portfolio (carry + 20d tactical, equal weight) ===")
        for label, series in [("carry sleeve", combo["carry"]), ("tactical sleeve", combo["tactical"]),
                              ("COMBINED", port)]:
            p = perf(series)
            print(f"  {label:16s} sharpe={p['sharpe']:+.2f}  ret={p['ret%']:+.1f}%  maxDD={p['maxDD%']:.1f}%")
        print(f"  carry/tactical correlation: {corr:+.2f}  (low = genuinely diversifying)")

    print("\nNB: carry has fat left-tail crash risk (unwinds in risk-off); this 2016-26 window")
    print("has no 2008-style event, so the Sharpe understates tail risk. AUD-only (yield data).")


if __name__ == "__main__":
    main()
