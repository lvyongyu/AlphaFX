#!/usr/bin/env python3
"""Portfolio replay — does breadth actually improve risk-adjusted return?

Takes the GO instruments, runs each pair's GATED adaptive signal through the
return-based BacktestAgent (holding_period=20 ≈ the paper engine's time barrier),
then combines the per-pair daily strategy-return series into an EQUAL-WEIGHT
portfolio. Reports per-pair Sharpe, the combined portfolio Sharpe / drawdown, the
cross-pair correlation matrix, and the realised breadth (avg pairs in a position).

Why return-space (not the dollar PnL of the per-instrument replay): pair prices
span 0.7 (AUD) to ~150 (JPY), so fixed-unit dollar PnL is not comparable across
pairs and cannot be summed into a portfolio. Daily strategy RETURNS are scale-free
and compound correctly — the right basis for Sharpe, drawdown and correlation.

This is the test of IR = IC x sqrt(breadth): if the combined Sharpe beats the
average (and the best) single pair because the pairs are weakly correlated, the
breadth thesis holds. If not, it doesn't — reported either way.

    python scripts/replay_portfolio.py
    python scripts/replay_portfolio.py --instruments AUDUSD,EURUSD,USDCHF,NZDUSD,GBPUSD --years 5
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

GO_DEFAULT = ["AUDUSD", "EURUSD", "USDCHF", "NZDUSD", "GBPUSD"]
ANNUAL = 252.0


def gated_signals(ctx) -> pd.DataFrame:
    """Neutralize any signal that wouldn't pass the live evidence+confidence gate."""
    sig = ctx.signals.copy()
    floor = RiskAgent.MIN_CONFIDENCE
    tradeable = (sig.get("probability_source") == "historical_calibration") & (sig.get("probability") >= floor)
    sig.loc[~tradeable.fillna(False), "signal"] = "neutral"
    return sig


def pair_daily_returns(name: str, start: date, end: date, db: Database) -> pd.Series:
    """Daily strategy return series for one pair's gated signal (NaN -> flat day)."""
    cfg = get_instrument(name)
    ctx = build_context(start, end, 2.0, use_llm=False, refresh=False, db=db, instrument=name)
    if ctx.status != "ok":
        return pd.Series(dtype=float, name=name)
    data, _ = ctx.backtest_agent.run(
        ctx.market_data, gated_signals(ctx), start, end, holding_period=20, leverage=2.0,
        target_symbol=cfg.fx_symbol,
    )
    if data.empty:
        return pd.Series(dtype=float, name=name)
    return pd.Series(data["strategy_return"].values, index=pd.to_datetime(data["date"]), name=name)


def sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(ANNUAL)) if sd and sd > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def main() -> None:
    load_local_env()
    ap = argparse.ArgumentParser(description="Equal-weight portfolio replay over the GO instruments")
    ap.add_argument("--instruments", default=",".join(GO_DEFAULT))
    ap.add_argument("--years", type=int, default=5)
    args = ap.parse_args()
    names = [s.strip().upper() for s in args.instruments.split(",") if s.strip()]

    end = date.today()
    start = end - timedelta(days=365 * args.years)
    db = Database()

    series = {}
    for name in names:
        s = pair_daily_returns(name, start, end, db)
        if not s.empty:
            series[name] = s
    if not series:
        print("No pair produced returns.")
        return

    rets = pd.DataFrame(series).sort_index()
    # Equal-weight across the pairs that have a return that day (a flat pair is 0,
    # not missing, so weights stay equal across the full set).
    port = rets.fillna(0.0).mean(axis=1)
    port_eq = (1.0 + port).cumprod()

    print(f"\n=== Portfolio replay ({args.years}y, equal-weight {len(rets.columns)} pairs) ===")
    print(f"window: {rets.index.min().date()} -> {rets.index.max().date()}\n")

    # Per-pair stats
    print("per-pair (gated adaptive signal, return space):")
    rows = []
    for name in rets.columns:
        s = rets[name].fillna(0.0)
        eq = (1.0 + s).cumprod()
        active = float((s != 0).mean())
        rows.append({"pair": name, "sharpe": round(sharpe(s), 2), "total_ret%": round((eq.iloc[-1] - 1) * 100, 1),
                     "maxDD%": round(max_drawdown(eq) * 100, 1), "days_in_mkt%": round(active * 100, 1)})
    per = pd.DataFrame(rows)
    print(per.to_string(index=False))

    avg_single = per["sharpe"].mean()
    best_single = per["sharpe"].max()
    aud_only = float(per.loc[per["pair"] == "AUDUSD", "sharpe"].iloc[0]) if "AUDUSD" in set(per["pair"]) else float("nan")
    breadth = float((rets.fillna(0.0) != 0).sum(axis=1).mean())

    print(f"\n=== PORTFOLIO ===")
    print(f"combined Sharpe : {sharpe(port):+.2f}")
    print(f"total return    : {(port_eq.iloc[-1]-1)*100:+.1f}%")
    print(f"max drawdown    : {max_drawdown(port_eq)*100:.1f}%")
    print(f"realised breadth: {breadth:.2f} pairs in a position on an avg day")
    print(f"\nbenchmark Sharpe — AUD-only: {aud_only:+.2f} | avg single pair: {avg_single:+.2f} | best single: {best_single:+.2f}")
    verdict = "breadth HELPS" if sharpe(port) > best_single else ("breadth NEUTRAL" if sharpe(port) >= avg_single else "breadth HURTS")
    print(f"verdict: portfolio Sharpe vs best single -> {verdict}")

    print("\ncross-pair correlation of daily strategy returns (low = independent bets = breadth works):")
    corr = rets.fillna(0.0).corr()
    print(corr.round(2).to_string())
    offdiag = corr.where(~np.eye(len(corr), dtype=bool))
    print(f"mean pairwise correlation: {np.nanmean(offdiag.values):.2f}")

    out = Path("data") / f"portfolio_replay_{args.years}y.csv"
    pd.DataFrame({"date": port_eq.index, "port_return": port.values, "port_equity": port_eq.values}).to_csv(out, index=False)
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
