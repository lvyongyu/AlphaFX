#!/usr/bin/env python3
"""Cross-sectional FX carry: rank the currency universe, go long the high yielders
and short the low yielders, dollar-neutral.

The question this answers: is the project's carry edge worth more as a ranked,
dollar-neutral portfolio than as a directional bet on one pair? The existing
sleeve times AUD/USD, which puts most of its variance on the US dollar and yields
only ~12 independent bets a year — too few to separate skill from luck in a
decade. Ranking removes the dollar and scales breadth with the universe.

Prints the headline comparison, a yearly breakdown (a persistent edge looks
different from one lucky regime), and a basket-size sweep.

    python scripts/carry_cross_sectional.py                 # 10y, cached rates
    python scripts/carry_cross_sectional.py --refresh       # re-download rates
    python scripts/carry_cross_sectional.py --years 5
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphafx.carry import (  # noqa: E402
    DEFAULT_COST_BPS,
    PUBLICATION_LAG_MONTHS,
    SHORT_RATE_SERIES,
    SPOT_SYMBOLS,
    build_rate_panel,
    build_spot_panel,
    cross_sectional_carry,
    decompose,
    download_short_rates,
    performance,
    rate_symbol,
    single_pair_carry,
    yearly_breakdown,
)
from alphafx.config import load_local_env  # noqa: E402
from alphafx.database import Database  # noqa: E402


def _row(label: str, stats: dict, turnover: float | None = None) -> str:
    tail = f"  {turnover:7.4f}" if turnover is not None else ""
    return (
        f"  {label:34s} {stats['sharpe']:+6.2f}  {stats['return_pct']:+8.1f}  "
        f"{stats['max_drawdown_pct']:7.1f}  {stats['hit_rate']:6.1%}{tail}"
    )


HEADER = f"  {'strategy':34s} {'sharpe':>6s}  {'ret%':>8s}  {'maxDD%':>7s}  {'hit':>6s}  {'turn/d':>7s}"


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="Cross-sectional dollar-neutral FX carry")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--basket", type=int, default=2, help="currencies per leg (default 2)")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--refresh", action="store_true", help="re-download short rates from FRED")
    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=365 * args.years)
    db = Database()

    if args.refresh:
        print("downloading short rates from FRED ...")
        downloaded = download_short_rates(db, start - timedelta(days=400), end)
        print(f"  stored {len(downloaded)} rate observations")

    market = db.load_market_data([symbol for symbol, _ in SPOT_SYMBOLS.values()])
    macro = db.load_macro_data([rate_symbol(c) for c in SHORT_RATE_SERIES])
    if market.empty or macro.empty:
        print("No data. Run once with --refresh, and make sure market data is downloaded.")
        return

    spot = build_spot_panel(market)
    spot = spot[(spot.index >= pd.Timestamp(start)) & (spot.index <= pd.Timestamp(end))]
    rates = build_rate_panel(macro, spot.index)
    usable = [c for c in spot.columns if c in rates.columns and rates[c].notna().any()]
    spot, rates = spot[usable], rates[usable + ["USD"]]

    print(f"\n=== Cross-sectional FX carry — {args.years}y, {args.cost_bps}bps/turnover ===")
    print(f"universe   : {', '.join(usable)}  (vs USD)")
    print(f"window     : {spot.index.min().date()} -> {spot.index.max().date()}  ({len(spot)} days)")
    print(f"rate lag   : {PUBLICATION_LAG_MONTHS} months (publication delay, no look-ahead)")
    stale = {c: rates[c].last_valid_index() for c in usable + ["USD"]}
    oldest = min(stale.values())
    if oldest is not None and oldest < spot.index.max() - pd.Timedelta(days=120):
        behind = [c for c, d in stale.items() if d is not None and d < spot.index.max() - pd.Timedelta(days=120)]
        print(f"NB         : rates carried forward for {', '.join(behind)} (source lags)")

    cross = cross_sectional_carry(spot, rates, basket=args.basket, cost_bps=args.cost_bps)
    aud = single_pair_carry(spot, rates, currency="AUD", cost_bps=args.cost_bps)

    print("\n" + HEADER)
    print(_row(f"cross-sectional (basket {args.basket})", performance(cross.returns), cross.turnover))
    print(_row("AUD/USD directional (baseline)", performance(aud.returns), aud.turnover))

    print("\n=== Basket-size sweep (is the result an artefact of one choice?) ===")
    print(HEADER)
    for basket in range(1, max(2, len(usable) // 2) + 1):
        swept = cross_sectional_carry(spot, rates, basket=basket, cost_bps=args.cost_bps)
        print(_row(f"basket {basket} long / {basket} short", performance(swept.returns), swept.turnover))

    parts = decompose(cross, spot, rates)
    print("\n=== Return decomposition (annualised) — is this actually carry? ===")
    print(f"  interest differential earned  {parts['interest_pct']:+6.2f} %/yr")
    print(f"  spot move                     {parts['spot_pct']:+6.2f} %/yr")
    print(f"  transaction costs             {parts['cost_pct']:+6.2f} %/yr")
    print(f"  {'NET':28s}  {parts['total_pct']:+6.2f} %/yr")

    stats = performance(cross.returns)
    print("\n=== Is it distinguishable from luck? ===")
    print(f"  t-stat of the Sharpe : {stats['t_stat']:.2f}   (needs ~2.0 to be evidence)")
    if stats["sharpe"] > 0:
        print(f"  years to reach t=2   : {(2.0 / stats['sharpe']) ** 2:.0f}")
    worst = cross.returns.rolling(252).sum().min()
    print(f"  worst 1y rolling     : {worst * 100:+.1f}%")

    print(f"\n=== Yearly breakdown — cross-sectional (basket {args.basket}) ===")
    table = yearly_breakdown(cross.returns)
    if not table.empty:
        print(table[["sharpe", "return_pct", "max_drawdown_pct", "days"]].round(2).to_string())
        print(f"positive years: {int((table['return_pct'] > 0).sum())} / {len(table)}")

    print("\n=== Average weight per currency (who is actually held) ===")
    if not cross.weights.empty:
        avg = cross.weights.mean().sort_values(ascending=False)
        print("  " + "  ".join(f"{c}:{w:+.2f}" for c, w in avg.items()))

    print("\nNB: carry crashes in risk-off; this window has no 2008-style unwind, so the")
    print("Sharpe understates tail risk. Rates are monthly, so ranks move slowly by design.")


if __name__ == "__main__":
    main()
