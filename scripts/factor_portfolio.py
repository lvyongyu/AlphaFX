#!/usr/bin/env python3
"""Three-factor cross-sectional FX portfolio: carry, momentum, value.

Replaces the 5-factor directional model's question ("will AUD/USD go up?") with
the one the currency literature actually answers ("which currencies are cheap,
trending and high-yielding relative to the others?"). Ranked, dollar-neutral,
across the G10 universe.

Reports each factor standalone, the blend, and the correlation between them —
because the whole case for holding three rests on them being different. Three
uncorrelated sleeves of Sharpe 0.25 blend to about 0.43; three copies of the
same sleeve stay at 0.25.

    python scripts/factor_portfolio.py                  # 10y, cached data
    python scripts/factor_portfolio.py --refresh        # re-download REER
    python scripts/factor_portfolio.py --years 15
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
    SHORT_RATE_SERIES,
    SPOT_SYMBOLS,
    build_rate_panel,
    build_spot_panel,
    performance,
    rate_symbol,
    yearly_breakdown,
)
from alphafx.config import load_local_env  # noqa: E402
from alphafx.database import Database  # noqa: E402
from alphafx.factors import (  # noqa: E402
    FACTORS,
    REER_SERIES,
    build_reer_panel,
    download_reer,
    factor_correlations,
    factor_portfolio,
    reer_symbol,
)

HEADER = (
    f"  {'strategy':30s} {'sharpe':>6s} {'t':>5s}  {'ret%':>8s}  "
    f"{'maxDD%':>7s}  {'hit':>6s}  {'turn/d':>7s}"
)


def _row(label: str, stats: dict, turnover: float) -> str:
    return (
        f"  {label:30s} {stats['sharpe']:+6.2f} {stats['t_stat']:5.2f}  "
        f"{stats['return_pct']:+8.1f}  {stats['max_drawdown_pct']:7.1f}  "
        f"{stats['hit_rate']:6.1%}  {turnover:7.4f}"
    )


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="Three-factor cross-sectional FX portfolio")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--basket", type=int, default=2, help="currencies per leg")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--refresh", action="store_true", help="re-download REER from FRED")
    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=365 * args.years)
    db = Database()

    if args.refresh:
        print("downloading BIS real effective exchange rates ...")
        print(f"  stored {len(download_reer(db, '1994-01-01', end))} observations")

    market = db.load_market_data([symbol for symbol, _ in SPOT_SYMBOLS.values()])
    macro = db.load_macro_data(
        [rate_symbol(c) for c in SHORT_RATE_SERIES] + [reer_symbol(c) for c in REER_SERIES]
    )
    if market.empty or macro.empty:
        print("No data. Run scripts/carry_cross_sectional.py --refresh first, then this with --refresh.")
        return

    # Build the panels over ALL history so the 12-month and 5-year lookbacks are
    # warm at the start of the reporting window, then clip to the window.
    spot_all = build_spot_panel(market)
    rates_all = build_rate_panel(macro, spot_all.index)
    # REER goes back to 1994, far earlier than the FX prices. Building it on the
    # price index would throw that away and leave the 5-year value lookback cold
    # for the first five years of the backtest, so give it its own longer index
    # and let `factor_portfolio` reindex the finished signal.
    reer_index = pd.date_range(pd.to_datetime(macro["date"]).min(), spot_all.index.max(), freq="B")
    reer_all = build_reer_panel(macro, reer_index)
    usable = [c for c in spot_all.columns if c in rates_all.columns and rates_all[c].notna().any()]

    result, signals = factor_portfolio(
        spot_all[usable], rates_all[usable + ["USD"]], reer_all,
        basket=args.basket, cost_bps=args.cost_bps,
    )
    window = (result.returns.index >= pd.Timestamp(start)) & (result.returns.index <= pd.Timestamp(end))
    blended = result.returns[window]

    print(f"\n=== Three-factor cross-sectional FX — {args.years}y, {args.cost_bps}bps/turnover ===")
    print(f"universe : {', '.join(usable)}  (vs USD, dollar-neutral)")
    print(f"window   : {blended.index.min().date()} -> {blended.index.max().date()}  ({len(blended)} days)")
    print(f"basket   : {args.basket} long / {args.basket} short")

    standalone: dict[str, pd.Series] = {}
    print("\n" + HEADER)
    for factor in FACTORS:
        solo, _ = factor_portfolio(
            spot_all[usable], rates_all[usable + ["USD"]], reer_all,
            weights={f: (1.0 if f == factor else 0.0) for f in FACTORS},
            basket=args.basket, cost_bps=args.cost_bps,
        )
        series = solo.returns[(solo.returns.index >= pd.Timestamp(start)) & (solo.returns.index <= pd.Timestamp(end))]
        standalone[factor] = series
        print(_row(f"{factor} only", performance(series), solo.turnover))
    print(_row("BLENDED (equal weight)", performance(blended), result.turnover))

    print("\n=== Factor return correlations (low = the blend is worth holding) ===")
    corr = factor_correlations(standalone)
    if not corr.empty:
        print(corr.round(3).to_string())

    print("\n=== Yearly breakdown — blended ===")
    table = yearly_breakdown(blended)
    if not table.empty:
        print(table[["sharpe", "return_pct", "max_drawdown_pct", "days"]].round(2).to_string())
        print(f"positive years: {int((table['return_pct'] > 0).sum())} / {len(table)}")

    print("\n=== Average weight per currency ===")
    held = result.weights[window]
    if not held.empty:
        avg = held.mean().sort_values(ascending=False)
        print("  " + "  ".join(f"{c}:{w:+.2f}" for c, w in avg.items()))

    print("\nFor reference, the 5-factor directional model this is meant to replace ran")
    print("net -23% with Sharpe -0.42 over ten years on the live 3-pair portfolio.")


if __name__ == "__main__":
    main()
