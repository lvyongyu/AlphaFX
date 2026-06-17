#!/usr/bin/env python3
"""Leverage stress demo for the paper-trading layer.

Replays a reproducible synthetic AUD/USD path through the *real* PaperBroker at
each requested leverage and reports how leverage amplifies losses. This
deliberately sizes positions at the requested leverage even above the RiskAgent's
5x recommendation cap — the whole point is to show *why* that cap exists.

    python scripts/leverage_sim.py            # compare 5x vs 20x over 300 paths
    python scripts/leverage_sim.py 2 5 10 20  # custom leverage ladder

The price paths are synthetic (geometric Brownian motion, ~10% annualised vol)
because this sandbox has no market-data network access. They are illustrative,
not a backtest of the real strategy. The signal is a transparent trend rule
(close vs its 60-day average) so the focus stays on leverage, not signal alpha.
A Monte-Carlo over many seeds is reported so one lucky/unlucky path does not
dominate — the point is the *distribution* of outcomes, especially blow-ups.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphafx.database import Database  # noqa: E402
from alphafx.risk import RiskAgent  # noqa: E402
from alphafx.trade.order import build_order_intent  # noqa: E402
from alphafx.trade.paper import PaperBroker  # noqa: E402

BASE_UNITS = 1000
ANNUAL_VOL = 0.10
START_PRICE = 0.6500
N_DAYS = 1260  # ~5 trading years
N_PATHS = 300


def synthetic_path(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    daily_sigma = ANNUAL_VOL / np.sqrt(252)
    shocks = rng.normal(0.0, daily_sigma, N_DAYS)
    close = START_PRICE * np.exp(np.cumsum(shocks))
    dates = pd.bdate_range("2019-01-01", periods=N_DAYS)
    df = pd.DataFrame({"date": dates, "close": close})
    ret = df["close"].pct_change()
    df["vol"] = (ret.rolling(20).std() * np.sqrt(252)).fillna(ANNUAL_VOL)
    df["sma"] = df["close"].rolling(60).mean()
    df["signal"] = np.where(df["close"] >= df["sma"], "bullish", "bearish")
    df.loc[df["sma"].isna(), "signal"] = "neutral"
    return df


def simulate(path: pd.DataFrame, leverage: float) -> dict:
    """Replay one path through the real PaperBroker at `leverage`.

    Open/closed state is tracked locally so the only per-day DB call is the
    broker's own stop/TP check (`update`); equity is recorded at trade closes
    (stops cap per-trade loss, so the close-event curve captures the drawdown).
    """
    capital = BASE_UNITS * START_PRICE  # 1x notional committed as margin
    risk_agent = RiskAgent()
    with tempfile.TemporaryDirectory() as tmp:
        broker = PaperBroker(Database(Path(tmp) / "sim.db"))
        realised = 0.0
        in_position = False
        equity_curve = [capital]
        pnls: list[float] = []
        blew_up = False
        for row in path.itertuples():
            price, when = float(row.close), str(row.date)
            closed = broker.update(price, when)
            if closed:
                realised += sum(float(c["realised_pnl"]) for c in closed)
                pnls.extend(float(c["realised_pnl"]) for c in closed)
                in_position = False
                equity_curve.append(capital + realised)
                if capital + realised <= 0:  # margin call -> account liquidated
                    blew_up = True
                    break
            if not in_position and row.signal != "neutral":
                risk = risk_agent.suggest(signal=row.signal, probability=0.65, volatility=float(row.vol))
                intent = build_order_intent(
                    {"signal": row.signal, "probability": 0.65}, risk, base_units=BASE_UNITS, leverage=leverage
                )
                if broker.place(intent, price, when).get("status") == "opened":
                    in_position = True

        eq = pd.Series(equity_curve)
        pnl = pd.Series(pnls)
        return {
            "trades": int(len(pnl)),
            "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
            "total_return_on_capital": -1.0 if blew_up else float(eq.iloc[-1] / capital - 1.0),
            "max_drawdown": float((eq / eq.cummax() - 1.0).min()),
            "worst_trade_pct_capital": float(pnl.min() / capital) if len(pnl) else 0.0,
            "blew_up": blew_up,
        }


def main() -> None:
    leverages = [float(x) for x in sys.argv[1:]] or [5.0, 20.0]
    paths = [synthetic_path(seed) for seed in range(N_PATHS)]
    print(f"Monte-Carlo: {N_PATHS} synthetic AUD/USD paths, {N_DAYS} days, ~{ANNUAL_VOL:.0%} annual vol.")
    print(f"Capital = 1x notional = {BASE_UNITS} x {START_PRICE} = {BASE_UNITS * START_PRICE:.0f} (quote ccy).")
    print("Same paths, same trend signal, same vol-based stops — only leverage differs.\n")
    header = f"{'lev':>5} {'median_ret':>11} {'p05_ret':>9} {'med_maxDD':>10} {'worst_trade':>12} {'blow_up_rate':>13}"
    print(header)
    print("-" * len(header))
    for lev in leverages:
        results = [simulate(p, lev) for p in paths]
        rets = np.array([r["total_return_on_capital"] for r in results])
        dds = np.array([r["max_drawdown"] for r in results])
        worst = np.array([r["worst_trade_pct_capital"] for r in results])
        blow_rate = float(np.mean([r["blew_up"] for r in results]))
        print(
            f"{lev:>5.0f} {np.median(rets):>+11.1%} {np.percentile(rets, 5):>+9.1%} "
            f"{np.median(dds):>+10.1%} {worst.min():>+12.1%} {blow_rate:>13.0%}"
        )
    print("\nAll figures are on the 1x capital. median_ret/p05_ret = median and 5th-percentile")
    print("total return across paths; med_maxDD = median max drawdown; worst_trade = worst single")
    print("closed trade across all paths; blow_up_rate = fraction of paths the account was")
    print("liquidated (equity <= 0). RiskAgent recommends <=5x; higher is shown to expose the")
    print("fat left tail and blow-up risk that leverage adds even to the same signal.")


if __name__ == "__main__":
    main()
