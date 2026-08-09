"""Three-factor cross-sectional FX model: carry, momentum, value.

The 5-factor directional model this replaces scored one pair with five votes —
momentum, DXY, yield-spread change, iron ore, VIX — and lost 23% over ten years.
Two things were wrong with it beyond the numbers. Three of the five votes (DXY,
VIX, iron ore) all proxy the same dollar/risk axis, so equal weighting quietly
put three votes on one idea; and the strongest measured signal, the rate
differential LEVEL, was never in the score at all — only its 20-day change was.

This model takes the three factors the currency literature actually converges on
and applies them the way the literature applies them: ranked across a universe,
dollar-neutral, so the common dollar move cancels and breadth scales with the
number of currencies.

    carry     rate differential vs USD (level, not change)
    momentum  trailing 12-month SPOT return, excluding the carry it earned, so
              the two factors stay distinct rather than measuring each other
    value     5-year real-exchange-rate reversal: a currency that has become
              expensive in real terms is expected to give it back

Value is measured on the BIS real broad effective exchange rate rather than
built from CPI. REER already embeds the inflation adjustment with one consistent
methodology across countries, and FRED's Japanese CPI stops in 2021 — forward
filling it through the 2022-24 inflation would have biased the yen's valuation
by several percent in exactly the period that matters most.

Every signal is lagged before use, and weights earn the NEXT day's return. See
`carry.py` for the shared portfolio machinery.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .carry import (
    CarryResult,
    excess_returns,
    gross_turnover,
    rank_weights,
)
from .data.fred_provider import FREDProvider
from .database import Database

# BIS real broad effective exchange rates. Monthly, 1994 to date, and still
# maintained — unlike FRED's OECD CPI family, which stops between 2021 and 2025
# depending on the country.
REER_SERIES: dict[str, str] = {
    "USD": "RBUSBIS",
    "EUR": "RBXMBIS",
    "GBP": "RBGBBIS",
    "JPY": "RBJPBIS",
    "CAD": "RBCABIS",
    "CHF": "RBCHBIS",
    "AUD": "RBAUBIS",
    "NZD": "RBNZBIS",
}

# BIS publishes a month's REER in the middle of the following month.
REER_PUBLICATION_LAG_MONTHS = 1

MOMENTUM_LOOKBACK_DAYS = 252  # ~12 months, the horizon with the strongest evidence
VALUE_LOOKBACK_DAYS = 1260  # ~5 years, the horizon used for currency value

FACTORS = ("carry", "momentum", "value")


def reer_symbol(currency: str) -> str:
    """macro_data symbol for a currency's REER, e.g. AUD -> AUD_REER."""
    return f"{currency}_REER"


def download_reer(db: Database, start: object, end: object | None = None) -> pd.DataFrame:
    """Fetch every currency's BIS real effective exchange rate into macro_data."""
    frames = []
    for currency, series_id in REER_SERIES.items():
        frame = FREDProvider(series_id, reer_symbol(currency), "monthly").download(start, end)
        if not frame.empty:
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    db.upsert_macro_data(data)
    return data


def build_reer_panel(macro: pd.DataFrame, index: pd.DatetimeIndex,
                     lag_months: int = REER_PUBLICATION_LAG_MONTHS) -> pd.DataFrame:
    """Monthly REER -> a daily panel of what was published by each date."""
    if macro.empty:
        return pd.DataFrame(index=index)
    wanted = {reer_symbol(c): c for c in REER_SERIES}
    frame = macro[macro["symbol"].isin(wanted)].copy()
    if frame.empty:
        return pd.DataFrame(index=index)
    frame["currency"] = frame["symbol"].map(wanted)
    frame["date"] = pd.to_datetime(frame["date"]) + pd.DateOffset(months=lag_months)
    panel = frame.pivot_table(index="date", columns="currency", values="value", aggfunc="last").sort_index()
    return panel.reindex(panel.index.union(index)).ffill().reindex(index)


# ---- the three signals ----

def carry_signal(rates: pd.DataFrame, currencies: list[str]) -> pd.DataFrame:
    """Rate differential against USD, in percentage points.

    The LEVEL, not its change. A positive differential is what a carry trade
    actually collects; the change is a momentum measure wearing carry's clothes,
    which is the confusion the old `yield_score` fell into.
    """
    return rates[currencies].sub(rates["USD"], axis=0)


def momentum_signal(spot: pd.DataFrame, lookback: int = MOMENTUM_LOOKBACK_DAYS) -> pd.DataFrame:
    """Trailing spot return over `lookback` trading days, as a log return.

    Spot only. Using the excess return here would fold the interest differential
    into momentum and leave it correlated with carry by construction, which
    defeats the point of holding both.
    """
    return np.log(spot / spot.shift(lookback))


def value_signal(reer: pd.DataFrame, currencies: list[str],
                 lookback: int = VALUE_LOOKBACK_DAYS) -> pd.DataFrame:
    """Negative of the 5-year real appreciation: cheap currencies score high.

    A currency whose real exchange rate has risen over five years has become
    expensive relative to its own history and is expected to give some of that
    back. The sign is flipped so that, like the other two, a higher score means
    "prefer to be long".
    """
    available = [c for c in currencies if c in reer.columns]
    return -np.log(reer[available] / reer[available].shift(lookback))


# ---- combination ----

def cross_sectional_zscore(signal: pd.DataFrame, min_names: int = 3) -> pd.DataFrame:
    """Standardise each row across currencies so factors are comparable.

    Carry is in percentage points, momentum and value in log returns; averaging
    them raw would silently weight by unit rather than by conviction. Rows with
    too few names, or with no dispersion, are dropped rather than amplified.
    """
    count = signal.notna().sum(axis=1)
    mean = signal.mean(axis=1)
    std = signal.std(axis=1)
    scaled = signal.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)
    return scaled.where((count >= min_names) & (std > 0), np.nan)


def combine_signals(signals: dict[str, pd.DataFrame],
                    weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Weighted average of per-factor z-scores, skipping factors not yet warm.

    A currency is scored on whatever factors have data for it on that date, so
    the model starts trading on carry alone and adds momentum and value as their
    lookbacks fill, instead of standing flat for the first five years.
    """
    if not signals:
        return pd.DataFrame()
    weights = weights or {name: 1.0 for name in signals}
    scaled = {name: cross_sectional_zscore(frame) for name, frame in signals.items()}
    total = None
    denominator = None
    for name, frame in scaled.items():
        weight = float(weights.get(name, 0.0))
        if weight == 0.0:
            continue
        contribution = frame * weight
        present = frame.notna() * abs(weight)
        total = contribution if total is None else total.add(contribution, fill_value=0.0)
        denominator = present if denominator is None else denominator.add(present, fill_value=0.0)
    if total is None:
        return pd.DataFrame()
    return total.div(denominator.replace(0.0, np.nan))


def hold_between_rebalances(weights: pd.DataFrame, freq: str | None = "MS") -> pd.DataFrame:
    """Re-rank only on rebalance dates, holding the book in between.

    Re-ranking daily is not what the currency literature measures: momentum is
    documented with a 12-month lookback and a ONE-MONTH holding period. Recomputed
    daily, a 252-day window drops one observation and adds another every session,
    so the ranks churn — 23% of days here versus 0.3% for carry — and the turnover
    is paid for without the signal having changed in any meaningful way.

    `freq=None` keeps the daily behaviour.
    """
    if freq is None or weights.empty:
        return weights
    marks = weights.groupby(pd.Grouper(freq=freq)).head(1).index
    return weights.loc[weights.index.isin(marks)].reindex(weights.index).ffill().fillna(0.0)


def factor_portfolio(
    spot: pd.DataFrame,
    rates: pd.DataFrame,
    reer: pd.DataFrame,
    weights: dict[str, float] | None = None,
    basket: int = 2,
    cost_bps: float = 2.0,
    rebalance: str | None = "MS",
) -> tuple[CarryResult, dict[str, pd.DataFrame]]:
    """Rank the universe on the combined score and hold the extremes.

    Returns the portfolio result and the individual factor signals, so a caller
    can report each factor standalone and their correlations without recomputing.
    """
    returns = excess_returns(spot, rates)
    if returns.empty:
        return CarryResult(pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), 0.0), {}
    currencies = list(returns.columns)
    signals = {
        "carry": carry_signal(rates, currencies).reindex(returns.index),
        "momentum": momentum_signal(spot[currencies]).reindex(returns.index),
        "value": value_signal(reer, currencies).reindex(returns.index),
    }
    combined = combine_signals(signals, weights)
    if combined.empty:
        return CarryResult(pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), 0.0), signals
    held_weights = hold_between_rebalances(rank_weights(combined, basket=basket), rebalance)
    held = held_weights.shift(1).fillna(0.0)
    traded = gross_turnover(held_weights)
    net = (held * returns).sum(axis=1) - traded * (cost_bps / 1e4)
    return (
        CarryResult(returns=net, weights=held, carry=combined, turnover=float(traded.mean())),
        signals,
    )


def factor_correlations(results: dict[str, pd.Series]) -> pd.DataFrame:
    """Correlation of the standalone factor return streams.

    The multi-factor case rests on these being small: three uncorrelated sleeves
    of Sharpe 0.25 combine to roughly 0.43, three identical ones stay at 0.25.
    """
    frame = pd.DataFrame(results).dropna()
    return frame.corr() if not frame.empty else pd.DataFrame()
