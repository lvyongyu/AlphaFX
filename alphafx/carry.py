"""Cross-sectional FX carry: rank a currency universe by short rate, go long the
high yielders and short the low yielders in equal size.

Why this shape rather than timing one pair. The existing carry sleeve asks "is
AUD's rate above USD's?" and takes a directional AUD/USD position. Almost all of
that position's variance is the US dollar, and it produces ~12 independent bets a
year — far too few to tell an edge from noise inside a decade.

Ranking instead produces a dollar-neutral portfolio: the long and short baskets
are equal size, so the US rate term cancels exactly between them and a common
dollar move largely cancels too. What is left is the carry factor itself, and the
number of independent bets scales with the size of the universe.

No look-ahead. OECD short rates are monthly and are published after the month
they describe, so a rate stamped month M is treated as unknown until M + 2
(`PUBLICATION_LAG_MONTHS`). Weights are applied to the NEXT day's return.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data.fred_provider import FREDProvider
from .database import Database

# OECD 3-month interbank rates, one per currency, all monthly and all on FRED.
# This is the rate a carry trade actually earns or pays, which is why it is
# preferred here over the 2Y government yields the tactical model uses.
SHORT_RATE_SERIES: dict[str, str] = {
    "USD": "IR3TIB01USM156N",
    "EUR": "IR3TIB01EZM156N",
    "GBP": "IR3TIB01GBM156N",
    "JPY": "IR3TIB01JPM156N",
    "CAD": "IR3TIB01CAM156N",
    "CHF": "IR3TIB01CHM156N",
    "AUD": "IR3TIB01AUM156N",
    "NZD": "IR3TIB01NZM156N",
}

# Yahoo symbol for each currency against USD, and whether it must be inverted to
# express the price as USD per unit of that currency. USD/JPY quotes JPY per USD,
# so it inverts; AUD/USD already quotes USD per AUD, so it does not.
SPOT_SYMBOLS: dict[str, tuple[str, bool]] = {
    "AUD": ("AUDUSD=X", False),
    "EUR": ("EURUSD=X", False),
    "GBP": ("GBPUSD=X", False),
    "NZD": ("NZDUSD=X", False),
    "JPY": ("USDJPY=X", True),
    "CAD": ("USDCAD=X", True),
    "CHF": ("USDCHF=X", True),
}

# Months a monthly OECD observation is withheld before we allow ourselves to see
# it. The month M figure describes M and lands mid-M+1 at the earliest; 2 months
# is deliberately conservative, since being early here would manufacture alpha.
PUBLICATION_LAG_MONTHS = 2

TRADING_DAYS = 252.0
DEFAULT_COST_BPS = 2.0  # per unit of turnover, matching BacktestAgent


def rate_symbol(currency: str) -> str:
    """macro_data symbol for a currency's short rate, e.g. AUD -> AUD_3M."""
    return f"{currency}_3M"


def download_short_rates(db: Database, start: object, end: object | None = None) -> pd.DataFrame:
    """Fetch every currency's 3M interbank rate from FRED into macro_data."""
    frames = []
    for currency, series_id in SHORT_RATE_SERIES.items():
        provider = FREDProvider(series_id, rate_symbol(currency), "monthly")
        frame = provider.download(start, end)
        if not frame.empty:
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    db.upsert_macro_data(data)
    return data


def build_rate_panel(macro: pd.DataFrame, index: pd.DatetimeIndex,
                     lag_months: int = PUBLICATION_LAG_MONTHS) -> pd.DataFrame:
    """Monthly rates -> a daily panel of what was KNOWN on each date, in percent.

    Each observation's date is pushed forward by `lag_months` before being
    forward-filled, so a month's rate cannot influence a decision taken before it
    was published. Columns are currencies; rows are `index`.
    """
    if macro.empty:
        return pd.DataFrame(index=index)
    wanted = {rate_symbol(c): c for c in SHORT_RATE_SERIES}
    frame = macro[macro["symbol"].isin(wanted)].copy()
    if frame.empty:
        return pd.DataFrame(index=index)
    frame["currency"] = frame["symbol"].map(wanted)
    frame["date"] = pd.to_datetime(frame["date"]) + pd.DateOffset(months=lag_months)
    panel = (
        frame.pivot_table(index="date", columns="currency", values="value", aggfunc="last")
        .sort_index()
    )
    # Union the two indexes first so a rate published between trading days is not
    # dropped, then forward-fill and restrict to the trading calendar.
    return panel.reindex(panel.index.union(index)).ffill().reindex(index)


def build_spot_panel(market: pd.DataFrame, currencies: list[str] | None = None) -> pd.DataFrame:
    """Daily USD-per-unit price for each currency, inverting USD-base quotes."""
    currencies = currencies or list(SPOT_SYMBOLS)
    series = {}
    for currency in currencies:
        symbol, invert = SPOT_SYMBOLS[currency]
        rows = market[market["symbol"] == symbol]
        if rows.empty:
            continue
        price = (
            rows.assign(date=lambda x: pd.to_datetime(x["date"]))
            .sort_values("date")
            .set_index("date")["close"]
            .astype(float)
        )
        price = price[price > 0]
        series[currency] = 1.0 / price if invert else price
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def excess_returns(spot: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Daily excess return of holding each currency against USD.

    spot appreciation vs USD, plus one day of the interest differential. The
    dollar leg is subtracted here for interpretability; it cancels anyway once
    the long and short baskets are differenced.
    """
    spot_return = spot.pct_change()
    common = [c for c in spot.columns if c in rates.columns]
    differential = rates[common].sub(rates["USD"], axis=0) / 100.0 / TRADING_DAYS
    return (spot_return[common] + differential).dropna(how="all")


@dataclass
class CarryResult:
    returns: pd.Series  # daily net portfolio return
    weights: pd.DataFrame  # per-currency weight actually held
    carry: pd.DataFrame  # the ranking signal (rate differential vs USD, percent)
    turnover: float  # average daily gross weight change

    @property
    def gross(self) -> pd.Series:
        return self.returns


def _turnover(weights: pd.DataFrame) -> pd.Series:
    """Gross weight change per day, counting the initial entry from flat.

    `DataFrame.diff()` leaves the first row NaN; zeroing it would make opening
    the book free, which understates cost for any short backtest window.
    """
    traded = weights.diff().abs().sum(axis=1)
    if len(weights):
        traded.iloc[0] = float(weights.iloc[0].abs().sum())
    return traded.fillna(0.0)


def rank_weights(carry: pd.DataFrame, basket: int = 2) -> pd.DataFrame:
    """Long the `basket` highest carries, short the `basket` lowest, equal size.

    Weights sum to zero on every date with enough currencies, which is what makes
    the book dollar-neutral. Dates with fewer than 2*basket observations are flat
    rather than partially invested.
    """
    weights = pd.DataFrame(0.0, index=carry.index, columns=carry.columns)
    for date, row in carry.iterrows():
        available = row.dropna()
        if len(available) < 2 * basket:
            continue
        ordered = available.sort_values(ascending=False)
        longs = ordered.index[:basket]
        shorts = ordered.index[-basket:]
        weights.loc[date, longs] = 1.0 / basket
        weights.loc[date, shorts] = -1.0 / basket
    return weights


def cross_sectional_carry(
    spot: pd.DataFrame,
    rates: pd.DataFrame,
    basket: int = 2,
    cost_bps: float = DEFAULT_COST_BPS,
) -> CarryResult:
    """Build the dollar-neutral carry portfolio and return its net daily returns."""
    returns = excess_returns(spot, rates)
    if returns.empty:
        return CarryResult(pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), 0.0)
    common = list(returns.columns)
    carry = rates[common].sub(rates["USD"], axis=0).reindex(returns.index)
    weights = rank_weights(carry, basket=basket)
    # Yesterday's weights earn today's return: the signal cannot use the return
    # it is being scored against.
    held = weights.shift(1).fillna(0.0)
    gross = (held * returns).sum(axis=1)
    traded = _turnover(weights)
    net = gross - traded * (cost_bps / 1e4)
    return CarryResult(
        returns=net,
        weights=held,
        carry=carry,
        turnover=float(traded.mean()),
    )


def single_pair_carry(
    spot: pd.DataFrame,
    rates: pd.DataFrame,
    currency: str = "AUD",
    cost_bps: float = DEFAULT_COST_BPS,
) -> CarryResult:
    """The directional baseline: long one currency vs USD when its carry is
    positive, short when negative. This is what the existing sleeve does, rebuilt
    on the same rate data so the comparison is like-for-like."""
    returns = excess_returns(spot[[currency]], rates)
    if returns.empty:
        return CarryResult(pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), 0.0)
    carry = rates[[currency]].sub(rates["USD"], axis=0).reindex(returns.index)
    weights = np.sign(carry).fillna(0.0)
    held = weights.shift(1).fillna(0.0)
    gross = (held * returns).sum(axis=1)
    traded = _turnover(weights)
    return CarryResult(
        returns=gross - traded * (cost_bps / 1e4),
        weights=held,
        carry=carry,
        turnover=float(traded.mean()),
    )


def is_degenerate(r: pd.Series) -> bool:
    """True when a Sharpe would be meaningless rather than merely large.

    Testing `std == 0.0` is not enough: a constant series carries floating-point
    noise of ~1e-19 rather than exact zero, which sails through the guard and
    yields a Sharpe around 1e16. Compare against the series' own scale instead.
    """
    if len(r) < 2:
        return True
    std = float(r.std())
    return not np.isfinite(std) or std < max(1e-15, abs(float(r.mean())) * 1e-9)


def performance(daily: pd.Series) -> dict[str, float]:
    """Annualised Sharpe, total return, max drawdown and hit rate."""
    r = daily.dropna()
    if r.empty or is_degenerate(r):
        return {"sharpe": 0.0, "return_pct": 0.0, "max_drawdown_pct": 0.0, "hit_rate": 0.0, "days": int(len(r))}
    equity = (1.0 + r).cumprod()
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))
    years = len(r) / TRADING_DAYS
    return {
        "sharpe": sharpe,
        # t of the Sharpe estimate. Roughly Sharpe * sqrt(years) — the number that
        # says whether a ratio is evidence or noise. Below ~2, it is noise.
        "t_stat": float(sharpe * np.sqrt(years)),
        "return_pct": float((equity.iloc[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float((equity / equity.cummax() - 1.0).min() * 100.0),
        "hit_rate": float((r > 0).mean()),
        "days": int(len(r)),
    }


def decompose(result: CarryResult, spot: pd.DataFrame, rates: pd.DataFrame) -> dict[str, float]:
    """Split the book's return into interest earned, spot move and costs, annualised.

    A carry trade is meant to collect the rate differential while the spot leg
    gives part of it back — high yielders tend to depreciate, but by less than
    they pay. Seeing that split is how you tell a working carry book from one that
    is accidentally trading momentum: if the interest line is small and the spot
    line carries the return, it is not carry.
    """
    held = result.weights
    if held.empty:
        return {"interest_pct": 0.0, "spot_pct": 0.0, "cost_pct": 0.0, "total_pct": 0.0}
    columns = list(held.columns)
    index = result.returns.index
    spot_leg = (held * spot[columns].pct_change().reindex(index)).sum(axis=1)
    rate_leg = (
        held * rates[columns].sub(rates["USD"], axis=0).div(100.0 * TRADING_DAYS).reindex(index)
    ).sum(axis=1)
    annualise = lambda s: float(s.dropna().mean() * TRADING_DAYS * 100.0)  # noqa: E731
    return {
        "interest_pct": annualise(rate_leg),
        "spot_pct": annualise(spot_leg),
        "cost_pct": annualise(result.returns - spot_leg - rate_leg),
        "total_pct": annualise(result.returns),
    }


def yearly_breakdown(daily: pd.Series) -> pd.DataFrame:
    """Per-calendar-year performance — the test that separates a persistent edge
    from a single lucky regime."""
    r = daily.dropna()
    if r.empty:
        return pd.DataFrame()
    rows = [{"year": year, **performance(r[r.index.year == year])} for year in sorted(set(r.index.year))]
    return pd.DataFrame(rows).set_index("year")
