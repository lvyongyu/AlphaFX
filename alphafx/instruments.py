"""Per-instrument configuration for the multi-instrument portfolio.

The research engine was originally hardcoded to AUD/USD. An `InstrumentConfig`
captures everything that varies BY currency pair so the same feature/signal/
backtest pipeline can run on any pair:

  - `fx_symbol`   the Yahoo price symbol for the pair (free via FX_DAILY too)
  - `foreign_yield` / `commodity`  the pair-specific macro factors. When a pair
    has no clean data source for one of these (e.g. NZD has no liquid 2Y on
    FRED, EUR has no single commodity), the field is None and that factor is
    simply absent — the pipeline already treats a missing factor as NaN and
    drops it from the equal-weight score, so a pair can run on the 3 generic
    factors (own-momentum, DXY, VIX) alone.

The three GENERIC factors (momentum, DXY, VIX) need only the pair's daily price
and the shared DXY/VIX series, so they work for every pair with zero new data
plumbing. That is the breadth that the multi-instrument thesis rests on.

Direction note: we do NOT special-case USD-base pairs (USD/JPY, USD/CAD,
USD/CHF) with hand-set factor signs. The walk-forward `adaptive_factor_signs`
learns each factor's sign per pair from its own trailing IC, so the quote
convention (USD as base vs quote) is handled by the data, not by config.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_SYMBOLS


@dataclass(frozen=True)
class InstrumentConfig:
    name: str  # canonical key used as the DB `instrument` value, e.g. "AUDUSD"
    fx_symbol: str  # Yahoo price symbol, e.g. "AUDUSD=X"
    label: str  # display label, e.g. "AUD/USD"
    oanda: str  # broker instrument id for order intents, e.g. "AUD_USD"
    foreign_yield: str | None = None  # macro symbol for the non-USD 2Y leg, or None
    commodity: str | None = None  # macro symbol for the pair's key commodity, or None

    @property
    def macro_symbols(self) -> list[str]:
        """Pair-specific macro series to load (US2Y is always loaded as the base leg)."""
        symbols = ["US2Y"]
        if self.foreign_yield:
            symbols.append(self.foreign_yield)
        if self.commodity:
            symbols.append(self.commodity)
        return symbols


# The 7 instruments approved for the first portfolio pass. AUD/USD is the
# baseline (must reproduce the existing single-instrument numbers exactly).
# foreign_yield / commodity are wired to real providers in Phase 1; where a
# source is not yet available the pair runs on the 3 generic factors.
INSTRUMENTS: dict[str, InstrumentConfig] = {
    "AUDUSD": InstrumentConfig("AUDUSD", DEFAULT_SYMBOLS.audusd, "AUD/USD", "AUD_USD", foreign_yield="AU2Y", commodity="IRON_ORE"),
    "EURUSD": InstrumentConfig("EURUSD", "EURUSD=X", "EUR/USD", "EUR_USD", foreign_yield="EU2Y"),
    "GBPUSD": InstrumentConfig("GBPUSD", "GBPUSD=X", "GBP/USD", "GBP_USD", foreign_yield="GB2Y"),
    "USDJPY": InstrumentConfig("USDJPY", "USDJPY=X", "USD/JPY", "USD_JPY", foreign_yield="JP2Y"),
    "USDCAD": InstrumentConfig("USDCAD", "USDCAD=X", "USD/CAD", "USD_CAD", foreign_yield="CA2Y", commodity="WTI"),
    "NZDUSD": InstrumentConfig("NZDUSD", "NZDUSD=X", "NZD/USD", "NZD_USD"),
    "USDCHF": InstrumentConfig("USDCHF", "USDCHF=X", "USD/CHF", "USD_CHF"),
}

DEFAULT_INSTRUMENT = "AUDUSD"


def get_instrument(name: str | None) -> InstrumentConfig:
    """Resolve an instrument by canonical name, defaulting to the AUD/USD baseline."""
    return INSTRUMENTS[name or DEFAULT_INSTRUMENT]
