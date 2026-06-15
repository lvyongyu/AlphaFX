from __future__ import annotations

import pandas as pd

from alphafx.data.fred_provider import FREDProvider
from alphafx.data.rba_provider import RBAProvider


def test_fred_parser_normalizes_macro_data():
    raw = pd.DataFrame({"observation_date": ["2024-01-01", "2024-01-02"], "DGS2": ["4.2", "."]})
    parsed = FREDProvider("DGS2", "US2Y", "daily").parse(raw)
    assert list(parsed["symbol"].unique()) == ["US2Y"]
    assert len(parsed) == 1


def test_rba_parser_normalizes_au2y_data():
    raw = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Australian Government 2 year bond": ["3.5", "3.6"]})
    parsed = RBAProvider().parse_au2y(raw)
    assert list(parsed["symbol"].unique()) == ["AU2Y"]
    assert len(parsed) == 2
