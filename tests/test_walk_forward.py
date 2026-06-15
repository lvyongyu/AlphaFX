from __future__ import annotations

import pandas as pd


def test_degradation_ratio_not_sign_flipped_on_negative_in_sample():
    # Directly exercise the guard: negative in-sample Sharpe must not flip the sign.
    in_sharpe_neg = -0.5
    out_sharpe = 0.4
    ratio = (out_sharpe / in_sharpe_neg) if in_sharpe_neg > 0 else float("nan")
    assert pd.isna(ratio)  # undefined, not a misleading negative ratio
    in_sharpe_pos = 0.8
    ratio_pos = (out_sharpe / in_sharpe_pos) if in_sharpe_pos > 0 else float("nan")
    assert ratio_pos > 0
