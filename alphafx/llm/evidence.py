from __future__ import annotations

from typing import Any

import pandas as pd

from ..agents import RiskSuggestion


def _num(value: Any) -> float | None:
    """Coerce a value to a plain float, or None when missing — never fabricate."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


def build_evidence_pack(
    signal: pd.Series | None,
    factors: pd.DataFrame | None = None,
    diagnostics: pd.DataFrame | None = None,
    backtest: dict[str, Any] | None = None,
    risk: RiskSuggestion | None = None,
) -> dict[str, Any]:
    """Serialize ONLY pre-computed numbers for the LLM.

    No raw price series and no future data ever enter the pack. Missing factors
    are kept and marked (stance "not available"), never silently dropped, so the
    model can say a factor is unavailable rather than invent it.
    """
    if signal is None or (isinstance(signal, pd.Series) and signal.empty):
        return {"signal": None}

    pack: dict[str, Any] = {
        "as_of": str(signal.get("date")),
        "signal": signal.get("signal"),
        "score": _num(signal.get("score")),
        "probability": _num(signal.get("probability")),
        "probability_source": signal.get("probability_source"),
        "calibration_sample_size": int(signal.get("calibration_sample_size") or 0),
        "confidence": signal.get("confidence"),
        "factors": [],
    }

    if factors is not None and not factors.empty:
        for _, row in factors.iterrows():
            pack["factors"].append(
                {
                    "factor": row.get("factor"),
                    "current_value": _num(row.get("current_value")),
                    "change_20d": _num(row.get("change_20d")),
                    "contribution": _num(row.get("contribution")),
                    "stance": row.get("stance"),
                }
            )

    if risk is not None:
        pack["risk"] = {
            "action": risk.action,
            "leverage": _num(risk.leverage),
            "stop_loss": _num(risk.stop_loss),
            "take_profit": _num(risk.take_profit),
            "regime": risk.regime,
        }

    if diagnostics is not None and not diagnostics.empty and pack["signal"] is not None:
        subset = diagnostics[diagnostics["signal"] == pack["signal"]]
        rows = []
        for _, row in subset.iterrows():
            rows.append({col: (row[col] if col == "signal" else _num(row[col])) for col in subset.columns})
        if rows:
            pack["diagnostics"] = rows

    if backtest is not None:
        pack["backtest"] = {k: _num(v) for k, v in backtest.items() if isinstance(v, (int, float))}

    return pack
