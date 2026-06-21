from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from alphafx.agents import (
    AIExplanationAgent,
    BacktestAgent,
    ContrarianAgent,
    DataAgent,
    FactorDiagnosticsAgent,
    FeatureAgent,
    JudgeAgent,
    PaperJournalAgent,
    QuantSignalAgent,
    RiskAgent,
    SignalDiagnosticsAgent,
    WalkForwardAgent,
)
from alphafx.config import DEFAULT_SYMBOLS
from alphafx.database import Database
from alphafx.llm import LLMContrarianAgent, LLMExplanationAgent, LLMJudgeAgent
from alphafx.ml import MLSignalAgent

# Status values returned by build_context.
NO_DATA = "no_data"
NO_SIGNAL = "no_signal"
OK = "ok"


@dataclass
class ResearchContext:
    """Everything the tabs need, computed once per run. No streamlit here."""

    status: str
    # sidebar params
    start: Any = None
    end: Any = None
    leverage: float = 2.0
    use_llm: bool = False
    # agents (tabs re-run some of these on widget input)
    data_agent: Any = None
    feature_agent: Any = None
    signal_agent: Any = None
    diagnostics_agent: Any = None
    risk_agent: Any = None
    backtest_agent: Any = None
    walk_forward_agent: Any = None
    factor_diagnostics_agent: Any = None
    paper_journal_agent: Any = None
    ml_agent: Any = None
    # data
    market_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    macro_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    forward_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    latest_feature: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    latest_signal: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    factor_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    risk: Any = None
    explanation: str = ""
    contrarian: dict[str, str] = field(default_factory=dict)
    judgement: dict[str, Any] = field(default_factory=dict)
    ml_result: dict[str, Any] = field(default_factory=dict)
    ml_signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    ml_latest_signal: str | None = None
    aud_latest: float | None = None
    warnings: list[str] = field(default_factory=list)


def _clip_dates(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Restrict a date-indexed frame to [start, end]. Empty/missing-column safe."""
    if frame.empty or "date" not in frame.columns:
        return frame
    dates = pd.to_datetime(frame["date"])
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return frame[mask].reset_index(drop=True)


def _latest_non_empty(
    signal_agent: QuantSignalAgent, features: pd.DataFrame, signals: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    latest_signal = signal_agent.latest_signal(signals)
    if latest_signal.empty:
        return pd.Series(dtype=object), pd.Series(dtype=object)
    latest_date = pd.to_datetime(latest_signal["date"])
    feature = features[pd.to_datetime(features["date"]) == latest_date]
    latest_feature = feature.iloc[-1] if not feature.empty else features.dropna(how="all").iloc[-1]
    return latest_feature, latest_signal


def _empty_ml_result(message: str) -> dict[str, Any]:
    return {
        "predictions": pd.DataFrame(),
        "fold_metrics": pd.DataFrame(),
        "feature_importance": pd.DataFrame(),
        "training_samples": 0,
        "effective_n": 0,
        "manifest": [],
        "warning": message,
    }


def build_context(
    start: date,
    end: date,
    leverage: float,
    use_llm: bool,
    refresh: bool = False,
    db: Database | None = None,
    compute_features: Any = None,
    compute_ml: Any = None,
) -> ResearchContext:
    """Load data and compute the full research context. Streamlit-free.

    Returns a ResearchContext whose `status` tells the caller whether to render
    (`ok`) or show a guard message (`no_data` / `no_signal`). The download on
    `refresh` is the only network call; the caller owns the spinner around it.
    """
    db = db or Database()
    data_agent = DataAgent(db=db)
    feature_agent = FeatureAgent(db=db)
    signal_agent = QuantSignalAgent(db=db)
    diagnostics_agent = SignalDiagnosticsAgent()
    risk_agent = RiskAgent()
    backtest_agent = BacktestAgent()
    walk_forward_agent = WalkForwardAgent()
    factor_diagnostics_agent = FactorDiagnosticsAgent()
    paper_journal_agent = PaperJournalAgent(db=db)
    ml_agent = MLSignalAgent()

    agents = dict(
        data_agent=data_agent,
        feature_agent=feature_agent,
        signal_agent=signal_agent,
        diagnostics_agent=diagnostics_agent,
        risk_agent=risk_agent,
        backtest_agent=backtest_agent,
        walk_forward_agent=walk_forward_agent,
        factor_diagnostics_agent=factor_diagnostics_agent,
        paper_journal_agent=paper_journal_agent,
        ml_agent=ml_agent,
    )

    if refresh:
        data_agent.download_market_data(start, end)
        data_agent.download_macro_data(start, end)

    # The DB holds whatever has ever been downloaded; restrict the analysis to the
    # requested [start, end] window so the calibration/backtest window is actually
    # controlled by start/end (and not silently the entire DB).
    market_data = _clip_dates(data_agent.load_market_data(), start, end)
    if market_data.empty:
        return ResearchContext(status=NO_DATA, start=start, end=end, leverage=leverage, use_llm=use_llm, **agents)

    macro_data = _clip_dates(data_agent.load_macro_data(), start, end)
    if compute_features is not None:
        features = compute_features(market_data, macro_data)
    else:
        features = feature_agent.build_features(market_data, macro_data=macro_data)
    # Walk-forward adaptive factor signs: flip any factor whose assumed direction
    # has been anti-predictive in the trailing window (point-in-time, no look-ahead).
    factor_signs = signal_agent.adaptive_factor_signs(features, market_data, horizon=20)
    raw_signals = signal_agent.generate_signals(features, factor_signs=factor_signs)
    calibration = diagnostics_agent.calibration_frame(market_data, raw_signals, horizon=20, min_samples=20)
    signals = signal_agent.generate_signals(features, calibration=calibration, factor_signs=factor_signs)

    # Surface a degraded model loudly instead of silently running on fewer factors.
    data_warnings: list[str] = []
    if macro_data.empty:
        data_warnings.append("Macro data is empty — yield-spread and iron-ore factors are dead.")
    dead = [c for c in signal_agent.score_columns if c in raw_signals and raw_signals[c].notna().sum() == 0]
    if dead:
        live = len(signal_agent.score_columns) - len(dead)
        data_warnings.append(f"Dead factors (no data): {', '.join(dead)} — running on {live}/5 factors.")
    forward_diagnostics = diagnostics_agent.forward_return_diagnostics(market_data, raw_signals)
    latest_feature, latest_signal = _latest_non_empty(signal_agent, features, signals)

    if latest_signal.empty:
        return ResearchContext(
            status=NO_SIGNAL,
            start=start,
            end=end,
            leverage=leverage,
            use_llm=use_llm,
            market_data=market_data,
            macro_data=macro_data,
            features=features,
            warnings=data_warnings,
            **agents,
        )

    factor_table = feature_agent.factor_table(latest_feature, latest_signal)
    risk = risk_agent.suggest(
        signal=latest_signal["signal"],
        probability=latest_signal["probability"],
        volatility=latest_feature.get("audusd_vol_20d"),
        user_leverage=leverage,
        probability_source=latest_signal.get("probability_source", "fallback_score_map"),
    )

    # The LLM only runs here, on the latest signal — never inside the Backtest,
    # Walk-Forward, or ML loops. LLM agents fall back to the template agents on
    # any error or when no API key is set.
    if use_llm:
        explanation = LLMExplanationAgent(fallback=AIExplanationAgent(), db=db).explain(latest_signal, factor_table)
        contrarian = LLMContrarianAgent(fallback=ContrarianAgent(), db=db).critique(latest_signal, factor_table)
        judgement = LLMJudgeAgent(fallback=JudgeAgent(), db=db).judge(latest_signal, risk, explanation, contrarian)
    else:
        explanation = AIExplanationAgent().explain(latest_signal, factor_table)
        contrarian = ContrarianAgent().critique(latest_signal, factor_table)
        judgement = JudgeAgent().judge(latest_signal, risk, explanation, contrarian)

    aud_latest = float(
        market_data[market_data["symbol"] == DEFAULT_SYMBOLS.audusd].sort_values("date").iloc[-1]["close"]
    )
    paper_journal_agent.record_signal(
        latest_feature, latest_signal, factor_table, risk, judgement["explanation"], audusd_price=aud_latest
    )

    # ML research comparison — computed once, reused by the AI Report and ML tabs.
    if compute_ml is not None:
        ml_result = compute_ml(features)
    else:
        try:
            ml_result = ml_agent.walk_forward_predict(features)
        except Exception as exc:  # noqa: BLE001 - keep the app running if ML is unavailable
            ml_result = _empty_ml_result(f"ML unavailable: {exc}")
    ml_signals = ml_agent.to_signals(ml_result.get("predictions", pd.DataFrame()))
    ml_latest_signal = ml_signals.iloc[-1]["signal"] if not ml_signals.empty else None

    return ResearchContext(
        status=OK,
        start=start,
        end=end,
        leverage=leverage,
        use_llm=use_llm,
        market_data=market_data,
        macro_data=macro_data,
        features=features,
        raw_signals=raw_signals,
        signals=signals,
        calibration=calibration,
        forward_diagnostics=forward_diagnostics,
        latest_feature=latest_feature,
        latest_signal=latest_signal,
        factor_table=factor_table,
        risk=risk,
        explanation=explanation,
        contrarian=contrarian,
        judgement=judgement,
        ml_result=ml_result,
        ml_signals=ml_signals,
        ml_latest_signal=ml_latest_signal,
        aud_latest=aud_latest,
        warnings=data_warnings,
        **agents,
    )
