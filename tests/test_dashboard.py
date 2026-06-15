from __future__ import annotations

from alphafx.dashboard import context as ctxmod
from alphafx.dashboard.tabs import (
    ai_report,
    backtest,
    chart,
    diagnostics,
    factor,
    hero,
    journal,
    ml,
    validation,
)
from alphafx.database import Database
from factories import sample_market_data


def _seeded_db(tmp_path) -> Database:
    db = Database(tmp_path / "dash.db")
    md = sample_market_data()
    md["date"] = md["date"].dt.strftime("%Y-%m-%d")
    db.upsert_market_data(md)
    return db


def test_build_context_no_data(tmp_path):
    db = Database(tmp_path / "empty.db")
    ctx = ctxmod.build_context("2024-01-01", "2024-06-01", 2.0, use_llm=False, db=db)
    assert ctx.status == ctxmod.NO_DATA


def test_build_context_ok(tmp_path):
    db = _seeded_db(tmp_path)
    ctx = ctxmod.build_context("2024-01-01", "2024-12-31", 2.0, use_llm=False, db=db)
    assert ctx.status == ctxmod.OK
    assert not ctx.latest_signal.empty
    assert not ctx.factor_table.empty
    assert ctx.aud_latest is not None
    assert "predictions" in ctx.ml_result  # ML ran (or returned its empty shape)
    assert ctx.judgement.get("final_signal") == ctx.latest_signal["signal"]  # rule owns the signal


def test_all_tabs_render_without_error(tmp_path):
    # Renders outside a Streamlit runtime: st.* calls are no-ops/warnings, but
    # any missing ctx binding or bad reference raises here.
    db = _seeded_db(tmp_path)
    ctx = ctxmod.build_context("2024-01-01", "2024-12-31", 2.0, use_llm=False, db=db)
    assert ctx.status == ctxmod.OK
    for renderer in (hero, chart, factor, backtest, diagnostics, validation, journal, ai_report, ml):
        renderer.render(ctx)
