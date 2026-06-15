"""Streamlit dashboard for AlphaFX.

The heavy compute lives in `context.build_context` (streamlit-free and
testable); each tab in `tabs/` only renders a `ResearchContext`. `app.py` at the
repo root is a thin orchestrator: sidebar -> build_context -> dispatch tabs.
"""
