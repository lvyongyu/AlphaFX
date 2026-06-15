from __future__ import annotations

import streamlit as st

from alphafx.database import Database
from alphafx.ml import ml_rule_agreement

from alphafx.dashboard.context import ResearchContext


def render(ctx: ResearchContext) -> None:
    leverage = ctx.leverage
    use_llm = ctx.use_llm
    latest_signal = ctx.latest_signal
    factor_table = ctx.factor_table
    risk = ctx.risk
    contrarian = ctx.contrarian
    ml_latest_signal = ctx.ml_latest_signal
    st.subheader("AI Report")
    report_mode = "LLM" if use_llm else "Template"
    st.caption(
        f"Mode: {report_mode}. The quant model sets the signal, score, and probability. "
        "The LLM only explains, challenges, and summarizes — it never sets the signal."
    )
    st.write(f"Probability source: `{latest_signal.get('probability_source', 'fallback_score_map')}`")
    bullish = factor_table[factor_table["stance"] == "bullish"]["factor"].tolist()
    bearish = factor_table[factor_table["stance"] == "bearish"]["factor"].tolist()

    st.markdown(
        f"""
### 1. Current Signal
{latest_signal["signal"].upper()} AUD/USD, probability {latest_signal["probability"]:.0%}, confidence {latest_signal["confidence"]}.

### 2. Key Bullish Drivers
{", ".join(bullish) if bullish else "No bullish drivers are currently active."}

### 3. Key Bearish Drivers
{", ".join(bearish) if bearish else "No bearish drivers are currently active."}

### 4. Contrarian View
{contrarian["main_risk"]}

{contrarian["alternative_scenario"]}

### 5. Risk Management
Suggested action: **{risk.action}**. Leverage: **{risk.leverage:.1f}x**. Stop loss: **{risk.stop_loss:.0%}**. Take profit: **{risk.take_profit:.0%}**.

### 6. What To Watch
{contrarian["watch"]}

### 7. Rule vs ML
{ml_rule_agreement(latest_signal["signal"], ml_latest_signal)}
"""
    )

    if use_llm:
        with st.expander("LLM call audit log"):
            calls = Database().load_llm_calls(limit=20)
            if calls.empty:
                st.info("No LLM calls logged yet — template fallback may be active (no API key or a call failed).")
            else:
                st.dataframe(
                    calls[
                        ["created_at", "date", "role", "model", "input_tokens", "output_tokens", "cache_read_tokens", "prompt_hash"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
