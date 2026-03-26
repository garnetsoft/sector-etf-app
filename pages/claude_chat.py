import os

import anthropic
import streamlit as st
from dotenv import load_dotenv

from utils import MODELS, estimate_cost

load_dotenv()

st.set_page_config(page_title="Claude Chat", layout="wide")

SYSTEM_PRESETS = {
    "General Assistant":   "You are a helpful assistant.",
    "Equity Analyst":      "You are a senior equity research analyst at a top investment bank. Provide rigorous, data-driven analysis with clear investment theses, risk factors, and valuation frameworks. Be specific and quantitative where possible.",
    "Macro Strategist":    "You are a macro strategist at a global asset manager. Analyze economic conditions, central bank policy, geopolitical risk, and cross-asset implications. Connect macro drivers to specific investment opportunities and risks.",
    "Risk Manager":        "You are a portfolio risk manager. Focus on downside scenarios, tail risks, correlation breakdowns, drawdown analysis, position sizing, and hedging strategies. Be conservative and thorough in identifying vulnerabilities.",
    "Sector Specialist":   "You are a sector specialist covering U.S. equities. Analyze industry dynamics, competitive positioning, regulatory environment, and valuation relative to history and peers. Provide actionable, sector-specific insights.",
    "Earnings Analyst":    "You are a corporate earnings analyst. Dissect financial statements, revenue trends, margin dynamics, capital allocation, and earnings quality. Flag accounting red flags and identify the key drivers of future earnings power.",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    # Silent API key: show input only if not in env
    _env_key = os.getenv("ANTHROPIC_API_KEY", "")
    if _env_key:
        api_key = _env_key
    else:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Set ANTHROPIC_API_KEY in .env or enter here",
        )

    model_label = st.selectbox("Model", list(MODELS.keys()))
    model       = MODELS[model_label]

    st.divider()
    preset_label  = st.selectbox("System Prompt Preset", list(SYSTEM_PRESETS.keys()))
    system_prompt = st.text_area(
        "System Prompt",
        value=SYSTEM_PRESETS[preset_label],
        height=120,
    )

    max_tokens  = st.slider("Max Tokens",  min_value=256, max_value=8096, value=1024, step=256)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0,  value=1.0,  step=0.1)

    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("usage", None)
        st.rerun()

    if "usage" in st.session_state:
        st.divider()
        u        = st.session_state.usage
        cost_str = estimate_cost(model, u.input_tokens, u.output_tokens)
        st.caption("**Last response**")
        st.caption(f"Input:  {u.input_tokens:,} tokens")
        st.caption(f"Output: {u.output_tokens:,} tokens")
        st.caption(f"Est. cost: {cost_str}")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Claude Chat")

if not api_key:
    st.warning("Enter your Anthropic API key in the sidebar to get started.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Message Claude..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder   = st.empty()
        full_response = ""

        try:
            client = anthropic.Anthropic(api_key=api_key)

            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=st.session_state.messages,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)
            st.session_state.usage = stream.get_final_message().usage

        except anthropic.AuthenticationError:
            st.error("Invalid API key. Please check your key in the sidebar.")
            st.session_state.messages.pop()
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.messages.pop()
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": full_response})
