import streamlit as st
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Claude Chat", layout="wide")

MODELS = {
    "Claude Sonnet 4.6 (Recommended)": "claude-sonnet-4-6",
    "Claude Opus 4.6 (Most Capable)":  "claude-opus-4-6",
    "Claude Haiku 4.5 (Fastest)":      "claude-haiku-4-5-20251001",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Anthropic API Key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Get your key at console.anthropic.com",
    )

    model_label = st.selectbox("Model", list(MODELS.keys()))
    model       = MODELS[model_label]

    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant.",
        height=120,
    )

    max_tokens   = st.slider("Max Tokens",   min_value=256, max_value=8096, value=1024, step=256)
    temperature  = st.slider("Temperature",  min_value=0.0, max_value=1.0,  value=1.0,  step=0.1)

    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Token usage display
    if "usage" in st.session_state:
        st.divider()
        st.caption("**Last response usage**")
        u = st.session_state.usage
        st.caption(f"Input tokens:  {u.input_tokens:,}")
        st.caption(f"Output tokens: {u.output_tokens:,}")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Claude Chat")

if not api_key:
    st.warning("Enter your Anthropic API key in the sidebar to get started.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Message Claude..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
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
