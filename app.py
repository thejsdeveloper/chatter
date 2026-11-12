"""Chatter: upload CSV files and ask questions answered only from their data.

Run with:  uv run streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import DEFAULT_PROVIDER, PROVIDERS, ask, build_agent

load_dotenv()

EXAMPLE_QUESTIONS = [
    "What is the return policy for electronics?",
    "What does the extended warranty cover?",
    "What are the visiting hours in the hospital?",
    "What is the API rate limit for the free plan?",
]

st.set_page_config(page_title="Chatter", page_icon="💬", layout="wide")
st.title("💬 Chatter")
st.caption("Upload your FAQ or policy CSV files, then ask a question. Answers come only from your files.")

# ---------- Sidebar: settings ----------
with st.sidebar:
    st.header("Settings")
    names = list(PROVIDERS)
    provider = st.selectbox("Provider", names, index=names.index(DEFAULT_PROVIDER))
    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["env_key"], "")
    if api_key:
        st.success(f"{provider} API key loaded from environment.")
    else:
        api_key = st.text_input(
            f"{provider} API key", type="password", help=f"Or set {cfg['env_key']} in a .env file."
        )
    model = st.text_input("Model", value=os.getenv("CHATTER_MODEL", cfg["model"]), help="Temperature is fixed at 0.")
    show_steps = st.toggle("Show how the answer was found", value=False)
    if st.button("Clear chat"):
        st.session_state.messages = []

# ---------- Step 1: upload ----------
uploaded_files = st.file_uploader("1. Upload CSV files", type="csv", accept_multiple_files=True)

dataframes: dict[str, pd.DataFrame] = {}
for f in uploaded_files or []:
    try:
        dataframes[f.name] = pd.read_csv(f)
    except Exception as e:  # bad encoding, empty file, etc.
        st.error(f"Could not read {f.name}: {e}")

if dataframes:
    cols = st.columns(min(len(dataframes), 4))
    for i, (name, df) in enumerate(dataframes.items()):
        with cols[i % len(cols)].expander(f"📄 {name} · {len(df)} rows"):
            st.dataframe(df.head(), hide_index=True)

# ---------- Build (or reuse) the agent ----------
st.session_state.setdefault("messages", [])
files_key = (provider, model, api_key, tuple(sorted((name, len(df)) for name, df in dataframes.items())))

ready = bool(dataframes and api_key and model)
if ready and st.session_state.get("files_key") != files_key:
    try:
        st.session_state.agent = build_agent(dataframes, api_key, provider=provider, model=model)
        st.session_state.files_key = files_key
        st.session_state.messages = []  # new data, fresh conversation
    except Exception as e:
        st.error(f"Could not start the agent: {e}")
        ready = False

if not dataframes:
    st.info("Start by uploading one or more CSV files above.")
elif not api_key:
    st.warning(f"Add your {provider} API key in the sidebar to start asking questions.")

# ---------- Step 2: ask ----------
st.subheader("2. Ask a question")

question = None
if ready and not st.session_state.messages:
    st.write("Try one of these:")
    ex_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, q in zip(ex_cols, EXAMPLE_QUESTIONS):
        if col.button(q, use_container_width=True):
            question = q

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_steps and msg.get("steps"):
            with st.expander("How the answer was found"):
                for action, observation in msg["steps"]:
                    st.code(str(action.tool_input), language="python")
                    st.text(str(observation)[:2000])

typed = st.chat_input("Ask about your uploaded files…", disabled=not ready)
question = typed or question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your files…"):
            try:
                answer, steps = ask(st.session_state.agent, question)
            except Exception as e:
                answer, steps = f"Sorry, something went wrong: {e}", []
        st.markdown(answer)
        if show_steps and steps:
            with st.expander("How the answer was found"):
                for action, observation in steps:
                    st.code(str(action.tool_input), language="python")
                    st.text(str(observation)[:2000])

    st.session_state.messages.append({"role": "assistant", "content": answer, "steps": steps})
