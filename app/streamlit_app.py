"""Streamlit chat experience for the synthetic HR assistant."""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("HR_API_URL", "http://127.0.0.1:8000").rstrip("/")
EXAMPLES = {
    "Remote work — eligible": "I am SYN-1001. Can I work remotely from New York?",
    "Remote work — needs review": "I am SYN-1003. Can I work remotely from Texas?",
    "PTO — eligible": "I am SYN-1002. Can I take 8 hours of PTO?",
}


def submit_message(
    message: str,
    history: list[dict[str, str]],
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Send one chat turn and return the assistant payload."""

    response = httpx.post(
        f"{API_URL}/chat",
        json={"message": message, "history": history, "context": context},
        timeout=90.0,
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("The API returned an unexpected response")
    return result


def _render_result(result: dict[str, Any]) -> None:
    st.markdown(result["answer"])
    if result.get("citations"):
        st.subheader("Policy sources")
        for citation in result["citations"]:
            with st.container(border=True):
                st.markdown(
                    f"**[{citation['citation_id']}] {citation['source']} — "
                    f"{citation['section']}**"
                )
                st.caption(citation["snippet"])
    with st.expander("Operational trace"):
        st.caption("Tool activity only; no hidden chain-of-thought is displayed.")
        for step in result["trace"]:
            tool = f" · `{step['tool']}`" if step.get("tool") else ""
            st.markdown(f"**{step['state']}**{tool} — {step['status']}")
            st.caption(step["result_summary"])
            if step.get("safe_arguments"):
                st.json(step["safe_arguments"])


def _history_before_latest() -> list[dict[str, str]]:
    return [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages[:-1]
    ]


def _finish_pending_question() -> None:
    """Answer the question already shown in the transcript."""

    message = st.session_state.pending_message
    try:
        with st.spinner("Checking policy and synthetic HR records…"):
            result = submit_message(message, _history_before_latest(), st.session_state.context)
    except (httpx.HTTPError, ValueError) as error:
        reply = {
            "role": "assistant",
            "content": f"The HR API could not complete the request: {error}",
            "result": None,
        }
    else:
        st.session_state.context = result.get("context")
        reply = {"role": "assistant", "content": result["answer"], "result": result}
    st.session_state.pending_message = None
    st.session_state.messages.append(reply)
    st.rerun()


st.set_page_config(page_title="RAGs to Riches", page_icon="💼", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #1b8a3e !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("RAGs to Riches")
st.caption("Synthetic HR policy guidance — not legal, employment, or benefits advice.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = None
if "pending_message" not in st.session_state:
    st.session_state.pending_message = None

with st.sidebar:
    st.header("Try a question")
    st.caption("These prompts fill the chat box. Send the message when you are ready.")
    for label, prompt in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state.chat_input = prompt
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context = None
        st.session_state.pending_message = None

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        if item["role"] == "assistant" and item.get("result"):
            _render_result(item["result"])
        else:
            st.markdown(item["content"])

prompt = st.chat_input(
    "Ask about HR policy, remote work, or PTO",
    key="chat_input",
)
if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "result": None}
    )
    st.session_state.pending_message = prompt
    st.rerun()

if st.session_state.pending_message:
    _finish_pending_question()
