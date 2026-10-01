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


def _ask(message: str) -> None:
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages
    ]
    try:
        with st.spinner("Checking policy and synthetic HR records…"):
            result = submit_message(message, history, st.session_state.context)
    except (httpx.HTTPError, ValueError) as error:
        st.session_state.messages.append(
            {"role": "user", "content": message, "result": None}
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"The HR API could not complete the request: {error}",
                "result": None,
            }
        )
        return
    st.session_state.context = result.get("context")
    st.session_state.messages.append(
        {"role": "user", "content": message, "result": None}
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "result": result}
    )


st.set_page_config(page_title="RAGs to Riches", page_icon="💼", layout="wide")
st.title("RAGs to Riches")
st.caption("Synthetic HR policy guidance — not legal, employment, or benefits advice.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = None

with st.sidebar:
    st.header("Try a question")
    st.caption("These prompts ask the assistant. You can also type your own.")
    for label, prompt in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state.pending_prompt = prompt
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context = None
        st.session_state.pending_prompt = ""

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        if item["role"] == "assistant" and item.get("result"):
            _render_result(item["result"])
        else:
            st.markdown(item["content"])

pending = st.session_state.pop("pending_prompt", None)
prompt = st.chat_input("Ask about HR policy, remote work, or PTO")
message = pending or prompt
if message:
    _ask(message)
    st.rerun()
