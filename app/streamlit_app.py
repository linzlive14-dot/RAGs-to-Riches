"""Streamlit chat experience for the synthetic HR workflow API."""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("HR_API_URL", "http://127.0.0.1:8000").rstrip("/")
PRESETS = {
    "Remote work — eligible": {
        "workflow": "remote_work",
        "employee_id": "SYN-1001",
        "requested_location_id": "US-NY",
    },
    "Remote work — needs review": {
        "workflow": "remote_work",
        "employee_id": "SYN-1003",
        "requested_location_id": "US-TX",
    },
    "PTO — eligible": {
        "workflow": "pto",
        "employee_id": "SYN-1002",
        "requested_hours": 8.0,
    },
}


def submit_workflow(payload: dict[str, Any]) -> dict[str, Any]:
    """Submit a workflow and return the API's validated JSON response."""

    response = httpx.post(f"{API_URL}/chat", json=payload, timeout=60.0)
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


st.set_page_config(page_title="RAGs to Riches", page_icon="💼", layout="wide")
st.title("RAGs to Riches")
st.caption("Synthetic HR policy guidance — not legal, employment, or benefits advice.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Demo preset")
    preset_name = st.selectbox("Scenario", list(PRESETS))
    if st.button("Load preset", use_container_width=True):
        st.session_state.form_values = PRESETS[preset_name]

values = st.session_state.get("form_values", PRESETS["Remote work — eligible"])
with st.form("workflow"):
    workflow = st.selectbox(
        "Workflow",
        ["remote_work", "pto"],
        index=0 if values["workflow"] == "remote_work" else 1,
        format_func=lambda item: item.replace("_", " ").title(),
    )
    employee_id = st.text_input("Synthetic employee ID", values["employee_id"])
    requested_location_id = None
    requested_hours = None
    if workflow == "remote_work":
        requested_location_id = st.text_input(
            "Requested location ID", values.get("requested_location_id", "")
        )
    else:
        requested_hours = st.number_input(
            "Requested PTO hours",
            min_value=0.5,
            value=float(values.get("requested_hours", 8.0)),
            step=0.5,
        )
    create_ticket = st.checkbox("Propose a mock HR ticket")
    confirmed = st.checkbox(
        "I explicitly confirm creation of the mock ticket",
        disabled=not create_ticket,
    )
    submitted = st.form_submit_button("Ask HR assistant", type="primary")

if submitted:
    payload: dict[str, Any] = {
        "workflow": workflow,
        "employee_id": employee_id,
        "create_ticket": create_ticket,
        "confirmed": confirmed,
    }
    if workflow == "remote_work":
        payload["requested_location_id"] = requested_location_id or None
    else:
        payload["requested_hours"] = requested_hours
    try:
        with st.spinner("Checking policy and synthetic HR records…"):
            result = submit_workflow(payload)
        st.session_state.history.insert(0, result)
    except (httpx.HTTPError, ValueError) as error:
        st.error(f"The HR API could not complete the request: {error}")

for item in st.session_state.history:
    _render_result(item)
