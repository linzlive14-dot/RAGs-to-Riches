"""Interpret a chat message into a workflow and slots.

Explicit IDs, locations, and hour amounts in the message win over model guesses.
A configured LLM may still choose the workflow when the wording is open-ended.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.guardrails import INJECTION_PATTERNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IntentLabel = Literal["policy_qa", "remote_work", "pto", "out_of_scope", "unsafe"]
EMPLOYEE_RE = re.compile(r"\bSYN-\d{4}\b", re.IGNORECASE)
HOURS_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*hours?\b", re.IGNORECASE)
REMOTE_RE = re.compile(r"\b(?:remote|remotely)\b", re.IGNORECASE)
PTO_RE = re.compile(r"\b(?:pto|paid time off|vacation)\b", re.IGNORECASE)
TICKET_RE = re.compile(r"\bticket\b", re.IGNORECASE)
QUESTION_RE = re.compile(
    r"^\s*(?:what|how|when|where|why|who|explain|describe)\b",
    re.IGNORECASE,
)
AFFIRM_RE = re.compile(
    r"\b(?:yes|yeah|yep|confirm|confirmed|go ahead)\b",
    re.IGNORECASE,
)


class ParsedIntent(BaseModel):
    intent: IntentLabel
    employee_id: str | None = None
    requested_location_id: str | None = None
    requested_hours: float | None = Field(default=None, gt=0)
    create_ticket: bool = False
    confirmed: bool = False


def _locations() -> list[dict[str, Any]]:
    payload = json.loads(
        (PROJECT_ROOT / "mock_data" / "locations.json").read_text(encoding="utf-8")
    )
    return payload["locations"]


def location_catalog_text() -> str:
    return "\n".join(
        f"{item['location_id']}: {item['region']}, {item['country']}"
        for item in _locations()
    )


def _location_id(message: str) -> str | None:
    folded = message.casefold()
    for item in _locations():
        location_id = str(item["location_id"])
        region = str(item["region"])
        if re.search(rf"\b{re.escape(location_id)}\b", message, flags=re.IGNORECASE):
            return location_id
        if re.search(rf"\b{re.escape(region)}\b", folded, flags=re.IGNORECASE):
            return location_id
    return None


def _valid_employee_id(value: str | None) -> str | None:
    if value and EMPLOYEE_RE.fullmatch(value.strip()):
        return value.strip().upper()
    return None


def _valid_location_id(value: str | None) -> str | None:
    if not value:
        return None
    known = {str(item["location_id"]) for item in _locations()}
    normalized = value.strip().upper()
    return normalized if normalized in known else None


def is_unsafe_message(message: str) -> bool:
    return any(pattern.search(message) for pattern in INJECTION_PATTERNS)


def deterministic_intent(message: str, context: dict[str, Any] | None = None) -> ParsedIntent:
    """Read workflow slots that are explicit in the message or saved context."""

    context = context or {}
    if is_unsafe_message(message):
        return ParsedIntent(intent="unsafe")

    if context.get("awaiting_confirmation") and AFFIRM_RE.search(message):
        workflow = context.get("workflow")
        intent: IntentLabel = workflow if workflow in {"remote_work", "pto"} else "policy_qa"
        return ParsedIntent(
            intent=intent,
            employee_id=_valid_employee_id(context.get("employee_id")),
            requested_location_id=_valid_location_id(context.get("requested_location_id")),
            requested_hours=context.get("requested_hours"),
            create_ticket=True,
            confirmed=True,
        )

    employee_match = EMPLOYEE_RE.search(message)
    employee_id = employee_match.group(0).upper() if employee_match else None
    location_id = _location_id(message)
    hours_match = HOURS_RE.search(message)
    hours = float(hours_match.group(1)) if hours_match else None
    create_ticket = bool(TICKET_RE.search(message))
    remote = REMOTE_RE.search(message) is not None
    pto = PTO_RE.search(message) is not None
    inherited = context.get("workflow") if context.get("workflow") in {"remote_work", "pto"} else None
    fills_saved_workflow = (
        inherited is not None
        and QUESTION_RE.search(message) is None
        and (hours_match is not None or location_id is not None or employee_match is not None)
        and not (remote or pto)
    )

    intent: IntentLabel
    if employee_id and remote:
        intent = "remote_work"
    elif employee_id and pto:
        intent = "pto"
    elif remote and location_id and not pto:
        intent = "remote_work"
    elif pto and hours is not None:
        intent = "pto"
    elif fills_saved_workflow:
        intent = inherited
        employee_id = employee_id or _valid_employee_id(context.get("employee_id"))
        location_id = location_id or _valid_location_id(context.get("requested_location_id"))
        if hours is None:
            hours = context.get("requested_hours")
        create_ticket = create_ticket or bool(context.get("create_ticket"))
    else:
        intent = "policy_qa"

    return ParsedIntent(
        intent=intent,
        employee_id=employee_id,
        requested_location_id=location_id,
        requested_hours=hours,
        create_ticket=create_ticket,
    )


def resolve_intent(
    message: str,
    context: dict[str, Any] | None = None,
    model: ParsedIntent | None = None,
) -> ParsedIntent:
    """Combine a model label with slots that appear explicitly in the message."""

    detected = deterministic_intent(message, context)
    if model is None or detected.intent == "unsafe" or detected.confirmed:
        return detected

    employee_id = detected.employee_id or _valid_employee_id(model.employee_id)
    if detected.intent in {"remote_work", "pto"}:
        intent: IntentLabel = detected.intent
    elif model.intent in {"remote_work", "pto"} and employee_id:
        intent = model.intent
    else:
        intent = "policy_qa"

    location_id = _location_id(message) or _valid_location_id(model.requested_location_id)
    if location_id is None:
        location_id = detected.requested_location_id
    hours_match = HOURS_RE.search(message)
    if hours_match:
        hours = float(hours_match.group(1))
    elif model.requested_hours is not None:
        hours = model.requested_hours
    else:
        hours = detected.requested_hours
    return ParsedIntent(
        intent=intent,
        employee_id=employee_id,
        requested_location_id=location_id,
        requested_hours=hours,
        create_ticket=detected.create_ticket or model.create_ticket,
    )
