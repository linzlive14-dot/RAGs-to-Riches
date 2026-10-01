"""Discoverable FastMCP tools backed only by synthetic HR data."""

from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from rag.index import PolicyIndex
from rag.ingestion import chunk_documents, load_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = PROJECT_ROOT / "policies"
DATA_DIR = Path(os.environ.get("HR_MOCK_DATA_DIR", PROJECT_ROOT / "mock_data"))
INDEX_PATH = Path(
    os.environ.get("HR_POLICY_INDEX", PROJECT_ROOT / "data" / "policy_index.sqlite3")
)

mcp = FastMCP(
    "Synthetic HR Tools",
    instructions=(
        "Tools operate on fictional policies and synthetic records only. "
        "Creating a mock ticket requires explicit user confirmation."
    ),
)


def _load_collection(filename: str, key: str) -> list[dict[str, Any]]:
    payload = json.loads((DATA_DIR / filename).read_text(encoding="utf-8"))
    return payload[key]


def _find(filename: str, key: str, field: str, value: str) -> dict[str, Any] | None:
    for record in _load_collection(filename, key):
        if record.get(field) == value:
            return record
    return None


def _missing_record(field: str, value: str) -> dict[str, Any]:
    """Return a negative finding the agent can explain, instead of raising."""

    return {
        "found": False,
        "reason": f"No synthetic record found for {field}={value}.",
        "synthetic": True,
    }


def _index() -> PolicyIndex:
    index = PolicyIndex(INDEX_PATH)
    if not INDEX_PATH.exists():
        chunks = chunk_documents(load_documents(POLICY_DIR), chunk_size=180, overlap=30)
        index.build(
            chunks,
            configuration={"chunk_size": 180, "overlap": 30, "retrieval": "sqlite-fts5-bm25"},
        )
    return index


def _public_employee(employee: dict[str, Any]) -> dict[str, Any]:
    """Exclude contact details because workflows do not need them."""

    return {key: value for key, value in employee.items() if key != "email"}


@mcp.tool()
def list_work_locations() -> dict[str, Any]:
    """Return the synthetic location register, including remote-work support."""

    locations = [
        {
            "location_id": item["location_id"],
            "region": item["region"],
            "country": item["country"],
            "regular_remote_employment_supported": item["regular_remote_employment_supported"],
            "international": item["international"],
        }
        for item in _load_collection("locations.json", "locations")
    ]
    return {"found": True, "locations": locations, "synthetic": True}


@mcp.tool()
def search_policy_documents(query: str, top_k: int = 5) -> dict[str, Any]:
    """Search policy chunks and return ranked text with citation metadata."""

    results = _index().export_results(query, top_k=top_k)
    return {"query": query, "results": results, "synthetic": True}


@mcp.tool()
def get_policy_section(source: str, section: str) -> dict[str, Any]:
    """Return one exact section from a policy source file."""

    allowed = {path.name: path for path in POLICY_DIR.iterdir() if path.suffix in {".md", ".txt"}}
    if source not in allowed:
        return {
            "found": False,
            "source": source,
            "section": section,
            "reason": f"Unknown policy source {source}.",
            "synthetic": True,
        }
    text = allowed[source].read_text(encoding="utf-8")
    headings = list(re.finditer(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE))
    for position, heading in enumerate(headings):
        if heading.group(1).strip().casefold() != section.strip().casefold():
            continue
        end = headings[position + 1].start() if position + 1 < len(headings) else len(text)
        return {
            "found": True,
            "source": source,
            "section": heading.group(1).strip(),
            "text": text[heading.end() : end].strip(),
            "synthetic": True,
        }
    return {
        "found": False,
        "source": source,
        "section": section,
        "reason": f"Section {section!r} was not found in {source}.",
        "synthetic": True,
    }


@mcp.tool()
def lookup_employee_profile(employee_id: str) -> dict[str, Any]:
    """Look up the minimum synthetic employee profile needed for HR guidance."""

    employee = _find("employees.json", "employees", "employee_id", employee_id)
    if employee is None:
        return {**_missing_record("employee_id", employee_id), "employee_id": employee_id}
    return {"found": True, "employee": _public_employee(employee), "synthetic": True}


@mcp.tool()
def check_pto_balance(employee_id: str) -> dict[str, Any]:
    """Return a synthetic employee's current PTO balance."""

    balance = _find("pto_balances.json", "balances", "employee_id", employee_id)
    if balance is None:
        return {**_missing_record("employee_id", employee_id), "employee_id": employee_id}
    return {"found": True, "balance": balance, "synthetic": True}


@mcp.tool()
def lookup_benefits_status(employee_id: str) -> dict[str, Any]:
    """Return a synthetic employee's benefits enrollment status."""

    status = _find("benefits_status.json", "statuses", "employee_id", employee_id)
    if status is None:
        return {**_missing_record("employee_id", employee_id), "employee_id": employee_id}
    return {"found": True, "benefits_status": status, "synthetic": True}


@mcp.tool()
def check_policy_compliance(
    workflow: Literal["remote_work", "pto"],
    employee_id: str,
    requested_location_id: str | None = None,
    requested_hours: float | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic policy prerequisites; this is guidance, not approval."""

    employee = _find("employees.json", "employees", "employee_id", employee_id)
    checks: list[dict[str, Any]] = []
    if employee is None:
        missing = _missing_record("employee_id", employee_id)
        return {
            "workflow": workflow,
            "decision": "needs_clarification",
            "checks": [],
            "found": False,
            "reason": missing["reason"],
            "synthetic": True,
        }
    if workflow == "remote_work":
        if not requested_location_id:
            return {
                "workflow": workflow,
                "decision": "needs_clarification",
                "checks": [],
                "found": True,
                "reason": "requested_location_id is required",
                "synthetic": True,
            }
        location = _find(
            "locations.json", "locations", "location_id", requested_location_id
        )
        if location is None:
            missing = _missing_record("location_id", requested_location_id)
            return {
                "workflow": workflow,
                "decision": "needs_clarification",
                "checks": [],
                "found": False,
                "reason": missing["reason"],
                "synthetic": True,
            }
        tenure_days = (date.today() - date.fromisoformat(employee["hire_date"])).days
        checks = [
            {"name": "180_day_tenure", "passed": tenure_days >= 180},
            {
                "name": "remote_capable_role",
                "passed": employee["remote_capability"] == "fully_remote",
            },
            {
                "name": "performance",
                "passed": employee["performance_status"] in {"meets_expectations", "exceeds_expectations"},
            },
            {"name": "no_final_warning", "passed": not employee["active_final_warning"]},
            {
                "name": "supported_location",
                "passed": location["regular_remote_employment_supported"],
            },
            {"name": "domestic_location", "passed": not location["international"]},
        ]
    else:
        if requested_hours is None or requested_hours <= 0:
            return {
                "workflow": workflow,
                "decision": "needs_clarification",
                "checks": [],
                "found": True,
                "reason": "requested_hours must be greater than zero",
                "synthetic": True,
            }
        balance = _find("pto_balances.json", "balances", "employee_id", employee_id)
        if balance is None:
            missing = _missing_record("employee_id", employee_id)
            return {
                "workflow": workflow,
                "decision": "needs_clarification",
                "checks": [],
                "found": False,
                "reason": missing["reason"],
                "synthetic": True,
            }
        covered = employee["employment_type"] in {"regular_full_time", "regular_part_time"}
        enough_balance = balance["available_hours"] >= requested_hours
        first_year = (date.today() - date.fromisoformat(employee["hire_date"])).days < 365
        checks = [
            {"name": "covered_employee", "passed": covered},
            {
                "name": "balance_or_possible_first_year_exception",
                "passed": enough_balance or first_year,
                "detail": "negative PTO still requires manager and HR approval"
                if not enough_balance and first_year
                else None,
            },
        ]
    decision = "eligible_for_review" if all(check["passed"] for check in checks) else "escalate"
    failed = [check["name"] for check in checks if not check["passed"]]
    return {
        "workflow": workflow,
        "decision": decision,
        "checks": checks,
        "found": True,
        "reason": (
            "Eligibility checks do not constitute manager or HR approval."
            if decision == "eligible_for_review"
            else f"These checks did not pass: {', '.join(failed)}."
        ),
        "disclaimer": "Eligibility checks do not constitute manager or HR approval.",
        "synthetic": True,
    }


@mcp.tool()
def create_mock_hr_ticket(
    employee_id: str,
    category: Literal["remote_work", "pto"],
    summary: str,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Create a mock-only HR ticket after explicit user confirmation."""

    if _find("employees.json", "employees", "employee_id", employee_id) is None:
        missing = _missing_record("employee_id", employee_id)
        return {"created": False, "found": False, "reason": missing["reason"], "synthetic": True}
    clean_summary = " ".join(summary.split())
    if not confirmed:
        return {
            "created": False,
            "found": True,
            "confirmation_required": True,
            "reason": "Mock ticket creation needs an explicit confirmation.",
            "proposed_action": {
                "employee_id": employee_id,
                "category": category,
                "summary": clean_summary,
            },
            "synthetic": True,
        }
    if not 10 <= len(clean_summary) <= 200:
        return {
            "created": False,
            "found": True,
            "rejected": True,
            "reason": "summary must contain 10 to 200 characters",
            "synthetic": True,
        }
    tickets_path = DATA_DIR / "tickets.json"
    payload = json.loads(tickets_path.read_text(encoding="utf-8"))
    ticket_id = f"MOCK-HR-{len(payload['tickets']) + 1:03d}"
    ticket = {
        "ticket_id": ticket_id,
        "employee_id": employee_id,
        "category": category,
        "status": "open",
        "summary": clean_summary,
        "created_at": f"{date.today().isoformat()}T00:00:00Z",
        "mock": True,
    }
    payload["tickets"].append(ticket)
    tickets_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"created": True, "found": True, "ticket": ticket, "synthetic": True}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
