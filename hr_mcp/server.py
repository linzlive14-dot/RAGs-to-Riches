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


def _find(filename: str, key: str, field: str, value: str) -> dict[str, Any]:
    for record in _load_collection(filename, key):
        if record.get(field) == value:
            return record
    raise ValueError(f"No synthetic record found for {field}={value!r}")


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
def search_policy_documents(query: str, top_k: int = 5) -> dict[str, Any]:
    """Search policy chunks and return ranked text with citation metadata."""

    results = _index().export_results(query, top_k=top_k)
    return {"query": query, "results": results, "synthetic": True}


@mcp.tool()
def get_policy_section(source: str, section: str) -> dict[str, Any]:
    """Return one exact section from a policy source file."""

    allowed = {path.name: path for path in POLICY_DIR.iterdir() if path.suffix in {".md", ".txt"}}
    if source not in allowed:
        raise ValueError("Unknown policy source")
    text = allowed[source].read_text(encoding="utf-8")
    headings = list(re.finditer(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE))
    for position, heading in enumerate(headings):
        if heading.group(1).strip().casefold() != section.strip().casefold():
            continue
        end = headings[position + 1].start() if position + 1 < len(headings) else len(text)
        return {
            "source": source,
            "section": heading.group(1).strip(),
            "text": text[heading.end() : end].strip(),
            "synthetic": True,
        }
    raise ValueError(f"Section {section!r} was not found in {source!r}")


@mcp.tool()
def lookup_employee_profile(employee_id: str) -> dict[str, Any]:
    """Look up the minimum synthetic employee profile needed for HR guidance."""

    employee = _find("employees.json", "employees", "employee_id", employee_id)
    return {"employee": _public_employee(employee), "synthetic": True}


@mcp.tool()
def check_pto_balance(employee_id: str) -> dict[str, Any]:
    """Return a synthetic employee's current PTO balance."""

    balance = _find("pto_balances.json", "balances", "employee_id", employee_id)
    return {"balance": balance, "synthetic": True}


@mcp.tool()
def lookup_benefits_status(employee_id: str) -> dict[str, Any]:
    """Return a synthetic employee's benefits enrollment status."""

    status = _find("benefits_status.json", "statuses", "employee_id", employee_id)
    return {"benefits_status": status, "synthetic": True}


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
    if workflow == "remote_work":
        if not requested_location_id:
            return {
                "workflow": workflow,
                "decision": "needs_clarification",
                "checks": [],
                "reason": "requested_location_id is required",
                "synthetic": True,
            }
        location = _find(
            "locations.json", "locations", "location_id", requested_location_id
        )
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
                "reason": "requested_hours must be greater than zero",
                "synthetic": True,
            }
        balance = _find("pto_balances.json", "balances", "employee_id", employee_id)
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
    return {
        "workflow": workflow,
        "decision": decision,
        "checks": checks,
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

    _find("employees.json", "employees", "employee_id", employee_id)
    clean_summary = " ".join(summary.split())
    if not confirmed:
        return {
            "created": False,
            "confirmation_required": True,
            "proposed_action": {
                "employee_id": employee_id,
                "category": category,
                "summary": clean_summary,
            },
            "synthetic": True,
        }
    if not 10 <= len(clean_summary) <= 200:
        raise ValueError("summary must contain 10 to 200 characters")
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
    return {"created": True, "ticket": ticket, "synthetic": True}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
