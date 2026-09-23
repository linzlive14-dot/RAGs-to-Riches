from fastapi.testclient import TestClient

from app.main import app


def test_chat_runs_cited_workflow_over_mcp() -> None:
    response = TestClient(app).post(
        "/chat",
        json={
            "workflow": "remote_work",
            "employee_id": "SYN-1001",
            "requested_location_id": "US-NY",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["citations"]
    assert "[P1]" in payload["answer"]
    assert [step["tool"] for step in payload["trace"] if step["tool"]] == [
        "lookup_employee_profile",
        "search_policy_documents",
        "check_policy_compliance",
    ]


def test_chat_returns_clarification_without_calling_tools() -> None:
    response = TestClient(app).post(
        "/chat",
        json={"workflow": "pto", "employee_id": "SYN-1002"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "needs_clarification"
    assert all(step["tool"] is None for step in payload["trace"])


def test_chat_rejects_invalid_employee_identifier() -> None:
    response = TestClient(app).post(
        "/chat",
        json={
            "workflow": "pto",
            "employee_id": "real-employee",
            "requested_hours": 8,
        },
    )

    assert response.status_code == 422
