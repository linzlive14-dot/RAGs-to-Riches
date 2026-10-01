from fastapi.testclient import TestClient

from app.main import app

EXPECTED_TOOLS = [
    "check_policy_compliance",
    "check_pto_balance",
    "create_mock_hr_ticket",
    "get_policy_section",
    "list_work_locations",
    "lookup_benefits_status",
    "lookup_employee_profile",
    "search_policy_documents",
]


def test_health_check_reports_mcp_connectivity_with_shared_session() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "RAGs to Riches"
    assert payload["mcp"]["status"] == "ok"
    assert payload["mcp"]["transport"] == "stdio"
    assert payload["mcp"]["tools"] == EXPECTED_TOOLS
    assert payload["index"]["status"] == "ok"
    assert payload["index"]["chunk_count"] > 0
    assert payload["index"]["retrieval_mode"] in {"hybrid", "bm25", "vector"}
    if payload["index"]["retrieval_mode"] != "bm25":
        assert payload["index"]["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert set(payload["llm"]) == {"configured", "model"}


def test_health_check_without_lifespan_starts_a_temporary_mcp_session() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["mcp"]["tool_count"] == len(EXPECTED_TOOLS)


def test_chat_reuses_the_shared_mcp_session() -> None:
    with TestClient(app) as client:
        first = client.post("/chat", json={"message": "I am SYN-1002. Can I take 8 hours of PTO?"})
        second = client.post("/chat", json={"message": "I am SYN-1002. Can I take 8 hours of PTO?"})

    assert first.status_code == second.status_code == 200
    discover_steps = [step for step in second.json()["trace"] if step["state"] == "discover"]
    assert discover_steps and discover_steps[0]["output_preview"]["tools"] == EXPECTED_TOOLS
