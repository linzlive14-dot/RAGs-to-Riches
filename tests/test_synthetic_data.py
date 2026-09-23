import json
from pathlib import Path


DATA_DIR = Path(__file__).parents[1] / "mock_data"


def test_all_mock_data_is_valid_and_explicitly_synthetic() -> None:
    files = sorted(DATA_DIR.glob("*.json"))

    assert {path.name for path in files} == {
        "benefits_status.json",
        "employees.json",
        "locations.json",
        "pto_balances.json",
        "tickets.json",
    }
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "SYNTHETIC DATA ONLY" in payload["_notice"]


def test_employee_references_are_consistent() -> None:
    employees = json.loads((DATA_DIR / "employees.json").read_text())["employees"]
    employee_ids = {employee["employee_id"] for employee in employees}
    for filename, key in (
        ("benefits_status.json", "statuses"),
        ("pto_balances.json", "balances"),
        ("tickets.json", "tickets"),
    ):
        records = json.loads((DATA_DIR / filename).read_text())[key]
        assert all(record["employee_id"] in employee_ids for record in records)

    assert all(employee["email"].endswith(".invalid") for employee in employees)
