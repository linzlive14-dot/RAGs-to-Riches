import asyncio
from pathlib import Path

import pytest

from app.config import Settings
from evaluation.runner import answer_match, load_tasks, run_ablation, run_evaluation


def test_gold_set_has_required_size_categories_and_unique_ids() -> None:
    tasks = load_tasks()

    assert len(tasks) == 30
    assert {task["kind"] for task in tasks} == {"workflow", "retrieval", "unsafe"}
    assert {task["category"] for task in tasks} == {
        "workflow",
        "policy",
        "multi_document",
        "ambiguous",
        "out_of_scope",
        "unsafe",
    }
    assert len({task["id"] for task in tasks}) == len(tasks)
    assert all(task.get("message") and task.get("gold_answer") for task in tasks)
    assert all(task.get("key_facts") for task in tasks)
    assert any(task["expected"].get("sources_all") for task in tasks)
    statuses = {task["expected"].get("status") for task in tasks}
    assert {"needs_clarification", "confirmation_required", "escalated"} <= statuses


def test_answer_match_gives_partial_credit_for_any_phrasing() -> None:
    task = {"key_facts": [["10 calendar days", "ten calendar days"], ["manager"]]}

    assert answer_match(task, "Submit at least ten calendar days ahead.") == 0.5
    assert answer_match(task, "Ask your Manager; 10 calendar days notice.") == 1.0
    assert answer_match({"key_facts": []}, "anything") is None


def test_evaluation_reports_all_metrics_and_latency(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    report = asyncio.run(run_evaluation(output_path=output))

    assert report["task_count"] == 30
    assert report["answer_source"] == "deterministic"
    assert report["passed"] >= 27
    assert {
        "groundedness",
        "citations",
        "tool_selection",
        "workflow",
        "clarification",
        "safety",
        "answer_match",
        "answer_full_match",
    } == set(report["metrics"])
    for metric in ("tool_selection", "workflow", "clarification", "safety"):
        assert report["metrics"][metric] == 1.0
    assert report["metrics"]["answer_match"] >= 0.85
    assert report["categories"]["multi_document"]["passed"] == 5
    assert report["categories"]["ambiguous"]["passed"] == 3
    assert report["latency"]["warm"]["samples"] == 30
    assert output.exists()


def test_graded_evaluation_requires_an_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "evaluation.runner.get_settings",
        lambda: Settings(_env_file=None),
    )

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        asyncio.run(run_evaluation(mode="llm"))


def test_blank_api_key_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "  ")

    assert Settings(_env_file=None).openrouter_api_key is None


def test_ablation_compares_retrievers_and_tool_availability(tmp_path: Path) -> None:
    output = tmp_path / "ablation.json"
    report = run_ablation(
        output_path=output,
        configurations=(
            {"name": "hybrid", "retrieval_mode": "hybrid"},
            {"name": "bm25", "retrieval_mode": "bm25"},
            {"name": "no section tool", "disabled_tools": ["get_policy_section"]},
        ),
    )

    rows = {row["name"]: row for row in report["configurations"]}
    assert rows["hybrid"]["passed"] >= rows["bm25"]["passed"]
    assert rows["no section tool"]["disabled_tools"] == ["get_policy_section"]
    assert rows["no section tool"]["passed"] < rows["hybrid"]["passed"]
    assert rows["no section tool"]["metrics"]["safety"] == 1.0
    assert report["recommended"] in {"hybrid", "bm25"}
    assert output.exists()
