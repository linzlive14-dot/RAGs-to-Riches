import asyncio
from pathlib import Path

from evaluation.runner import load_tasks, run_ablation, run_evaluation


def test_gold_set_has_required_size_categories_and_unique_ids() -> None:
    tasks = load_tasks()

    assert len(tasks) == 25
    assert {task["kind"] for task in tasks} == {"workflow", "retrieval", "unsafe"}
    assert len({task["id"] for task in tasks}) == len(tasks)
    assert any(
        task.get("expected", {}).get("status") == "needs_clarification"
        for task in tasks
    )
    assert any(
        task.get("expected", {}).get("status") == "confirmation_required"
        for task in tasks
    )


def test_evaluation_reports_all_metrics_and_latency(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    report = asyncio.run(run_evaluation(output_path=output))

    assert report["task_count"] == 25
    assert report["passed"] == 25
    assert set(report["metrics"]) == {
        "groundedness",
        "citations",
        "tool_selection",
        "workflow",
        "clarification",
        "safety",
    }
    assert all(value == 1.0 for value in report["metrics"].values())
    assert report["latency"]["cold"]["samples"] == 2
    assert report["latency"]["warm"]["samples"] == 25
    assert output.exists()


def test_ablation_compares_chunk_size_and_retrieval_k(tmp_path: Path) -> None:
    output = tmp_path / "ablation.json"
    report = run_ablation(output_path=output)

    assert report["task_count"] == 7
    assert len(report["configurations"]) == 9
    assert {item["chunk_size"] for item in report["configurations"]} == {100, 180, 260}
    assert {item["top_k"] for item in report["configurations"]} == {3, 5, 8}
    assert report["recommended"] in report["configurations"]
    assert output.exists()
