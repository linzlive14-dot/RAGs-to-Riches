"""Gold-task evaluation, latency reporting, and retrieval ablation."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from app.agent import HRAgent, WorkflowResult
from app.agent import ChatRequest as AgentChatRequest
from app.config import get_settings
from app.llm import (
    OpenRouterClient,
    OpenRouterIntentExtractor,
    OpenRouterJudge,
    OpenRouterSynthesizer,
)
from hr_mcp.client import HRMCPClient
from rag.answering import answer_query
from rag.index import PolicyIndex
from rag.ingestion import chunk_documents, load_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = Path(__file__).with_name("gold_tasks.json")
DEFAULT_REPORT = Path(__file__).with_name("report.json")
DEFAULT_ABLATION = Path(__file__).with_name("ablation.json")


def load_tasks(path: Path = DEFAULT_TASKS) -> list[dict[str, Any]]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not 20 <= len(tasks) <= 30:
        raise ValueError("The gold set must contain 20 to 30 tasks")
    identifiers = [task["id"] for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Gold task IDs must be unique")
    return tasks


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 3)


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }


def _build_index(path: Path, *, chunk_size: int = 180, overlap: int = 30) -> PolicyIndex:
    chunks = chunk_documents(
        load_documents(PROJECT_ROOT / "policies"),
        chunk_size=chunk_size,
        overlap=overlap,
    )
    index = PolicyIndex(path)
    index.build(
        chunks,
        configuration={
            "chunk_size": chunk_size,
            "overlap": overlap,
            "retrieval": "sqlite-fts5-bm25",
        },
    )
    return index


def _tool_names(result: WorkflowResult) -> list[str]:
    return [step.tool for step in result.trace if step.tool]


def _answer_source(result: WorkflowResult) -> str:
    for step in result.trace:
        if step.state == "synthesize" and "configured LLM" in step.result_summary:
            return "llm"
        if step.state == "synthesize" and "failed validation" in step.result_summary:
            return "fallback"
    return "deterministic"


def _citations_valid(result: WorkflowResult) -> bool:
    if not result.citations:
        return False
    return all(f"[{citation.citation_id}]" in result.answer for citation in result.citations)


def _structural_scores(task: dict[str, Any], result: WorkflowResult) -> dict[str, bool]:
    expected = task["expected"]
    tools = _tool_names(result)
    scores: dict[str, bool] = {}
    if task["kind"] == "workflow":
        status_match = result.status == expected["status"]
        scores["tool_selection"] = tools == expected["tools"]
        scores["workflow"] = status_match
        scores["clarification"] = (
            status_match if expected["status"] == "needs_clarification" else True
        )
        scores["safety"] = not result.mock_action or not result.mock_action.get("created", False)
        if "search_policy_documents" in expected["tools"]:
            scores["groundedness"] = bool(result.citations)
            scores["citations"] = _citations_valid(result)
    elif task["kind"] == "retrieval":
        actual_sources = {citation.source for citation in result.citations}
        expected_sources = set(expected["sources_any"])
        if expected["grounded"]:
            scores["groundedness"] = (
                result.status == "completed" and bool(actual_sources & expected_sources)
            )
            scores["citations"] = _citations_valid(result)
        else:
            scores["groundedness"] = result.status == "escalated" and not result.citations
            scores["citations"] = not result.citations
        scores["tool_selection"] = tools == ["search_policy_documents"]
        scores["safety"] = True
    else:
        blocked = any(
            "Blocked unsafe request" in step.result_summary for step in result.trace
        )
        scores["safety"] = blocked and result.status == "escalated"
        scores["tool_selection"] = tools == []
    return scores


def _judge_evidence(result: WorkflowResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "workflow": result.workflow.value,
        "citations": [citation.model_dump() for citation in result.citations],
        "trace": [
            {
                "state": step.state.value,
                "tool": step.tool,
                "result_summary": step.result_summary,
            }
            for step in result.trace
        ],
    }


def _llm_clients() -> tuple[OpenRouterSynthesizer, OpenRouterIntentExtractor, OpenRouterJudge]:
    settings = get_settings()
    if settings.openrouter_api_key is None:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required so evaluation scores LLM answers "
            "instead of deterministic fallback text."
        )
    client = OpenRouterClient(
        settings.openrouter_api_key.get_secret_value(),
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        timeout_seconds=max(settings.openrouter_timeout_seconds, 45.0),
    )
    return (
        OpenRouterSynthesizer(client),
        OpenRouterIntentExtractor(client),
        OpenRouterJudge(client),
    )


async def run_evaluation(
    *,
    tasks_path: Path = DEFAULT_TASKS,
    output_path: Path | None = None,
    top_k: int = 5,
    mode: Literal["offline", "llm"] = "offline",
) -> dict[str, Any]:
    """Execute the gold set and return a machine-readable score report.

    ``offline`` checks tool behavior and citation structure without a model.
    ``llm`` scores the configured model's answers with an LLM judge.
    ``top_k`` is retained for callers; chat retrieval uses the agent's top_k.
    """

    del top_k
    synthesizer = None
    intent_extractor = None
    judge = None
    if mode == "llm":
        synthesizer, intent_extractor, judge = _llm_clients()

    tasks = load_tasks(tasks_path)
    task_results: list[dict[str, Any]] = []
    warm_latencies: list[float] = []
    metric_values: dict[str, list[bool]] = {
        name: []
        for name in (
            "groundedness",
            "citations",
            "tool_selection",
            "workflow",
            "clarification",
            "safety",
        )
    }

    with tempfile.TemporaryDirectory(prefix="hr-evaluation-") as directory:
        build_started = time.perf_counter()
        index_path = Path(directory) / "policy.sqlite3"
        _build_index(index_path)
        index_build_ms = (time.perf_counter() - build_started) * 1_000

        startup_started = time.perf_counter()
        async with HRMCPClient(env={"HR_POLICY_INDEX": str(index_path)}) as client:
            mcp_startup_ms = (time.perf_counter() - startup_started) * 1_000
            agent = HRAgent(
                client,
                synthesizer=synthesizer,
                intent_extractor=intent_extractor,
            )
            for task in tasks:
                started = time.perf_counter()
                result = await agent.chat(AgentChatRequest(message=task["message"]))
                scores = _structural_scores(task, result)
                source = _answer_source(result)
                if mode == "llm" and source != "llm":
                    if "groundedness" in scores:
                        scores["groundedness"] = False
                    if "citations" in scores:
                        scores["citations"] = False
                elif judge is not None and "groundedness" in scores:
                    judged = await judge.score(
                        answer=result.answer,
                        evidence=_judge_evidence(result),
                        rubric={
                            "facts": task.get("facts", []),
                            "expected_status": task["expected"].get("status"),
                            "sources_any": task["expected"].get("sources_any", []),
                        },
                    )
                    scores["groundedness"] = scores["groundedness"] and judged["grounded"]
                    scores["citations"] = scores["citations"] and judged["citations_accurate"]
                elapsed_ms = (time.perf_counter() - started) * 1_000
                warm_latencies.append(elapsed_ms)
                for metric, passed in scores.items():
                    metric_values[metric].append(passed)
                task_results.append(
                    {
                        "id": task["id"],
                        "kind": task["kind"],
                        "passed": all(scores.values()),
                        "scores": scores,
                        "answer_source": source,
                        "actual": {
                            "status": result.status,
                            "tools": _tool_names(result),
                            "citation_sources": [item.source for item in result.citations],
                            "answer": result.answer,
                        },
                        "latency_ms": round(elapsed_ms, 3),
                    }
                )

    metrics = {
        name: round(sum(values) / len(values), 4) if values else None
        for name, values in metric_values.items()
    }
    report: dict[str, Any] = {
        "task_count": len(tasks),
        "passed": sum(result["passed"] for result in task_results),
        "answer_source": "llm" if mode == "llm" else "deterministic",
        "metrics": metrics,
        "latency": {
            "cold": _latency_summary([index_build_ms, mcp_startup_ms]),
            "cold_components_ms": {
                "index_build": round(index_build_ms, 3),
                "mcp_startup": round(mcp_startup_ms, 3),
            },
            "warm": _latency_summary(warm_latencies),
        },
        "tasks": task_results,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_ablation(
    *,
    tasks_path: Path = DEFAULT_TASKS,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Compare chunk-size and retrieval-k settings on retrieval gold tasks."""

    retrieval_tasks = [
        task
        for task in load_tasks(tasks_path)
        if task["kind"] == "retrieval" and task["expected"]["grounded"]
    ]
    configurations: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="hr-ablation-") as directory:
        for chunk_size, overlap in ((100, 20), (180, 30), (260, 40)):
            index = _build_index(
                Path(directory) / f"policy-{chunk_size}.sqlite3",
                chunk_size=chunk_size,
                overlap=overlap,
            )
            for top_k in (3, 5, 8):
                hits = 0
                grounded = 0
                latencies: list[float] = []
                for task in retrieval_tasks:
                    started = time.perf_counter()
                    answer = answer_query(index, task["query"], top_k=top_k)
                    latencies.append((time.perf_counter() - started) * 1_000)
                    sources = {citation.source for citation in answer.citations}
                    expected_sources = set(task["expected"]["sources_any"])
                    hits += bool(sources & expected_sources)
                    grounded += answer.grounded
                configurations.append(
                    {
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "top_k": top_k,
                        "source_hit_rate": round(hits / len(retrieval_tasks), 4),
                        "grounded_rate": round(grounded / len(retrieval_tasks), 4),
                        "mean_latency_ms": round(statistics.fmean(latencies), 3),
                    }
                )
    best = max(
        configurations,
        key=lambda item: (
            item["source_hit_rate"],
            item["grounded_rate"],
            -item["mean_latency_ms"],
        ),
    )
    report = {
        "task_count": len(retrieval_tasks),
        "configurations": configurations,
        "recommended": best,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ablation-output", type=Path, default=DEFAULT_ABLATION)
    args = parser.parse_args()
    try:
        report = asyncio.run(
            run_evaluation(tasks_path=args.tasks, output_path=args.output, mode="llm")
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    ablation = run_ablation(tasks_path=args.tasks, output_path=args.ablation_output)
    print(
        json.dumps(
            {
                "evaluation": {
                    "passed": report["passed"],
                    "task_count": report["task_count"],
                    "metrics": report["metrics"],
                },
                "recommended_ablation": ablation["recommended"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
