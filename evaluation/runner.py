"""Gold-task evaluation, latency reporting, and ablations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import httpx

from app.agent import HRAgent, WorkflowResult
from app.agent import ChatRequest as AgentChatRequest
from app.config import get_settings
from app.llm import (
    DailyLimitReached,
    OpenRouterClient,
    OpenRouterIntentExtractor,
    OpenRouterJudge,
    OpenRouterSynthesizer,
)
from hr_mcp.client import HRMCPClient
from rag.index import PolicyIndex, RetrievalMode
from rag.ingestion import chunk_documents, load_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = Path(__file__).with_name("gold_tasks.json")
RESULTS_DIR = Path(__file__).with_name("results")
DEFAULT_REPORT = RESULTS_DIR / "llm-report.json"
DEFAULT_ABLATION = RESULTS_DIR / "ablation.json"
DEFAULT_HTTP_REPORT = RESULTS_DIR / "http-latency.json"
PASSING_ANSWER_MATCH = 0.5
JUDGE_BATCH_SIZE = 5
EVALUATION_MIN_INTERVAL_SECONDS = 3.5
METRICS = (
    "groundedness",
    "citations",
    "tool_selection",
    "workflow",
    "clarification",
    "safety",
)
ABLATIONS: tuple[dict[str, Any], ...] = (
    {"name": "hybrid retrieval (default)", "retrieval_mode": "hybrid"},
    {"name": "BM25 only", "retrieval_mode": "bm25"},
    {"name": "vector only", "retrieval_mode": "vector"},
    {"name": "hybrid, 100-word chunks", "chunk_size": 100, "overlap": 20},
    {"name": "hybrid, 260-word chunks", "chunk_size": 260, "overlap": 40},
    {"name": "without get_policy_section", "disabled_tools": ["get_policy_section"]},
    {"name": "without check_policy_compliance", "disabled_tools": ["check_policy_compliance"]},
    {"name": "without search_policy_documents", "disabled_tools": ["search_policy_documents"]},
)


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
            "retrieval": "hybrid-bm25-faiss-rrf",
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


def _normalized(text: str) -> str:
    return " ".join(text.replace("\u2019", "'").casefold().split())


def answer_match(task: dict[str, Any], answer: str) -> float | None:
    """Fraction of the task's key facts present in the answer (partial credit).

    Each key fact is a list of acceptable phrasings; any one of them counts.
    """

    facts = task.get("key_facts") or []
    if not facts:
        return None
    text = _normalized(answer)
    found = sum(any(_normalized(option) in text for option in fact) for fact in facts)
    return round(found / len(facts), 4)


def _structural_scores(task: dict[str, Any], result: WorkflowResult) -> dict[str, bool]:
    expected = task["expected"]
    tools = _tool_names(result)
    cited = {citation.source for citation in result.citations}
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
            scores["groundedness"] = bool(result.citations) and set(
                expected.get("sources_all", [])
            ) <= cited
            scores["citations"] = _citations_valid(result)
    elif task["kind"] == "retrieval":
        if expected["grounded"]:
            scores["groundedness"] = (
                result.status == "completed"
                and bool(cited & set(expected["sources_any"]))
                and set(expected.get("sources_all", [])) <= cited
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


def _llm_client() -> OpenRouterClient:
    settings = get_settings()
    if settings.openrouter_api_key is None:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required so evaluation scores LLM answers "
            "instead of deterministic fallback text."
        )
    return OpenRouterClient(
        settings.openrouter_api_key.get_secret_value(),
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        timeout_seconds=max(settings.openrouter_timeout_seconds, 45.0),
        min_interval_seconds=EVALUATION_MIN_INTERVAL_SECONDS,
    )


def _rubric(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "gold_answer": task.get("gold_answer"),
        "key_facts": task.get("key_facts", []),
        "expected_status": task["expected"].get("status"),
        "sources_any": task["expected"].get("sources_any", []),
    }


def _passed(entry: dict[str, Any]) -> bool:
    match = entry["answer_match"]
    return all(entry["scores"].values()) and (match is None or match >= PASSING_ANSWER_MATCH)


def _metrics(task_results: list[dict[str, Any]]) -> dict[str, float | None]:
    values: dict[str, list[bool]] = {name: [] for name in METRICS}
    for entry in task_results:
        for metric, passed in entry["scores"].items():
            values[metric].append(passed)
    metrics: dict[str, float | None] = {
        name: round(sum(scores) / len(scores), 4) if scores else None
        for name, scores in values.items()
    }
    matches = [entry["answer_match"] for entry in task_results if entry["answer_match"] is not None]
    metrics["answer_match"] = round(sum(matches) / len(matches), 4) if matches else None
    metrics["answer_full_match"] = (
        round(sum(match == 1.0 for match in matches) / len(matches), 4) if matches else None
    )
    return metrics


def _category_summary(task_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in task_results:
        grouped[result["category"]].append(result)
    summary: dict[str, dict[str, Any]] = {}
    for category, results in sorted(grouped.items()):
        matches = [r["answer_match"] for r in results if r["answer_match"] is not None]
        summary[category] = {
            "tasks": len(results),
            "passed": sum(r["passed"] for r in results),
            "answer_match": round(sum(matches) / len(matches), 4) if matches else None,
        }
    return summary


async def run_evaluation(
    *,
    tasks_path: Path = DEFAULT_TASKS,
    output_path: Path | None = None,
    top_k: int = 5,
    mode: Literal["offline", "llm"] = "offline",
    retrieval_mode: RetrievalMode | None = None,
    chunk_size: int = 180,
    overlap: int = 30,
    disabled_tools: list[str] | None = None,
    index_path: Path | None = None,
    resume: bool = False,
    llm_client: OpenRouterClient | None = None,
) -> dict[str, Any]:
    """Execute the gold set through the MCP-backed agent and score it.

    ``offline`` checks tool behavior, citations, and key facts without a model.
    ``llm`` scores the configured model's answers with a batched LLM judge.
    When the provider's daily limit is reached, the report is saved with
    ``complete: false``; ``resume=True`` later re-runs only the unfinished
    tasks from ``output_path``. ``top_k`` is retained for callers; chat
    retrieval uses the agent's top_k.
    """

    del top_k
    llm = None
    synthesizer = None
    intent_extractor = None
    judge = None
    if mode == "llm":
        llm = llm_client or _llm_client()
        synthesizer = OpenRouterSynthesizer(llm)
        intent_extractor = OpenRouterIntentExtractor(llm)
        judge = OpenRouterJudge(llm)

    tasks = load_tasks(tasks_path)
    finished: dict[str, dict[str, Any]] = {}
    if resume and output_path is not None and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        finished = {entry["id"]: entry for entry in previous.get("tasks", [])}
    task_results: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], dict[str, Any], WorkflowResult]] = []
    unfinished: list[str] = []
    stopped = False

    async def judge_pending() -> None:
        nonlocal stopped
        batch = list(pending)
        pending.clear()
        if not batch or judge is None:
            return
        try:
            verdicts = await judge.score_batch(
                [
                    {
                        "id": task["id"],
                        "answer": result.answer,
                        "evidence": _judge_evidence(result),
                        "rubric": _rubric(task),
                    }
                    for entry, task, result in batch
                ]
            )
        except DailyLimitReached:
            stopped = True
            for entry, _, _ in batch:
                task_results.remove(entry)
                unfinished.append(entry["id"])
            return
        except Exception as error:
            verdicts = {}
            for entry, _, _ in batch:
                entry["judge_error"] = f"{type(error).__name__}: {error}"
        for entry, task, _ in batch:
            verdict = verdicts.get(task["id"], {"grounded": False, "citations_accurate": False})
            entry["judge"] = verdict
            entry["scores"]["groundedness"] = entry["scores"]["groundedness"] and verdict["grounded"]
            entry["scores"]["citations"] = (
                entry["scores"]["citations"] and verdict["citations_accurate"]
            )

    with tempfile.TemporaryDirectory(prefix="hr-evaluation-") as directory:
        index_build_ms = 0.0
        if index_path is None:
            build_started = time.perf_counter()
            index_path = Path(directory) / "policy.sqlite3"
            _build_index(index_path, chunk_size=chunk_size, overlap=overlap)
            index_build_ms = (time.perf_counter() - build_started) * 1_000

        server_env = {"HR_POLICY_INDEX": str(index_path)}
        if retrieval_mode is not None:
            server_env["HR_RETRIEVAL_MODE"] = retrieval_mode
        startup_started = time.perf_counter()
        async with HRMCPClient(env=server_env) as client:
            mcp_startup_ms = (time.perf_counter() - startup_started) * 1_000
            agent = HRAgent(
                client,
                synthesizer=synthesizer,
                intent_extractor=intent_extractor,
                disabled_tools=set(disabled_tools or ()),
            )
            for task in tasks:
                if task["id"] in finished:
                    task_results.append(finished[task["id"]])
                    continue
                if stopped:
                    unfinished.append(task["id"])
                    continue
                waited_before = llm.waited_seconds if llm else 0.0
                started = time.perf_counter()
                result = await agent.chat(AgentChatRequest(message=task["message"]))
                waited_ms = ((llm.waited_seconds if llm else 0.0) - waited_before) * 1_000
                elapsed_ms = (time.perf_counter() - started) * 1_000 - waited_ms
                if llm is not None and llm.daily_limit_reached:
                    stopped = True
                    unfinished.append(task["id"])
                    continue
                scores = _structural_scores(task, result)
                source = _answer_source(result)
                if mode == "llm" and source == "fallback":
                    if "groundedness" in scores:
                        scores["groundedness"] = False
                    if "citations" in scores:
                        scores["citations"] = False
                entry: dict[str, Any] = {
                    "id": task["id"],
                    "kind": task["kind"],
                    "category": task.get("category", task["kind"]),
                    "scores": scores,
                    "answer_match": answer_match(task, result.answer),
                    "answer_source": source,
                    "actual": {
                        "status": result.status,
                        "tools": _tool_names(result),
                        "citation_sources": [item.source for item in result.citations],
                        "answer": result.answer,
                    },
                    "latency_ms": round(elapsed_ms, 3),
                    "rate_limit_wait_ms": round(waited_ms, 3),
                }
                task_results.append(entry)
                if judge is not None and source == "llm" and "groundedness" in scores:
                    pending.append((entry, task, result))
                    if len(pending) >= JUDGE_BATCH_SIZE:
                        await judge_pending()
            await judge_pending()

    for entry in task_results:
        entry["passed"] = _passed(entry)
    order = {task["id"]: position for position, task in enumerate(tasks)}
    unfinished.sort(key=order.__getitem__)
    metrics = _metrics(task_results)
    cold_samples = [mcp_startup_ms] + ([index_build_ms] if index_build_ms else [])
    report: dict[str, Any] = {
        "task_count": len(tasks),
        "tasks_scored": len(task_results),
        "complete": not unfinished,
        "unfinished_tasks": unfinished,
        "passed": sum(entry["passed"] for entry in task_results),
        "answer_source": "llm" if mode == "llm" else "deterministic",
        "llm": None
        if llm is None
        else {
            "model": llm.model,
            "requests_this_run": llm.requests,
            "judge_batch_size": JUDGE_BATCH_SIZE,
            "daily_limit_reached": llm.daily_limit_reached,
        },
        "configuration": {
            "retrieval_mode": retrieval_mode or os.environ.get("HR_RETRIEVAL_MODE", "hybrid"),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "disabled_tools": sorted(disabled_tools or []),
        },
        "metrics": metrics,
        "categories": _category_summary(task_results),
        "latency": {
            "cold": _latency_summary(cold_samples),
            "cold_components_ms": {
                "index_build": round(index_build_ms, 3),
                "mcp_startup": round(mcp_startup_ms, 3),
            },
            "warm": _latency_summary([entry["latency_ms"] for entry in task_results]),
            "warm_excludes_rate_limit_waits": True,
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
    configurations: tuple[dict[str, Any], ...] = ABLATIONS,
) -> dict[str, Any]:
    """Run the full offline gold set once per retriever, chunking, or tool setting."""

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="hr-ablation-") as directory:
        indexes: dict[tuple[int, int], Path] = {}
        for configuration in configurations:
            chunking = (configuration.get("chunk_size", 180), configuration.get("overlap", 30))
            if chunking not in indexes:
                indexes[chunking] = Path(directory) / f"policy-{chunking[0]}.sqlite3"
                _build_index(indexes[chunking], chunk_size=chunking[0], overlap=chunking[1])
            report = asyncio.run(
                run_evaluation(
                    tasks_path=tasks_path,
                    retrieval_mode=configuration.get("retrieval_mode", "hybrid"),
                    chunk_size=chunking[0],
                    overlap=chunking[1],
                    disabled_tools=configuration.get("disabled_tools"),
                    index_path=indexes[chunking],
                )
            )
            rows.append(
                {
                    "name": configuration["name"],
                    **report["configuration"],
                    "passed": report["passed"],
                    "task_count": report["task_count"],
                    "metrics": report["metrics"],
                    "categories": report["categories"],
                    "warm_p50_ms": report["latency"]["warm"]["p50_ms"],
                    "failed_tasks": [task["id"] for task in report["tasks"] if not task["passed"]],
                }
            )
    retrieval_rows = [row for row in rows if not row["disabled_tools"]]
    best = max(
        retrieval_rows,
        key=lambda row: (
            row["passed"],
            row["metrics"]["answer_match"] or 0.0,
            row["metrics"]["groundedness"] or 0.0,
        ),
    )
    report = {
        "answer_source": "deterministic",
        "configurations": rows,
        "recommended": best["name"],
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def measure_http_latency(
    *,
    tasks_path: Path = DEFAULT_TASKS,
    base_url: str | None = None,
    rounds: int = 2,
    use_llm: bool = False,
    output_path: Path | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Time `/health` and `/chat` over real HTTP.

    Without `base_url`, starts a local uvicorn server (with its MCP session)
    and includes that start-up in the cold measurement.
    """

    messages = [task["message"] for task in load_tasks(tasks_path)]
    process: subprocess.Popen[bytes] | None = None
    startup_ms: float | None = None
    with tempfile.TemporaryDirectory(prefix="hr-http-") as directory:
        if base_url is None:
            index_path = Path(directory) / "policy.sqlite3"
            _build_index(index_path)
            port = _free_port()
            base_url = f"http://127.0.0.1:{port}"
            environment = os.environ.copy()
            environment["HR_POLICY_INDEX"] = str(index_path)
            if not use_llm:
                environment["OPENROUTER_API_KEY"] = ""
            started = time.perf_counter()
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        try:
            with httpx.Client(base_url=base_url, timeout=timeout) as http:
                deadline = time.monotonic() + timeout
                health: dict[str, Any] = {}
                while time.monotonic() < deadline:
                    try:
                        response = http.get("/health")
                        if response.status_code == 200:
                            health = response.json()
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.25)
                else:
                    raise TimeoutError(f"{base_url}/health did not respond within {timeout:g}s")
                if process is not None:
                    startup_ms = (time.perf_counter() - started) * 1_000

                first_started = time.perf_counter()
                http.post("/chat", json={"message": messages[0]}).raise_for_status()
                first_chat_ms = (time.perf_counter() - first_started) * 1_000

                health_ms: list[float] = []
                chat_ms: list[float] = []
                for _ in range(rounds):
                    started_health = time.perf_counter()
                    http.get("/health").raise_for_status()
                    health_ms.append((time.perf_counter() - started_health) * 1_000)
                    for message in messages:
                        started_chat = time.perf_counter()
                        http.post("/chat", json={"message": message}).raise_for_status()
                        chat_ms.append((time.perf_counter() - started_chat) * 1_000)
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()

    report = {
        "target": "local uvicorn" if startup_ms is not None else base_url,
        "answer_source": "llm" if health.get("llm", {}).get("configured") else "deterministic",
        "health_at_start": {
            "status": health.get("status"),
            "mcp": health.get("mcp", {}).get("status"),
            "tool_count": health.get("mcp", {}).get("tool_count"),
        },
        "cold": {
            "startup_until_healthy_ms": round(startup_ms, 3) if startup_ms is not None else None,
            "first_chat_ms": round(first_chat_ms, 3),
        },
        "warm": {
            "health": _latency_summary(health_ms),
            "chat": _latency_summary(chat_ms),
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument(
        "--mode",
        choices=("llm", "offline"),
        default="llm",
        help="llm scores model answers with a judge; offline needs no API key.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ablation-output", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument(
        "--http",
        nargs="?",
        const="local",
        default=None,
        help="Also time /health and /chat over HTTP: 'local' or a base URL.",
    )
    parser.add_argument("--http-output", type=Path, default=DEFAULT_HTTP_REPORT)
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=None,
        help="Exit non-zero when fewer than this fraction of tasks pass.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep finished tasks from --output and run only the unfinished ones.",
    )
    args = parser.parse_args()
    output = args.output or RESULTS_DIR / f"{args.mode}-report.json"
    try:
        report = asyncio.run(
            run_evaluation(
                tasks_path=args.tasks,
                output_path=output,
                mode=args.mode,
                resume=args.resume,
            )
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    summary: dict[str, Any] = {
        "evaluation": {
            "complete": report["complete"],
            "passed": report["passed"],
            "tasks_scored": report["tasks_scored"],
            "task_count": report["task_count"],
            "metrics": report["metrics"],
            "llm": report["llm"],
            "report": str(output),
        }
    }
    if not report["complete"]:
        print(
            f"Daily model limit reached; {len(report['unfinished_tasks'])} tasks unfinished. "
            "Run again with --resume after the limit resets.",
            file=sys.stderr,
        )
    if not args.skip_ablation:
        ablation = run_ablation(tasks_path=args.tasks, output_path=args.ablation_output)
        summary["ablation"] = {
            row["name"]: f"{row['passed']}/{row['task_count']}" for row in ablation["configurations"]
        }
        summary["recommended_ablation"] = ablation["recommended"]
    if args.http:
        http_report = measure_http_latency(
            tasks_path=args.tasks,
            base_url=None if args.http == "local" else args.http,
            use_llm=args.mode == "llm",
            output_path=args.http_output,
        )
        summary["http"] = {"cold": http_report["cold"], "warm_chat": http_report["warm"]["chat"]}
    print(json.dumps(summary, indent=2))
    if (
        args.min_pass_rate is not None
        and report["complete"]
        and report["passed"] < args.min_pass_rate * report["task_count"]
    ):
        raise SystemExit(
            f"Evaluation passed {report['passed']}/{report['task_count']}, "
            f"below the required {args.min_pass_rate:.0%}"
        )


if __name__ == "__main__":
    main()
