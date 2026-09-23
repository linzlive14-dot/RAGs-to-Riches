"""Explicit, inspectable state-machine orchestration for HR workflows."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ToolClient(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class AnswerSynthesizer(Protocol):
    async def synthesize(self, evidence: dict[str, Any]) -> str: ...


class Workflow(StrEnum):
    REMOTE_WORK = "remote_work"
    PTO = "pto"


class AgentState(StrEnum):
    CLASSIFY = "classify"
    RETRIEVE = "retrieve"
    VALIDATE = "validate"
    CONFIRM = "confirm"
    SYNTHESIZE = "synthesize"
    COMPLETE = "complete"
    ESCALATE = "escalate"


class WorkflowRequest(BaseModel):
    workflow: Workflow
    employee_id: str = Field(pattern=r"^SYN-\d{4}$")
    requested_location_id: str | None = None
    requested_hours: float | None = Field(default=None, gt=0)
    create_ticket: bool = False
    confirmed: bool = False


class TraceStep(BaseModel):
    state: AgentState
    tool: str | None = None
    safe_arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "error", "confirmation_required", "skipped"] = "ok"
    result_summary: str
    sources: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    citation_id: str
    source: str
    section: str
    snippet: str


class WorkflowResult(BaseModel):
    workflow: Workflow
    status: Literal["completed", "needs_clarification", "confirmation_required", "escalated"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace: list[TraceStep]
    mock_action: dict[str, Any] | None = None


class HRAgent:
    """Run fixed workflow states without exposing hidden model reasoning."""

    def __init__(
        self,
        tools: ToolClient,
        synthesizer: AnswerSynthesizer | None = None,
    ) -> None:
        self.tools = tools
        self.synthesizer = synthesizer

    async def _call(
        self,
        trace: list[TraceStep],
        state: AgentState,
        tool: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        safe_arguments = {
            key: value
            for key, value in arguments.items()
            if key not in {"confirmed"}
        }
        try:
            result = await self.tools.call_tool(tool, arguments)
        except Exception as error:
            trace.append(
                TraceStep(
                    state=state,
                    tool=tool,
                    safe_arguments=safe_arguments,
                    status="error",
                    result_summary=f"{type(error).__name__}: tool unavailable",
                )
            )
            raise
        sources = sorted(
            {
                item["source"]
                for item in result.get("results", [])
                if isinstance(item, dict) and item.get("source")
            }
        )
        trace.append(
            TraceStep(
                state=state,
                tool=tool,
                safe_arguments=safe_arguments,
                result_summary=self._summary(tool, result),
                sources=sources,
            )
        )
        return result

    @staticmethod
    def _summary(tool: str, result: dict[str, Any]) -> str:
        if tool == "search_policy_documents":
            return f"Retrieved {len(result.get('results', []))} policy chunks"
        if tool == "check_policy_compliance":
            return f"Compliance outcome: {result.get('decision', 'unknown')}"
        if tool == "create_mock_hr_ticket":
            return (
                "Mock ticket created"
                if result.get("created")
                else "Mock ticket awaiting explicit confirmation"
            )
        return f"{tool} returned a synthetic record"

    @staticmethod
    def _citations(policy_result: dict[str, Any], *, limit: int = 3) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, str]] = set()
        for result in policy_result.get("results", []):
            key = (result["source"], result["section"])
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    citation_id=f"P{len(citations) + 1}",
                    source=result["source"],
                    section=result["section"],
                    snippet=result["snippet"],
                )
            )
            if len(citations) == limit:
                break
        return citations

    async def run(self, request: WorkflowRequest) -> WorkflowResult:
        trace = [
            TraceStep(
                state=AgentState.CLASSIFY,
                result_summary=f"Selected {request.workflow.value} workflow from structured input",
            )
        ]
        missing = (
            request.workflow == Workflow.REMOTE_WORK and not request.requested_location_id
        ) or (request.workflow == Workflow.PTO and request.requested_hours is None)
        if missing:
            needed = (
                "requested work location"
                if request.workflow == Workflow.REMOTE_WORK
                else "requested PTO hours"
            )
            trace.append(
                TraceStep(
                    state=AgentState.ESCALATE,
                    status="skipped",
                    result_summary=f"Clarification required: missing {needed}",
                )
            )
            return WorkflowResult(
                workflow=request.workflow,
                status="needs_clarification",
                answer=f"Please provide the {needed} before I check policy eligibility.",
                trace=trace,
            )

        try:
            employee = await self._call(
                trace,
                AgentState.RETRIEVE,
                "lookup_employee_profile",
                {"employee_id": request.employee_id},
            )
            if request.workflow == Workflow.REMOTE_WORK:
                policy_query = (
                    "fully remote eligibility tenure performance supported location "
                    "location change request approval"
                )
                compliance_arguments = {
                    "workflow": request.workflow.value,
                    "employee_id": request.employee_id,
                    "requested_location_id": request.requested_location_id,
                }
            else:
                policy_query = (
                    "PTO available balance request notice manager approval protected leave"
                )
                await self._call(
                    trace,
                    AgentState.RETRIEVE,
                    "check_pto_balance",
                    {"employee_id": request.employee_id},
                )
                compliance_arguments = {
                    "workflow": request.workflow.value,
                    "employee_id": request.employee_id,
                    "requested_hours": request.requested_hours,
                }
            policies = await self._call(
                trace,
                AgentState.RETRIEVE,
                "search_policy_documents",
                {"query": policy_query, "top_k": 8},
            )
            compliance = await self._call(
                trace,
                AgentState.VALIDATE,
                "check_policy_compliance",
                compliance_arguments,
            )
        except Exception:
            return WorkflowResult(
                workflow=request.workflow,
                status="escalated",
                answer="A required HR tool was unavailable. No action was taken; contact HR.",
                trace=trace,
            )

        citations = self._citations(policies)
        references = " ".join(f"[{item.citation_id}]" for item in citations)
        decision = compliance["decision"]
        name = employee["employee"]["name"]
        if decision == "eligible_for_review":
            fallback_answer = (
                f"{name} meets the automated prerequisites for {request.workflow.value.replace('_', ' ')} "
                f"review, but manager/HR approval is still required. {references}"
            )
            status: Literal["completed", "escalated"] = "completed"
        else:
            failed = [
                check["name"] for check in compliance.get("checks", []) if not check["passed"]
            ]
            fallback_answer = (
                f"The request needs HR review because these checks did not pass: "
                f"{', '.join(failed) or 'manual review required'}. {references}"
            )
            status = "escalated"

        answer = fallback_answer
        synthesis_summary = "Assembled cited guidance deterministically"
        if self.synthesizer is not None:
            evidence = {
                "workflow": request.workflow.value,
                "employee_name": name,
                "decision": decision,
                "checks": compliance.get("checks", []),
                "citations": [citation.model_dump() for citation in citations],
            }
            try:
                answer = await self.synthesizer.synthesize(evidence)
                synthesis_summary = "Generated grounded guidance with the configured LLM"
            except Exception:
                synthesis_summary = (
                    "LLM generation failed validation; used safe deterministic guidance"
                )
        trace.append(
            TraceStep(
                state=AgentState.SYNTHESIZE,
                result_summary=synthesis_summary,
                sources=[citation.source for citation in citations],
            )
        )

        mock_action: dict[str, Any] | None = None
        if request.create_ticket:
            summary = (
                f"Review {request.workflow.value.replace('_', ' ')} guidance for "
                f"{request.employee_id}"
            )
            try:
                mock_action = await self._call(
                    trace,
                    AgentState.CONFIRM,
                    "create_mock_hr_ticket",
                    {
                        "employee_id": request.employee_id,
                        "category": request.workflow.value,
                        "summary": summary,
                        "confirmed": request.confirmed,
                    },
                )
            except Exception:
                return WorkflowResult(
                    workflow=request.workflow,
                    status="escalated",
                    answer=answer + " The mock ticket action failed; no action was taken.",
                    citations=citations,
                    trace=trace,
                )
            if not mock_action.get("created"):
                trace[-1].status = "confirmation_required"
                return WorkflowResult(
                    workflow=request.workflow,
                    status="confirmation_required",
                    answer=answer + " Confirm explicitly if you want me to create the displayed mock ticket.",
                    citations=citations,
                    trace=trace,
                    mock_action=mock_action,
                )

        trace.append(
            TraceStep(
                state=AgentState.COMPLETE if status == "completed" else AgentState.ESCALATE,
                result_summary="Workflow completed without hidden reasoning",
            )
        )
        return WorkflowResult(
            workflow=request.workflow,
            status=status,
            answer=answer,
            citations=citations,
            trace=trace,
            mock_action=mock_action,
        )
