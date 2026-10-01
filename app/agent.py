"""Explicit, inspectable orchestration for conversational HR workflows."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, field_validator

from app.intent import ParsedIntent, asks_for_location_list, is_unsafe_message, resolve_intent
from rag.answering import REFUSAL
from rag.guardrails import has_sufficient_evidence


class ToolClient(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class AnswerSynthesizer(Protocol):
    async def synthesize(self, evidence: dict[str, Any]) -> str: ...


class IntentExtractor(Protocol):
    async def extract(
        self,
        message: str,
        history: list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class Workflow(StrEnum):
    REMOTE_WORK = "remote_work"
    PTO = "pto"
    POLICY_QA = "policy_qa"
    UNSUPPORTED = "unsupported"


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
    user_message: str | None = None

    @field_validator("workflow")
    @classmethod
    def _structured_workflow(cls, value: Workflow) -> Workflow:
        if value not in {Workflow.REMOTE_WORK, Workflow.PTO}:
            raise ValueError("workflow must be remote_work or pto")
        return value


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2_000)


class ChatContext(BaseModel):
    workflow: Workflow | None = None
    employee_id: str | None = None
    requested_location_id: str | None = None
    requested_hours: float | None = None
    create_ticket: bool = False
    awaiting_confirmation: bool = False


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2_000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    context: ChatContext | None = None

    @field_validator("message")
    @classmethod
    def _message_not_blank(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Message cannot be empty")
        return cleaned


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
    context: ChatContext | None = None


class HRAgent:
    """Run fixed workflow states without exposing hidden model reasoning."""

    def __init__(
        self,
        tools: ToolClient,
        synthesizer: AnswerSynthesizer | None = None,
        intent_extractor: IntentExtractor | None = None,
    ) -> None:
        self.tools = tools
        self.synthesizer = synthesizer
        self.intent_extractor = intent_extractor

    async def _call(
        self,
        trace: list[TraceStep],
        state: AgentState,
        tool: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        safe_arguments = {
            key: value for key, value in arguments.items() if key not in {"confirmed"}
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
        if result.get("found") is False:
            return str(result.get("reason") or f"{tool} found no matching record")
        if tool == "search_policy_documents":
            return f"Retrieved {len(result.get('results', []))} policy chunks"
        if tool == "list_work_locations":
            count = len(result.get("locations", []))
            return f"Location register contains {count} entries"
        if tool == "check_policy_compliance":
            reason = result.get("reason")
            decision = result.get("decision", "unknown")
            return f"Compliance outcome: {decision}" + (f". {reason}" if reason else "")
        if tool == "create_mock_hr_ticket":
            if result.get("created"):
                return "Mock ticket created"
            if result.get("rejected"):
                return str(result.get("reason") or "Mock ticket was rejected")
            return "Mock ticket awaiting explicit confirmation"
        return f"{tool} returned a synthetic record"

    @staticmethod
    def _citations(policy_result: dict[str, Any], *, limit: int = 3) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[str] = set()
        for result in policy_result.get("results", []):
            key = result["source"]
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

    @staticmethod
    def _fallback_answer(evidence: dict[str, Any], citations: list[Citation]) -> str:
        references = " ".join(f"[{item.citation_id}]" for item in citations)
        findings = [
            finding
            for finding in evidence.get("findings", [])
            if isinstance(finding, dict) and finding.get("reason") and finding.get("found") is False
        ]
        if findings:
            text = str(findings[0]["reason"])
            supported = evidence.get("supported_regular_remote_locations") or []
            if supported:
                text += " Regular remote employment is supported in " + ", ".join(supported) + "."
                if references:
                    text = f"{text} {references}"
                return text
            return f"{text} No action was taken."
        missing = evidence.get("missing") or []
        if missing:
            return (
                "Please provide the "
                + ", ".join(str(item) for item in missing)
                + " before I check policy eligibility."
            )
        if evidence.get("blocked"):
            return (
                "I can help with synthetic HR policy and workflow questions. "
                "I can't override system instructions or reveal secrets."
            )
        if evidence.get("grounded") is False:
            return REFUSAL
        decision = evidence.get("decision")
        if decision == "needs_clarification":
            return str(
                evidence.get("reason")
                or "More information is required before I can check eligibility."
            )
        name = evidence.get("employee_name") or "The employee"
        workflow = str(evidence.get("workflow", "request")).replace("_", " ")
        if decision == "eligible_for_review":
            return (
                f"{name} meets the automated prerequisites for {workflow} review, "
                f"but manager/HR approval is still required. {references}"
            ).strip()
        if decision == "escalate":
            failed = [
                check["name"]
                for check in evidence.get("checks", [])
                if isinstance(check, dict) and not check.get("passed", True)
            ]
            return (
                "The request needs HR review because these checks did not pass: "
                f"{', '.join(failed) or 'manual review required'}. {references}"
            ).strip()
        supported = evidence.get("supported_regular_remote_locations") or []
        if supported:
            listed = ", ".join(str(item) for item in supported)
            return (
                f"Regular remote employment is supported in {listed}. {references}"
            ).strip()
        if citations:
            claims = " ".join(f"{item.snippet} [{item.citation_id}]" for item in citations)
            return f"Policy guidance: {claims}"
        return "I need a more specific HR policy question before I can help."

    def _context_for(
        self,
        request: WorkflowRequest,
        *,
        awaiting_confirmation: bool = False,
    ) -> ChatContext:
        return ChatContext(
            workflow=request.workflow,
            employee_id=request.employee_id,
            requested_location_id=request.requested_location_id,
            requested_hours=request.requested_hours,
            create_ticket=request.create_ticket,
            awaiting_confirmation=awaiting_confirmation,
        )

    async def _speak(
        self,
        trace: list[TraceStep],
        evidence: dict[str, Any],
        citations: list[Citation],
    ) -> str:
        fallback = self._fallback_answer(evidence, citations)
        summary = "Assembled cited guidance deterministically"
        answer = fallback
        if self.synthesizer is not None:
            try:
                answer = await self.synthesizer.synthesize(evidence)
                summary = "Generated grounded guidance with the configured LLM"
            except Exception:
                summary = "LLM generation failed validation; used safe deterministic guidance"
                answer = fallback
        trace.append(
            TraceStep(
                state=AgentState.SYNTHESIZE,
                result_summary=summary,
                sources=[citation.source for citation in citations],
            )
        )
        return answer

    async def _finish(
        self,
        *,
        workflow: Workflow,
        status: Literal["completed", "needs_clarification", "confirmation_required", "escalated"],
        trace: list[TraceStep],
        evidence: dict[str, Any],
        citations: list[Citation] | None = None,
        mock_action: dict[str, Any] | None = None,
        context: ChatContext | None = None,
        terminal_state: AgentState | None = None,
    ) -> WorkflowResult:
        citations = citations or []
        answer = await self._speak(trace, evidence, citations)
        if terminal_state is not None:
            trace.append(
                TraceStep(
                    state=terminal_state,
                    result_summary="Workflow completed without hidden reasoning",
                )
            )
        return WorkflowResult(
            workflow=workflow,
            status=status,
            answer=answer,
            citations=citations,
            trace=trace,
            mock_action=mock_action,
            context=context,
        )

    async def chat(self, request: ChatRequest) -> WorkflowResult:
        context_data = request.context.model_dump() if request.context else None
        if is_unsafe_message(request.message):
            trace = [
                TraceStep(
                    state=AgentState.CLASSIFY,
                    status="skipped",
                    result_summary="Blocked unsafe request before any tool call",
                )
            ]
            return await self._finish(
                workflow=Workflow.UNSUPPORTED,
                status="escalated",
                trace=trace,
                evidence={
                    "workflow": Workflow.UNSUPPORTED.value,
                    "blocked": True,
                    "findings": [
                        {
                            "found": False,
                            "reason": "The request tries to override controls or reveal secrets.",
                        }
                    ],
                    "user_message_redacted": True,
                },
                context=ChatContext(workflow=Workflow.UNSUPPORTED),
                terminal_state=AgentState.ESCALATE,
            )

        model_intent: ParsedIntent | None = None
        note = "Resolved the workflow from the user message"
        if self.intent_extractor is not None:
            try:
                model_intent = ParsedIntent.model_validate(
                    await self.intent_extractor.extract(
                        request.message,
                        [turn.model_dump() for turn in request.history],
                        context_data,
                    )
                )
                note = "Resolved user intent with the configured LLM"
            except Exception:
                note = "LLM intent extraction failed; resolved the user message deterministically"
        parsed = resolve_intent(request.message, context_data, model_intent)
        if model_intent is not None and parsed.intent != model_intent.intent:
            note += f"; kept {parsed.intent} from explicit wording in the message"
        trace = [TraceStep(state=AgentState.CLASSIFY, result_summary=note)]

        if parsed.intent == "unsafe":
            trace.append(
                TraceStep(
                    state=AgentState.ESCALATE,
                    status="skipped",
                    result_summary="Blocked unsafe request before any tool call",
                )
            )
            return await self._finish(
                workflow=Workflow.UNSUPPORTED,
                status="escalated",
                trace=trace,
                evidence={"workflow": "unsupported", "blocked": True},
                context=ChatContext(workflow=Workflow.UNSUPPORTED),
                terminal_state=AgentState.ESCALATE,
            )

        if (
            asks_for_location_list(request.message)
            and not parsed.requested_location_id
            and not parsed.unrecognized_location
        ):
            return await self._location_catalog(request.message, trace, parsed.employee_id)

        if parsed.intent in {"policy_qa", "out_of_scope"}:
            return await self._policy_qa(request.message, trace)

        workflow = Workflow(parsed.intent)
        if workflow == Workflow.REMOTE_WORK and parsed.unrecognized_location:
            return await self._unknown_location(parsed, trace, request.message)

        missing: list[str] = []
        if not parsed.employee_id:
            missing.append("synthetic employee ID (SYN-####)")
        if workflow == Workflow.REMOTE_WORK and not parsed.requested_location_id:
            missing.append("requested work location")
        if workflow == Workflow.PTO and parsed.requested_hours is None:
            missing.append("requested PTO hours")
        if missing:
            trace.append(
                TraceStep(
                    state=AgentState.ESCALATE,
                    status="skipped",
                    result_summary="Clarification required: missing " + ", ".join(missing),
                )
            )
            return await self._finish(
                workflow=workflow,
                status="needs_clarification",
                trace=trace,
                evidence={
                    "workflow": workflow.value,
                    "missing": missing,
                    "answer_instruction": (
                        "Ask only for the missing details. Do not say the policy corpus lacks support."
                    ),
                    "user_message": request.message,
                },
                context=ChatContext(
                    workflow=workflow,
                    employee_id=parsed.employee_id,
                    requested_location_id=parsed.requested_location_id,
                    requested_hours=parsed.requested_hours,
                    create_ticket=parsed.create_ticket,
                ),
            )

        employee_id = parsed.employee_id
        assert employee_id is not None
        structured = WorkflowRequest(
            workflow=workflow,
            employee_id=employee_id,
            requested_location_id=parsed.requested_location_id,
            requested_hours=parsed.requested_hours,
            create_ticket=parsed.create_ticket,
            confirmed=parsed.confirmed,
            user_message=request.message,
        )
        return await self.run(structured, trace=trace)

    @staticmethod
    def _split_locations(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
        supported: list[str] = []
        others: list[str] = []
        for item in payload.get("locations", []):
            if not isinstance(item, dict):
                continue
            label = f"{item.get('region')} ({item.get('location_id')})"
            if item.get("regular_remote_employment_supported"):
                supported.append(label)
            else:
                others.append(label)
        return supported, others

    async def _load_location_register(
        self, trace: list[TraceStep]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        locations = await self._call(trace, AgentState.RETRIEVE, "list_work_locations", {})
        policies = await self._call(
            trace,
            AgentState.RETRIEVE,
            "search_policy_documents",
            {
                "query": (
                    "regular remote employment location register California "
                    "New York Texas Florida London"
                ),
                "top_k": 8,
            },
        )
        return locations, policies

    async def _location_catalog(
        self,
        message: str,
        trace: list[TraceStep],
        employee_id: str | None,
    ) -> WorkflowResult:
        try:
            locations, policies = await self._load_location_register(trace)
        except Exception:
            return WorkflowResult(
                workflow=Workflow.POLICY_QA,
                status="escalated",
                answer="A required HR tool was unavailable. No action was taken; contact HR.",
                trace=trace,
                context=ChatContext(workflow=Workflow.POLICY_QA),
            )
        supported, others = self._split_locations(locations)
        citations = self._citations(policies)
        return await self._finish(
            workflow=Workflow.POLICY_QA,
            status="completed",
            trace=trace,
            evidence={
                "workflow": Workflow.POLICY_QA.value,
                "decision": "policy_guidance",
                "grounded": True,
                "supported_regular_remote_locations": supported,
                "other_register_locations": others,
                "citations": [citation.model_dump() for citation in citations],
                "answer_instruction": (
                    "List the locations where regular remote employment is supported. "
                    "Also name register locations that are not supported. "
                    "Cite the remote-work policy."
                ),
                "user_message": message,
            },
            citations=citations,
            context=ChatContext(
                workflow=Workflow.REMOTE_WORK,
                employee_id=employee_id,
            ),
            terminal_state=AgentState.COMPLETE,
        )

    async def _unknown_location(
        self,
        parsed: ParsedIntent,
        trace: list[TraceStep],
        message: str,
    ) -> WorkflowResult:
        place = parsed.unrecognized_location or "That place"
        try:
            if parsed.employee_id:
                await self._call(
                    trace,
                    AgentState.RETRIEVE,
                    "lookup_employee_profile",
                    {"employee_id": parsed.employee_id},
                )
            locations, policies = await self._load_location_register(trace)
        except Exception:
            return WorkflowResult(
                workflow=Workflow.REMOTE_WORK,
                status="escalated",
                answer="A required HR tool was unavailable. No action was taken; contact HR.",
                trace=trace,
                context=ChatContext(
                    workflow=Workflow.REMOTE_WORK,
                    employee_id=parsed.employee_id,
                ),
            )
        supported, others = self._split_locations(locations)
        citations = self._citations(policies)
        return await self._finish(
            workflow=Workflow.REMOTE_WORK,
            status="escalated",
            trace=trace,
            evidence={
                "workflow": Workflow.REMOTE_WORK.value,
                "decision": "escalate",
                "grounded": True,
                "findings": [
                    {
                        "found": False,
                        "reason": (
                            f"{place} is not in the synthetic location register. "
                            "The request names a place, so this is not a missing location."
                        ),
                    }
                ],
                "supported_regular_remote_locations": supported,
                "other_register_locations": others,
                "citations": [citation.model_dump() for citation in citations],
                "answer_instruction": (
                    f"Tell the user that {place} is not one of the register locations. "
                    "List where regular remote employment is supported, and mention "
                    "register locations that are not supported. Do not ask for a "
                    "location as if the user left it blank."
                ),
                "user_message": message,
            },
            citations=citations,
            context=ChatContext(
                workflow=Workflow.REMOTE_WORK,
                employee_id=parsed.employee_id,
            ),
            terminal_state=AgentState.ESCALATE,
        )

    async def _policy_qa(self, message: str, trace: list[TraceStep]) -> WorkflowResult:
        try:
            policies = await self._call(
                trace,
                AgentState.RETRIEVE,
                "search_policy_documents",
                {"query": message, "top_k": 8},
            )
        except Exception:
            return WorkflowResult(
                workflow=Workflow.POLICY_QA,
                status="escalated",
                answer="A required HR tool was unavailable. No action was taken; contact HR.",
                trace=trace,
                context=ChatContext(workflow=Workflow.POLICY_QA),
            )
        texts = [
            str(item.get("text") or item.get("snippet") or "")
            for item in policies.get("results", [])
            if isinstance(item, dict)
        ]
        grounded = bool(texts) and has_sufficient_evidence(message, texts)
        citations = self._citations(policies) if grounded else []
        status: Literal["completed", "escalated"] = "completed" if grounded else "escalated"
        return await self._finish(
            workflow=Workflow.POLICY_QA,
            status=status,
            trace=trace,
            evidence={
                "workflow": Workflow.POLICY_QA.value,
                "decision": "policy_guidance" if grounded else "unsupported",
                "grounded": grounded,
                "citations": [citation.model_dump() for citation in citations],
                "user_message": message,
                "findings": []
                if grounded
                else [
                    {
                        "found": False,
                        "reason": "The policy corpus does not contain enough support for this question.",
                    }
                ],
            },
            citations=citations,
            context=ChatContext(workflow=Workflow.POLICY_QA),
            terminal_state=AgentState.COMPLETE if grounded else AgentState.ESCALATE,
        )

    async def run(
        self,
        request: WorkflowRequest,
        *,
        trace: list[TraceStep] | None = None,
    ) -> WorkflowResult:
        if trace is None:
            trace = [
                TraceStep(
                    state=AgentState.CLASSIFY,
                    result_summary=(
                        f"Selected {request.workflow.value} workflow from structured input"
                    ),
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
            return await self._finish(
                workflow=request.workflow,
                status="needs_clarification",
                trace=trace,
                evidence={
                    "workflow": request.workflow.value,
                    "missing": [needed],
                    "answer_instruction": (
                        "Ask only for the missing details. Do not say the policy corpus lacks support."
                    ),
                    "user_message": request.user_message,
                },
                context=self._context_for(request),
            )

        employee: dict[str, Any] | None = None
        policies: dict[str, Any] | None = None
        compliance: dict[str, Any] | None = None
        try:
            employee = await self._call(
                trace,
                AgentState.RETRIEVE,
                "lookup_employee_profile",
                {"employee_id": request.employee_id},
            )
            if employee.get("found") is False:
                return await self._finish(
                    workflow=request.workflow,
                    status="needs_clarification",
                    trace=trace,
                    evidence={
                        "workflow": request.workflow.value,
                        "findings": [employee],
                        "missing": ["synthetic employee ID (SYN-####) that exists in the directory"],
                        "answer_instruction": (
                            "Explain that this employee ID was not found and ask for a valid "
                            "SYN-#### ID. Do not say the policy corpus lacks support."
                        ),
                        "user_message": request.user_message,
                    },
                    context=self._context_for(request),
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
                balance = await self._call(
                    trace,
                    AgentState.RETRIEVE,
                    "check_pto_balance",
                    {"employee_id": request.employee_id},
                )
                if balance.get("found") is False:
                    return await self._finish(
                        workflow=request.workflow,
                        status="needs_clarification",
                        trace=trace,
                        evidence={
                            "workflow": request.workflow.value,
                            "employee_name": employee["employee"]["name"],
                            "findings": [balance],
                            "answer_instruction": (
                                "Explain the missing PTO record and ask the user to correct the "
                                "employee ID. Do not say the policy corpus lacks support."
                            ),
                            "user_message": request.user_message,
                        },
                        context=self._context_for(request),
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
                context=self._context_for(request),
            )

        assert employee is not None and policies is not None and compliance is not None
        citations = self._citations(policies)
        decision = compliance.get("decision")
        if decision == "needs_clarification":
            status: Literal[
                "completed", "needs_clarification", "confirmation_required", "escalated"
            ] = "needs_clarification"
        elif decision == "eligible_for_review":
            status = "completed"
        else:
            status = "escalated"
        evidence = {
            "workflow": request.workflow.value,
            "employee_name": employee["employee"]["name"],
            "decision": decision,
            "checks": compliance.get("checks", []),
            "findings": [compliance] if compliance.get("found") is False else [],
            "citations": [citation.model_dump() for citation in citations],
            "grounded": True,
            "user_message": request.user_message,
            "reason": compliance.get("reason"),
        }
        if status == "needs_clarification":
            return await self._finish(
                workflow=request.workflow,
                status=status,
                trace=trace,
                evidence=evidence,
                citations=citations,
                context=self._context_for(request),
            )

        answer = await self._speak(trace, evidence, citations)
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
                    context=self._context_for(request),
                )
            if mock_action.get("found") is False or mock_action.get("rejected"):
                reason = str(mock_action.get("reason") or "The mock ticket was not created.")
                if reason not in answer:
                    answer = f"{answer} {reason}"
                return WorkflowResult(
                    workflow=request.workflow,
                    status="needs_clarification" if mock_action.get("found") is False else "escalated",
                    answer=answer,
                    citations=citations,
                    trace=trace,
                    mock_action=mock_action,
                    context=self._context_for(request),
                )
            if not mock_action.get("created"):
                trace[-1].status = "confirmation_required"
                suffix = " Confirm explicitly if you want me to create the displayed mock ticket."
                if suffix.strip() not in answer:
                    answer += suffix
                return WorkflowResult(
                    workflow=request.workflow,
                    status="confirmation_required",
                    answer=answer,
                    citations=citations,
                    trace=trace,
                    mock_action=mock_action,
                    context=self._context_for(request, awaiting_confirmation=True),
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
            context=self._context_for(request),
        )
