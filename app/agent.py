"""Explicit, inspectable orchestration for conversational HR workflows."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, field_validator

from app.intent import ParsedIntent, asks_for_location_list, is_unsafe_message, resolve_intent
from rag.answering import REFUSAL
from rag.guardrails import has_sufficient_evidence


class ToolClient(Protocol):
    async def list_tool_schemas(self) -> list[dict[str, Any]]: ...

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
    BENEFITS = "benefits"
    POLICY_QA = "policy_qa"
    UNSUPPORTED = "unsupported"


SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

STRUCTURED_WORKFLOWS = {Workflow.REMOTE_WORK, Workflow.PTO, Workflow.BENEFITS}

RECORD_TOOLS = {Workflow.PTO: "check_pto_balance", Workflow.BENEFITS: "lookup_benefits_status"}

CHECK_DESCRIPTIONS = {
    "180_day_tenure": "at least 180 days of employment",
    "remote_capable_role": "a role classified as fully remote-capable",
    "performance": "a Meets Expectations or higher rating",
    "no_final_warning": "no active final written warning",
    "supported_location": "a location supported for regular remote work",
    "domestic_location": "a domestic location",
    "covered_employee": "regular full- or part-time status",
    "balance_or_possible_first_year_exception": "enough available PTO",
    "benefits_eligible_schedule": "a regular schedule of 30 or more hours",
    "enrollment_complete_or_window_open": "completed or still-open initial enrollment",
}


LOCATION_QUERY = (
    "regular remote employment location register California New York Texas Florida London"
)
WORKFLOW_LABELS = {"remote_work": "remote work", "pto": "PTO", "benefits": "benefits"}
TERM_RE = re.compile(r"[a-z0-9]+")


FAILED_CHECK_SEARCHES = {
    "180_day_tenure": (
        "remote-work.md",
        "fully remote arrangement requires 180 calendar days waiting period waived",
    ),
    "remote_capable_role": (
        "remote-work.md",
        "role must be classified as remote-capable in the job profile",
    ),
    "performance": ("remote-work.md", "current performance rating Meets Expectations or higher"),
    "no_final_warning": ("remote-work.md", "no active final written warning"),
    "supported_location": (
        "remote-work.md",
        "regular remote employment supported only locations listed active register",
    ),
    "domestic_location": (
        "remote-work.md",
        "international remote work prohibited unless Legal Information Security Payroll written approval",
    ),
    "covered_employee": ("paid-time-off.md", "covered employees regular full-time part-time accrue"),
    "balance_or_possible_first_year_exception": (
        "paid-time-off.md",
        "negative PTO first-year planned absence manager HR approval",
    ),
    "benefits_eligible_schedule": (
        "benefits.md",
        "scheduled 20 to 29 hours employee assistance program voluntary benefits",
    ),
    "enrollment_complete_or_window_open": (
        "benefits.md",
        "no election made employer-paid default wait annual enrollment qualifying life event",
    ),
}


class RetrievalPlan(BaseModel):
    """Targeted (source, query, top_k) searches plus the section the answer rests on."""

    searches: list[tuple[str, str, int]]
    section: tuple[str, str]


def supporting_sentence(text: str, query: str, *, max_characters: int = 300) -> str:
    """Return the sentence in a chunk that shares the most terms with the query."""

    terms = {term for term in TERM_RE.findall(query.lower()) if len(term) >= 4}
    sentences = [part.strip() for part in SENTENCE_END.split(text) if part.strip()]
    if not sentences:
        return text[:max_characters].rstrip()
    best = max(
        enumerate(sentences),
        key=lambda item: (len(terms & set(TERM_RE.findall(item[1].lower()))), -item[0]),
    )[1]
    return best if len(best) <= max_characters else best[: max_characters - 1].rstrip() + "…"


class AgentState(StrEnum):
    CLASSIFY = "classify"
    DISCOVER = "discover"
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
        if value not in STRUCTURED_WORKFLOWS:
            raise ValueError("workflow must be remote_work, pto, or benefits")
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
    output_preview: dict[str, Any] = Field(default_factory=dict)
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
        disabled_tools: set[str] | None = None,
    ) -> None:
        self.tools = tools
        self.synthesizer = synthesizer
        self.intent_extractor = intent_extractor
        self.disabled_tools = frozenset(disabled_tools or ())
        self._discovered: list[str] | None = None

    async def _discover(self, trace: list[TraceStep]) -> set[str]:
        """List the MCP server's tools once per agent and record them in the trace."""

        first_time = self._discovered is None
        if self._discovered is None:
            try:
                schemas = await self.tools.list_tool_schemas()
            except Exception as error:
                trace.append(
                    TraceStep(
                        state=AgentState.DISCOVER,
                        status="error",
                        result_summary=f"{type(error).__name__}: MCP tool discovery failed",
                    )
                )
                raise
            self._discovered = sorted(schema["name"] for schema in schemas)
        available = [name for name in self._discovered if name not in self.disabled_tools]
        verb = "Discovered" if first_time else "Using"
        trace.append(
            TraceStep(
                state=AgentState.DISCOVER,
                result_summary=f"{verb} {len(available)} MCP tools from the HR server",
                output_preview={"tools": available},
            )
        )
        return set(available)

    async def _missing_tools(
        self, trace: list[TraceStep], required: list[str]
    ) -> list[str] | None:
        """Return required tools the server does not expose, or None if discovery failed."""

        try:
            available = await self._discover(trace)
        except Exception:
            return None
        return [name for name in required if name not in available]

    def _tools_unavailable(
        self,
        workflow: Workflow,
        trace: list[TraceStep],
        missing: list[str] | None,
        context: ChatContext | None = None,
    ) -> WorkflowResult:
        if missing is None:
            answer = "The HR tool server is unavailable. No action was taken; please try again or contact HR."
            summary = "Escalated because MCP tool discovery failed"
        else:
            answer = (
                "A required HR tool is unavailable ("
                + ", ".join(missing)
                + "). No action was taken; please contact HR."
            )
            summary = "Escalated because required MCP tools are missing: " + ", ".join(missing)
        trace.append(TraceStep(state=AgentState.ESCALATE, status="skipped", result_summary=summary))
        return WorkflowResult(
            workflow=workflow,
            status="escalated",
            answer=answer,
            trace=trace,
            context=context or ChatContext(workflow=workflow),
        )

    @staticmethod
    def _preview(tool: str, result: dict[str, Any]) -> dict[str, Any]:
        """Keep the decision-relevant fields of a tool result for the trace."""

        if result.get("found") is False:
            return {"found": False, "reason": result.get("reason")}
        if tool == "lookup_employee_profile":
            employee = result.get("employee", {})
            keys = (
                "employee_id",
                "role",
                "employment_type",
                "hire_date",
                "work_location_id",
                "remote_capability",
                "performance_status",
                "active_final_warning",
            )
            return {key: employee.get(key) for key in keys}
        if tool == "check_pto_balance":
            balance = result.get("balance", {})
            return {
                key: balance.get(key)
                for key in ("available_hours", "accrued_hours", "approved_future_hours")
            }
        if tool == "lookup_benefits_status":
            status = result.get("benefits_status", {})
            return {
                key: status.get(key)
                for key in ("eligibility", "medical", "coverage_effective_date", "next_action")
            }
        if tool == "search_policy_documents":
            return {
                "results": [
                    {
                        "source": item.get("source"),
                        "section": item.get("section"),
                        "score": item.get("score"),
                    }
                    for item in result.get("results", [])[:5]
                    if isinstance(item, dict)
                ]
            }
        if tool == "get_policy_section":
            return {
                "source": result.get("source"),
                "section": result.get("section"),
                "characters": len(str(result.get("text", ""))),
            }
        if tool == "list_work_locations":
            locations = [item for item in result.get("locations", []) if isinstance(item, dict)]
            return {
                "supported": [
                    item["location_id"]
                    for item in locations
                    if item.get("regular_remote_employment_supported")
                ],
                "not_supported": [
                    item["location_id"]
                    for item in locations
                    if not item.get("regular_remote_employment_supported")
                ],
            }
        if tool == "check_policy_compliance":
            return {
                "decision": result.get("decision"),
                "checks": {
                    check["name"]: check["passed"]
                    for check in result.get("checks", [])
                    if isinstance(check, dict)
                },
            }
        if tool == "create_mock_hr_ticket":
            if result.get("created"):
                return {"created": True, "ticket_id": result.get("ticket", {}).get("ticket_id")}
            return {
                "created": False,
                "confirmation_required": bool(result.get("confirmation_required")),
                "proposed_action": result.get("proposed_action"),
            }
        return {}

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
                output_preview=self._preview(tool, result),
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
        if tool == "lookup_employee_profile":
            employee = result.get("employee", {})
            return f"Found {employee.get('employee_id')}: {employee.get('role')}"
        if tool == "check_pto_balance":
            return f"Available PTO: {result.get('balance', {}).get('available_hours')} hours"
        if tool == "lookup_benefits_status":
            return f"Benefits eligibility: {result.get('benefits_status', {}).get('eligibility')}"
        if tool == "get_policy_section":
            return f"Read section '{result.get('section')}' of {result.get('source')}"
        return f"{tool} returned a synthetic record"

    @staticmethod
    def _retrieval_plan(
        request: WorkflowRequest,
        employee: dict[str, Any],
        compliance: dict[str, Any],
    ) -> RetrievalPlan:
        """Choose source-filtered searches from the workflow and the compliance outcome.

        Each failed check gets its own search so the answer can cite the rule
        behind it; the remaining searches cover the workflow's related policies.
        """

        searches: list[tuple[str, str, int]] = [
            (*FAILED_CHECK_SEARCHES[check["name"]], 1)
            for check in compliance.get("checks", [])
            if isinstance(check, dict)
            and not check.get("passed", True)
            and check.get("name") in FAILED_CHECK_SEARCHES
        ]
        if request.workflow == Workflow.REMOTE_WORK:
            searches += [
                (
                    "remote-work.md",
                    "fully remote arrangement requires 180 calendar days remote-capable performance",
                    1,
                ),
                ("information-security.md", "remote workers private network VPN devices", 1),
            ]
            if (compliance.get("details") or {}).get("location_change"):
                searches.append(
                    ("payroll-and-working-time.md", "address changes tax forms benefits", 1)
                )
            section = ("remote-work.md", "Request and approval process")
        elif request.workflow == Workflow.PTO:
            searches += [
                (
                    "paid-time-off.md",
                    "requests scheduled workdays submitted at least calendar days in advance",
                    1,
                ),
                ("paid-time-off.md", "manager approves requests based on available balance", 1),
            ]
            section = ("paid-time-off.md", "Requesting planned time")
        else:
            searches.append(
                (
                    "benefits.md",
                    "eligible medical coverage scheduled 30 hours initial enrollment hire date",
                    2,
                )
            )
            section = ("benefits.md", "Enrollment")
        return RetrievalPlan(searches=searches, section=section)

    @staticmethod
    def _citations(
        policy_result: dict[str, Any],
        *,
        limit: int = 3,
        distinct: Literal["source", "snippet"] = "source",
        query: str = "",
    ) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[str] = set()
        for result in policy_result.get("results", []):
            snippet = str(result.get("passage") or "") or supporting_sentence(
                str(result.get("text") or result["snippet"]),
                str(result.get("query") or query),
            )
            key = result["source"] if distinct == "source" else snippet
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    citation_id=f"P{len(citations) + 1}",
                    source=result["source"],
                    section=result["section"],
                    snippet=snippet,
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
        if decision in {"eligible_for_review", "escalate"}:
            return HRAgent._labelled_answer(
                HRAgent._policy_lines(citations),
                HRAgent._records_line(evidence),
                HRAgent._next_step(evidence),
            )
        supported = evidence.get("supported_regular_remote_locations") or []
        if supported:
            listed = ", ".join(str(item) for item in supported)
            return (
                f"Regular remote employment is supported in {listed}. {references}"
            ).strip()
        if citations:
            return HRAgent._labelled_answer(
                HRAgent._policy_lines(citations),
                None,
                "Contact HR for a decision about your specific situation.",
            )
        return "I need a more specific HR policy question before I can help."

    @staticmethod
    def _labelled_answer(policy: str, records: str | None, next_step: str | None) -> str:
        parts = [f"**Policy:** {policy}"]
        if records:
            parts.append(f"**Your records:** {records}")
        if next_step:
            parts.append(f"**Recommended next step:** {next_step}")
        return "\n\n".join(parts)

    @staticmethod
    def _policy_lines(citations: list[Citation]) -> str:
        return " ".join(f"{citation.snippet} [{citation.citation_id}]" for citation in citations)

    @staticmethod
    def _records_line(evidence: dict[str, Any]) -> str:
        name = evidence.get("employee_name") or "The employee"
        workflow = WORKFLOW_LABELS.get(str(evidence.get("workflow")), "this")
        checks = [check for check in evidence.get("checks", []) if isinstance(check, dict)]
        records = evidence.get("records") or {}
        if evidence.get("decision") == "eligible_for_review" and evidence.get("workflow") == "benefits":
            return (
                f"{name} is benefits-eligible with medical coverage "
                f"{str(records.get('medical', 'on file')).replace('_', ' ')}, effective "
                f"{records.get('coverage_effective_date')}."
            )
        if evidence.get("decision") == "eligible_for_review":
            if evidence.get("workflow") == "pto":
                name = f"{name} has {records.get('available_hours')} available hours and"
            met = ", ".join(CHECK_DESCRIPTIONS.get(c["name"], c["name"]) for c in checks)
            return (
                f"{name} meets the automated prerequisites for {workflow} review ({met}). "
                "This is not an approval; manager/HR approval is still required."
            )
        failed = [
            CHECK_DESCRIPTIONS.get(c["name"], c["name"]) for c in checks if not c.get("passed", True)
        ]
        return (
            f"The {workflow} request needs HR review because these checks did not pass: "
            f"{', '.join(failed) or 'manual review required'}."
        )

    @staticmethod
    def _next_step(evidence: dict[str, Any]) -> str:
        workflow = evidence.get("workflow")
        eligible = evidence.get("decision") == "eligible_for_review"
        details = evidence.get("details") or {}
        if workflow == Workflow.PTO.value:
            if eligible:
                notice = details.get("required_notice_calendar_days")
                return (
                    f"Submit the request in the HR system at least {notice} calendar days ahead. "
                    "It is not approved until the HR system shows your manager's approval."
                )
            return "Talk with your manager and HR before making plans that depend on this time off."
        if workflow == Workflow.BENEFITS.value:
            if eligible:
                return "Review your elections in the benefits portal and report any error to Benefits."
            failed = {
                check["name"]
                for check in evidence.get("checks", [])
                if isinstance(check, dict) and not check.get("passed", True)
            }
            if "benefits_eligible_schedule" in failed:
                return "Contact Benefits about the employee assistance program and voluntary benefits."
            deadline = details.get("initial_enrollment_deadline")
            return (
                f"The initial enrollment deadline ({deadline}) has passed. Contact Benefits about "
                "annual enrollment or a qualifying life event."
            )
        if eligible:
            return (
                "Submit a remote work request in the HR system with your schedule, work address, "
                "start date, and coverage plan, and wait for a written decision."
            )
        return "Talk with your manager or HR before making plans; HR can explain your options."

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

        if len(parsed.candidates) > 1:
            options = " or ".join(WORKFLOW_LABELS[name] for name in parsed.candidates)
            trace.append(
                TraceStep(
                    state=AgentState.ESCALATE,
                    status="skipped",
                    result_summary=f"Clarification required: the message mentions {options}",
                )
            )
            return await self._finish(
                workflow=Workflow.POLICY_QA,
                status="needs_clarification",
                trace=trace,
                evidence={
                    "workflow": Workflow.POLICY_QA.value,
                    "decision": "needs_clarification",
                    "reason": (
                        f"Your message mentions {options}. Which one should I check first? "
                        "I'll handle one request at a time."
                    ),
                    "answer_instruction": "Ask which request to handle first. Do not answer either.",
                    "user_message": request.message,
                },
                context=ChatContext(employee_id=parsed.employee_id),
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
            {"query": LOCATION_QUERY, "top_k": 8},
        )
        return locations, policies

    async def _location_catalog(
        self,
        message: str,
        trace: list[TraceStep],
        employee_id: str | None,
    ) -> WorkflowResult:
        missing_tools = await self._missing_tools(
            trace, ["list_work_locations", "search_policy_documents"]
        )
        if missing_tools != []:
            return self._tools_unavailable(Workflow.POLICY_QA, trace, missing_tools)
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
        citations = self._citations(policies, query=LOCATION_QUERY)
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
        required = ["list_work_locations", "search_policy_documents"]
        if parsed.employee_id:
            required.insert(0, "lookup_employee_profile")
        missing_tools = await self._missing_tools(trace, required)
        if missing_tools != []:
            return self._tools_unavailable(Workflow.REMOTE_WORK, trace, missing_tools)
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
        citations = self._citations(policies, query=LOCATION_QUERY)
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
        missing_tools = await self._missing_tools(trace, ["search_policy_documents"])
        if missing_tools != []:
            return self._tools_unavailable(Workflow.POLICY_QA, trace, missing_tools)
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
        items = [item for item in policies.get("results", []) if isinstance(item, dict)]
        texts = [str(item.get("text") or item.get("snippet") or "") for item in items]
        similarities = [
            float(item["similarity"]) for item in items if item.get("similarity") is not None
        ]
        grounded = bool(texts) and has_sufficient_evidence(
            message, texts, similarities=similarities
        )
        by_passage = {
            **policies,
            "results": sorted(
                items, key=lambda item: -float(item.get("passage_similarity") or 0.0)
            ),
        }
        citations = self._citations(by_passage, query=message) if grounded else []
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

        required = [
            "lookup_employee_profile",
            "search_policy_documents",
            "get_policy_section",
            "check_policy_compliance",
        ]
        if request.workflow in RECORD_TOOLS:
            required.insert(1, RECORD_TOOLS[request.workflow])
        if request.create_ticket:
            required.append("create_mock_hr_ticket")
        missing_tools = await self._missing_tools(trace, required)
        if missing_tools != []:
            return self._tools_unavailable(
                request.workflow, trace, missing_tools, self._context_for(request)
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
            record: dict[str, Any] | None = None
            record_tool = RECORD_TOOLS.get(request.workflow)
            if record_tool is not None:
                record = await self._call(
                    trace,
                    AgentState.RETRIEVE,
                    record_tool,
                    {"employee_id": request.employee_id},
                )
                if record.get("found") is False:
                    return await self._finish(
                        workflow=request.workflow,
                        status="needs_clarification",
                        trace=trace,
                        evidence={
                            "workflow": request.workflow.value,
                            "employee_name": employee["employee"]["name"],
                            "findings": [record],
                            "answer_instruction": (
                                "Explain which synthetic record is missing and ask the user to "
                                "correct the employee ID. Do not say the policy corpus lacks support."
                            ),
                            "user_message": request.user_message,
                        },
                        context=self._context_for(request),
                    )
            compliance_arguments: dict[str, Any] = {
                "workflow": request.workflow.value,
                "employee_id": request.employee_id,
            }
            if request.workflow == Workflow.REMOTE_WORK:
                compliance_arguments["requested_location_id"] = request.requested_location_id
            if request.workflow == Workflow.PTO:
                compliance_arguments["requested_hours"] = request.requested_hours

            compliance = await self._call(
                trace,
                AgentState.VALIDATE,
                "check_policy_compliance",
                compliance_arguments,
            )
            plan = self._retrieval_plan(request, employee["employee"], compliance)
            ranked: list[list[dict[str, Any]]] = []
            for source, query, top_k in plan.searches:
                found = await self._call(
                    trace,
                    AgentState.RETRIEVE,
                    "search_policy_documents",
                    {"query": query, "top_k": top_k, "sources": [source]},
                )
                ranked.append([{**item, "query": query} for item in found.get("results", [])])
            policies = {
                "results": [
                    group[rank]
                    for rank in range(max((len(group) for group in ranked), default=0))
                    for group in ranked
                    if rank < len(group)
                ]
            }
            section = await self._call(
                trace,
                AgentState.RETRIEVE,
                "get_policy_section",
                {"source": plan.section[0], "section": plan.section[1]},
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
        citations = self._citations(policies, distinct="snippet", limit=5)
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
            "employee": self._preview("lookup_employee_profile", employee),
            "records": self._preview(record_tool, record) if record_tool and record else {},
            "decision": decision,
            "checks": compliance.get("checks", []),
            "details": compliance.get("details", {}),
            "as_of": compliance.get("as_of"),
            "findings": [compliance] if compliance.get("found") is False else [],
            "citations": [citation.model_dump() for citation in citations],
            "policy_section": {
                "source": section.get("source"),
                "section": section.get("section"),
                "text": str(section.get("text", ""))[:1500],
            }
            if section.get("found")
            else None,
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
