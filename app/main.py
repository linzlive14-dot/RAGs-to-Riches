import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request, status
from pydantic import BaseModel

from app.agent import AgentState, ChatRequest, HRAgent, ToolClient, WorkflowResult
from app.config import get_settings
from app.llm import (
    OpenRouterClient,
    OpenRouterIntentExtractor,
    OpenRouterSynthesizer,
)
from hr_mcp.client import HRMCPClient, MCPSessionManager
from rag.index import PolicyIndex, configured_mode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEALTH_TIMEOUT_SECONDS = 10.0


class MCPHealth(BaseModel):
    status: Literal["ok", "unavailable"]
    transport: str = HRMCPClient.transport
    tool_count: int = 0
    tools: list[str] = []
    error: str | None = None


class IndexHealth(BaseModel):
    status: Literal["ok", "missing"]
    chunk_count: int | None = None
    retrieval_mode: str | None = None
    embedding_model: str | None = None


class LLMHealth(BaseModel):
    configured: bool
    model: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    environment: str
    mcp: MCPHealth
    index: IndexHealth
    llm: LLMHealth


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    manager = MCPSessionManager()
    await manager.start()
    app.state.mcp = manager
    try:
        yield
    finally:
        app.state.mcp = None
        await manager.stop()


app = FastAPI(title=settings.app_name, version="0.2.0", lifespan=lifespan)


def _manager(request: Request) -> MCPSessionManager | None:
    return getattr(request.app.state, "mcp", None)


def _index_health() -> IndexHealth:
    path = Path(
        os.environ.get("HR_POLICY_INDEX", PROJECT_ROOT / "data" / "policy_index.sqlite3")
    )
    index = PolicyIndex(path)
    try:
        manifest = index.manifest()
    except (FileNotFoundError, RuntimeError):
        return IndexHealth(status="missing")
    count = manifest.get("chunk_count")
    configuration = manifest.get("configuration") or {}
    vectors = index.has_vectors()
    return IndexHealth(
        status="ok",
        chunk_count=count if isinstance(count, int) else None,
        retrieval_mode=(configured_mode() or "hybrid") if vectors else "bm25",
        embedding_model=configuration.get("embedding_model") if vectors else None,
    )


async def _mcp_health(manager: MCPSessionManager | None) -> MCPHealth:
    async def discover(client: HRMCPClient) -> list[str]:
        return await client.list_tools()

    try:
        if manager is not None:
            client = await asyncio.wait_for(manager.client(), HEALTH_TIMEOUT_SECONDS)
            tools = await asyncio.wait_for(discover(client), HEALTH_TIMEOUT_SECONDS)
        else:
            async with HRMCPClient() as client:
                tools = await asyncio.wait_for(discover(client), HEALTH_TIMEOUT_SECONDS)
    except Exception as error:
        return MCPHealth(status="unavailable", error=f"{type(error).__name__}: {error}")
    return MCPHealth(status="ok", tool_count=len(tools), tools=sorted(tools))


@app.get("/health", response_model=HealthResponse, tags=["operations"])
async def health(request: Request) -> HealthResponse:
    """Report API readiness, MCP connectivity, index state, and LLM configuration."""

    mcp = await _mcp_health(_manager(request))
    index = _index_health()
    return HealthResponse(
        status="ok" if mcp.status == "ok" and index.status == "ok" else "degraded",
        service=settings.app_name,
        environment=settings.app_env,
        mcp=mcp,
        index=index,
        llm=LLMHealth(
            configured=settings.openrouter_api_key is not None,
            model=settings.openrouter_model if settings.openrouter_api_key else None,
        ),
    )


class _UnavailableTools:
    """Stand-in client that lets the agent report an MCP outage as an escalation."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def list_tool_schemas(self) -> list[dict[str, Any]]:
        raise RuntimeError(self.reason)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(self.reason)


async def _live_tools(manager: MCPSessionManager) -> ToolClient:
    try:
        return await manager.client()
    except RuntimeError as error:
        return _UnavailableTools(str(error))


def _discovery_failed(result: WorkflowResult) -> bool:
    return any(
        step.state == AgentState.DISCOVER and step.status == "error" for step in result.trace
    )


@app.post(
    "/chat",
    response_model=WorkflowResult,
    status_code=status.HTTP_200_OK,
    tags=["assistant"],
)
async def chat(chat_request: ChatRequest, request: Request) -> WorkflowResult:
    """Answer a natural-language HR question through the MCP tool boundary."""

    agent_options: dict[str, Any] = {}
    if settings.openrouter_api_key is not None:
        client = OpenRouterClient(
            settings.openrouter_api_key.get_secret_value(),
            model=settings.openrouter_model,
            base_url=settings.openrouter_base_url,
            timeout_seconds=settings.openrouter_timeout_seconds,
        )
        agent_options = {
            "synthesizer": OpenRouterSynthesizer(client),
            "intent_extractor": OpenRouterIntentExtractor(client),
        }

    run: Callable[[ToolClient], Awaitable[WorkflowResult]] = lambda tools: HRAgent(
        tools, **agent_options
    ).chat(chat_request)

    manager = _manager(request)
    if manager is None:
        async with HRMCPClient() as tools:
            return await run(tools)

    result = await run(await _live_tools(manager))
    if _discovery_failed(result):
        await manager.restart()
        result = await run(await _live_tools(manager))
    return result
