from typing import Literal

from fastapi import FastAPI, status
from pydantic import BaseModel

from app.agent import ChatRequest, HRAgent, WorkflowResult
from app.config import get_settings
from app.llm import (
    OpenRouterClient,
    OpenRouterIntentExtractor,
    OpenRouterSynthesizer,
)
from hr_mcp.client import HRMCPClient


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    environment: str


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    """Report that the API process is ready to accept requests."""

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.app_env,
    )


@app.post(
    "/chat",
    response_model=WorkflowResult,
    status_code=status.HTTP_200_OK,
    tags=["assistant"],
)
async def chat(request: ChatRequest) -> WorkflowResult:
    """Answer a natural-language HR question through the MCP tool boundary."""

    synthesizer = None
    intent_extractor = None
    if settings.openrouter_api_key is not None:
        client = OpenRouterClient(
            settings.openrouter_api_key.get_secret_value(),
            model=settings.openrouter_model,
            base_url=settings.openrouter_base_url,
            timeout_seconds=settings.openrouter_timeout_seconds,
        )
        synthesizer = OpenRouterSynthesizer(client)
        intent_extractor = OpenRouterIntentExtractor(client)
    async with HRMCPClient() as tools:
        return await HRAgent(
            tools,
            synthesizer=synthesizer,
            intent_extractor=intent_extractor,
        ).chat(request)
