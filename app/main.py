from typing import Literal

from fastapi import FastAPI, status
from pydantic import BaseModel

from app.agent import HRAgent, WorkflowRequest, WorkflowResult
from app.config import get_settings
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
async def chat(request: WorkflowRequest) -> WorkflowResult:
    """Run one validated HR workflow through the real MCP client boundary."""

    async with HRMCPClient() as tools:
        return await HRAgent(tools).run(request)
