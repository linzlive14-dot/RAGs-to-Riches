from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import get_settings


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
