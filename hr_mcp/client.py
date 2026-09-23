"""Small MCP client adapter used by the explicit agent orchestrator."""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HRMCPClient:
    """Own one stdio MCP session and expose JSON-shaped tool results."""

    def __init__(self, *, env: dict[str, str] | None = None) -> None:
        self._extra_env = env or {}
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "HRMCPClient":
        self._stack = AsyncExitStack()
        environment = os.environ.copy()
        environment.update(self._extra_env)
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "hr_mcp.server"],
            cwd=PROJECT_ROOT,
            env=environment,
        )
        read_stream, write_stream = await self._stack.enter_async_context(stdio_client(parameters))
        self._session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._session = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("Use HRMCPClient as an async context manager")
        return self._session

    async def list_tools(self) -> list[str]:
        response = await self._require_session().list_tools()
        return [tool.name for tool in response.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        response = await self._require_session().call_tool(name, arguments)
        if response.isError:
            message = "MCP tool call failed"
            if response.content and hasattr(response.content[0], "text"):
                message = response.content[0].text
            raise RuntimeError(message)
        if response.structuredContent is not None:
            payload = response.structuredContent
            # FastMCP wraps structured output in {"result": ...} in some SDK versions.
            if set(payload) == {"result"} and isinstance(payload["result"], dict):
                return payload["result"]
            return payload
        if response.content and hasattr(response.content[0], "text"):
            decoded = json.loads(response.content[0].text)
            if isinstance(decoded, dict):
                return decoded
        raise RuntimeError(f"Tool {name!r} returned no JSON object")
