"""Small MCP client adapter used by the explicit agent orchestrator."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_CALL_TIMEOUT = timedelta(seconds=30)


class HRMCPClient:
    """Own one stdio MCP session and expose JSON-shaped tool results."""

    transport = "stdio"

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
            ClientSession(read_stream, write_stream, read_timeout_seconds=TOOL_CALL_TIMEOUT)
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

    async def list_tool_schemas(self) -> list[dict[str, Any]]:
        """Return each tool's name, description, and JSON input schema."""

        response = await self._require_session().list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

    async def list_tools(self) -> list[str]:
        return [tool["name"] for tool in await self.list_tool_schemas()]

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


class MCPSessionManager:
    """Keep one MCP server process alive for the API and restart it on demand.

    A background task owns the stdio context, because anyio requires the
    context to be entered and exited by the same task.
    """

    def __init__(self, *, env: dict[str, str] | None = None, startup_timeout: float = 30.0) -> None:
        self._env = env
        self._startup_timeout = startup_timeout
        self._client: HRMCPClient | None = None
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._cycle = asyncio.Event()
        self._stopping = False
        self.last_error: str | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        await self._wait_ready()

    async def _wait_ready(self) -> None:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=self._startup_timeout)
        except TimeoutError:
            self.last_error = "MCP server did not start in time"

    async def _run(self) -> None:
        while not self._stopping:
            self._cycle.clear()
            try:
                async with HRMCPClient(env=self._env) as client:
                    self._client = client
                    self.last_error = None
                    self._ready.set()
                    await self._cycle.wait()
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                self._ready.set()
                if not self._stopping:
                    await asyncio.sleep(1.0)
            finally:
                self._client = None
                if not self._stopping:
                    self._ready.clear()

    async def client(self) -> HRMCPClient:
        """Return the live client, starting the server if needed."""

        if self._task is None or self._task.done():
            self._task = None
            self._ready.clear()
            await self.start()
        elif self._client is None:
            await self._wait_ready()
        if self._client is None:
            raise RuntimeError(self.last_error or "MCP server is unavailable")
        return self._client

    async def restart(self) -> None:
        self._ready.clear()
        self._cycle.set()
        await self._wait_ready()

    async def stop(self) -> None:
        self._stopping = True
        self._cycle.set()
        if self._task is not None:
            await self._task
            self._task = None
