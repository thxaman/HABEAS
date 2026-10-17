"""MCP stdio client helper for the HABEAS investigation tools (PLAN.md §3).

The investigation agent reaches the graph ONLY through this client: it spawns
``python -m src.tools.server``, speaks the MCP protocol over stdio, discovers
tools, and calls them. Nothing here imports the graph driver for tool calls.

Design points:
- deterministic startup: explicit command/args/cwd/env, so the child always
  resolves the same interpreter, repo root, and graph name;
- clear failures: startup/initialize problems raise :class:`MCPClientError`
  with the child's own stderr tail included;
- clean shutdown: the stdio_client/ClientSession context managers terminate
  and reap the child process on both success and failure;
- envelopes are preserved verbatim: the full Phase 2 evidence envelope comes
  back as parsed JSON, never reduced to prose.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from src.config import Settings, get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_MODULE = "src.tools.server"

DEFAULT_STARTUP_TIMEOUT_S = 30.0
DEFAULT_CALL_TIMEOUT_S = 60.0


class MCPClientError(RuntimeError):
    """The MCP server could not be started, initialized, or called."""


@dataclass
class ToolCallResult:
    """One MCP tool call result, envelope-preserving."""

    tool: str
    ok: bool
    envelope: dict[str, Any] | None
    text: str
    is_error: bool
    error: str | None = None

    @property
    def query_evidence(self) -> list[dict[str, Any]]:
        return list((self.envelope or {}).get("query_evidence") or [])

    @property
    def interpretation(self) -> str:
        return str((self.envelope or {}).get("interpretation") or "")


def server_parameters(
    settings: Settings | None = None,
    *,
    graph_name: str | None = None,
    args: list[str] | None = None,
    command: str | None = None,
) -> StdioServerParameters:
    """Build deterministic child-process parameters.

    ``graph_name``/``args``/``command`` are overridable so tests can point the
    client at a controlled graph or at a deliberately broken server.
    """
    settings = settings or get_settings()
    env = dict(os.environ)
    env.update({
        "GRAPH_NAME": graph_name or settings.graph_name,
        "FALKORDB_HOST": settings.falkordb_host,
        "FALKORDB_PORT": str(settings.falkordb_port),
        "FALKORDB_QUERY_TIMEOUT_MS": str(settings.falkordb_query_timeout_ms),
        # The child must inherit the sandbox configuration, otherwise the
        # action tools would never be registered and the parent's
        # HABEAS_ENABLE_ACTIONS intent would be silently lost.
        "HABEAS_DEMO_MODE": "true" if settings.habeas_demo_mode else "false",
        "HABEAS_ENABLE_ACTIONS": "true" if settings.habeas_enable_actions else "false",
        "LLM_MODE": settings.llm_mode,
        "TOOL_BUDGET": str(settings.tool_budget),
        "PYTHONPATH": str(REPO_ROOT),
        # keep the child's diagnostics on stderr and unbuffered
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    return StdioServerParameters(
        command=command or sys.executable,
        args=args if args is not None else ["-m", SERVER_MODULE],
        env=env,
        cwd=str(REPO_ROOT),
    )


@asynccontextmanager
async def open_session(
    settings: Settings | None = None,
    *,
    graph_name: str | None = None,
    params: StdioServerParameters | None = None,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
) -> AsyncIterator[ClientSession]:
    """Start the MCP server over stdio and yield an initialized session.

    Terminates and reaps the child process on exit, whether the body
    succeeded or raised.
    """
    served = params or server_parameters(settings, graph_name=graph_name)
    # Only the startup/handshake phase is relabelled as a startup failure: once
    # the session is live, an error raised by the caller's own code must
    # propagate unchanged instead of masquerading as "could not start server".
    initialized = False
    try:
        async with stdio_client(served) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                try:
                    await asyncio.wait_for(
                        session.initialize(), timeout=startup_timeout_s
                    )
                except asyncio.TimeoutError as exc:
                    raise MCPClientError(
                        f"HABEAS MCP server did not initialize within "
                        f"{startup_timeout_s:.0f}s (command: {served.command} "
                        f"{' '.join(served.args or [])})"
                    ) from exc
                except Exception as exc:
                    raise MCPClientError(
                        f"HABEAS MCP server failed to initialize "
                        f"({type(exc).__name__}: {exc}); command: {served.command} "
                        f"{' '.join(served.args or [])}"
                    ) from exc
                initialized = True
                yield session
    except MCPClientError:
        raise
    except BaseExceptionGroup as exc:  # anyio may group child-process failures
        if initialized:
            raise
        raise MCPClientError(
            f"could not start HABEAS MCP server ({exc!r}); command: "
            f"{served.command} {' '.join(served.args or [])}"
        ) from exc
    except Exception as exc:
        if initialized:
            raise
        raise MCPClientError(
            f"could not start HABEAS MCP server "
            f"({type(exc).__name__}: {exc}); command: {served.command} "
            f"{' '.join(served.args or [])}"
        ) from exc


async def list_tools_async(
    settings: Settings | None = None, *, graph_name: str | None = None
) -> list[dict[str, Any]]:
    async with open_session(settings, graph_name=graph_name) as session:
        result = await session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in result.tools
        ]


def result_from_raw(name: str, raw: Any) -> ToolCallResult:
    """Parse a raw MCP CallToolResult into an envelope-preserving result."""
    text = "".join(
        part.text for part in raw.content if getattr(part, "type", None) == "text"
    )
    envelope: dict[str, Any] | None = None
    parse_error: str | None = None
    if text:
        try:
            parsed = json.loads(text)
            envelope = parsed if isinstance(parsed, dict) else {"result": parsed}
        except json.JSONDecodeError as exc:
            parse_error = f"tool {name!r} returned non-JSON content: {exc}"
    elif raw.isError:
        parse_error = f"tool {name!r} reported an error with no content"

    ok = bool(envelope and envelope.get("ok") is True and not raw.isError)
    return ToolCallResult(
        tool=name,
        ok=ok,
        envelope=envelope,
        text=text,
        is_error=bool(raw.isError),
        error=parse_error,
    )


async def call_tool_async(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    settings: Settings | None = None,
    graph_name: str | None = None,
    timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
) -> ToolCallResult:
    """Call one tool over the real protocol and return its parsed envelope."""
    async with open_session(settings, graph_name=graph_name) as session:
        try:
            raw = await asyncio.wait_for(
                session.call_tool(name, arguments or {}), timeout=timeout_s
            )
        except asyncio.TimeoutError as exc:
            raise MCPClientError(f"tool {name!r} timed out after {timeout_s:.0f}s") from exc
        except Exception as exc:
            raise MCPClientError(
                f"tool {name!r} call failed ({type(exc).__name__}: {exc})"
            ) from exc
    return result_from_raw(name, raw)


# ---- synchronous convenience wrappers (tests, CLI) ------------------------


def list_tools_sync(
    settings: Settings | None = None, *, graph_name: str | None = None
) -> list[dict[str, Any]]:
    return asyncio.run(list_tools_async(settings, graph_name=graph_name))


def call_tool_sync(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    settings: Settings | None = None,
    graph_name: str | None = None,
    timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
) -> ToolCallResult:
    return asyncio.run(
        call_tool_async(
            name, arguments, settings=settings, graph_name=graph_name,
            timeout_s=timeout_s,
        )
    )