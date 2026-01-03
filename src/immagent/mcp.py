"""MCP (Model Context Protocol) client integration for tool calling."""

import json
import time
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import immagent.exceptions as exc
from immagent.logging import logger


def tool_to_openai_format(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an MCP tool definition to OpenAI function calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
        },
    }


async def _execute_tool(
    session: ClientSession,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """Execute a tool call via MCP.

    Args:
        session: An active MCP ClientSession
        tool_name: Name of the tool to call
        arguments: Tool arguments as a dict

    Returns:
        The tool result as a string
    """
    result = await session.call_tool(tool_name, arguments)

    # MCP returns a list of content items
    if result.content:
        # Concatenate all text content
        texts = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text is not None:
                texts.append(text)
            else:
                # For non-text content, serialize to JSON
                texts.append(json.dumps(item.model_dump()))
        return "\n".join(texts)
    return ""


class MCPManager:
    """Manages connections to multiple MCP servers.

    Use as an async context manager for proper resource cleanup:

        async with MCPManager() as mcp:
            await mcp.connect("server", "command", ["args"])
            tools = mcp.get_all_tools()

    Or manually manage lifecycle:

        mcp = MCPManager()
        await mcp.connect("server", "command", ["args"])
        # ... use mcp ...
        await mcp.close()
    """

    def __init__(self):
        self._exit_stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[
            str, tuple[str, dict[str, Any]]
        ] = {}  # tool_name -> (server_key, tool_def)

    async def __aenter__(self) -> "MCPManager":
        await self._exit_stack.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()

    async def connect(
        self,
        server_key: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Connect to an MCP server.

        Args:
            server_key: A unique key to identify this server
            command: The command to run the MCP server
            args: Optional arguments to the command
            env: Optional environment variables
        """
        server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )

        # Use exit stack for proper cleanup on errors
        read, write = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()

        self._sessions[server_key] = session

        # Discover and index tools
        result = await session.list_tools()
        for tool in result.tools:
            tool_def = tool_to_openai_format(tool.model_dump())
            self._tools[tool.name] = (server_key, tool_def)

        logger.debug(
            "MCP connected: server=%s, tools=%d (%s)",
            server_key,
            len(result.tools),
            [t.name for t in result.tools],
        )

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Get all available tools across all connected servers."""
        return [tool_def for _, tool_def in self._tools.values()]

    async def execute(self, tool_name: str, arguments: str) -> str:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool
            arguments: JSON string of arguments

        Returns:
            Tool result as a string

        Raises:
            ToolExecutionError: If the tool is unknown, arguments are invalid,
                or execution fails
        """
        if tool_name not in self._tools:
            raise exc.ToolExecutionError(tool_name, "Unknown tool")

        server_key, _ = self._tools[tool_name]
        session = self._sessions.get(server_key)
        if session is None:
            raise exc.ToolExecutionError(
                tool_name, f"Server '{server_key}' is no longer connected"
            )

        try:
            args_dict = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as e:
            raise exc.ToolExecutionError(tool_name, f"Invalid arguments JSON: {e}") from e

        logger.debug("MCP execute: tool=%s, server=%s", tool_name, server_key)
        start_time = time.perf_counter()

        try:
            result = await _execute_tool(session, tool_name, args_dict)
        except Exception as e:
            raise exc.ToolExecutionError(tool_name, str(e)) from e

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(
            "MCP result: tool=%s, elapsed=%.0fms, result_len=%d",
            tool_name,
            elapsed_ms,
            len(result),
        )

        return result

    async def close(self) -> None:
        """Close all MCP sessions and their underlying connections."""
        await self._exit_stack.aclose()
        self._sessions.clear()
        self._tools.clear()
