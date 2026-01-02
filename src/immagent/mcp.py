"""MCP (Model Context Protocol) client integration for tool calling."""

import json
import time
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from immagent.logging import logger


@asynccontextmanager
async def connect_server(
    command: str, args: list[str] | None = None, env: dict[str, str] | None = None
):
    """Connect to an MCP server via stdio.

    Args:
        command: The command to run the MCP server
        args: Optional arguments to the command
        env: Optional environment variables

    Yields:
        An MCP ClientSession
    """
    server_params = StdioServerParameters(
        command=command,
        args=args or [],
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


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


async def get_tools_from_session(session: ClientSession) -> list[dict[str, Any]]:
    """Get all available tools from an MCP session in OpenAI format."""
    result = await session.list_tools()
    return [tool_to_openai_format(t.model_dump()) for t in result.tools]


async def execute_tool(
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


class _ServerConnection:
    """Internal class to track a server's context managers."""

    def __init__(self, stdio_cm: Any, session: ClientSession):
        self.stdio_cm = stdio_cm
        self.session = session


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
        self._connections: dict[str, _ServerConnection] = {}
        self._tools: dict[
            str, tuple[str, dict[str, Any]]
        ] = {}  # tool_name -> (server_key, tool_def)

    async def __aenter__(self) -> "MCPManager":
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

        # Create and enter the stdio context manager
        stdio_cm = stdio_client(server_params)
        read, write = await stdio_cm.__aenter__()

        # Create and enter the session context manager
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()

        # Store both for proper cleanup
        self._connections[server_key] = _ServerConnection(stdio_cm, session)

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
        """
        if tool_name not in self._tools:
            logger.warning("MCP unknown tool: %s", tool_name)
            return f"Error: Unknown tool '{tool_name}'"

        server_key, _ = self._tools[tool_name]
        conn = self._connections[server_key]

        args_dict = json.loads(arguments) if arguments else {}

        logger.debug("MCP execute: tool=%s, server=%s", tool_name, server_key)
        start_time = time.perf_counter()

        result = await execute_tool(conn.session, tool_name, args_dict)

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
        for conn in self._connections.values():
            # Exit session first, then stdio
            await conn.session.__aexit__(None, None, None)
            await conn.stdio_cm.__aexit__(None, None, None)
        self._connections.clear()
        self._tools.clear()
