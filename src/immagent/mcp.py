"""MCP (Model Context Protocol) client integration for tool calling."""

import json
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


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
            if hasattr(item, "text"):
                texts.append(item.text)
            else:
                # For non-text content, serialize to JSON
                texts.append(json.dumps(item.model_dump()))
        return "\n".join(texts)
    return ""


class MCPManager:
    """Manages connections to multiple MCP servers."""

    def __init__(self):
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[
            str, tuple[str, dict[str, Any]]
        ] = {}  # tool_name -> (server_key, tool_def)

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

        # Note: For proper resource management, this should be used with
        # an async context manager. This simplified version assumes
        # the caller will handle cleanup.
        read, write = await stdio_client(server_params).__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()

        self._sessions[server_key] = session

        # Discover and index tools
        result = await session.list_tools()
        for tool in result.tools:
            tool_def = tool_to_openai_format(tool.model_dump())
            self._tools[tool.name] = (server_key, tool_def)

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
            return f"Error: Unknown tool '{tool_name}'"

        server_key, _ = self._tools[tool_name]
        session = self._sessions[server_key]

        args_dict = json.loads(arguments) if arguments else {}
        return await execute_tool(session, tool_name, args_dict)

    async def close(self) -> None:
        """Close all MCP sessions."""
        for session in self._sessions.values():
            await session.__aexit__(None, None, None)
        self._sessions.clear()
        self._tools.clear()
