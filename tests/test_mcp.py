"""Tests for MCP integration."""

import os
import sys
from pathlib import Path

import pytest

from immagent.mcp import MCPManager, tool_to_openai_format


class TestToolToOpenAIFormat:
    def test_basic_tool(self):
        """Converts basic MCP tool to OpenAI format."""
        mcp_tool = {
            "name": "get_weather",
            "description": "Get the current weather",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        }

        result = tool_to_openai_format(mcp_tool)

        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["description"] == "Get the current weather"
        assert result["function"]["parameters"]["type"] == "object"
        assert "city" in result["function"]["parameters"]["properties"]

    def test_tool_without_description(self):
        """Tool without description gets empty string."""
        mcp_tool = {
            "name": "do_something",
            "inputSchema": {"type": "object", "properties": {}},
        }

        result = tool_to_openai_format(mcp_tool)

        assert result["function"]["description"] == ""

    def test_tool_without_input_schema(self):
        """Tool without inputSchema gets default empty object."""
        mcp_tool = {
            "name": "simple_tool",
            "description": "A simple tool",
        }

        result = tool_to_openai_format(mcp_tool)

        assert result["function"]["parameters"] == {"type": "object", "properties": {}}


class TestMCPManager:
    def test_init(self):
        """MCPManager initializes with empty state."""
        manager = MCPManager()

        assert manager.get_all_tools() == []

    def test_get_all_tools_empty(self):
        """get_all_tools returns empty list when no servers connected."""
        manager = MCPManager()

        assert manager.get_all_tools() == []

    async def test_execute_unknown_tool(self):
        """execute returns error for unknown tool."""
        manager = MCPManager()

        result = await manager.execute("unknown_tool", "{}")

        assert "Error" in result
        assert "unknown_tool" in result


# Check if weather token is available for integration tests
WEATHER_TOKEN = os.environ.get("WEATHER_TOKEN")


@pytest.mark.skipif(not WEATHER_TOKEN, reason="WEATHER_TOKEN not set")
class TestMCPIntegration:
    """Integration tests using the weather MCP server."""

    async def test_connect_and_list_tools(self):
        """Can connect to weather server and list tools."""
        async with MCPManager() as manager:
            await manager.connect(
                "weather",
                sys.executable,
                [str(Path(__file__).parent / "weather_server.py")],
                env={**os.environ, "WEATHER_TOKEN": WEATHER_TOKEN},
            )

            tools = manager.get_all_tools()
            assert len(tools) == 1
            assert tools[0]["function"]["name"] == "get_weather"

    async def test_execute_weather_tool(self):
        """Can execute weather tool and get result."""
        async with MCPManager() as manager:
            await manager.connect(
                "weather",
                sys.executable,
                [str(Path(__file__).parent / "weather_server.py")],
                env={**os.environ, "WEATHER_TOKEN": WEATHER_TOKEN},
            )

            result = await manager.execute("get_weather", '{"city": "London"}')

            assert "London" in result
            assert "Temperature" in result or "Error" not in result
