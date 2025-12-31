"""Tests for MCP integration."""

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


# Integration tests for MCP would require actual MCP servers
# These are marked as skip by default
@pytest.mark.skip(reason="Requires MCP server (npx)")
class TestMCPIntegration:
    async def test_connect_to_echo_server(self):
        """Can connect to an MCP server and list tools."""
        # This would require an actual MCP server like the echo server
        # manager = MCPManager()
        # await manager.connect("echo", "npx", ["-y", "@modelcontextprotocol/server-echo"])
        # tools = manager.get_all_tools()
        # assert len(tools) > 0
        # await manager.close()
        pass
