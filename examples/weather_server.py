#!/usr/bin/env python3
"""Example MCP server that wraps the WeatherAPI.

This demonstrates how to create a simple MCP (Model Context Protocol) server
in Python that an agent can use as a tool. The server exposes a single tool
`get_weather` that fetches current weather data for a city.

Setup:
    1. Get a free API key from https://www.weatherapi.com/
    2. Set the WEATHER_TOKEN environment variable

Usage:
    # Run the server directly (for testing)
    WEATHER_TOKEN=your_key python examples/weather_server.py

    # Use with immagent
    async with immagent.MCPManager() as mcp:
        await mcp.connect("weather", "python", ["examples/weather_server.py"])
        agent = await store.advance(agent, "What's the weather in Tokyo?", mcp=mcp)

The server communicates via stdio using the MCP protocol, so it can be used
with any MCP-compatible client.
"""

import os
import urllib.request
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

WEATHER_API_URL = "https://api.weatherapi.com/v1/current.json"

server = Server("weather")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., 'London', 'New York')",
                    },
                },
                "required": ["city"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "get_weather":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    city = arguments.get("city", "London")
    token = os.environ.get("WEATHER_TOKEN", "")

    if not token:
        return [TextContent(type="text", text="Error: WEATHER_TOKEN not set")]

    url = f"{WEATHER_API_URL}?key={token}&q={city}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())

        location = data["location"]
        current = data["current"]

        result = (
            f"Weather in {location['name']}, {location['country']}:\n"
            f"  Temperature: {current['temp_c']}°C ({current['temp_f']}°F)\n"
            f"  Condition: {current['condition']['text']}\n"
            f"  Humidity: {current['humidity']}%\n"
            f"  Wind: {current['wind_kph']} km/h {current['wind_dir']}"
        )
        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error fetching weather: {e}")]


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
