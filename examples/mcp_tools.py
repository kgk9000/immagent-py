#!/usr/bin/env python3
"""Example using MCP tools with an agent.

Demonstrates connecting to an MCP server and letting the agent use tools.

Requires:
    - Docker running (for automatic PostgreSQL)
    - ANTHROPIC_API_KEY in .env
    - WEATHER_TOKEN in .weather.env (get one free at https://www.weatherapi.com/)

Usage:
    make mcp-tools
"""

import asyncio
import os

import immagent

from helpers import get_store


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY")
        return

    if not os.environ.get("WEATHER_TOKEN"):
        print("Please set WEATHER_TOKEN (get one free at https://www.weatherapi.com/)")
        return

    async with get_store() as store:
        # Connect to the weather MCP server
        async with immagent.MCPManager() as mcp:
            await mcp.connect(
                "weather",
                "uv",
                ["run", "python", "weather_server.py"],
            )

            # Show available tools
            tools = mcp.get_all_tools()
            print(f"Available tools: {[t['function']['name'] for t in tools]}\n")

            # Create an agent (auto-saved)
            agent = await store.create_agent(
                name="WeatherBot",
                system_prompt="You are a helpful assistant with access to weather data. Use the get_weather tool when asked about weather.",
                model=immagent.Model.CLAUDE_3_5_HAIKU,
            )

            # Ask about weather - the agent will use the tool (auto-saved)
            agent = await agent.advance(
                "What's the weather like in Tokyo?",
                mcp=mcp,
            )

            # Show the conversation
            print("Conversation:")
            for msg in await agent.get_messages():
                if msg.tool_calls:
                    print(f"  {msg.role}: [called {msg.tool_calls[0].name}]")
                elif msg.role == "tool":
                    print(f"  {msg.role}: {msg.content[:50]}...")
                else:
                    print(f"  {msg.role}: {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
