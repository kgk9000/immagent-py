#!/usr/bin/env python3
"""Simple agent example - no database, no UUIDs, just chat.

This demonstrates SimpleAgent for quick scripts and experimentation.
No Store needed, no persistence, just in-memory chat.

Requires:
    - ANTHROPIC_API_KEY in environment

Usage:
    uv run python examples/simple.py
"""

import asyncio
import os

from immagent import SimpleAgent


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY")
        return

    # Create a simple agent - no store, no database
    agent = SimpleAgent(
        name="Calculator",
        system_prompt="You are a calculator. Only respond with numbers.",
        model="anthropic/claude-3-5-haiku-20241022",
    )

    # advance() returns a new agent (same pattern as PersistentAgent)
    agent = await agent.advance("What is 2 + 2?")
    agent = await agent.advance("Multiply that by 10")

    # Show conversation
    print("Conversation:")
    for msg in agent.messages():
        print(f"  {msg.role}: {msg.content}")

    # Get total token usage
    input_tokens, output_tokens = await agent.token_usage()
    print(f"\nToken usage: {input_tokens} input, {output_tokens} output")

    # Get last response
    print(f"\nLast response: {agent.last_response()}")


if __name__ == "__main__":
    asyncio.run(main())
