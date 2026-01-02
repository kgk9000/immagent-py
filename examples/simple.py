#!/usr/bin/env python3
"""Simple non-interactive example.

Requires:
    - Docker running (for automatic PostgreSQL)
    - ANTHROPIC_API_KEY in .env

Usage:
    make simple
"""

import asyncio
import os

import immagent

from helpers import get_store


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY")
        return

    async with get_store() as store:
        # Create agent (auto-saved)
        agent = await store.create_agent(
            name="Calculator",
            system_prompt="You are a calculator. Only respond with numbers.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )
        print(f"Agent v1: {agent.id}")

        # First turn (auto-saved)
        agent = await agent.advance("What is 2 + 2?")
        print(f"Agent v2: {agent.id}")

        # Second turn (auto-saved)
        agent = await agent.advance("Multiply that by 10")
        print(f"Agent v3: {agent.id}")

        # Show conversation
        print("\nConversation:")
        for msg in await agent.get_messages():
            print(f"  {msg.role}: {msg.content}")

        # Show lineage
        print("\nLineage:")
        for a in await agent.get_lineage():
            print(f"  {a.id} (parent: {a.parent_id})")


if __name__ == "__main__":
    asyncio.run(main())
