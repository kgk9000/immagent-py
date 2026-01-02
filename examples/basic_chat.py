#!/usr/bin/env python3
"""Interactive chat example using ImmAgent.

Requires:
    - Docker running (for automatic PostgreSQL)
    - ANTHROPIC_API_KEY in .env

Usage:
    make chat
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
        # Create a new agent (auto-saved)
        agent = await store.create_agent(
            name="Assistant",
            system_prompt="You are a helpful assistant. Be concise.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )
        print(f"Created agent: {agent.id}\n")

        # Chat loop
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            # Advance the agent (auto-saved)
            agent = await agent.advance(user_input)

            # Get the last message (assistant's response)
            messages = await agent.get_messages()
            if messages:
                last_message = messages[-1]
                print(f"Assistant: {last_message.content}\n")

        # Show lineage
        lineage = await agent.get_lineage()
        print(f"\nAgent went through {len(lineage)} states")


if __name__ == "__main__":
    asyncio.run(main())
