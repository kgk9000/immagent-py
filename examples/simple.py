#!/usr/bin/env python3
"""Simple non-interactive example.

Requires:
    - PostgreSQL running
    - ANTHROPIC_API_KEY environment variable set
"""

import asyncio
import os

import immagent


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY")
        return

    db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/postgres",
    )

    async with await immagent.Store.connect(db_url) as store:
        await store.init_schema()

        # Create agent
        agent = store.create_agent(
            name="Calculator",
            system_prompt="You are a calculator. Only respond with numbers.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )
        await store.save(agent)
        print(f"Agent v1: {agent.id}")

        # First turn
        agent = await store.advance(agent, "What is 2 + 2?")
        await store.save(agent)
        print(f"Agent v2: {agent.id}")

        # Second turn
        agent = await store.advance(agent, "Multiply that by 10")
        await store.save(agent)
        print(f"Agent v3: {agent.id}")

        # Show conversation
        print("\nConversation:")
        for msg in await store.get_messages(agent):
            print(f"  {msg.role}: {msg.content}")

        # Show lineage
        print("\nLineage:")
        for a in await store.get_lineage(agent):
            print(f"  {a.id} (parent: {a.parent_id})")


if __name__ == "__main__":
    asyncio.run(main())
