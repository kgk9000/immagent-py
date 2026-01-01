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

    async with await immagent.Database.connect(db_url) as db:
        await db.init_schema()

        # Create agent
        agent, assets = immagent.create_agent(
            name="Calculator",
            system_prompt="You are a calculator. Only respond with numbers.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )
        await db.save(*assets)
        print(f"Agent v1: {agent.id}")

        # First turn
        agent, assets = await immagent.advance(db, agent, "What is 2 + 2?")
        await db.save(*assets)
        print(f"Agent v2: {agent.id}")

        # Second turn
        agent, assets = await immagent.advance(db, agent, "Multiply that by 10")
        await db.save(*assets)
        print(f"Agent v3: {agent.id}")

        # Show conversation
        print("\nConversation:")
        for msg in await immagent.get_messages(db, agent):
            print(f"  {msg.role}: {msg.content}")

        # Show lineage
        print("\nLineage:")
        for a in await immagent.get_lineage(db, agent):
            print(f"  {a.id} (parent: {a.parent_id})")


if __name__ == "__main__":
    asyncio.run(main())
