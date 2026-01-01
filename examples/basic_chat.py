#!/usr/bin/env python3
"""Basic chat example using ImmAgent.

Requires:
    - PostgreSQL running (or use Docker)
    - ANTHROPIC_API_KEY environment variable set

Usage:
    # Start PostgreSQL
    docker run -d --name immagent-postgres \
        -e POSTGRES_PASSWORD=postgres \
        -p 5432:5432 postgres:16

    # Run the example
    ANTHROPIC_API_KEY=sk-ant-... python examples/basic_chat.py
"""

import asyncio
import os

import immagent


async def main():
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        return

    # Connect to PostgreSQL
    db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/postgres",
    )

    async with await immagent.Store.connect(db_url) as store:
        await store.init_schema()

        # Create a new agent
        agent = store.create_agent(
            name="Assistant",
            system_prompt="You are a helpful assistant. Be concise.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )
        await store.save(agent)
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

            # Advance the agent
            agent = await store.advance(agent, user_input)
            await store.save(agent)

            # Get the last message (assistant's response)
            messages = await store.get_messages(agent)
            if messages:
                last_message = messages[-1]
                print(f"Assistant: {last_message.content}\n")

        # Show lineage
        lineage = await store.get_lineage(agent)
        print(f"\nAgent went through {len(lineage)} states")


if __name__ == "__main__":
    asyncio.run(main())
