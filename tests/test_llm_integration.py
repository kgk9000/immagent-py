"""Integration tests for LLM calls (requires ANTHROPIC_API_KEY)."""

import os

import pytest

import immagent
import immagent.db as db_mod
from immagent.llm import complete
from immagent.messages import Message


needs_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@needs_api_key
async def test_simple_completion():
    """Claude Haiku can convert Fahrenheit to Celsius."""
    msgs = [Message.user("Convert 32F to Celsius. Reply with just the number.")]

    response = await complete(
        model="anthropic/claude-3-5-haiku-20241022",
        msgs=msgs,
        system="You are a helpful assistant. Be concise.",
    )

    assert response.role == "assistant"
    assert response.content is not None
    assert "0" in response.content  # 32F = 0C


@needs_api_key
async def test_advance_creates_new_agent(db: db_mod.Database):
    """advance() calls LLM and returns new agent with different ID."""
    # Create agent
    agent1, assets1 = immagent.create_agent(
        name="Calculator",
        system_prompt="You are a calculator. Only respond with numbers.",
        model=immagent.Model.CLAUDE_3_5_HAIKU,
    )
    await immagent.save(db, *assets1)

    # Advance
    agent2, assets2 = await immagent.advance(agent1, "What is 2 + 2?", db)
    await immagent.save(db, *assets2)

    # Verify new agent
    assert agent2.id != agent1.id
    assert agent2.parent_id == agent1.id

    # Verify messages
    messages = await immagent.get_messages(agent2, db)
    assert len(messages) == 2  # user + assistant
    assert messages[0].role == "user"
    assert messages[0].content == "What is 2 + 2?"
    assert messages[1].role == "assistant"
    assert "4" in messages[1].content


@needs_api_key
async def test_advance_multi_turn(db: db_mod.Database):
    """advance() maintains conversation context across turns."""
    # Create agent
    agent, assets = immagent.create_agent(
        name="Assistant",
        system_prompt="You are helpful. Be very concise.",
        model=immagent.Model.CLAUDE_3_5_HAIKU,
    )
    await immagent.save(db, *assets)

    # First turn
    agent, assets = await immagent.advance(agent, "My name is Alice.", db)
    await immagent.save(db, *assets)

    # Second turn - should remember context
    agent, assets = await immagent.advance(agent, "What is my name?", db)
    await immagent.save(db, *assets)

    # Verify it remembers
    messages = await immagent.get_messages(agent, db)
    last_response = messages[-1].content
    assert "Alice" in last_response

    # Verify lineage
    lineage = await immagent.get_lineage(agent, db)
    assert len(lineage) == 3  # original + 2 advances
