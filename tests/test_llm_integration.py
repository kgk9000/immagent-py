"""Integration tests for LLM calls (requires ANTHROPIC_API_KEY)."""

import os

import pytest

import immagent
from immagent import Store
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
async def test_advance_creates_new_agent(store: Store):
    """advance() calls LLM and returns new agent with different ID."""
    agent1 = store.create_agent(
        name="Calculator",
        system_prompt="You are a calculator. Only respond with numbers.",
        model=immagent.Model.CLAUDE_3_5_HAIKU,
    )

    agent2 = await store.advance(agent1, "What is 2 + 2?")

    # Verify new agent
    assert agent2.id != agent1.id
    assert agent2.parent_id == agent1.id

    # Verify messages
    messages = await store.get_messages(agent2)
    assert len(messages) == 2  # user + assistant
    assert messages[0].role == "user"
    assert messages[0].content == "What is 2 + 2?"
    assert messages[1].role == "assistant"
    assert "4" in messages[1].content


@needs_api_key
async def test_advance_multi_turn(store: Store):
    """advance() maintains conversation context across turns."""
    agent = store.create_agent(
        name="Assistant",
        system_prompt="You are helpful. Be very concise.",
        model=immagent.Model.CLAUDE_3_5_HAIKU,
    )

    # First turn
    agent = await store.advance(agent, "My name is Alice.")

    # Second turn - should remember context
    agent = await store.advance(agent, "What is my name?")

    # Verify it remembers
    messages = await store.get_messages(agent)
    last_response = messages[-1].content
    assert "Alice" in last_response

    # Verify lineage
    lineage = await store.get_lineage(agent)
    assert len(lineage) == 3  # original + 2 advances
