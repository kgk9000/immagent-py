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
    agent1 = await store.create_agent(
        name="Calculator",
        system_prompt="You are a calculator. Only respond with numbers.",
        model=immagent.Model.CLAUDE_3_5_HAIKU,
    )

    agent2 = await agent1.advance("What is 2 + 2?")

    # Verify new agent
    assert agent2.id != agent1.id
    assert agent2.parent_id == agent1.id

    # Verify messages
    messages = await agent2.messages()
    assert len(messages) == 2  # user + assistant
    assert messages[0].role == "user"
    assert messages[0].content == "What is 2 + 2?"
    assert messages[1].role == "assistant"
    assert "4" in messages[1].content


@needs_api_key
async def test_advance_multi_turn(store: Store):
    """advance() maintains conversation context across turns."""
    agent = await store.create_agent(
        name="Assistant",
        system_prompt="You are helpful. Be very concise.",
        model=immagent.Model.CLAUDE_3_5_HAIKU,
    )

    # First turn
    agent = await agent.advance("My name is Alice.")

    # Second turn - should remember context
    agent = await agent.advance("What is my name?")

    # Verify it remembers
    messages = await agent.messages()
    last_response = messages[-1].content
    assert "Alice" in last_response

    # Verify lineage
    lineage = await agent.lineage()
    assert len(lineage) == 3  # original + 2 advances


@needs_api_key
async def test_token_tracking(store: Store):
    """advance() tracks token usage on assistant messages."""
    agent = await store.create_agent(
        name="Calculator",
        system_prompt="You are a calculator. Only respond with numbers.",
        model=immagent.Model.CLAUDE_3_5_HAIKU,
    )

    agent = await agent.advance("What is 2 + 2?")

    messages = await agent.messages()
    assistant_msg = messages[-1]

    # Verify token counts are present
    assert assistant_msg.input_tokens is not None
    assert assistant_msg.output_tokens is not None
    assert assistant_msg.input_tokens > 0
    assert assistant_msg.output_tokens > 0

    # User messages should not have token counts
    user_msg = messages[0]
    assert user_msg.input_tokens is None
    assert user_msg.output_tokens is None
