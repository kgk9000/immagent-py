"""Integration tests for LLM calls (requires ANTHROPIC_API_KEY)."""

import os

import pytest

from immagent.llm import complete
from immagent.messages import Message


needs_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@needs_api_key
async def test_simple_completion():
    """Claude Haiku can convert Fahrenheit to Celsius."""
    messages = [Message.user("Convert 32F to Celsius. Reply with just the number.")]

    response = await complete(
        model="anthropic/claude-3-5-haiku-20241022",
        messages=messages,
        system="You are a helpful assistant. Be concise.",
    )

    assert response.role == "assistant"
    assert response.content is not None
    assert "0" in response.content  # 32F = 0C
