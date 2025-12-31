"""LLM integration via LiteLLM."""

from enum import Enum
from typing import Any

import litellm

import immagent.messages as messages


class Model(str, Enum):
    """Common LLM models supported via LiteLLM.

    This is a curated list for convenience. Any LiteLLM model string works.
    """

    # Anthropic
    CLAUDE_3_5_HAIKU = "anthropic/claude-3-5-haiku-20241022"
    CLAUDE_SONNET_4 = "anthropic/claude-sonnet-4-20250514"
    CLAUDE_OPUS_4 = "anthropic/claude-opus-4-20250514"

    # OpenAI
    GPT_4O = "openai/gpt-4o"
    GPT_4O_MINI = "openai/gpt-4o-mini"
    O1 = "openai/o1"
    O1_MINI = "openai/o1-mini"


async def complete(
    model: str,
    msgs: list[messages.Message],
    system: str,
    tools: list[dict[str, Any]] | None = None,
) -> messages.Message:
    """Call an LLM via LiteLLM and return the response as a Message.

    Args:
        model: LiteLLM model string (e.g., "anthropic/claude-sonnet-4-20250514")
        msgs: Conversation history as Message objects
        system: System prompt content
        tools: Optional list of tools in OpenAI function format

    Returns:
        An assistant Message with the response
    """
    # Convert messages to LiteLLM format
    litellm_messages = [{"role": "system", "content": system}]
    litellm_messages.extend(msg.to_litellm_dict() for msg in msgs)

    # Build kwargs
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": litellm_messages,
    }
    if tools:
        kwargs["tools"] = tools

    # Call LiteLLM
    response = await litellm.acompletion(**kwargs)
    choice = response.choices[0].message

    # Parse tool calls if present
    tool_calls: tuple[messages.ToolCall, ...] | None = None
    if choice.tool_calls:
        tool_calls = tuple(
            messages.ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments,
            )
            for tc in choice.tool_calls
        )

    return messages.Message.assistant(
        content=choice.content,
        tool_calls=tool_calls,
    )
