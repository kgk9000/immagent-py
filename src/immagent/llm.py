"""LLM integration via LiteLLM."""

from typing import Any

import litellm

from immagent.messages import Message, ToolCall


async def complete(
    model: str,
    messages: list[Message],
    system: str,
    tools: list[dict[str, Any]] | None = None,
) -> Message:
    """Call an LLM via LiteLLM and return the response as a Message.

    Args:
        model: LiteLLM model string (e.g., "anthropic/claude-sonnet-4-20250514")
        messages: Conversation history as Message objects
        system: System prompt content
        tools: Optional list of tools in OpenAI function format

    Returns:
        An assistant Message with the response
    """
    # Convert messages to LiteLLM format
    litellm_messages = [{"role": "system", "content": system}]
    litellm_messages.extend(msg.to_litellm_dict() for msg in messages)

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
    tool_calls: tuple[ToolCall, ...] | None = None
    if choice.tool_calls:
        tool_calls = tuple(
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments,
            )
            for tc in choice.tool_calls
        )

    return Message.assistant(
        content=choice.content,
        tool_calls=tool_calls,
    )
