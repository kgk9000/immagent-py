"""LLM integration via LiteLLM."""

import time
from typing import Any

import litellm

import immagent.exceptions as exc
import immagent.messages as messages
from immagent.logging import logger


class Model:
    """Common LLM model strings for convenience.

    These are just string constants — any LiteLLM model string works directly.
    See https://docs.litellm.ai/docs/providers for all supported models.
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
    max_retries: int = 3,
    timeout: float | None = 120.0,
    model_config: dict[str, Any] | None = None,
) -> messages.Message:
    """Call an LLM via LiteLLM and return the response as a Message.

    Uses exponential backoff for retries on rate limits and transient errors.

    Args:
        model: LiteLLM model string (e.g., "anthropic/claude-sonnet-4-20250514")
        msgs: Conversation history as Message objects
        system: System prompt content
        tools: Optional list of tools in OpenAI function format
        max_retries: Number of retry attempts for transient failures (default: 3)
        timeout: Request timeout in seconds (default: 120). None for no timeout.
        model_config: Optional LLM configuration (temperature, max_tokens, top_p, etc.)

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
        "num_retries": max_retries,
    }
    if tools:
        kwargs["tools"] = tools
    if timeout is not None:
        kwargs["timeout"] = timeout

    # Apply model config (temperature, max_tokens, top_p, etc.)
    if model_config:
        kwargs.update(model_config)

    # Call LiteLLM (handles retries with exponential backoff internally)
    logger.debug(
        "LLM request: model=%s, messages=%d, tools=%d",
        model,
        len(litellm_messages),
        len(tools) if tools else 0,
    )
    start_time = time.perf_counter()

    try:
        response = await litellm.acompletion(**kwargs)
    except Exception as e:
        # Handle specific error types with better messages
        error_type = type(e).__name__
        if error_type == "AuthenticationError":
            # Extract provider from model string (e.g., "anthropic/claude-..." -> "anthropic")
            provider = model.split("/")[0] if "/" in model else model
            env_var_hints = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "azure": "AZURE_API_KEY",
                "cohere": "COHERE_API_KEY",
                "google": "GOOGLE_API_KEY",
                "mistral": "MISTRAL_API_KEY",
            }
            hint = env_var_hints.get(provider, f"{provider.upper()}_API_KEY")
            raise exc.LLMError(
                f"Authentication failed for {provider}. "
                f"Check that {hint} is set and valid."
            ) from e
        elif error_type == "RateLimitError":
            raise exc.LLMError(f"Rate limit exceeded: {e}") from e
        else:
            raise exc.LLMError(f"LLM call failed: {e}") from e

    # Check for empty response (can happen with content filtering)
    choices = getattr(response, "choices", None)
    if not choices:
        raise exc.LLMError("LLM returned no choices (possibly filtered)")

    choice = choices[0].message

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    usage = getattr(response, "usage", None)
    logger.debug(
        "LLM response: model=%s, elapsed=%.0fms, input_tokens=%s, output_tokens=%s",
        model,
        elapsed_ms,
        getattr(usage, "prompt_tokens", "?") if usage else "?",
        getattr(usage, "completion_tokens", "?") if usage else "?",
    )

    # Parse tool calls if present
    tool_calls: tuple[messages.ToolCall, ...] | None = None
    if choice.tool_calls:
        tool_calls = tuple(
            messages.ToolCall(
                id=tc.id,
                name=tc.function.name or "",
                arguments=tc.function.arguments,
            )
            for tc in choice.tool_calls
        )
        logger.debug("LLM requested %d tool call(s): %s", len(tool_calls), [tc.name for tc in tool_calls])

    # Extract token usage
    input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage else None

    return messages.Message.assistant(
        content=choice.content,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
