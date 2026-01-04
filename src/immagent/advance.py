"""Agent advancement logic - LLM orchestration and tool execution.

This module contains pure orchestration logic with no persistence concerns.
It takes conversation data in and returns new messages out.
"""

import asyncio
from typing import TYPE_CHECKING, Any

import immagent.exceptions as exc
import immagent.llm as llm
import immagent.messages as messages
from immagent.logging import logger

if TYPE_CHECKING:
    from immagent.mcp import MCPManager


async def advance(
    *,
    model: str,
    system_prompt: str,
    history: tuple[messages.Message, ...],
    user_input: str,
    mcp: "MCPManager | None" = None,
    max_tool_rounds: int = 10,
    max_retries: int = 3,
    timeout: float | None = 120.0,
    model_config: dict[str, Any] | None = None,
) -> list[messages.Message]:
    """Run the LLM orchestration loop and return new messages.

    This is a pure orchestration function - it takes conversation data in
    and returns new messages out. It has no knowledge of persistence.

    Args:
        model: LiteLLM model string
        system_prompt: The system prompt content
        history: Existing conversation messages
        user_input: The user's new message
        mcp: Optional MCP manager for tool execution
        max_tool_rounds: Maximum tool call iterations (default: 10)
        max_retries: LLM retry attempts on failure (default: 3)
        timeout: LLM request timeout in seconds (default: 120)
        model_config: LLM configuration (temperature, max_tokens, etc.)

    Returns:
        List of new messages created during this turn (user message,
        assistant responses, and any tool results)

    Raises:
        ValidationError: If inputs are invalid
        LLMError: If LLM call fails after retries
    """
    # Validate inputs
    if not user_input or not user_input.strip():
        raise exc.ValidationError("user_input", "must not be empty")
    if max_tool_rounds < 1:
        raise exc.ValidationError("max_tool_rounds", "must be at least 1")
    if max_retries < 0:
        raise exc.ValidationError("max_retries", "must be non-negative")
    if timeout is not None and timeout <= 0:
        raise exc.ValidationError("timeout", "must be positive")

    # Build message list: history + new user message
    user_message = messages.Message.user(user_input)
    msgs = list(history)
    msgs.append(user_message)

    # Get tools if MCP is available
    tools = mcp.get_all_tools() if mcp else None

    # Track new messages created in this turn
    new_messages: list[messages.Message] = [user_message]

    # Tool loop - each iteration is one LLM call, possibly followed by tool execution
    last_assistant_message: messages.Message | None = None
    llm_calls = 0
    for _ in range(max_tool_rounds):
        # Call LLM
        assistant_message = await llm.complete(
            model=model,
            msgs=msgs,
            system=system_prompt,
            tools=tools,
            max_retries=max_retries,
            timeout=timeout,
            model_config=model_config,
        )
        llm_calls += 1
        last_assistant_message = assistant_message
        msgs.append(assistant_message)
        new_messages.append(assistant_message)

        # Check for tool calls
        if not assistant_message.tool_calls or not mcp:
            break

        # Execute tool calls concurrently
        async def execute_one(tc: messages.ToolCall) -> messages.Message:
            try:
                result = await mcp.execute(tc.name, tc.arguments)
            except exc.ToolExecutionError as e:
                result = f"Error: {e}"
            return messages.Message.tool_result(tc.id, result)

        tool_results = await asyncio.gather(
            *(execute_one(tc) for tc in assistant_message.tool_calls)
        )
        for tool_result_message in tool_results:
            msgs.append(tool_result_message)
            new_messages.append(tool_result_message)
    else:
        # Loop completed without break - hit max_tool_rounds
        # Check if LLM still wanted to call tools
        if last_assistant_message and last_assistant_message.tool_calls and mcp:
            logger.warning(
                "Reached max_tool_rounds=%d but LLM still requesting %d tool(s): %s",
                max_tool_rounds,
                len(last_assistant_message.tool_calls),
                [tc.name for tc in last_assistant_message.tool_calls],
            )

    logger.debug(
        "Advance complete: llm_calls=%d, new_messages=%d",
        llm_calls,
        len(new_messages),
    )

    return new_messages
