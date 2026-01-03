"""Simple agent for quick scripts and experimentation.

This module provides a simpler API for agents that don't need persistence.
No Store, no database, no UUIDs - just an in-memory agent that talks to an LLM.
"""

from typing import TYPE_CHECKING, Any

import immagent.advance as advance_mod
from immagent.messages import Message

if TYPE_CHECKING:
    from immagent.mcp import MCPManager


class SimpleAgent:
    """In-memory agent without persistence.

    Like PersistentAgent, advance() returns a new agent instance.
    No database, no UUIDs, no history tracking.

    Usage:
        agent = SimpleAgent(
            name="Bot",
            system_prompt="You are helpful.",
            model="anthropic/claude-3-5-haiku-20241022",
        )
        agent = await agent.advance("Hello!")
        agent = await agent.advance("What's 2+2?")
        for msg in agent.messages:
            print(f"{msg.role}: {msg.content}")
    """

    def __init__(
        self,
        *,
        name: str,
        system_prompt: str,
        model: str,
        model_config: dict[str, Any] | None = None,
        _messages: list[Message] | None = None,
    ):
        """Create a new simple agent.

        Args:
            name: Human-readable name for the agent
            system_prompt: The system prompt content
            model: LiteLLM model string (e.g., "anthropic/claude-3-5-haiku-20241022")
            model_config: Optional LLM configuration (temperature, max_tokens, etc.)
        """
        self._name = name
        self._model = model
        self._model_config = model_config or {}
        self._system_prompt = system_prompt
        self._messages: tuple[Message, ...] = tuple(_messages) if _messages else ()

    @property
    def name(self) -> str:
        """Get the agent name."""
        return self._name

    @property
    def model(self) -> str:
        """Get the model string."""
        return self._model

    @property
    def model_config(self) -> dict[str, Any]:
        """Get the model configuration."""
        return dict(self._model_config)

    @property
    def system_prompt(self) -> str:
        """Get the system prompt."""
        return self._system_prompt

    @property
    def messages(self) -> list[Message]:
        """Get all messages in the conversation."""
        return list(self._messages)

    @property
    def last_response(self) -> str | None:
        """Get the last assistant response, or None if no responses yet."""
        for msg in reversed(self._messages):
            if msg.role == "assistant" and msg.content:
                return msg.content
        return None

    async def advance(
        self,
        user_input: str,
        *,
        mcp: "MCPManager | None" = None,
        max_tool_rounds: int = 10,
        max_retries: int = 3,
        timeout: float | None = 120.0,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ) -> "SimpleAgent":
        """Process a user message and return a new agent with the response.

        Args:
            user_input: The user's message
            mcp: Optional MCP manager for tool execution
            max_tool_rounds: Maximum tool call iterations (default: 10)
            max_retries: LLM retry attempts on failure (default: 3)
            timeout: LLM request timeout in seconds (default: 120)
            temperature: Override temperature for this call
            max_tokens: Override max_tokens for this call
            top_p: Override top_p for this call

        Returns:
            A new SimpleAgent with the updated conversation
        """
        # Build effective config
        effective_config = dict(self._model_config)
        if temperature is not None:
            effective_config["temperature"] = temperature
        if max_tokens is not None:
            effective_config["max_tokens"] = max_tokens
        if top_p is not None:
            effective_config["top_p"] = top_p

        # Call the pure advance function
        new_messages = await advance_mod.advance(
            model=self._model,
            system_prompt=self._system_prompt,
            history=list(self._messages),
            user_input=user_input,
            mcp=mcp,
            max_tool_rounds=max_tool_rounds,
            max_retries=max_retries,
            timeout=timeout,
            model_config=effective_config,
        )

        # Return new agent with updated messages
        return SimpleAgent(
            name=self._name,
            system_prompt=self._system_prompt,
            model=self._model,
            model_config=self._model_config,
            _messages=list(self._messages) + new_messages,
        )

    def get_token_usage(self) -> tuple[int, int]:
        """Get total token usage for this conversation.

        Returns:
            A tuple of (input_tokens, output_tokens) summed across all
            assistant messages in the conversation.
        """
        input_tokens = sum(m.input_tokens or 0 for m in self._messages if m.role == "assistant")
        output_tokens = sum(m.output_tokens or 0 for m in self._messages if m.role == "assistant")
        return input_tokens, output_tokens
