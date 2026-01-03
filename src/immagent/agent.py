"""The immutable agent type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

import immagent.assets as assets
import immagent.messages as messages

if TYPE_CHECKING:
    from immagent.mcp import MCPManager


@dataclass(frozen=True)
class ImmAgent(assets.Asset):
    """An immutable agent.

    Every state transition (e.g., processing a turn) creates a new ImmAgent
    with a new UUID. The parent_id links to the previous state.

    Attributes:
        name: Human-readable name for the agent
        system_prompt_id: UUID of the SystemPrompt asset
        parent_id: UUID of the previous agent state (None for initial agent)
        conversation_id: UUID of the Conversation asset
        model: LiteLLM model string (e.g., "anthropic/claude-sonnet-4-20250514")
        metadata: Custom key-value data attached to the agent
        model_config: LLM configuration (temperature, max_tokens, etc.)
    """

    name: str
    system_prompt_id: UUID
    parent_id: UUID | None
    conversation_id: UUID
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _create(
        cls,
        *,
        name: str,
        system_prompt_id: UUID,
        conversation_id: UUID,
        model: str,
        metadata: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> ImmAgent:
        """Create a new agent (internal).

        Takes IDs of pre-created dependencies. Caller is responsible for
        creating and persisting the system prompt and conversation.
        """
        return cls(
            id=assets.new_id(),
            created_at=assets.now(),
            name=name,
            system_prompt_id=system_prompt_id,
            parent_id=None,
            conversation_id=conversation_id,
            model=model,
            metadata=metadata or {},
            model_config=model_config or {},
        )

    def _evolve(
        self,
        conversation: messages.Conversation,
        metadata: dict[str, Any] | None = None,
    ) -> ImmAgent:
        """Create a new agent state with an updated conversation (internal).

        The new agent links back to this one via parent_id.
        Metadata and model_config are inherited from the current agent.
        """
        from immagent.store import _get_store, _register_agent

        new_agent = ImmAgent(
            id=assets.new_id(),
            created_at=assets.now(),
            name=self.name,
            system_prompt_id=self.system_prompt_id,
            parent_id=self.id,
            conversation_id=conversation.id,
            model=self.model,
            metadata=metadata if metadata is not None else self.metadata,
            model_config=self.model_config,
        )
        # Register new agent with the same store as the parent
        _register_agent(new_agent.id, _get_store(self.id))
        return new_agent

    async def with_metadata(self, metadata: dict[str, Any]) -> ImmAgent:
        """Create a new agent with updated metadata.

        The new agent has the same conversation but new metadata.
        Useful for updating agent state between turns.

        Args:
            metadata: New metadata dict (replaces existing metadata)

        Returns:
            A new agent with updated metadata
        """
        from immagent.store import _get_store

        return await _get_store(self.id)._update_metadata(self, metadata)

    async def advance(
        self,
        user_input: str,
        *,
        mcp: MCPManager | None = None,
        max_retries: int = 3,
        timeout: float | None = 120.0,
        max_tool_rounds: int = 10,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ) -> ImmAgent:
        """Process a user message and return a new agent with the response.

        Calls the LLM, handles any tool calls, and creates a new agent
        with the updated conversation. The new agent is automatically saved.

        Args:
            user_input: The user's message
            mcp: Optional MCP manager for tool execution
            max_retries: Number of retries for LLM calls (default: 3)
            timeout: Request timeout in seconds, or None for no timeout (default: 120)
            max_tool_rounds: Maximum tool call rounds (default: 10)
            temperature: Override temperature for this call (default: use agent's model_config)
            max_tokens: Override max_tokens for this call (default: use agent's model_config)
            top_p: Override top_p for this call (default: use agent's model_config)
        """
        from immagent.store import _get_store

        return await _get_store(self.id)._advance(
            self,
            user_input,
            mcp=mcp,
            max_retries=max_retries,
            timeout=timeout,
            max_tool_rounds=max_tool_rounds,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

    async def get_messages(self) -> list[messages.Message]:
        """Get all messages in this agent's conversation."""
        from immagent.store import _get_store

        return await _get_store(self.id)._get_agent_messages(self)

    async def get_lineage(self) -> list[ImmAgent]:
        """Get the chain of agents from root to this agent."""
        from immagent.store import _get_store

        return await _get_store(self.id)._get_agent_lineage(self)

    async def clone(self) -> ImmAgent:
        """Create a clone of this agent for branching.

        The clone shares the same parent, conversation, and system prompt,
        allowing you to advance it in a different direction.
        """
        from immagent.store import _get_store

        return await _get_store(self.id)._clone_agent(self)

    async def get_token_usage(self) -> tuple[int, int]:
        """Get total token usage for this agent's conversation.

        Returns:
            A tuple of (input_tokens, output_tokens) summed across all
            assistant messages in the conversation.
        """
        msgs = await self.get_messages()
        input_tokens = sum(m.input_tokens or 0 for m in msgs if m.role == "assistant")
        output_tokens = sum(m.output_tokens or 0 for m in msgs if m.role == "assistant")
        return input_tokens, output_tokens
