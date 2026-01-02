"""The immutable agent type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

import immagent.assets as assets
import immagent.messages as messages

if TYPE_CHECKING:
    from immagent.mcp import MCPManager
    from immagent.store import Store


@dataclass(frozen=True)
class ImmAgent(assets.Asset):
    """An immutable agent.

    Every state transition (e.g., processing a turn) creates a new ImmAgent
    with a new UUID. The parent_id links to the previous state.

    Attributes:
        name: Human-readable name for the agent
        system_prompt_id: UUID of the TextAsset containing the system prompt
        parent_id: UUID of the previous agent state (None for initial agent)
        conversation_id: UUID of the Conversation asset
        model: LiteLLM model string (e.g., "anthropic/claude-sonnet-4-20250514")
    """

    name: str
    system_prompt_id: UUID
    parent_id: UUID | None
    conversation_id: UUID
    model: str
    _store: Store | None = field(default=None, compare=False, hash=False, repr=False)

    @classmethod
    def _create(
        cls,
        name: str,
        system_prompt: assets.TextAsset,
        model: str,
        store: Store,
        conversation: messages.Conversation | None = None,
    ) -> tuple[ImmAgent, messages.Conversation]:
        """Create a new agent with an empty conversation.

        Returns both the agent and the conversation asset for persistence.
        """
        conv = conversation or messages.Conversation.create()
        agent = cls(
            id=assets.new_id(),
            created_at=assets.now(),
            name=name,
            system_prompt_id=system_prompt.id,
            parent_id=None,
            conversation_id=conv.id,
            model=model,
            _store=store,
        )
        return agent, conv

    def evolve(
        self,
        conversation: messages.Conversation,
    ) -> ImmAgent:
        """Create a new agent state with an updated conversation.

        The new agent links back to this one via parent_id.
        """
        return ImmAgent(
            id=assets.new_id(),
            created_at=assets.now(),
            name=self.name,
            system_prompt_id=self.system_prompt_id,
            parent_id=self.id,
            conversation_id=conversation.id,
            model=self.model,
            _store=self._store,
        )

    async def advance(
        self,
        user_input: str,
        *,
        mcp: MCPManager | None = None,
        max_retries: int = 3,
        timeout: float = 120.0,
        max_tool_rounds: int = 10,
    ) -> ImmAgent:
        """Process a user message and return a new agent with the response.

        Calls the LLM, handles any tool calls, and creates a new agent
        with the updated conversation. The new agent is automatically saved.
        """
        if self._store is None:
            raise RuntimeError("Agent not bound to a store")
        return await self._store._advance(
            self,
            user_input,
            mcp=mcp,
            max_retries=max_retries,
            timeout=timeout,
            max_tool_rounds=max_tool_rounds,
        )

    async def get_messages(self) -> list[messages.Message]:
        """Get all messages in this agent's conversation."""
        if self._store is None:
            raise RuntimeError("Agent not bound to a store")
        return await self._store._get_agent_messages(self)

    async def get_lineage(self) -> list[ImmAgent]:
        """Get the chain of agents from root to this agent."""
        if self._store is None:
            raise RuntimeError("Agent not bound to a store")
        return await self._store._get_agent_lineage(self)

    async def copy(self) -> ImmAgent:
        """Create a copy of this agent with a new ID."""
        if self._store is None:
            raise RuntimeError("Agent not bound to a store")
        return await self._store._copy_agent(self)
