"""Persistent agent with database storage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Mapping
from uuid import UUID

import immagent.assets as assets
import immagent.messages as messages
from immagent.registry import get_store, register_agent

if TYPE_CHECKING:
    from immagent.mcp import MCPManager


def _empty_mapping() -> MappingProxyType[str, Any]:
    """Create an empty immutable mapping for dataclass defaults."""
    return MappingProxyType({})


@dataclass(frozen=True)
class PersistentAgent(assets.Asset):
    """Database-backed agent with full history tracking.

    Every state transition creates a new PersistentAgent with a new UUID.
    The parent_id links to the previous state, enabling full lineage traversal.

    Create via Store:
        agent = await store.create_agent(name="Bot", system_prompt="...", model="...")
        agent = await agent.advance("Hello!")

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
    # MappingProxyType ensures these can't be mutated, preserving immutability
    metadata: MappingProxyType[str, Any] = field(default_factory=_empty_mapping)
    model_config: MappingProxyType[str, Any] = field(default_factory=_empty_mapping)

    TABLE: ClassVar[str] = "agents"
    COLUMNS: ClassVar[str] = "id, created_at, name, system_prompt_id, parent_id, conversation_id, model, metadata, model_config"
    SELECT_SQL: ClassVar[str] = f"SELECT {COLUMNS} FROM agents WHERE id = $1"

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def from_row(cls, row: Any) -> "PersistentAgent":
        """Construct from a database row."""
        return cls(
            id=row["id"],
            created_at=row["created_at"],
            name=row["name"],
            system_prompt_id=row["system_prompt_id"],
            parent_id=row["parent_id"],
            conversation_id=row["conversation_id"],
            model=row["model"],
            metadata=MappingProxyType(json.loads(row["metadata"]) if row["metadata"] else {}),
            model_config=MappingProxyType(json.loads(row["model_config"]) if row["model_config"] else {}),
        )

    def to_insert_params(self) -> tuple[str, tuple[Any, ...]]:
        """Return (INSERT SQL, parameters) for this asset."""
        return (
            """INSERT INTO agents (id, created_at, name, system_prompt_id, parent_id,
                                   conversation_id, model, metadata, model_config)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) ON CONFLICT (id) DO NOTHING""",
            (
                self.id,
                self.created_at,
                self.name,
                self.system_prompt_id,
                self.parent_id,
                self.conversation_id,
                self.model,
                json.dumps(dict(self.metadata)),
                json.dumps(dict(self.model_config)),
            ),
        )

    @classmethod
    def _create(
        cls,
        *,
        name: str,
        system_prompt_id: UUID,
        conversation_id: UUID,
        model: str,
        metadata: Mapping[str, Any] | None = None,
        model_config: Mapping[str, Any] | None = None,
    ) -> PersistentAgent:
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
            metadata=MappingProxyType(dict(metadata) if metadata else {}),
            model_config=MappingProxyType(dict(model_config) if model_config else {}),
        )

    def _evolve(
        self,
        conversation: messages.Conversation,
        metadata: Mapping[str, Any] | None = None,
    ) -> PersistentAgent:
        """Create a new agent state with an updated conversation (internal).

        The new agent links back to this one via parent_id.
        Metadata and model_config are inherited from the current agent.
        """
        new_agent = PersistentAgent(
            id=assets.new_id(),
            created_at=assets.now(),
            name=self.name,
            system_prompt_id=self.system_prompt_id,
            parent_id=self.id,
            conversation_id=conversation.id,
            model=self.model,
            metadata=MappingProxyType(dict(metadata)) if metadata is not None else self.metadata,
            model_config=self.model_config,
        )
        # Register new agent with the same store as the parent
        register_agent(new_agent, get_store(self))
        return new_agent

    async def with_metadata(self, metadata: Mapping[str, Any]) -> PersistentAgent:
        """Create a new agent with updated metadata.

        The new agent has the same conversation but new metadata.
        Useful for updating agent state between turns.

        Args:
            metadata: New metadata (replaces existing metadata)

        Returns:
            A new agent with updated metadata
        """
        return await get_store(self)._update_metadata(self, metadata)

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
    ) -> PersistentAgent:
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
        return await get_store(self)._advance(
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

    async def messages(self) -> tuple[messages.Message, ...]:
        """Get all messages in this agent's conversation."""
        return await get_store(self)._agent_messages(self)

    async def last_response(self) -> str | None:
        """Get the last assistant response, or None if no responses yet."""
        msgs = await self.messages()
        for msg in reversed(msgs):
            if msg.role == "assistant" and msg.content:
                return msg.content
        return None

    async def lineage(self) -> list[PersistentAgent]:
        """Get the chain of agents from root to this agent."""
        return await get_store(self)._agent_lineage(self)

    async def clone(self) -> PersistentAgent:
        """Create a sibling clone of this agent for branching.

        The clone has a new UUID but shares the same parent_id, conversation,
        and system prompt. This creates a sibling in the lineage tree, not a child.

        Use this to explore different conversation branches from the same point:
            agent_a = await agent.clone()
            agent_b = await agent.clone()
            agent_a = await agent_a.advance("Option A")
            agent_b = await agent_b.advance("Option B")
        """
        return await get_store(self)._clone_agent(self)

    async def token_usage(self) -> tuple[int, int]:
        """Get total token usage for this agent's conversation.

        Returns:
            A tuple of (input_tokens, output_tokens) summed across all
            assistant messages in the conversation.
        """
        msgs = await self.messages()
        input_tokens = sum(m.input_tokens or 0 for m in msgs if m.role == "assistant")
        output_tokens = sum(m.output_tokens or 0 for m in msgs if m.role == "assistant")
        return input_tokens, output_tokens
