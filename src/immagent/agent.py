"""The immutable agent type."""

from dataclasses import dataclass
from uuid import UUID

import immagent.assets as assets
import immagent.messages as messages


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

    @classmethod
    def _create(
        cls,
        name: str,
        system_prompt: assets.TextAsset,
        model: str,
        conversation: messages.Conversation | None = None,
    ) -> tuple["ImmAgent", messages.Conversation]:
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
        )
        return agent, conv

    def _evolve(
        self,
        conversation: messages.Conversation,
    ) -> "ImmAgent":
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
        )
