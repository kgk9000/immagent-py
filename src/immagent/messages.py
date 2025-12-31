"""Message types for conversations."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from immagent.assets import Asset, new_id, now


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the assistant.

    This is not an Asset because it's always embedded in a Message.
    """

    id: str  # Tool call ID from the LLM
    name: str  # Tool name
    arguments: str  # JSON string of arguments


@dataclass(frozen=True)
class Message(Asset):
    """An immutable message in a conversation.

    Messages can be from the user, assistant, or tool (for tool results).
    """

    role: Literal["user", "assistant", "tool"]
    content: str | None
    tool_calls: tuple[ToolCall, ...] | None = None
    tool_call_id: str | None = None  # For tool role messages, references the tool call

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(
            id=new_id(),
            created_at=now(),
            role="user",
            content=content,
        )

    @classmethod
    def assistant(
        cls,
        content: str | None,
        tool_calls: tuple[ToolCall, ...] | None = None,
    ) -> "Message":
        """Create an assistant message."""
        return cls(
            id=new_id(),
            created_at=now(),
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        )

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> "Message":
        """Create a tool result message."""
        return cls(
            id=new_id(),
            created_at=now(),
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
        )

    def to_litellm_dict(self) -> dict:
        """Convert to LiteLLM message format."""
        msg: dict = {"role": self.role}

        if self.content is not None:
            msg["content"] = self.content

        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]

        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id

        return msg


@dataclass(frozen=True)
class Conversation(Asset):
    """An immutable conversation, which is an ordered list of message IDs.

    New messages create a new Conversation with a new ID.
    """

    message_ids: tuple[UUID, ...]

    @classmethod
    def create(cls, message_ids: tuple[UUID, ...] | None = None) -> "Conversation":
        """Create a new conversation."""
        return cls(
            id=new_id(),
            created_at=now(),
            message_ids=message_ids or (),
        )

    def with_messages(self, *new_message_ids: UUID) -> "Conversation":
        """Create a new conversation with additional messages appended."""
        return Conversation(
            id=new_id(),
            created_at=now(),
            message_ids=self.message_ids + new_message_ids,
        )
