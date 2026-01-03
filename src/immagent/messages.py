"""Message types for conversations."""

import json
from dataclasses import dataclass
from typing import Any, ClassVar, Literal
from uuid import UUID

import immagent.assets as assets


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the assistant.

    This is not an Asset because it's always embedded in a Message.
    """

    id: str  # Tool call ID from the LLM
    name: str  # Tool name
    arguments: str  # JSON string of arguments


@dataclass(frozen=True)
class Message(assets.Asset):
    """An immutable message in a conversation.

    Messages can be from the user, assistant, or tool (for tool results).
    Assistant messages include token usage from the LLM call.
    """

    role: Literal["user", "assistant", "tool"]
    content: str | None
    tool_calls: tuple[ToolCall, ...] | None = None
    tool_call_id: str | None = None  # For tool role messages, references the tool call
    input_tokens: int | None = None  # Token usage for assistant messages
    output_tokens: int | None = None  # Token usage for assistant messages

    TABLE: ClassVar[str] = "messages"
    SELECT_SQL: ClassVar[str] = """
        SELECT id, created_at, role, content, tool_calls, tool_call_id,
               input_tokens, output_tokens
        FROM messages WHERE id = $1
    """

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(
            id=assets.new_id(),
            created_at=assets.now(),
            role="user",
            content=content,
        )

    @classmethod
    def assistant(
        cls,
        content: str | None,
        tool_calls: tuple[ToolCall, ...] | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> "Message":
        """Create an assistant message."""
        return cls(
            id=assets.new_id(),
            created_at=assets.now(),
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> "Message":
        """Create a tool result message."""
        return cls(
            id=assets.new_id(),
            created_at=assets.now(),
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
        )

    @classmethod
    def from_row(cls, row: Any) -> "Message":
        """Construct from a database row."""
        tool_calls = None
        if row["tool_calls"]:
            tool_calls = tuple(
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                for tc in json.loads(row["tool_calls"])
            )
        return cls(
            id=row["id"],
            created_at=row["created_at"],
            role=row["role"],
            content=row["content"],
            tool_calls=tool_calls,
            tool_call_id=row["tool_call_id"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
        )

    def to_insert_params(self) -> tuple[str, tuple[Any, ...]]:
        """Return (INSERT SQL, parameters) for this asset."""
        tool_calls_json = None
        if self.tool_calls:
            tool_calls_json = json.dumps(
                [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in self.tool_calls]
            )
        return (
            """INSERT INTO messages (id, created_at, role, content, tool_calls,
                                     tool_call_id, input_tokens, output_tokens)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8) ON CONFLICT (id) DO NOTHING""",
            (
                self.id,
                self.created_at,
                self.role,
                self.content,
                tool_calls_json,
                self.tool_call_id,
                self.input_tokens,
                self.output_tokens,
            ),
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
class Conversation(assets.Asset):
    """Internal: ordered list of message IDs for an agent.

    Users interact with messages via agent.get_messages(), not directly
    with Conversation objects. Not part of the public API.

    Why a separate Conversation type instead of storing message_ids on Agent?
    The conversation_id provides identity: if two agents share the same
    conversation_id, they have identical history—checkable in O(1) without
    comparing message lists. This matters for clone() where siblings share
    history until one advances.
    """

    message_ids: tuple[UUID, ...]

    TABLE: ClassVar[str] = "conversations"
    SELECT_SQL: ClassVar[str] = "SELECT id, created_at, message_ids FROM conversations WHERE id = $1"

    @classmethod
    def create(cls, message_ids: tuple[UUID, ...] | None = None) -> "Conversation":
        """Create a new conversation."""
        return cls(
            id=assets.new_id(),
            created_at=assets.now(),
            message_ids=message_ids or (),
        )

    @classmethod
    def from_row(cls, row: Any) -> "Conversation":
        """Construct from a database row."""
        return cls(
            id=row["id"],
            created_at=row["created_at"],
            message_ids=tuple(row["message_ids"]),
        )

    def to_insert_params(self) -> tuple[str, tuple[Any, ...]]:
        """Return (INSERT SQL, parameters) for this asset."""
        return (
            """INSERT INTO conversations (id, created_at, message_ids)
               VALUES ($1, $2, $3) ON CONFLICT (id) DO NOTHING""",
            (self.id, self.created_at, list(self.message_ids)),
        )

    def with_messages(self, *new_message_ids: UUID) -> "Conversation":
        """Create a new conversation with additional messages appended."""
        return Conversation(
            id=assets.new_id(),
            created_at=assets.now(),
            message_ids=self.message_ids + new_message_ids,
        )
