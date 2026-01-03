"""Custom exceptions for immagent."""

from uuid import UUID


class ImmAgentError(Exception):
    """Base exception for all immagent errors."""

    pass


class ValidationError(ImmAgentError):
    """Raised when input validation fails."""

    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(f"{field}: {message}")


class AssetNotFoundError(ImmAgentError):
    """Raised when an asset cannot be found in the database or cache."""

    def __init__(self, asset_type: str, asset_id: UUID):
        self.asset_type = asset_type
        self.asset_id = asset_id
        super().__init__(f"{asset_type} {asset_id} not found")


class ConversationNotFoundError(AssetNotFoundError):
    """Internal error: agent's conversation is missing.

    This indicates database corruption or a bug. Not part of the public API.
    """

    def __init__(self, conversation_id: UUID):
        super().__init__("Conversation", conversation_id)


class SystemPromptNotFoundError(AssetNotFoundError):
    """Internal error: agent's system prompt is missing.

    This indicates database corruption or a bug. Not part of the public API.
    """

    def __init__(self, prompt_id: UUID):
        super().__init__("System prompt", prompt_id)


class AgentNotFoundError(AssetNotFoundError):
    """Raised when an agent cannot be found."""

    def __init__(self, agent_id: UUID):
        super().__init__("Agent", agent_id)


class MessageNotFoundError(AssetNotFoundError):
    """Raised when a message cannot be found."""

    def __init__(self, message_id: UUID):
        super().__init__("Message", message_id)


class LLMError(ImmAgentError):
    """Raised when an LLM call fails."""

    pass


class ToolExecutionError(ImmAgentError):
    """Raised when MCP tool execution fails."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class AgentNotRegisteredError(ImmAgentError):
    """Raised when an agent is not associated with a store.

    This typically means the agent was created outside the normal flow
    (not via Store.create_agent or loaded via Store.load_agent).
    """

    def __init__(self, agent_id: UUID):
        self.agent_id = agent_id
        super().__init__(f"Agent {agent_id} is not associated with a store")
