"""Custom exceptions for immagent."""

from uuid import UUID


class ImmAgentError(Exception):
    """Base exception for all immagent errors."""

    pass


class AssetNotFoundError(ImmAgentError):
    """Raised when an asset cannot be found in the database or cache."""

    def __init__(self, asset_type: str, asset_id: UUID):
        self.asset_type = asset_type
        self.asset_id = asset_id
        super().__init__(f"{asset_type} {asset_id} not found")


class ConversationNotFoundError(AssetNotFoundError):
    """Raised when a conversation cannot be found."""

    def __init__(self, conversation_id: UUID):
        super().__init__("Conversation", conversation_id)


class SystemPromptNotFoundError(AssetNotFoundError):
    """Raised when a system prompt (TextAsset) cannot be found."""

    def __init__(self, prompt_id: UUID):
        super().__init__("System prompt", prompt_id)


class AgentNotFoundError(AssetNotFoundError):
    """Raised when an agent cannot be found."""

    def __init__(self, agent_id: UUID):
        super().__init__("Agent", agent_id)


class LLMError(ImmAgentError):
    """Raised when an LLM call fails."""

    pass


class ToolExecutionError(ImmAgentError):
    """Raised when MCP tool execution fails."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")
