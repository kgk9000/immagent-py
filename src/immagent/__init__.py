"""ImmAgent - Immutable Agent Architecture.

A Python library implementing immutable agents where every state transition
creates a new agent with a fresh UUID4.
"""

from immagent.assets import SystemPrompt
from immagent.exceptions import (
    AgentNotFoundError,
    AssetNotFoundError,
    ConversationNotFoundError,
    ImmAgentError,
    LLMError,
    MessageNotFoundError,
    SystemPromptNotFoundError,
    ToolExecutionError,
    ValidationError,
)
from immagent.llm import Model
from immagent.mcp import MCPManager
from immagent.messages import Conversation, Message, ToolCall
from immagent.persistent import PersistentAgent
from immagent.simple import SimpleAgent
from immagent.store import Store

__all__ = [
    # Core types
    "SimpleAgent",
    "PersistentAgent",
    "Message",
    "Conversation",
    "ToolCall",
    "SystemPrompt",
    # Store (main interface)
    "Store",
    # MCP
    "MCPManager",
    # Models
    "Model",
    # Exceptions
    "ImmAgentError",
    "AssetNotFoundError",
    "ConversationNotFoundError",
    "SystemPromptNotFoundError",
    "AgentNotFoundError",
    "MessageNotFoundError",
    "LLMError",
    "ToolExecutionError",
    "ValidationError",
]
