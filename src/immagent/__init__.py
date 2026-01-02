"""ImmAgent - Immutable Agent Architecture.

A Python library implementing immutable agents where every state transition
creates a new agent with a fresh UUID4.
"""

from immagent.agent import ImmAgent
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
from immagent.messages import Message
from immagent.store import MemoryStore, Store

__all__ = [
    # Core types
    "ImmAgent",
    "Message",
    # Store (main interface)
    "Store",
    "MemoryStore",
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
