"""ImmAgent - Immutable Agent Architecture.

A Python library implementing immutable agents where every state transition
creates a new agent with a fresh UUID4.
"""

from immagent.agent import ImmAgent
from immagent.db import Database
from immagent.exceptions import (
    AgentNotFoundError,
    AssetNotFoundError,
    ConversationNotFoundError,
    ImmAgentError,
    SystemPromptNotFoundError,
)
from immagent.llm import Model
from immagent.mcp import MCPManager
from immagent.messages import Message
from immagent.api import (
    advance,
    create_agent,
    get_lineage,
    get_messages,
    load_agent,
)
from immagent.cache import clear_cache

__all__ = [
    # Core types
    "ImmAgent",
    "Message",
    # Database
    "Database",
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
    # Functions
    "advance",
    "clear_cache",
    "create_agent",
    "get_lineage",
    "get_messages",
    "load_agent",
]
