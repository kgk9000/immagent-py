"""ImmAgent - Immutable Agent Architecture.

A Python library implementing immutable agents where every state transition
creates a new agent with a fresh UUID4.
"""

from immagent.agent import ImmAgent
from immagent.assets import Asset, TextAsset
from immagent.db import Database
from immagent.llm import Model
from immagent.mcp import MCPManager
from immagent.messages import Conversation, Message, ToolCall
from immagent.turn import (
    advance,
    create_agent,
    get_lineage,
    get_messages,
    load_agent,
    save,
)

__all__ = [
    # Core types
    "Asset",
    "TextAsset",
    "ImmAgent",
    "Message",
    "ToolCall",
    "Conversation",
    # Database
    "Database",
    # MCP
    "MCPManager",
    # Models
    "Model",
    # Functions
    "advance",
    "create_agent",
    "get_lineage",
    "get_messages",
    "load_agent",
    "save",
]
