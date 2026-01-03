"""Agent-to-store registry.

This module breaks the circular import between agent.py and store.py
by providing a central registry that both can import.
"""

from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from immagent.store import Store

_agent_stores: dict[UUID, "Store"] = {}


def get_store(agent_id: UUID) -> "Store":
    """Get the store for an agent by ID."""
    store = _agent_stores.get(agent_id)
    if store is None:
        raise RuntimeError(f"Agent {agent_id} not associated with a store")
    return store


def register_agent(agent_id: UUID, store: "Store") -> None:
    """Register an agent with its store."""
    _agent_stores[agent_id] = store
