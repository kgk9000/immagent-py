"""Agent-to-store registry.

This module breaks the circular import between persistent.py and store.py
by providing a central registry that both can import.

Uses WeakKeyDictionary so agents are automatically removed from the
registry when they are garbage collected.
"""

import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from immagent.persistent import PersistentAgent
    from immagent.store import Store

_agent_stores: weakref.WeakKeyDictionary["PersistentAgent", "Store"] = weakref.WeakKeyDictionary()


def get_store(agent: "PersistentAgent") -> "Store":
    """Get the store for an agent."""
    store = _agent_stores.get(agent)
    if store is None:
        raise RuntimeError(f"Agent {agent.id} not associated with a store")
    return store


def register_agent(agent: "PersistentAgent", store: "Store") -> None:
    """Register an agent with its store."""
    _agent_stores[agent] = store
