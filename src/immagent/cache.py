"""In-memory cache for immutable assets.

Since assets are immutable, caching is safe and efficient.
Once an asset is loaded, it never changes.

Uses a thread-safe LRU cache with a configurable max size.
"""

import threading
from uuid import UUID

from cachetools import LRUCache

import immagent.agent as agent_mod
import immagent.assets as assets
import immagent.db as db_mod
import immagent.messages as messages

# Thread-safe LRU cache with 10,000 entry limit
_cache: LRUCache[UUID, assets.Asset] = LRUCache(maxsize=10_000)
_lock = threading.RLock()


def get_cached(asset_id: UUID) -> assets.Asset | None:
    """Get an asset from cache if present."""
    with _lock:
        return _cache.get(asset_id)


def cache(asset: assets.Asset) -> None:
    """Add an asset to the cache."""
    with _lock:
        _cache[asset.id] = asset


def cache_all(*assets_to_cache: assets.Asset) -> None:
    """Add multiple assets to the cache."""
    with _lock:
        for asset in assets_to_cache:
            _cache[asset.id] = asset


def clear_cache() -> None:
    """Clear the entire cache. Useful for testing."""
    with _lock:
        _cache.clear()


async def get_text_asset(db: db_mod.Database, asset_id: UUID) -> assets.TextAsset | None:
    """Get a TextAsset from cache or database."""
    cached = get_cached(asset_id)
    if cached is not None:
        return cached if isinstance(cached, assets.TextAsset) else None

    asset = await db.load_text_asset(asset_id)
    if asset:
        cache(asset)
    return asset


async def get_message(db: db_mod.Database, message_id: UUID) -> messages.Message | None:
    """Get a Message from cache or database."""
    cached = get_cached(message_id)
    if cached is not None:
        return cached if isinstance(cached, messages.Message) else None

    message = await db.load_message(message_id)
    if message:
        cache(message)
    return message


async def get_messages(db: db_mod.Database, message_ids: tuple[UUID, ...]) -> list[messages.Message]:
    """Get multiple Messages, using cache where possible."""
    if not message_ids:
        return []

    msgs_by_id: dict[UUID, messages.Message] = {}
    to_load: list[UUID] = []

    # Check cache first
    for mid in message_ids:
        cached = get_cached(mid)
        if cached is not None and isinstance(cached, messages.Message):
            msgs_by_id[mid] = cached
        else:
            to_load.append(mid)

    # Load missing from database
    if to_load:
        loaded = await db.load_messages(tuple(to_load))
        for msg in loaded:
            cache(msg)
            msgs_by_id[msg.id] = msg

    # Return in original order
    return [msgs_by_id[mid] for mid in message_ids if mid in msgs_by_id]


async def get_conversation(db: db_mod.Database, conversation_id: UUID) -> messages.Conversation | None:
    """Get a Conversation from cache or database."""
    cached = get_cached(conversation_id)
    if cached is not None:
        return cached if isinstance(cached, messages.Conversation) else None

    conversation = await db.load_conversation(conversation_id)
    if conversation:
        cache(conversation)
    return conversation


async def get_agent(db: db_mod.Database, agent_id: UUID) -> agent_mod.ImmAgent | None:
    """Get an ImmAgent from cache or database."""
    cached = get_cached(agent_id)
    if cached is not None:
        return cached if isinstance(cached, agent_mod.ImmAgent) else None

    agent = await db.load_agent(agent_id)
    if agent:
        cache(agent)
    return agent
