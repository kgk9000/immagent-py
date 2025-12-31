"""In-memory cache for immutable assets.

Since assets are immutable, caching is safe and efficient.
Once an asset is loaded, it never changes.
"""

from uuid import UUID

from immagent.agent import ImmAgent
from immagent.assets import Asset, TextAsset
from immagent.db import Database
from immagent.messages import Conversation, Message

# Global cache - safe because assets are immutable
_cache: dict[UUID, Asset] = {}


def get_cached(asset_id: UUID) -> Asset | None:
    """Get an asset from cache if present."""
    return _cache.get(asset_id)


def cache(asset: Asset) -> None:
    """Add an asset to the cache."""
    _cache[asset.id] = asset


def cache_all(*assets: Asset) -> None:
    """Add multiple assets to the cache."""
    for asset in assets:
        _cache[asset.id] = asset


def clear_cache() -> None:
    """Clear the entire cache. Useful for testing."""
    _cache.clear()


async def get_text_asset(db: Database, asset_id: UUID) -> TextAsset | None:
    """Get a TextAsset from cache or database."""
    cached = get_cached(asset_id)
    if cached is not None:
        return cached if isinstance(cached, TextAsset) else None

    asset = await db.load_text_asset(asset_id)
    if asset:
        cache(asset)
    return asset


async def get_message(db: Database, message_id: UUID) -> Message | None:
    """Get a Message from cache or database."""
    cached = get_cached(message_id)
    if cached is not None:
        return cached if isinstance(cached, Message) else None

    message = await db.load_message(message_id)
    if message:
        cache(message)
    return message


async def get_messages(db: Database, message_ids: tuple[UUID, ...]) -> list[Message]:
    """Get multiple Messages, using cache where possible."""
    result: list[Message] = []
    to_load: list[UUID] = []

    # Check cache first
    for mid in message_ids:
        cached = get_cached(mid)
        if cached is not None and isinstance(cached, Message):
            result.append(cached)
        else:
            to_load.append(mid)

    # Load missing from database
    if to_load:
        loaded = await db.load_messages(tuple(to_load))
        for msg in loaded:
            cache(msg)

    # Rebuild in correct order
    messages_by_id = {m.id: m for m in result}
    for msg in await db.load_messages(tuple(to_load)):
        messages_by_id[msg.id] = msg

    return [messages_by_id[mid] for mid in message_ids if mid in messages_by_id]


async def get_conversation(db: Database, conversation_id: UUID) -> Conversation | None:
    """Get a Conversation from cache or database."""
    cached = get_cached(conversation_id)
    if cached is not None:
        return cached if isinstance(cached, Conversation) else None

    conversation = await db.load_conversation(conversation_id)
    if conversation:
        cache(conversation)
    return conversation


async def get_agent(db: Database, agent_id: UUID) -> ImmAgent | None:
    """Get an ImmAgent from cache or database."""
    cached = get_cached(agent_id)
    if cached is not None:
        return cached if isinstance(cached, ImmAgent) else None

    agent = await db.load_agent(agent_id)
    if agent:
        cache(agent)
    return agent
