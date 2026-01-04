"""Store - unified cache and database access for agents.

The Store is the main interface for working with agents. It combines:
- Database persistence (PostgreSQL)
- In-memory weak reference caching
- Agent lifecycle operations (create, advance, load)
"""

import threading
import weakref
from collections.abc import Mapping, MutableMapping
from types import MappingProxyType
from typing import Any
from uuid import UUID

import asyncpg

import immagent.advance as advance_mod
import immagent.assets as assets
import immagent.exceptions as exc
import immagent.mcp as mcp_mod
import immagent.messages as messages
from immagent.persistent import PersistentAgent
from immagent.logging import logger
from immagent.registry import register_agent

SCHEMA = """
-- Text assets (system prompts, etc.)
CREATE TABLE IF NOT EXISTS text_assets (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    content TEXT NOT NULL
);

-- Messages
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_calls JSONB,
    tool_call_id TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER
);

-- Conversations (ordered list of message IDs)
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    message_ids UUID[] NOT NULL
);

-- Agents
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    name TEXT NOT NULL,
    system_prompt_id UUID NOT NULL REFERENCES text_assets(id),
    parent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    model TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    model_config JSONB NOT NULL DEFAULT '{}'
);

-- Indexes for common lookups
CREATE INDEX IF NOT EXISTS idx_agents_parent_id ON agents(parent_id);
CREATE INDEX IF NOT EXISTS idx_agents_conversation_id ON agents(conversation_id);
CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name);
"""


class Store:
    """Unified cache and database access for agents.

    The Store manages both persistence (PostgreSQL) and caching (weak refs).
    It's the main interface for creating, loading, and advancing agents.

    Usage:
        async with await Store.connect("postgresql://...") as store:
            await store.init_schema()
            agent = await store.create_agent(
                name="Bot",
                system_prompt="You are helpful.",
                model=Model.CLAUDE_3_5_HAIKU,
            )
            agent = await agent.advance("Hello!")
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool
        self._cache: MutableMapping[UUID, assets.Asset] = weakref.WeakValueDictionary()
        # threading.RLock (not asyncio.Lock) because: all locked operations are sync
        # dict ops with no await inside, and this protects against multi-threaded access
        self._lock = threading.RLock()

    @classmethod
    async def connect(
        cls,
        dsn: str,
        *,
        min_size: int = 2,
        max_size: int = 10,
        max_inactive_connection_lifetime: float = 300.0,
    ) -> "Store":
        """Connect to PostgreSQL and return a Store instance.

        Args:
            dsn: PostgreSQL connection string
            min_size: Minimum pool connections (default: 2)
            max_size: Maximum pool connections (default: 10)
            max_inactive_connection_lifetime: Idle timeout in seconds (default: 300)

        Returns:
            A Store instance ready to use
        """
        pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            max_inactive_connection_lifetime=max_inactive_connection_lifetime,
        )
        if pool is None:
            raise RuntimeError("Failed to create database connection pool")
        return cls(pool)


    async def close(self) -> None:
        """Close the database connection pool."""
        await self._pool.close()

    async def ping(self) -> bool:
        """Check if the database connection is alive.

        Returns:
            True if the connection is healthy, False otherwise.

        Example:
            if not await store.ping():
                logger.error("Database connection lost")
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def __aenter__(self) -> "Store":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def init_schema(self) -> None:
        """Initialize the database schema (creates tables if not exist)."""
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA)

    # -- Cache operations --

    def _get_cached(self, asset_id: UUID) -> assets.Asset | None:
        with self._lock:
            return self._cache.get(asset_id)

    def _cache_asset(self, asset: assets.Asset) -> None:
        with self._lock:
            self._cache[asset.id] = asset

    def _cache_all(self, *assets_to_cache: assets.Asset) -> None:
        with self._lock:
            for asset in assets_to_cache:
                self._cache[asset.id] = asset

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        with self._lock:
            self._cache.clear()

    # -- Load operations (cache + db) --

    def _get_or_build_agent(self, row: asyncpg.Record) -> PersistentAgent:
        """Get agent from cache or build from row and cache it."""
        cached = self._get_cached(row["id"])
        if cached is not None and isinstance(cached, PersistentAgent):
            return cached
        agent = PersistentAgent.from_row(row)
        register_agent(agent, self)
        self._cache_asset(agent)
        return agent

    async def _get_system_prompt(self, asset_id: UUID) -> assets.SystemPrompt | None:
        cached = self._get_cached(asset_id)
        if cached is not None:
            return cached if isinstance(cached, assets.SystemPrompt) else None

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(assets.SystemPrompt.SELECT_SQL, asset_id)
            if row:
                asset = assets.SystemPrompt.from_row(row)
                self._cache_asset(asset)
                return asset
        return None

    async def _get_message(self, message_id: UUID) -> messages.Message | None:
        cached = self._get_cached(message_id)
        if cached is not None:
            return cached if isinstance(cached, messages.Message) else None

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(messages.Message.SELECT_SQL, message_id)
            if row:
                msg = messages.Message.from_row(row)
                self._cache_asset(msg)
                return msg
        return None

    async def _get_messages(self, message_ids: tuple[UUID, ...]) -> tuple[messages.Message, ...]:
        if not message_ids:
            return ()

        msgs_by_id: dict[UUID, messages.Message] = {}
        to_load: list[UUID] = []

        for mid in message_ids:
            cached = self._get_cached(mid)
            if cached is not None and isinstance(cached, messages.Message):
                msgs_by_id[mid] = cached
            else:
                to_load.append(mid)

        if to_load:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT id, created_at, role, content, tool_calls, tool_call_id,
                              input_tokens, output_tokens
                       FROM messages WHERE id = ANY($1)""",
                    to_load,
                )
            for row in rows:
                msg = messages.Message.from_row(row)
                self._cache_asset(msg)
                msgs_by_id[msg.id] = msg

        # Verify all messages were found
        for mid in message_ids:
            if mid not in msgs_by_id:
                raise exc.MessageNotFoundError(mid)

        return tuple(msgs_by_id[mid] for mid in message_ids)

    async def _get_conversation(self, conversation_id: UUID) -> messages.Conversation | None:
        cached = self._get_cached(conversation_id)
        if cached is not None:
            return cached if isinstance(cached, messages.Conversation) else None

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(messages.Conversation.SELECT_SQL, conversation_id)
            if row:
                conv = messages.Conversation.from_row(row)
                self._cache_asset(conv)
                return conv
        return None

    async def _get_agent(self, agent_id: UUID) -> PersistentAgent | None:
        cached = self._get_cached(agent_id)
        if cached is not None:
            return cached if isinstance(cached, PersistentAgent) else None

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(PersistentAgent.SELECT_SQL, agent_id)
            if row:
                agent = PersistentAgent.from_row(row)
                register_agent(agent, self)
                self._cache_asset(agent)
                return agent
        return None

    # -- Save operations --

    async def _save_one(
        self, conn: asyncpg.Connection | asyncpg.pool.PoolConnectionProxy, asset: assets.Asset
    ) -> None:
        sql, params = asset.to_insert_params()
        await conn.execute(sql, *params)

    async def _save(self, *assets_to_save: assets.Asset) -> None:
        """Save assets to the database atomically (internal).

        All assets are saved in a single transaction.
        When saving an PersistentAgent, its dependencies (system prompt, conversation)
        are automatically saved first if they're in the cache.
        """
        if not assets_to_save:
            return

        # Collect all assets to save, including dependencies
        all_assets: list[assets.Asset] = []
        seen: set[UUID] = set()

        for asset in assets_to_save:
            if asset.id in seen:
                continue

            # For agents, add dependencies first (order matters for foreign keys)
            if isinstance(asset, PersistentAgent):
                # Add system prompt if in cache
                prompt = self._get_cached(asset.system_prompt_id)
                if prompt is not None and prompt.id not in seen:
                    all_assets.append(prompt)
                    seen.add(prompt.id)

                # Add conversation and its messages if in cache
                conv = self._get_cached(asset.conversation_id)
                if conv is not None and conv.id not in seen:
                    if isinstance(conv, messages.Conversation):
                        # Add messages first
                        for msg_id in conv.message_ids:
                            msg = self._get_cached(msg_id)
                            if msg is not None and msg.id not in seen:
                                all_assets.append(msg)
                                seen.add(msg.id)
                    all_assets.append(conv)
                    seen.add(conv.id)

            all_assets.append(asset)
            seen.add(asset.id)

        # Write to database
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for asset in all_assets:
                    await self._save_one(conn, asset)

        # Cache them
        self._cache_all(*all_assets)

    # -- Public API --

    async def create_agent(
        self,
        *,
        name: str,
        system_prompt: str,
        model: str,
        metadata: Mapping[str, Any] | None = None,
        model_config: Mapping[str, Any] | None = None,
    ) -> PersistentAgent:
        """Create a new agent with an empty conversation.

        The agent is saved immediately and cached.

        Args:
            name: Human-readable name for the agent
            system_prompt: The system prompt content
            model: LiteLLM model string (e.g., Model.CLAUDE_3_5_HAIKU)
            metadata: Optional custom key-value data for the agent
            model_config: Optional LLM configuration (temperature, max_tokens, top_p, etc.)

        Returns:
            The new agent

        Raises:
            ValidationError: If any input is invalid
        """
        # Validate inputs
        if not name or not name.strip():
            raise exc.ValidationError("name", "must not be empty")
        if not system_prompt or not system_prompt.strip():
            raise exc.ValidationError("system_prompt", "must not be empty")
        if not model or not model.strip():
            raise exc.ValidationError("model", "must not be empty")

        prompt_asset = assets.SystemPrompt.create(system_prompt)
        conversation = messages.Conversation.create()
        agent = PersistentAgent._create(
            name=name,
            system_prompt_id=prompt_asset.id,
            conversation_id=conversation.id,
            model=model,
            metadata=metadata,
            model_config=model_config,
        )

        # Register agent with this store
        register_agent(agent, self)

        # Cache first (_save() looks up dependencies in cache)
        self._cache_all(prompt_asset, conversation, agent)

        # Save to database
        await self._save(agent)

        return agent

    async def load_agent(self, agent_id: UUID) -> PersistentAgent:
        """Load an agent by ID.

        Args:
            agent_id: The agent's UUID

        Returns:
            The agent

        Raises:
            AgentNotFoundError: If no agent exists with the given ID
        """
        agent = await self._get_agent(agent_id)
        if agent is None:
            raise exc.AgentNotFoundError(agent_id)
        return agent

    async def load_agents(self, agent_ids: list[UUID]) -> list[PersistentAgent]:
        """Load multiple agents by ID in a single batch.

        More efficient than calling load_agent() multiple times.

        Args:
            agent_ids: List of agent UUIDs to load

        Returns:
            List of agents in the same order as the input IDs

        Raises:
            AgentNotFoundError: If any agent ID is not found
        """
        if not agent_ids:
            return []

        agents_by_id: dict[UUID, PersistentAgent] = {}
        to_load: list[UUID] = []

        # Check cache first
        for aid in agent_ids:
            cached = self._get_cached(aid)
            if cached is not None and isinstance(cached, PersistentAgent):
                agents_by_id[aid] = cached
            else:
                to_load.append(aid)

        # Batch load from DB
        if to_load:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    f"SELECT {PersistentAgent.COLUMNS} FROM agents WHERE id = ANY($1)",
                    to_load,
                )
            for row in rows:
                agent = PersistentAgent.from_row(row)
                register_agent(agent, self)
                self._cache_asset(agent)
                agents_by_id[agent.id] = agent

        # Verify all agents were found and return in order
        result: list[PersistentAgent] = []
        for aid in agent_ids:
            if aid not in agents_by_id:
                raise exc.AgentNotFoundError(aid)
            result.append(agents_by_id[aid])

        return result

    async def delete(self, agent: PersistentAgent) -> None:
        """Delete an agent from the database and cache.

        Only deletes the agent record. Use gc() to clean up orphaned assets.

        Args:
            agent: The agent to delete
        """
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM agents WHERE id = $1", agent.id)

        with self._lock:
            self._cache.pop(agent.id, None)

    async def list_agents(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        name: str | None = None,
    ) -> list[PersistentAgent]:
        """List agents with pagination and optional filtering.

        Args:
            limit: Maximum number of agents to return (default: 100)
            offset: Number of agents to skip (default: 0)
            name: Optional name filter (substring match, case-insensitive)

        Returns:
            List of agents ordered by created_at descending (newest first)
        """
        if name:
            query = f"""
                SELECT {PersistentAgent.COLUMNS} FROM agents
                WHERE name ILIKE $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
            """
            args = (f"%{name}%", limit, offset)
        else:
            query = f"""
                SELECT {PersistentAgent.COLUMNS} FROM agents
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
            """
            args = (limit, offset)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *args)

        return [self._get_or_build_agent(row) for row in rows]

    async def count_agents(self, *, name: str | None = None) -> int:
        """Count total number of agents.

        Args:
            name: Optional name filter (substring match, case-insensitive)

        Returns:
            Total count of matching agents
        """
        async with self._pool.acquire() as conn:
            if name:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM agents WHERE name ILIKE $1",
                    f"%{name}%",
                )
            else:
                count = await conn.fetchval("SELECT COUNT(*) FROM agents")

        return count or 0

    async def find_by_name(self, name: str) -> list[PersistentAgent]:
        """Find agents by exact name match.

        Args:
            name: Exact name to match (case-sensitive)

        Returns:
            List of agents with the given name, ordered by created_at descending
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {PersistentAgent.COLUMNS} FROM agents WHERE name = $1 ORDER BY created_at DESC",
                name,
            )

        return [self._get_or_build_agent(row) for row in rows]

    async def gc(self) -> dict[str, int]:
        """Garbage collect orphaned assets.

        Deletes conversations, messages, and text_assets that are no longer
        referenced by any agent. Safe to call anytime.

        Returns:
            Dict with counts of deleted assets by type.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Delete orphaned text_assets (system prompts not used by any agent)
                deleted = await conn.fetch("""
                    DELETE FROM text_assets t
                    WHERE NOT EXISTS (
                        SELECT 1 FROM agents a WHERE a.system_prompt_id = t.id
                    )
                    RETURNING id
                """)
                text_assets_count = len(deleted)

                # Delete orphaned conversations
                deleted = await conn.fetch("""
                    DELETE FROM conversations c
                    WHERE NOT EXISTS (
                        SELECT 1 FROM agents a WHERE a.conversation_id = c.id
                    )
                    RETURNING id
                """)
                conversations_count = len(deleted)

                # Delete orphaned messages
                deleted = await conn.fetch("""
                    DELETE FROM messages m
                    WHERE NOT EXISTS (
                        SELECT 1 FROM conversations c WHERE m.id = ANY(c.message_ids)
                    )
                    RETURNING id
                """)
                messages_count = len(deleted)

        return {
            "text_assets": text_assets_count,
            "conversations": conversations_count,
            "messages": messages_count,
        }

    async def _advance(
        self,
        agent: PersistentAgent,
        user_input: str,
        *,
        mcp: mcp_mod.MCPManager | None = None,
        max_tool_rounds: int = 10,
        max_retries: int = 3,
        timeout: float | None = 120.0,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ) -> PersistentAgent:
        """Advance the agent with a user message (internal).

        Use agent.advance() instead.
        """
        logger.info(
            "Advancing agent: id=%s, name=%s, model=%s",
            agent.id,
            agent.name,
            agent.model,
        )

        # Load conversation and system prompt
        conversation = await self._get_conversation(agent.conversation_id)
        if conversation is None:
            raise exc.ConversationNotFoundError(agent.conversation_id)

        system_prompt = await self._get_system_prompt(agent.system_prompt_id)
        if system_prompt is None:
            raise exc.SystemPromptNotFoundError(agent.system_prompt_id)

        # Load existing messages
        history = await self._get_messages(conversation.message_ids)

        # Build effective model config: agent defaults + call overrides
        effective_config = dict(agent.model_config)
        if temperature is not None:
            effective_config["temperature"] = temperature
        if max_tokens is not None:
            effective_config["max_tokens"] = max_tokens
        if top_p is not None:
            effective_config["top_p"] = top_p

        # Run LLM orchestration (pure function, no persistence)
        new_messages = await advance_mod.advance(
            model=agent.model,
            system_prompt=system_prompt.content,
            history=history,
            user_input=user_input,
            mcp=mcp,
            max_tool_rounds=max_tool_rounds,
            max_retries=max_retries,
            timeout=timeout,
            model_config=effective_config,
        )

        # Create new conversation with all message IDs
        new_conversation = conversation.with_messages(*[m.id for m in new_messages])

        # Create new agent state
        new_agent = agent._evolve(new_conversation)

        # Cache and save
        self._cache_all(*new_messages, new_conversation, new_agent)
        await self._save(new_agent)

        logger.info(
            "Agent advanced: old_id=%s, new_id=%s, new_messages=%d",
            agent.id,
            new_agent.id,
            len(new_messages),
        )

        return new_agent

    async def _agent_messages(self, agent: PersistentAgent) -> tuple[messages.Message, ...]:
        """Get all messages in an agent's conversation (internal).

        Use agent.get_messages() instead.
        """
        conversation = await self._get_conversation(agent.conversation_id)
        if conversation is None:
            raise exc.ConversationNotFoundError(agent.conversation_id)
        return await self._get_messages(conversation.message_ids)

    async def _clone_agent(self, agent: PersistentAgent) -> PersistentAgent:
        """Create a clone of an agent for branching.

        The clone shares the same parent, conversation, and system prompt,
        allowing you to advance it in a different direction from the original.
        """
        new_agent = PersistentAgent(
            id=assets.new_id(),
            created_at=assets.now(),
            name=agent.name,
            system_prompt_id=agent.system_prompt_id,
            parent_id=agent.parent_id,
            conversation_id=agent.conversation_id,
            model=agent.model,
            metadata=agent.metadata,
            model_config=agent.model_config,
        )
        register_agent(new_agent, self)
        self._cache_asset(new_agent)
        await self._save(new_agent)
        return new_agent

    async def _update_metadata(self, agent: PersistentAgent, metadata: Mapping[str, Any]) -> PersistentAgent:
        """Create a new agent with updated metadata (internal).

        Use agent.with_metadata() instead.
        """
        new_agent = PersistentAgent(
            id=assets.new_id(),
            created_at=assets.now(),
            name=agent.name,
            system_prompt_id=agent.system_prompt_id,
            parent_id=agent.id,
            conversation_id=agent.conversation_id,
            model=agent.model,
            metadata=MappingProxyType(dict(metadata)),
            model_config=agent.model_config,
        )
        register_agent(new_agent, self)
        self._cache_asset(new_agent)
        await self._save(new_agent)
        return new_agent

    async def _agent_lineage(self, agent: PersistentAgent) -> list[PersistentAgent]:
        """Get the agent's lineage (internal).

        Use agent.get_lineage() instead.

        Uses a recursive CTE for efficient single-query traversal.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH RECURSIVE lineage AS (
                    SELECT id, created_at, name, system_prompt_id, parent_id,
                           conversation_id, model, metadata, model_config
                    FROM agents WHERE id = $1
                    UNION ALL
                    SELECT a.id, a.created_at, a.name, a.system_prompt_id, a.parent_id,
                           a.conversation_id, a.model, a.metadata, a.model_config
                    FROM agents a
                    INNER JOIN lineage l ON a.id = l.parent_id
                )
                SELECT * FROM lineage
                """,
                agent.id,
            )

        if not rows:
            raise exc.AgentNotFoundError(agent.id)

        # Build agents and cache them (rows are child-first, reverse for root-first)
        lineage = [self._get_or_build_agent(row) for row in rows]
        lineage.reverse()
        return lineage
