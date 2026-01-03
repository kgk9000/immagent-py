"""Store - unified cache and database access for agents.

The Store is the main interface for working with agents. It combines:
- Database persistence (PostgreSQL)
- In-memory weak reference caching
- Agent lifecycle operations (create, advance, load)
"""

import asyncio
import json
import threading
import weakref
from collections.abc import MutableMapping
from typing import Any
from uuid import UUID

import asyncpg

import immagent.assets as assets
import immagent.exceptions as exc
import immagent.llm as llm
import immagent.mcp as mcp_mod
import immagent.messages as messages
from immagent.agent import ImmAgent
from immagent.logging import logger

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

    Usage with PostgreSQL:
        async with await Store.connect("postgresql://...") as store:
            agent = await store.create_agent(
                name="Bot",
                system_prompt="You are helpful.",
                model=Model.CLAUDE_3_5_HAIKU,
            )
            agent = await agent.advance("Hello!")

    For in-memory stores without persistence, use MemoryStore instead.
    """

    def __init__(self, pool: asyncpg.Pool | None = None):
        self._pool = pool
        self._cache: MutableMapping[UUID, assets.Asset] = weakref.WeakValueDictionary()
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
        """Close the database connection pool (if any)."""
        if self._pool is not None:
            await self._pool.close()

    async def __aenter__(self) -> "Store":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def init_schema(self) -> None:
        """Initialize the database schema (creates tables if not exist).

        No-op for in-memory stores.
        """
        if self._pool is None:
            return
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

    def _message_from_row(self, row: asyncpg.Record) -> messages.Message:
        """Build a Message from a database row."""
        tool_calls = None
        if row["tool_calls"]:
            tc_data = json.loads(row["tool_calls"])
            tool_calls = tuple(
                messages.ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                for tc in tc_data
            )
        return messages.Message(
            id=row["id"],
            created_at=row["created_at"],
            role=row["role"],
            content=row["content"],
            tool_calls=tool_calls,
            tool_call_id=row["tool_call_id"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
        )

    async def _get_system_prompt(self, asset_id: UUID) -> assets.SystemPrompt | None:
        cached = self._get_cached(asset_id)
        if cached is not None:
            return cached if isinstance(cached, assets.SystemPrompt) else None

        if self._pool is None:
            return None

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, created_at, content FROM text_assets WHERE id = $1",
                asset_id,
            )
            if row:
                asset = assets.SystemPrompt(
                    id=row["id"],
                    created_at=row["created_at"],
                    content=row["content"],
                )
                self._cache_asset(asset)
                return asset
        return None

    async def _get_message(self, message_id: UUID) -> messages.Message | None:
        cached = self._get_cached(message_id)
        if cached is not None:
            return cached if isinstance(cached, messages.Message) else None

        if self._pool is None:
            return None

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT id, created_at, role, content, tool_calls, tool_call_id,
                          input_tokens, output_tokens
                   FROM messages WHERE id = $1""",
                message_id,
            )
            if row:
                msg = self._message_from_row(row)
                self._cache_asset(msg)
                return msg
        return None

    async def _get_messages(self, message_ids: tuple[UUID, ...]) -> list[messages.Message]:
        if not message_ids:
            return []

        msgs_by_id: dict[UUID, messages.Message] = {}
        to_load: list[UUID] = []

        for mid in message_ids:
            cached = self._get_cached(mid)
            if cached is not None and isinstance(cached, messages.Message):
                msgs_by_id[mid] = cached
            else:
                to_load.append(mid)

        if to_load and self._pool is not None:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT id, created_at, role, content, tool_calls, tool_call_id,
                              input_tokens, output_tokens
                       FROM messages WHERE id = ANY($1)""",
                    to_load,
                )
            for row in rows:
                msg = self._message_from_row(row)
                self._cache_asset(msg)
                msgs_by_id[msg.id] = msg

        # Verify all messages were found
        for mid in message_ids:
            if mid not in msgs_by_id:
                raise exc.MessageNotFoundError(mid)

        return [msgs_by_id[mid] for mid in message_ids]

    async def _get_conversation(self, conversation_id: UUID) -> messages.Conversation | None:
        cached = self._get_cached(conversation_id)
        if cached is not None:
            return cached if isinstance(cached, messages.Conversation) else None

        if self._pool is None:
            return None

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, created_at, message_ids FROM conversations WHERE id = $1",
                conversation_id,
            )
            if row:
                conv = messages.Conversation(
                    id=row["id"],
                    created_at=row["created_at"],
                    message_ids=tuple(row["message_ids"]),
                )
                self._cache_asset(conv)
                return conv
        return None

    async def _get_agent(self, agent_id: UUID) -> ImmAgent | None:
        cached = self._get_cached(agent_id)
        if cached is not None:
            return cached if isinstance(cached, ImmAgent) else None

        if self._pool is None:
            return None

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, created_at, name, system_prompt_id, parent_id, conversation_id, model, metadata, model_config
                FROM agents WHERE id = $1
                """,
                agent_id,
            )
            if row:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                model_config = json.loads(row["model_config"]) if row["model_config"] else {}
                agent = ImmAgent(
                    id=row["id"],
                    created_at=row["created_at"],
                    name=row["name"],
                    system_prompt_id=row["system_prompt_id"],
                    parent_id=row["parent_id"],
                    conversation_id=row["conversation_id"],
                    model=row["model"],
                    metadata=metadata,
                    model_config=model_config,
                    _store=self,
                )
                self._cache_asset(agent)
                return agent
        return None

    # -- Save operations --

    async def _save_one(
        self, conn: asyncpg.Connection | asyncpg.pool.PoolConnectionProxy, asset: assets.Asset
    ) -> None:
        match asset:
            case ImmAgent():
                await conn.execute(
                    """
                    INSERT INTO agents (id, created_at, name, system_prompt_id, parent_id, conversation_id, model, metadata, model_config)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    asset.id,
                    asset.created_at,
                    asset.name,
                    asset.system_prompt_id,
                    asset.parent_id,
                    asset.conversation_id,
                    asset.model,
                    json.dumps(asset.metadata),
                    json.dumps(asset.model_config),
                )
            case messages.Conversation():
                await conn.execute(
                    """
                    INSERT INTO conversations (id, created_at, message_ids)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    asset.id,
                    asset.created_at,
                    list(asset.message_ids),
                )
            case messages.Message():
                tool_calls_json = None
                if asset.tool_calls:
                    tool_calls_json = json.dumps(
                        [
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in asset.tool_calls
                        ]
                    )
                await conn.execute(
                    """
                    INSERT INTO messages (id, created_at, role, content, tool_calls, tool_call_id,
                                          input_tokens, output_tokens)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    asset.id,
                    asset.created_at,
                    asset.role,
                    asset.content,
                    tool_calls_json,
                    asset.tool_call_id,
                    asset.input_tokens,
                    asset.output_tokens,
                )
            case assets.SystemPrompt():
                await conn.execute(
                    """
                    INSERT INTO text_assets (id, created_at, content)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    asset.id,
                    asset.created_at,
                    asset.content,
                )
            case _:
                raise TypeError(f"Unknown asset type: {type(asset)}")

    async def _save(self, *assets_to_save: assets.Asset) -> None:
        """Save assets to the database atomically (internal).

        All assets are saved in a single transaction.
        When saving an ImmAgent, its dependencies (system prompt, conversation)
        are automatically saved first if they're in the cache.

        For in-memory stores (no pool), only caches the assets.
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
            if isinstance(asset, ImmAgent):
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

        # Write to database if we have a pool
        if self._pool is not None:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    for asset in all_assets:
                        await self._save_one(conn, asset)

        # Always cache them
        self._cache_all(*all_assets)

    # -- Public API --

    async def create_agent(
        self,
        *,
        name: str,
        system_prompt: str,
        model: str,
        metadata: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> ImmAgent:
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
        model_str = model
        agent = ImmAgent._create(
            name=name,
            system_prompt_id=prompt_asset.id,
            conversation_id=conversation.id,
            model=model_str,
            store=self,
            metadata=metadata,
            model_config=model_config,
        )

        # Cache first (_save() looks up dependencies in cache)
        self._cache_all(prompt_asset, conversation, agent)

        # Save to database
        await self._save(agent)

        return agent

    async def load_agent(self, agent_id: UUID) -> ImmAgent:
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

    async def load_agents(self, agent_ids: list[UUID]) -> list[ImmAgent]:
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

        agents_by_id: dict[UUID, ImmAgent] = {}
        to_load: list[UUID] = []

        # Check cache first
        for aid in agent_ids:
            cached = self._get_cached(aid)
            if cached is not None and isinstance(cached, ImmAgent):
                agents_by_id[aid] = cached
            else:
                to_load.append(aid)

        # Batch load from DB
        if to_load and self._pool is not None:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, created_at, name, system_prompt_id, parent_id,
                           conversation_id, model, metadata, model_config
                    FROM agents WHERE id = ANY($1)
                    """,
                    to_load,
                )
            for row in rows:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                model_config = json.loads(row["model_config"]) if row["model_config"] else {}
                agent = ImmAgent(
                    id=row["id"],
                    created_at=row["created_at"],
                    name=row["name"],
                    system_prompt_id=row["system_prompt_id"],
                    parent_id=row["parent_id"],
                    conversation_id=row["conversation_id"],
                    model=row["model"],
                    metadata=metadata,
                    model_config=model_config,
                    _store=self,
                )
                self._cache_asset(agent)
                agents_by_id[agent.id] = agent

        # Verify all agents were found and return in order
        result: list[ImmAgent] = []
        for aid in agent_ids:
            if aid not in agents_by_id:
                raise exc.AgentNotFoundError(aid)
            result.append(agents_by_id[aid])

        return result

    async def delete(self, agent: ImmAgent) -> None:
        """Delete an agent from the database and cache.

        Only deletes the agent record. Use gc() to clean up orphaned assets.

        Args:
            agent: The agent to delete
        """
        if self._pool is not None:
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
    ) -> list[ImmAgent]:
        """List agents with pagination and optional filtering.

        Args:
            limit: Maximum number of agents to return (default: 100)
            offset: Number of agents to skip (default: 0)
            name: Optional name filter (substring match, case-insensitive)

        Returns:
            List of agents ordered by created_at descending (newest first)
        """
        if self._pool is None:
            # MemoryStore: filter in-memory cache
            agents = [a for a in self._cache.values() if isinstance(a, ImmAgent)]
            if name:
                name_lower = name.lower()
                agents = [a for a in agents if name_lower in a.name.lower()]
            agents.sort(key=lambda a: a.created_at, reverse=True)
            return agents[offset : offset + limit]

        # PostgreSQL
        if name:
            query = """
                SELECT id, created_at, name, system_prompt_id, parent_id,
                       conversation_id, model, metadata, model_config
                FROM agents
                WHERE name ILIKE $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
            """
            args = (f"%{name}%", limit, offset)
        else:
            query = """
                SELECT id, created_at, name, system_prompt_id, parent_id,
                       conversation_id, model, metadata, model_config
                FROM agents
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
            """
            args = (limit, offset)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *args)

        agents = []
        for row in rows:
            # Check cache first
            cached = self._get_cached(row["id"])
            if cached is not None and isinstance(cached, ImmAgent):
                agents.append(cached)
            else:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                model_config = json.loads(row["model_config"]) if row["model_config"] else {}
                agent = ImmAgent(
                    id=row["id"],
                    created_at=row["created_at"],
                    name=row["name"],
                    system_prompt_id=row["system_prompt_id"],
                    parent_id=row["parent_id"],
                    conversation_id=row["conversation_id"],
                    model=row["model"],
                    metadata=metadata,
                    model_config=model_config,
                    _store=self,
                )
                self._cache_asset(agent)
                agents.append(agent)

        return agents

    async def count_agents(self, *, name: str | None = None) -> int:
        """Count total number of agents.

        Args:
            name: Optional name filter (substring match, case-insensitive)

        Returns:
            Total count of matching agents
        """
        if self._pool is None:
            # MemoryStore: count in-memory cache
            agents = [a for a in self._cache.values() if isinstance(a, ImmAgent)]
            if name:
                name_lower = name.lower()
                agents = [a for a in agents if name_lower in a.name.lower()]
            return len(agents)

        # PostgreSQL
        async with self._pool.acquire() as conn:
            if name:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM agents WHERE name ILIKE $1",
                    f"%{name}%",
                )
            else:
                count = await conn.fetchval("SELECT COUNT(*) FROM agents")

        return count or 0

    async def find_by_name(self, name: str) -> list[ImmAgent]:
        """Find agents by exact name match.

        Args:
            name: Exact name to match (case-sensitive)

        Returns:
            List of agents with the given name, ordered by created_at descending
        """
        if self._pool is None:
            # MemoryStore: filter in-memory cache
            agents = [
                a for a in self._cache.values()
                if isinstance(a, ImmAgent) and a.name == name
            ]
            agents.sort(key=lambda a: a.created_at, reverse=True)
            return agents

        # PostgreSQL
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, created_at, name, system_prompt_id, parent_id,
                       conversation_id, model, metadata, model_config
                FROM agents
                WHERE name = $1
                ORDER BY created_at DESC
                """,
                name,
            )

        agents = []
        for row in rows:
            cached = self._get_cached(row["id"])
            if cached is not None and isinstance(cached, ImmAgent):
                agents.append(cached)
            else:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                model_config = json.loads(row["model_config"]) if row["model_config"] else {}
                agent = ImmAgent(
                    id=row["id"],
                    created_at=row["created_at"],
                    name=row["name"],
                    system_prompt_id=row["system_prompt_id"],
                    parent_id=row["parent_id"],
                    conversation_id=row["conversation_id"],
                    model=row["model"],
                    metadata=metadata,
                    model_config=model_config,
                    _store=self,
                )
                self._cache_asset(agent)
                agents.append(agent)

        return agents

    async def gc(self) -> dict[str, int]:
        """Garbage collect orphaned assets.

        Deletes conversations, messages, and text_assets that are no longer
        referenced by any agent. Safe to call anytime.

        No-op for in-memory stores.

        Returns:
            Dict with counts of deleted assets by type.
        """
        if self._pool is None:
            return {"text_assets": 0, "conversations": 0, "messages": 0}

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Delete orphaned text_assets (system prompts not used by any agent)
                deleted = await conn.fetch("""
                    DELETE FROM text_assets
                    WHERE id NOT IN (SELECT system_prompt_id FROM agents)
                    RETURNING id
                """)
                text_assets_count = len(deleted)

                # Delete orphaned conversations
                deleted = await conn.fetch("""
                    DELETE FROM conversations
                    WHERE id NOT IN (SELECT conversation_id FROM agents)
                    RETURNING id
                """)
                conversations_count = len(deleted)

                # Delete orphaned messages
                deleted = await conn.fetch("""
                    DELETE FROM messages
                    WHERE id NOT IN (
                        SELECT unnest(message_ids) FROM conversations
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
        agent: ImmAgent,
        user_input: str,
        *,
        mcp: mcp_mod.MCPManager | None = None,
        max_tool_rounds: int = 10,
        max_retries: int = 3,
        timeout: float | None = 120.0,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ) -> ImmAgent:
        """Advance the agent with a user message (internal).

        Use agent.advance() instead.
        """
        # Validate inputs
        if not user_input or not user_input.strip():
            raise exc.ValidationError("user_input", "must not be empty")
        if max_tool_rounds < 1:
            raise exc.ValidationError("max_tool_rounds", "must be at least 1")
        if max_retries < 0:
            raise exc.ValidationError("max_retries", "must be non-negative")
        if timeout is not None and timeout <= 0:
            raise exc.ValidationError("timeout", "must be positive")

        # Build effective model config: agent defaults + call overrides
        effective_config = dict(agent.model_config)
        if temperature is not None:
            effective_config["temperature"] = temperature
        if max_tokens is not None:
            effective_config["max_tokens"] = max_tokens
        if top_p is not None:
            effective_config["top_p"] = top_p

        logger.info(
            "Advancing agent: id=%s, name=%s, model=%s",
            agent.id,
            agent.name,
            agent.model,
        )

        # Load existing conversation and system prompt
        conversation = await self._get_conversation(agent.conversation_id)
        if conversation is None:
            raise exc.ConversationNotFoundError(agent.conversation_id)

        system_prompt = await self._get_system_prompt(agent.system_prompt_id)
        if system_prompt is None:
            raise exc.SystemPromptNotFoundError(agent.system_prompt_id)

        # Load existing messages
        msgs = await self._get_messages(conversation.message_ids)
        logger.debug("Loaded %d existing messages", len(msgs))

        # Create user message
        user_message = messages.Message.user(user_input)
        msgs.append(user_message)

        # Get tools if MCP is available
        tools = mcp.get_all_tools() if mcp else None

        # New messages created in this turn
        new_messages: list[messages.Message] = [user_message]

        # Tool loop - each iteration is one LLM call, possibly followed by tool execution
        llm_calls = 0
        for _ in range(max_tool_rounds):
            # Call LLM
            assistant_message = await llm.complete(
                model=agent.model,
                msgs=msgs,
                system=system_prompt.content,
                tools=tools,
                max_retries=max_retries,
                timeout=timeout,
                model_config=effective_config,
            )
            llm_calls += 1
            msgs.append(assistant_message)
            new_messages.append(assistant_message)

            # Check for tool calls
            if not assistant_message.tool_calls or not mcp:
                break

            # Execute tool calls concurrently
            async def execute_one(tc: messages.ToolCall) -> messages.Message:
                try:
                    result = await mcp.execute(tc.name, tc.arguments)
                except exc.ToolExecutionError as e:
                    result = f"Error: {e}"
                return messages.Message.tool_result(tc.id, result)

            tool_results = await asyncio.gather(
                *(execute_one(tc) for tc in assistant_message.tool_calls)
            )
            for tool_result_message in tool_results:
                msgs.append(tool_result_message)
                new_messages.append(tool_result_message)

        # Create new conversation with all message IDs
        new_conversation = conversation.with_messages(*[m.id for m in new_messages])

        # Create new agent state
        new_agent = agent._evolve(new_conversation)

        # Cache new assets first (save() looks up dependencies in cache)
        self._cache_all(*new_messages, new_conversation, new_agent)

        # Save to database
        await self._save(new_agent)

        logger.info(
            "Agent advanced: old_id=%s, new_id=%s, llm_calls=%d, new_messages=%d",
            agent.id,
            new_agent.id,
            llm_calls,
            len(new_messages),
        )

        return new_agent

    async def _get_agent_messages(self, agent: ImmAgent) -> list[messages.Message]:
        """Get all messages in an agent's conversation (internal).

        Use agent.get_messages() instead.
        """
        conversation = await self._get_conversation(agent.conversation_id)
        if conversation is None:
            raise exc.ConversationNotFoundError(agent.conversation_id)
        return await self._get_messages(conversation.message_ids)

    async def _clone_agent(self, agent: ImmAgent) -> ImmAgent:
        """Create a clone of an agent for branching.

        The clone shares the same parent, conversation, and system prompt,
        allowing you to advance it in a different direction from the original.
        """
        new_agent = ImmAgent(
            id=assets.new_id(),
            created_at=assets.now(),
            name=agent.name,
            system_prompt_id=agent.system_prompt_id,
            parent_id=agent.parent_id,
            conversation_id=agent.conversation_id,
            model=agent.model,
            metadata=agent.metadata,
            model_config=agent.model_config,
            _store=self,
        )
        self._cache_asset(new_agent)
        await self._save(new_agent)
        return new_agent

    async def _update_metadata(self, agent: ImmAgent, metadata: dict[str, Any]) -> ImmAgent:
        """Create a new agent with updated metadata (internal).

        Use agent.with_metadata() instead.
        """
        new_agent = ImmAgent(
            id=assets.new_id(),
            created_at=assets.now(),
            name=agent.name,
            system_prompt_id=agent.system_prompt_id,
            parent_id=agent.id,
            conversation_id=agent.conversation_id,
            model=agent.model,
            metadata=metadata,
            model_config=agent.model_config,
            _store=self,
        )
        self._cache_asset(new_agent)
        await self._save(new_agent)
        return new_agent

    async def _get_agent_lineage(self, agent: ImmAgent) -> list[ImmAgent]:
        """Get the agent's lineage (internal).

        Use agent.get_lineage() instead.

        Uses a recursive CTE for efficient single-query traversal.
        """
        if self._pool is None:
            # MemoryStore: fall back to iterative traversal
            lineage: list[ImmAgent] = [agent]
            current = agent
            while current.parent_id is not None:
                parent = await self._get_agent(current.parent_id)
                if parent is None:
                    raise exc.AgentNotFoundError(current.parent_id)
                lineage.append(parent)
                current = parent
            lineage.reverse()
            return lineage

        # PostgreSQL: use recursive CTE for single-query traversal
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
        lineage = []
        for row in rows:
            # Check cache first
            cached = self._get_cached(row["id"])
            if cached is not None and isinstance(cached, ImmAgent):
                lineage.append(cached)
            else:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                model_config = json.loads(row["model_config"]) if row["model_config"] else {}
                loaded = ImmAgent(
                    id=row["id"],
                    created_at=row["created_at"],
                    name=row["name"],
                    system_prompt_id=row["system_prompt_id"],
                    parent_id=row["parent_id"],
                    conversation_id=row["conversation_id"],
                    model=row["model"],
                    metadata=metadata,
                    model_config=model_config,
                    _store=self,
                )
                self._cache_asset(loaded)
                lineage.append(loaded)

        lineage.reverse()
        return lineage


class MemoryStore(Store):
    """In-memory store with no database persistence.

    Assets persist until explicitly deleted or the store is closed.
    Useful for experimentation, testing, or stateless use cases.

    Usage:
        async with MemoryStore() as store:
            agent = await store.create_agent(
                name="Bot",
                system_prompt="You are helpful.",
                model=Model.CLAUDE_3_5_HAIKU,
            )
            agent = await agent.advance("Hello!")
    """

    def __init__(self) -> None:
        super().__init__(pool=None)
        # Use strong references since there's no DB fallback
        self._cache: MutableMapping[UUID, assets.Asset] = {}
