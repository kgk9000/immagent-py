"""PostgreSQL database persistence for assets."""

import json
from uuid import UUID

import asyncpg

from immagent.agent import ImmAgent
from immagent.assets import Asset, TextAsset
from immagent.messages import Conversation, Message, ToolCall

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
    tool_call_id TEXT
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
    parent_id UUID REFERENCES agents(id),
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    model TEXT NOT NULL
);

-- Indexes for common lookups
CREATE INDEX IF NOT EXISTS idx_agents_parent_id ON agents(parent_id);
CREATE INDEX IF NOT EXISTS idx_agents_conversation_id ON agents(conversation_id);
"""


class Database:
    """Async PostgreSQL database connection for asset persistence."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str) -> "Database":
        """Connect to PostgreSQL and return a Database instance."""
        pool = await asyncpg.create_pool(dsn)
        return cls(pool)

    async def close(self) -> None:
        """Close the database connection pool."""
        await self._pool.close()

    async def init_schema(self) -> None:
        """Initialize the database schema."""
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA)

    # -- TextAsset --

    async def save_text_asset(self, asset: TextAsset) -> None:
        """Save a TextAsset to the database."""
        async with self._pool.acquire() as conn:
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

    async def load_text_asset(self, asset_id: UUID) -> TextAsset | None:
        """Load a TextAsset by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, created_at, content FROM text_assets WHERE id = $1",
                asset_id,
            )
            if row:
                return TextAsset(
                    id=row["id"],
                    created_at=row["created_at"],
                    content=row["content"],
                )
            return None

    # -- Message --

    async def save_message(self, message: Message) -> None:
        """Save a Message to the database."""
        tool_calls_json = None
        if message.tool_calls:
            tool_calls_json = json.dumps(
                [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in message.tool_calls
                ]
            )

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO messages (id, created_at, role, content, tool_calls, tool_call_id)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO NOTHING
                """,
                message.id,
                message.created_at,
                message.role,
                message.content,
                tool_calls_json,
                message.tool_call_id,
            )

    async def load_message(self, message_id: UUID) -> Message | None:
        """Load a Message by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, created_at, role, content, tool_calls, tool_call_id FROM messages WHERE id = $1",
                message_id,
            )
            if row:
                tool_calls = None
                if row["tool_calls"]:
                    tc_data = json.loads(row["tool_calls"])
                    tool_calls = tuple(
                        ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                        for tc in tc_data
                    )
                return Message(
                    id=row["id"],
                    created_at=row["created_at"],
                    role=row["role"],
                    content=row["content"],
                    tool_calls=tool_calls,
                    tool_call_id=row["tool_call_id"],
                )
            return None

    async def load_messages(self, message_ids: tuple[UUID, ...]) -> list[Message]:
        """Load multiple messages by ID, preserving order."""
        if not message_ids:
            return []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, created_at, role, content, tool_calls, tool_call_id FROM messages WHERE id = ANY($1)",
                list(message_ids),
            )

        # Build a dict for ordering
        messages_by_id: dict[UUID, Message] = {}
        for row in rows:
            tool_calls = None
            if row["tool_calls"]:
                tc_data = json.loads(row["tool_calls"])
                tool_calls = tuple(
                    ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                    for tc in tc_data
                )
            messages_by_id[row["id"]] = Message(
                id=row["id"],
                created_at=row["created_at"],
                role=row["role"],
                content=row["content"],
                tool_calls=tool_calls,
                tool_call_id=row["tool_call_id"],
            )

        # Return in the original order
        return [messages_by_id[mid] for mid in message_ids if mid in messages_by_id]

    # -- Conversation --

    async def save_conversation(self, conversation: Conversation) -> None:
        """Save a Conversation to the database."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO conversations (id, created_at, message_ids)
                VALUES ($1, $2, $3)
                ON CONFLICT (id) DO NOTHING
                """,
                conversation.id,
                conversation.created_at,
                list(conversation.message_ids),
            )

    async def load_conversation(self, conversation_id: UUID) -> Conversation | None:
        """Load a Conversation by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, created_at, message_ids FROM conversations WHERE id = $1",
                conversation_id,
            )
            if row:
                return Conversation(
                    id=row["id"],
                    created_at=row["created_at"],
                    message_ids=tuple(row["message_ids"]),
                )
            return None

    # -- ImmAgent --

    async def save_agent(self, agent: ImmAgent) -> None:
        """Save an ImmAgent to the database."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agents (id, created_at, name, system_prompt_id, parent_id, conversation_id, model)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO NOTHING
                """,
                agent.id,
                agent.created_at,
                agent.name,
                agent.system_prompt_id,
                agent.parent_id,
                agent.conversation_id,
                agent.model,
            )

    async def load_agent(self, agent_id: UUID) -> ImmAgent | None:
        """Load an ImmAgent by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, created_at, name, system_prompt_id, parent_id, conversation_id, model
                FROM agents WHERE id = $1
                """,
                agent_id,
            )
            if row:
                return ImmAgent(
                    id=row["id"],
                    created_at=row["created_at"],
                    name=row["name"],
                    system_prompt_id=row["system_prompt_id"],
                    parent_id=row["parent_id"],
                    conversation_id=row["conversation_id"],
                    model=row["model"],
                )
            return None

    # -- Generic save for any asset --

    async def save(self, asset: Asset) -> None:
        """Save any asset to the appropriate table."""
        if isinstance(asset, ImmAgent):
            await self.save_agent(asset)
        elif isinstance(asset, Conversation):
            await self.save_conversation(asset)
        elif isinstance(asset, Message):
            await self.save_message(asset)
        elif isinstance(asset, TextAsset):
            await self.save_text_asset(asset)
        else:
            raise TypeError(f"Unknown asset type: {type(asset)}")

    async def save_all(self, *assets: Asset) -> None:
        """Save multiple assets."""
        for asset in assets:
            await self.save(asset)
