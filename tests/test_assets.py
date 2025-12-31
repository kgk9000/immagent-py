"""Tests for asset persistence and retrieval."""

import pytest

from immagent import Database, ImmAgent, Conversation, Message, TextAsset
from immagent.assets import new_id
from immagent.cache import get_agent, get_conversation, get_message, get_text_asset


class TestTextAsset:
    async def test_save_and_load(self, db: Database):
        """TextAsset can be saved and loaded."""
        asset = TextAsset.create("Hello, world!")

        await db.save_text_asset(asset)
        loaded = await db.load_text_asset(asset.id)

        assert loaded is not None
        assert loaded.id == asset.id
        assert loaded.content == "Hello, world!"

    async def test_cache_hit(self, db: Database):
        """TextAsset is cached after first load."""
        asset = TextAsset.create("Cached content")
        await db.save_text_asset(asset)

        # First load - from DB
        loaded1 = await get_text_asset(db, asset.id)
        # Second load - from cache
        loaded2 = await get_text_asset(db, asset.id)

        assert loaded1 is loaded2  # Same object reference


class TestMessage:
    async def test_user_message(self, db: Database):
        """User message can be saved and loaded."""
        msg = Message.user("What's the weather?")

        await db.save_message(msg)
        loaded = await db.load_message(msg.id)

        assert loaded is not None
        assert loaded.role == "user"
        assert loaded.content == "What's the weather?"

    async def test_assistant_message_with_tool_calls(self, db: Database):
        """Assistant message with tool calls persists correctly."""
        from immagent.messages import ToolCall

        tool_calls = (
            ToolCall(id="call_123", name="get_weather", arguments='{"city": "NYC"}'),
        )
        msg = Message.assistant("Let me check the weather.", tool_calls=tool_calls)

        await db.save_message(msg)
        loaded = await db.load_message(msg.id)

        assert loaded is not None
        assert loaded.role == "assistant"
        assert loaded.tool_calls is not None
        assert len(loaded.tool_calls) == 1
        assert loaded.tool_calls[0].name == "get_weather"

    async def test_tool_result_message(self, db: Database):
        """Tool result message persists correctly."""
        msg = Message.tool_result("call_123", "Sunny, 72°F")

        await db.save_message(msg)
        loaded = await db.load_message(msg.id)

        assert loaded is not None
        assert loaded.role == "tool"
        assert loaded.tool_call_id == "call_123"
        assert loaded.content == "Sunny, 72°F"


class TestConversation:
    async def test_empty_conversation(self, db: Database):
        """Empty conversation can be saved and loaded."""
        conv = Conversation.create()

        await db.save_conversation(conv)
        loaded = await db.load_conversation(conv.id)

        assert loaded is not None
        assert loaded.message_ids == ()

    async def test_conversation_with_messages(self, db: Database):
        """Conversation preserves message order."""
        msg1 = Message.user("Hello")
        msg2 = Message.assistant("Hi there!")
        await db.save_message(msg1)
        await db.save_message(msg2)

        conv = Conversation.create((msg1.id, msg2.id))
        await db.save_conversation(conv)
        loaded = await db.load_conversation(conv.id)

        assert loaded is not None
        assert loaded.message_ids == (msg1.id, msg2.id)

    async def test_with_messages_creates_new_conversation(self, db: Database):
        """with_messages returns a new conversation with new ID."""
        conv1 = Conversation.create()
        msg = Message.user("Hello")

        conv2 = conv1.with_messages(msg.id)

        assert conv2.id != conv1.id
        assert conv2.message_ids == (msg.id,)


class TestImmAgent:
    async def test_create_agent(self, db: Database):
        """Agent can be created and saved."""
        prompt = TextAsset.create("You are helpful.")
        await db.save_text_asset(prompt)

        agent, conv = ImmAgent.create(
            name="TestBot",
            system_prompt=prompt,
            model="anthropic/claude-sonnet-4-20250514",
        )
        await db.save_conversation(conv)
        await db.save_agent(agent)

        loaded = await db.load_agent(agent.id)

        assert loaded is not None
        assert loaded.name == "TestBot"
        assert loaded.system_prompt_id == prompt.id
        assert loaded.parent_id is None

    async def test_evolve_creates_new_agent(self, db: Database):
        """evolve creates a new agent with parent link."""
        prompt = TextAsset.create("You are helpful.")
        await db.save_text_asset(prompt)

        agent1, conv1 = ImmAgent.create(
            name="TestBot",
            system_prompt=prompt,
            model="anthropic/claude-sonnet-4-20250514",
        )
        await db.save_conversation(conv1)
        await db.save_agent(agent1)

        # Evolve with new conversation
        msg = Message.user("Hello")
        await db.save_message(msg)
        conv2 = conv1.with_messages(msg.id)
        await db.save_conversation(conv2)

        agent2 = agent1.evolve(conv2)
        await db.save_agent(agent2)

        assert agent2.id != agent1.id
        assert agent2.parent_id == agent1.id
        assert agent2.conversation_id == conv2.id
        assert agent2.name == agent1.name  # Inherited
