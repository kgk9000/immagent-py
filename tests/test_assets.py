"""Tests for asset persistence and retrieval."""

from immagent import Store
from immagent.agent import ImmAgent
from immagent.assets import TextAsset
from immagent.messages import Conversation, Message, ToolCall


class TestTextAsset:
    async def test_save_and_load(self, store: Store):
        """TextAsset can be saved and loaded."""
        asset = TextAsset.create("Hello, world!")

        await store.save(asset)
        # Clear cache to force db load
        store.clear_cache()
        loaded = await store._get_text_asset(asset.id)

        assert loaded is not None
        assert loaded.id == asset.id
        assert loaded.content == "Hello, world!"

    async def test_cache_hit(self, store: Store):
        """TextAsset is cached after first load."""
        asset = TextAsset.create("Cached content")
        await store.save(asset)
        store.clear_cache()

        # First load - from DB
        loaded1 = await store._get_text_asset(asset.id)
        # Second load - from cache
        loaded2 = await store._get_text_asset(asset.id)

        assert loaded1 is loaded2  # Same object reference


class TestMessage:
    async def test_user_message(self, store: Store):
        """User message can be saved and loaded."""
        msg = Message.user("What's the weather?")

        await store.save(msg)
        store.clear_cache()
        loaded = await store._get_message(msg.id)

        assert loaded is not None
        assert loaded.role == "user"
        assert loaded.content == "What's the weather?"

    async def test_assistant_message_with_tool_calls(self, store: Store):
        """Assistant message with tool calls persists correctly."""
        tool_calls = (
            ToolCall(id="call_123", name="get_weather", arguments='{"city": "NYC"}'),
        )
        msg = Message.assistant("Let me check the weather.", tool_calls=tool_calls)

        await store.save(msg)
        store.clear_cache()
        loaded = await store._get_message(msg.id)

        assert loaded is not None
        assert loaded.role == "assistant"
        assert loaded.tool_calls is not None
        assert len(loaded.tool_calls) == 1
        assert loaded.tool_calls[0].name == "get_weather"

    async def test_tool_result_message(self, store: Store):
        """Tool result message persists correctly."""
        msg = Message.tool_result("call_123", "Sunny, 72°F")

        await store.save(msg)
        store.clear_cache()
        loaded = await store._get_message(msg.id)

        assert loaded is not None
        assert loaded.role == "tool"
        assert loaded.tool_call_id == "call_123"
        assert loaded.content == "Sunny, 72°F"


class TestConversation:
    async def test_empty_conversation(self, store: Store):
        """Empty conversation can be saved and loaded."""
        conv = Conversation.create()

        await store.save(conv)
        store.clear_cache()
        loaded = await store._get_conversation(conv.id)

        assert loaded is not None
        assert loaded.message_ids == ()

    async def test_conversation_with_messages(self, store: Store):
        """Conversation preserves message order."""
        msg1 = Message.user("Hello")
        msg2 = Message.assistant("Hi there!")
        await store.save(msg1, msg2)

        conv = Conversation.create((msg1.id, msg2.id))
        await store.save(conv)
        store.clear_cache()
        loaded = await store._get_conversation(conv.id)

        assert loaded is not None
        assert loaded.message_ids == (msg1.id, msg2.id)

    async def test_with_messages_creates_new_conversation(self, store: Store):
        """with_messages returns a new conversation with new ID."""
        conv1 = Conversation.create()
        msg = Message.user("Hello")

        conv2 = conv1.with_messages(msg.id)

        assert conv2.id != conv1.id
        assert conv2.message_ids == (msg.id,)


class TestImmAgent:
    async def test_create_agent(self, store: Store):
        """Agent can be created and saved."""
        prompt = TextAsset.create("You are helpful.")
        await store.save(prompt)

        agent, conv = ImmAgent._create(
            name="TestBot",
            system_prompt=prompt,
            model="anthropic/claude-sonnet-4-20250514",
            store=store,
        )
        await store.save(conv, agent)
        store.clear_cache()

        loaded = await store._get_agent(agent.id)

        assert loaded is not None
        assert loaded.name == "TestBot"
        assert loaded.system_prompt_id == prompt.id
        assert loaded.parent_id is None

    async def test_evolve_creates_new_agent(self, store: Store):
        """evolve creates a new agent with parent link."""
        prompt = TextAsset.create("You are helpful.")
        await store.save(prompt)

        agent1, conv1 = ImmAgent._create(
            name="TestBot",
            system_prompt=prompt,
            model="anthropic/claude-sonnet-4-20250514",
            store=store,
        )
        await store.save(conv1, agent1)

        # Evolve with new conversation
        msg = Message.user("Hello")
        await store.save(msg)
        conv2 = conv1.with_messages(msg.id)
        await store.save(conv2)

        agent2 = agent1.evolve(conv2)
        await store.save(agent2)

        assert agent2.id != agent1.id
        assert agent2.parent_id == agent1.id
        assert agent2.conversation_id == conv2.id
        assert agent2.name == agent1.name  # Inherited
        assert agent2._store is store  # Store is preserved
