"""Tests for asset persistence and retrieval."""

from immagent import Store
from immagent.persistent import PersistentAgent
from immagent.assets import SystemPrompt
from immagent.messages import Conversation, Message, ToolCall


class TestSystemPrompt:
    async def test_save_and_load(self, store: Store):
        """SystemPrompt can be saved and loaded."""
        asset = SystemPrompt.create("Hello, world!")

        await store._save(asset)
        # Clear cache to force db load
        store.clear_cache()
        loaded = await store._get_system_prompt(asset.id)

        assert loaded is not None
        assert loaded.id == asset.id
        assert loaded.content == "Hello, world!"

    async def test_cache_hit(self, store: Store):
        """SystemPrompt is cached after first load."""
        asset = SystemPrompt.create("Cached content")
        await store._save(asset)
        store.clear_cache()

        # First load - from DB
        loaded1 = await store._get_system_prompt(asset.id)
        # Second load - from cache
        loaded2 = await store._get_system_prompt(asset.id)

        assert loaded1 is loaded2  # Same object reference


class TestMessage:
    async def test_user_message(self, store: Store):
        """User message can be saved and loaded."""
        msg = Message.user("What's the weather?")

        await store._save(msg)
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

        await store._save(msg)
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

        await store._save(msg)
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

        await store._save(conv)
        store.clear_cache()
        loaded = await store._get_conversation(conv.id)

        assert loaded is not None
        assert loaded.message_ids == ()

    async def test_conversation_with_messages(self, store: Store):
        """Conversation preserves message order."""
        msg1 = Message.user("Hello")
        msg2 = Message.assistant("Hi there!")
        await store._save(msg1, msg2)

        conv = Conversation.create((msg1.id, msg2.id))
        await store._save(conv)
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


class TestPersistentAgent:
    async def test_create_agent(self, store: Store):
        """Agent can be created and saved."""
        from immagent.registry import register_agent

        prompt = SystemPrompt.create("You are helpful.")
        conv = Conversation.create()
        await store._save(prompt, conv)

        agent = PersistentAgent._create(
            name="TestBot",
            system_prompt_id=prompt.id,
            conversation_id=conv.id,
            model="anthropic/claude-sonnet-4-20250514",
        )
        register_agent(agent, store)
        await store._save(agent)
        store.clear_cache()

        loaded = await store._get_agent(agent.id)

        assert loaded is not None
        assert loaded.name == "TestBot"
        assert loaded.system_prompt_id == prompt.id
        assert loaded.parent_id is None

    async def test_evolve_creates_new_agent(self, store: Store):
        """_evolve creates a new agent with parent link."""
        from immagent.registry import register_agent

        prompt = SystemPrompt.create("You are helpful.")
        conv1 = Conversation.create()
        await store._save(prompt, conv1)

        agent1 = PersistentAgent._create(
            name="TestBot",
            system_prompt_id=prompt.id,
            conversation_id=conv1.id,
            model="anthropic/claude-sonnet-4-20250514",
        )
        register_agent(agent1, store)
        await store._save(agent1)

        # Evolve with new conversation
        msg = Message.user("Hello")
        await store._save(msg)
        conv2 = conv1.with_messages(msg.id)
        await store._save(conv2)

        agent2 = agent1._evolve(conv2)
        await store._save(agent2)

        assert agent2.id != agent1.id
        assert agent2.parent_id == agent1.id
        assert agent2.conversation_id == conv2.id
        assert agent2.name == agent1.name  # Inherited
