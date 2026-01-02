"""Tests for the public API."""

import uuid

import pytest

import immagent
from immagent import MemoryStore, Store
from immagent.messages import Conversation


class TestCreateAgent:
    async def test_creates_agent(self, store: Store):
        """create_agent returns an agent."""
        agent = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        assert agent.name == "TestBot"

    async def test_accepts_string_model(self, store: Store):
        """create_agent accepts a string model."""
        agent = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model="anthropic/claude-3-5-haiku-20241022",
        )

        assert agent.model == "anthropic/claude-3-5-haiku-20241022"

    async def test_accepts_model_enum(self, store: Store):
        """create_agent accepts a Model enum."""
        agent = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        assert agent.model == "anthropic/claude-3-5-haiku-20241022"

    async def test_agent_is_auto_saved(self, store: Store):
        """create_agent auto-saves to database."""
        agent = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        # Clear cache and reload from DB
        store.clear_cache()
        loaded = await store.load_agent(agent.id)

        assert loaded.id == agent.id
        assert loaded.name == "TestBot"


class TestSaveAndLoad:
    async def test_save_and_load_agent(self, store: Store):
        """Agent can be saved and loaded."""
        agent = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        # Auto-saved, so just load
        loaded = await store.load_agent(agent.id)

        assert loaded.id == agent.id
        assert loaded.name == "TestBot"

    async def test_load_nonexistent_agent(self, store: Store):
        """Loading nonexistent agent raises AgentNotFoundError."""
        with pytest.raises(immagent.AgentNotFoundError):
            await store.load_agent(uuid.uuid4())


class TestGetMessages:
    async def test_empty_conversation(self, store: Store):
        """New agent has no messages."""
        agent = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        messages = await agent.get_messages()

        assert messages == []


class TestDelete:
    async def test_delete_removes_agent(self, store: Store):
        """delete() removes agent from database."""
        agent = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )
        agent_id = agent.id

        await store.delete(agent)

        # Should no longer be loadable
        with pytest.raises(immagent.AgentNotFoundError):
            await store.load_agent(agent_id)

    async def test_delete_removes_from_cache(self, store: Store):
        """delete() removes agent from cache."""
        agent = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        await store.delete(agent)

        # Cache should not have the agent
        assert store._get_cached(agent.id) is None


class TestGC:
    async def test_gc_cleans_orphaned_assets(self, store: Store):
        """gc() removes assets not referenced by any agent."""
        agent = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        # Delete the agent
        await store.delete(agent)

        # Run gc
        result = await store.gc()

        # Should have cleaned up the orphaned assets
        assert result["text_assets"] == 1
        assert result["conversations"] == 1
        assert result["messages"] == 0  # No messages yet

    async def test_gc_preserves_shared_assets(self, store: Store):
        """gc() keeps assets still referenced by other agents."""
        agent1 = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        # Copy shares the same system prompt and conversation
        await agent1.copy()

        # Delete the original
        await store.delete(agent1)

        # Run gc
        result = await store.gc()

        # Nothing should be deleted - agent2 still references them
        assert result["text_assets"] == 0
        assert result["conversations"] == 0
        assert result["messages"] == 0


class TestCopy:
    async def test_copy_creates_new_id(self, store: Store):
        """copy() creates agent with new ID but same parent (sibling)."""
        agent1 = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        agent2 = await agent1.copy()

        assert agent2.id != agent1.id
        assert agent2.name == agent1.name
        assert agent2.model == agent1.model
        assert agent2.parent_id == agent1.parent_id  # Sibling, same parent

    async def test_copy_keeps_conversation(self, store: Store):
        """copy() keeps same conversation by default."""
        agent1 = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        agent2 = await agent1.copy()

        assert agent2.conversation_id == agent1.conversation_id



class TestGetLineage:
    async def test_single_agent_lineage(self, store: Store):
        """Single agent's lineage is just itself."""
        agent = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        lineage = await agent.get_lineage()

        assert len(lineage) == 1
        assert lineage[0].id == agent.id

    async def test_evolved_agent_lineage(self, store: Store):
        """Evolved agent's lineage includes parent."""
        agent1 = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        # Manually evolve (simulating what advance would do)
        conv = Conversation.create()
        store._cache_all(conv)
        agent2 = agent1._evolve(conv)
        await store._save(agent2)

        lineage = await agent2.get_lineage()

        assert len(lineage) == 2
        assert lineage[0].id == agent1.id
        assert lineage[1].id == agent2.id


class TestValidation:
    """Tests for input validation."""

    async def test_create_agent_empty_name(self):
        """create_agent rejects empty name."""
        async with MemoryStore() as store:
            with pytest.raises(immagent.ValidationError) as exc_info:
                await store.create_agent(
                    name="",
                    system_prompt="You are helpful.",
                    model=immagent.Model.CLAUDE_3_5_HAIKU,
                )
            assert exc_info.value.field == "name"

    async def test_create_agent_whitespace_name(self):
        """create_agent rejects whitespace-only name."""
        async with MemoryStore() as store:
            with pytest.raises(immagent.ValidationError) as exc_info:
                await store.create_agent(
                    name="   ",
                    system_prompt="You are helpful.",
                    model=immagent.Model.CLAUDE_3_5_HAIKU,
                )
            assert exc_info.value.field == "name"

    async def test_create_agent_empty_system_prompt(self):
        """create_agent rejects empty system_prompt."""
        async with MemoryStore() as store:
            with pytest.raises(immagent.ValidationError) as exc_info:
                await store.create_agent(
                    name="TestBot",
                    system_prompt="",
                    model=immagent.Model.CLAUDE_3_5_HAIKU,
                )
            assert exc_info.value.field == "system_prompt"

    async def test_create_agent_empty_model(self):
        """create_agent rejects empty model."""
        async with MemoryStore() as store:
            with pytest.raises(immagent.ValidationError) as exc_info:
                await store.create_agent(
                    name="TestBot",
                    system_prompt="You are helpful.",
                    model="",
                )
            assert exc_info.value.field == "model"

    async def test_advance_empty_input(self):
        """advance rejects empty user_input."""
        async with MemoryStore() as store:
            agent = await store.create_agent(
                name="TestBot",
                system_prompt="You are helpful.",
                model=immagent.Model.CLAUDE_3_5_HAIKU,
            )
            with pytest.raises(immagent.ValidationError) as exc_info:
                await agent.advance("")
            assert exc_info.value.field == "user_input"

    async def test_advance_invalid_max_tool_rounds(self):
        """advance rejects max_tool_rounds < 1."""
        async with MemoryStore() as store:
            agent = await store.create_agent(
                name="TestBot",
                system_prompt="You are helpful.",
                model=immagent.Model.CLAUDE_3_5_HAIKU,
            )
            with pytest.raises(immagent.ValidationError) as exc_info:
                await agent.advance("Hello", max_tool_rounds=0)
            assert exc_info.value.field == "max_tool_rounds"

    async def test_advance_invalid_max_retries(self):
        """advance rejects negative max_retries."""
        async with MemoryStore() as store:
            agent = await store.create_agent(
                name="TestBot",
                system_prompt="You are helpful.",
                model=immagent.Model.CLAUDE_3_5_HAIKU,
            )
            with pytest.raises(immagent.ValidationError) as exc_info:
                await agent.advance("Hello", max_retries=-1)
            assert exc_info.value.field == "max_retries"

    async def test_advance_invalid_timeout(self):
        """advance rejects non-positive timeout."""
        async with MemoryStore() as store:
            agent = await store.create_agent(
                name="TestBot",
                system_prompt="You are helpful.",
                model=immagent.Model.CLAUDE_3_5_HAIKU,
            )
            with pytest.raises(immagent.ValidationError) as exc_info:
                await agent.advance("Hello", timeout=0)
            assert exc_info.value.field == "timeout"


class TestMetadata:
    """Tests for agent metadata."""

    async def test_create_agent_with_metadata(self, store: Store):
        """Agent can be created with metadata."""
        agent = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
            metadata={"task_id": "abc123", "step": 1},
        )

        assert agent.metadata == {"task_id": "abc123", "step": 1}

    async def test_create_agent_without_metadata(self, store: Store):
        """Agent without metadata has empty dict."""
        agent = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        assert agent.metadata == {}

    async def test_metadata_persists(self, store: Store):
        """Metadata is saved and loaded from database."""
        agent = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
            metadata={"key": "value"},
        )

        store.clear_cache()
        loaded = await store.load_agent(agent.id)

        assert loaded.metadata == {"key": "value"}

    async def test_with_metadata_creates_new_agent(self):
        """with_metadata creates new agent with updated metadata."""
        async with MemoryStore() as store:
            agent1 = await store.create_agent(
                name="TestBot",
                system_prompt="You are helpful.",
                model=immagent.Model.CLAUDE_3_5_HAIKU,
                metadata={"step": 1},
            )

            agent2 = await agent1.with_metadata({"step": 2})

            assert agent2.id != agent1.id
            assert agent2.parent_id == agent1.id
            assert agent2.metadata == {"step": 2}
            assert agent1.metadata == {"step": 1}  # Original unchanged

    async def test_metadata_inherited_on_evolve(self, store: Store):
        """Metadata is inherited when agent evolves."""
        agent1 = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
            metadata={"persistent": True},
        )

        # Manually evolve
        conv = Conversation.create()
        store._cache_all(conv)
        agent2 = agent1._evolve(conv)
        await store._save(agent2)

        assert agent2.metadata == {"persistent": True}

    async def test_copy_preserves_metadata(self, store: Store):
        """Copy preserves metadata."""
        agent1 = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
            metadata={"copied": True},
        )

        agent2 = await agent1.copy()

        assert agent2.metadata == {"copied": True}


class TestTokenUsage:
    """Tests for token usage tracking."""

    async def test_empty_conversation_zero_tokens(self):
        """New agent with no messages has zero tokens."""
        async with MemoryStore() as store:
            agent = await store.create_agent(
                name="TestBot",
                system_prompt="You are helpful.",
                model=immagent.Model.CLAUDE_3_5_HAIKU,
            )

            input_tokens, output_tokens = await agent.get_token_usage()

            assert input_tokens == 0
            assert output_tokens == 0

    async def test_token_usage_sums_assistant_messages(self, store: Store):
        """Token usage sums across assistant messages."""
        from immagent.messages import Message

        agent = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        # Manually add messages with token counts
        user_msg = Message.user("Hello")
        asst_msg1 = Message.assistant("Hi!", input_tokens=10, output_tokens=5)
        asst_msg2 = Message.assistant("How can I help?", input_tokens=15, output_tokens=8)
        await store._save(user_msg, asst_msg1, asst_msg2)

        conv = Conversation.create((user_msg.id, asst_msg1.id, asst_msg2.id))
        store._cache_all(conv)
        agent2 = agent._evolve(conv)
        await store._save(agent2)

        input_tokens, output_tokens = await agent2.get_token_usage()

        assert input_tokens == 25  # 10 + 15
        assert output_tokens == 13  # 5 + 8

    async def test_token_usage_ignores_user_and_tool_messages(self, store: Store):
        """Token usage only counts assistant messages."""
        from immagent.messages import Message

        agent = await store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        # Add various message types
        user_msg = Message.user("Hello")
        asst_msg = Message.assistant("Let me check.", input_tokens=10, output_tokens=5)
        tool_msg = Message.tool_result("call_123", "Result")
        await store._save(user_msg, asst_msg, tool_msg)

        conv = Conversation.create((user_msg.id, asst_msg.id, tool_msg.id))
        store._cache_all(conv)
        agent2 = agent._evolve(conv)
        await store._save(agent2)

        input_tokens, output_tokens = await agent2.get_token_usage()

        assert input_tokens == 10
        assert output_tokens == 5
