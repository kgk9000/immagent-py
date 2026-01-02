"""Tests for the public API."""

import uuid

import pytest

import immagent
from immagent import Store
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
        agent2 = agent1.evolve(conv)
        await store._save(agent2)

        lineage = await agent2.get_lineage()

        assert len(lineage) == 2
        assert lineage[0].id == agent1.id
        assert lineage[1].id == agent2.id
