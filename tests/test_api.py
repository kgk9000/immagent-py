"""Tests for the public API."""

import uuid

import pytest

import immagent
from immagent import Store
from immagent.messages import Conversation


class TestCreateAgent:
    def test_creates_agent(self, store: Store):
        """create_agent returns an agent."""
        agent = store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        assert agent.name == "TestBot"

    def test_accepts_string_model(self, store: Store):
        """create_agent accepts a string model."""
        agent = store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model="anthropic/claude-3-5-haiku-20241022",
        )

        assert agent.model == "anthropic/claude-3-5-haiku-20241022"

    def test_accepts_model_enum(self, store: Store):
        """create_agent accepts a Model enum."""
        agent = store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        assert agent.model == "anthropic/claude-3-5-haiku-20241022"


class TestSaveAndLoad:
    async def test_save_and_load_agent(self, store: Store):
        """Agent can be saved and loaded."""
        agent = store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )
        await store.save(agent)

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
        agent = store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        messages = await store.get_messages(agent)

        assert messages == []


class TestGetLineage:
    async def test_single_agent_lineage(self, store: Store):
        """Single agent's lineage is just itself."""
        agent = store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        lineage = await store.get_lineage(agent)

        assert len(lineage) == 1
        assert lineage[0].id == agent.id

    async def test_evolved_agent_lineage(self, store: Store):
        """Evolved agent's lineage includes parent."""
        agent1 = store.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )
        await store.save(agent1)

        # Manually evolve (simulating what advance would do)
        conv = Conversation.create()
        await store.save(conv)
        agent2 = agent1.evolve(conv)
        await store.save(agent2)

        lineage = await store.get_lineage(agent2)

        assert len(lineage) == 2
        assert lineage[0].id == agent1.id
        assert lineage[1].id == agent2.id
