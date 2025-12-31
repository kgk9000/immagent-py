"""Tests for the public API."""

import immagent
import immagent.db as db_mod


class TestCreateAgent:
    def test_returns_agent_and_assets(self):
        """create_agent returns an agent and assets to save."""
        agent, assets = immagent.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        assert agent.name == "TestBot"
        assert len(assets) == 3  # prompt, conversation, agent

    def test_accepts_string_model(self):
        """create_agent accepts a string model."""
        agent, _ = immagent.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model="anthropic/claude-3-5-haiku-20241022",
        )

        assert agent.model == "anthropic/claude-3-5-haiku-20241022"

    def test_accepts_model_enum(self):
        """create_agent accepts a Model enum."""
        agent, _ = immagent.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        assert agent.model == "anthropic/claude-3-5-haiku-20241022"


class TestSaveAndLoad:
    async def test_save_and_load_agent(self, db: db_mod.Database):
        """Agent can be saved and loaded."""
        agent, assets = immagent.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        await immagent.save(db, *assets)
        loaded = await immagent.load_agent(db, agent.id)

        assert loaded is not None
        assert loaded.id == agent.id
        assert loaded.name == "TestBot"

    async def test_load_nonexistent_agent(self, db: db_mod.Database):
        """Loading nonexistent agent returns None."""
        import uuid

        loaded = await immagent.load_agent(db, uuid.uuid4())

        assert loaded is None


class TestGetMessages:
    async def test_empty_conversation(self, db: db_mod.Database):
        """New agent has no messages."""
        agent, assets = immagent.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )
        await immagent.save(db, *assets)

        messages = await immagent.get_messages(agent, db)

        assert messages == []


class TestGetLineage:
    async def test_single_agent_lineage(self, db: db_mod.Database):
        """Single agent's lineage is just itself."""
        agent, assets = immagent.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )
        await immagent.save(db, *assets)

        lineage = await immagent.get_lineage(agent, db)

        assert len(lineage) == 1
        assert lineage[0].id == agent.id

    async def test_evolved_agent_lineage(self, db: db_mod.Database):
        """Evolved agent's lineage includes parent."""
        # Create first agent
        agent1, assets1 = immagent.create_agent(
            name="TestBot",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )
        await immagent.save(db, *assets1)

        # Manually evolve (simulating what advance would do)
        conv = immagent.Conversation.create()
        await db.save_conversation(conv)
        agent2 = agent1._evolve(conv)
        await db.save_agent(agent2)

        lineage = await immagent.get_lineage(agent2, db)

        assert len(lineage) == 2
        assert lineage[0].id == agent1.id
        assert lineage[1].id == agent2.id
