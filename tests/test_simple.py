"""Tests for SimpleAgent."""

import os

import pytest

import immagent
from immagent import SimpleAgent


class TestSimpleAgentValidation:
    """Tests for SimpleAgent input validation."""

    def test_empty_name(self):
        """SimpleAgent rejects empty name."""
        with pytest.raises(immagent.ValidationError) as exc_info:
            SimpleAgent(
                name="",
                system_prompt="You are helpful.",
                model="anthropic/claude-3-5-haiku-20241022",
            )
        assert exc_info.value.field == "name"

    def test_whitespace_name(self):
        """SimpleAgent rejects whitespace-only name."""
        with pytest.raises(immagent.ValidationError) as exc_info:
            SimpleAgent(
                name="   ",
                system_prompt="You are helpful.",
                model="anthropic/claude-3-5-haiku-20241022",
            )
        assert exc_info.value.field == "name"

    def test_empty_system_prompt(self):
        """SimpleAgent rejects empty system_prompt."""
        with pytest.raises(immagent.ValidationError) as exc_info:
            SimpleAgent(
                name="TestBot",
                system_prompt="",
                model="anthropic/claude-3-5-haiku-20241022",
            )
        assert exc_info.value.field == "system_prompt"

    def test_whitespace_system_prompt(self):
        """SimpleAgent rejects whitespace-only system_prompt."""
        with pytest.raises(immagent.ValidationError) as exc_info:
            SimpleAgent(
                name="TestBot",
                system_prompt="   ",
                model="anthropic/claude-3-5-haiku-20241022",
            )
        assert exc_info.value.field == "system_prompt"

    def test_empty_model(self):
        """SimpleAgent rejects empty model."""
        with pytest.raises(immagent.ValidationError) as exc_info:
            SimpleAgent(
                name="TestBot",
                system_prompt="You are helpful.",
                model="",
            )
        assert exc_info.value.field == "model"

    def test_whitespace_model(self):
        """SimpleAgent rejects whitespace-only model."""
        with pytest.raises(immagent.ValidationError) as exc_info:
            SimpleAgent(
                name="TestBot",
                system_prompt="You are helpful.",
                model="   ",
            )
        assert exc_info.value.field == "model"


class TestSimpleAgentProperties:
    """Tests for SimpleAgent properties."""

    def test_name(self):
        """name property returns the agent name."""
        agent = SimpleAgent(
            name="TestBot",
            system_prompt="You are helpful.",
            model="anthropic/claude-3-5-haiku-20241022",
        )
        assert agent.name == "TestBot"

    def test_model(self):
        """model property returns the model string."""
        agent = SimpleAgent(
            name="TestBot",
            system_prompt="You are helpful.",
            model="anthropic/claude-3-5-haiku-20241022",
        )
        assert agent.model == "anthropic/claude-3-5-haiku-20241022"

    def test_system_prompt(self):
        """system_prompt property returns the system prompt."""
        agent = SimpleAgent(
            name="TestBot",
            system_prompt="You are helpful.",
            model="anthropic/claude-3-5-haiku-20241022",
        )
        assert agent.system_prompt == "You are helpful."

    def test_messages_empty(self):
        """messages property returns empty tuple for new agent."""
        agent = SimpleAgent(
            name="TestBot",
            system_prompt="You are helpful.",
            model="anthropic/claude-3-5-haiku-20241022",
        )
        assert agent.messages() == ()

    def test_last_response_none(self):
        """last_response() returns None for new agent."""
        agent = SimpleAgent(
            name="TestBot",
            system_prompt="You are helpful.",
            model="anthropic/claude-3-5-haiku-20241022",
        )
        assert agent.last_response() is None

    def test_model_config_default(self):
        """model_config returns empty dict by default."""
        agent = SimpleAgent(
            name="TestBot",
            system_prompt="You are helpful.",
            model="anthropic/claude-3-5-haiku-20241022",
        )
        assert agent.model_config == {}

    def test_model_config_custom(self):
        """model_config returns provided config."""
        agent = SimpleAgent(
            name="TestBot",
            system_prompt="You are helpful.",
            model="anthropic/claude-3-5-haiku-20241022",
            model_config={"temperature": 0.5},
        )
        assert agent.model_config == {"temperature": 0.5}

    def test_model_config_returns_copy(self):
        """model_config returns a copy, not the internal dict."""
        agent = SimpleAgent(
            name="TestBot",
            system_prompt="You are helpful.",
            model="anthropic/claude-3-5-haiku-20241022",
            model_config={"temperature": 0.5},
        )
        config = agent.model_config
        config["temperature"] = 1.0
        # Internal state should be unchanged
        assert agent.model_config == {"temperature": 0.5}


class TestSimpleAgentTokenUsage:
    """Tests for SimpleAgent.token_usage()."""

    async def test_token_usage_is_async(self):
        """token_usage is an async method for API consistency."""
        agent = SimpleAgent(
            name="TestBot",
            system_prompt="You are helpful.",
            model="anthropic/claude-3-5-haiku-20241022",
        )
        # This would fail if token_usage was not async
        result = agent.token_usage()
        assert hasattr(result, "__await__")
        # Actually await it
        input_tokens, output_tokens = await result
        assert input_tokens == 0
        assert output_tokens == 0

    async def test_token_usage_empty(self):
        """token_usage returns (0, 0) for new agent."""
        agent = SimpleAgent(
            name="TestBot",
            system_prompt="You are helpful.",
            model="anthropic/claude-3-5-haiku-20241022",
        )
        input_tokens, output_tokens = await agent.token_usage()
        assert input_tokens == 0
        assert output_tokens == 0


# Integration tests that require API key
needs_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@needs_api_key
async def test_simple_agent_advance():
    """SimpleAgent.advance() calls LLM and returns new agent."""
    agent1 = SimpleAgent(
        name="Calculator",
        system_prompt="You are a calculator. Only respond with numbers.",
        model=immagent.Model.CLAUDE_3_5_HAIKU,
    )

    agent2 = await agent1.advance("What is 2 + 2?")

    # Original agent unchanged
    assert agent1.messages() == ()

    # New agent has messages
    assert len(agent2.messages()) == 2  # user + assistant
    assert agent2.messages()[0].role == "user"
    assert agent2.messages()[0].content == "What is 2 + 2?"
    assert agent2.messages()[1].role == "assistant"
    assert "4" in agent2.messages()[1].content


@needs_api_key
async def test_simple_agent_last_response():
    """SimpleAgent.last_response() returns the last assistant message."""
    agent = SimpleAgent(
        name="Calculator",
        system_prompt="You are a calculator. Only respond with numbers.",
        model=immagent.Model.CLAUDE_3_5_HAIKU,
    )

    agent = await agent.advance("What is 2 + 2?")

    assert agent.last_response() is not None
    assert "4" in agent.last_response()


@needs_api_key
async def test_simple_agent_token_tracking():
    """SimpleAgent tracks token usage after advance()."""
    agent = SimpleAgent(
        name="Calculator",
        system_prompt="You are a calculator. Only respond with numbers.",
        model=immagent.Model.CLAUDE_3_5_HAIKU,
    )

    agent = await agent.advance("What is 2 + 2?")

    input_tokens, output_tokens = await agent.token_usage()
    assert input_tokens > 0
    assert output_tokens > 0


@needs_api_key
async def test_simple_agent_multi_turn():
    """SimpleAgent maintains context across turns."""
    agent = SimpleAgent(
        name="Assistant",
        system_prompt="You are helpful. Be very concise.",
        model=immagent.Model.CLAUDE_3_5_HAIKU,
    )

    # First turn
    agent = await agent.advance("My name is Alice.")

    # Second turn - should remember context
    agent = await agent.advance("What is my name?")

    assert "Alice" in agent.last_response()
    assert len(agent.messages()) == 4  # 2 user + 2 assistant
