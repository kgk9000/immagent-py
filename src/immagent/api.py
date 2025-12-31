"""Turn processing - the core agent loop."""

from uuid import UUID

import immagent.agent as agent_mod
import immagent.assets as assets
import immagent.cache as cache
import immagent.db as db_mod
import immagent.llm as llm
import immagent.mcp as mcp_mod
import immagent.messages as messages


async def advance(
    agent: agent_mod.ImmAgent,
    user_input: str,
    db: db_mod.Database,
    mcp: mcp_mod.MCPManager | None = None,
    max_tool_rounds: int = 10,
    max_retries: int = 3,
) -> tuple[agent_mod.ImmAgent, list[assets.Asset]]:
    """Advance the agent with a user message.

    This is the main entry point for interacting with an agent. It:
    1. Loads the conversation history
    2. Adds the user message
    3. Calls the LLM (with retries for transient failures)
    4. If tools are requested, executes them and loops
    5. Creates a new agent with the updated conversation

    The caller is responsible for persisting the returned assets.

    Args:
        agent: The current agent state
        user_input: The user's message
        db: Database connection for loading conversation history
        mcp: Optional MCP manager for tool calling
        max_tool_rounds: Maximum number of tool-use rounds (safety limit)
        max_retries: Number of retry attempts for LLM calls (default: 3)

    Returns:
        A tuple of (new_agent, assets) where assets is a list of all new
        assets created during this turn (messages, conversation, agent).
    """
    # Load existing conversation and system prompt
    conversation = await cache.get_conversation(db, agent.conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {agent.conversation_id} not found")

    system_prompt = await cache.get_text_asset(db, agent.system_prompt_id)
    if system_prompt is None:
        raise ValueError(f"System prompt {agent.system_prompt_id} not found")

    # Load existing messages
    msgs = await cache.get_messages(db, conversation.message_ids)

    # Create user message
    user_message = messages.Message.user(user_input)
    msgs.append(user_message)

    # Get tools if MCP is available
    tools = mcp.get_all_tools() if mcp else None

    # New messages created in this turn
    new_messages: list[messages.Message] = [user_message]

    # Tool loop
    for _ in range(max_tool_rounds):
        # Call LLM (with retries for transient failures)
        assistant_message = await llm.complete(
            model=agent.model,
            msgs=msgs,
            system=system_prompt.content,
            tools=tools,
            max_retries=max_retries,
        )
        msgs.append(assistant_message)
        new_messages.append(assistant_message)

        # Check for tool calls
        if not assistant_message.tool_calls or not mcp:
            break

        # Execute tool calls
        for tool_call in assistant_message.tool_calls:
            result = await mcp.execute(tool_call.name, tool_call.arguments)
            tool_result_message = messages.Message.tool_result(tool_call.id, result)
            msgs.append(tool_result_message)
            new_messages.append(tool_result_message)

    # Create new conversation with all message IDs
    new_conversation = conversation.with_messages(*[m.id for m in new_messages])

    # Create new agent state
    new_agent = agent._evolve(new_conversation)

    # Collect all new assets
    new_assets: list[assets.Asset] = [*new_messages, new_conversation, new_agent]

    # Cache the new assets
    cache.cache_all(*new_assets)

    return new_agent, new_assets


def create_agent(
    name: str,
    system_prompt: str,
    model: str | llm.Model,
) -> tuple[agent_mod.ImmAgent, list[assets.Asset]]:
    """Create a new agent with an empty conversation.

    This is a convenience function for creating a fresh agent.
    The caller is responsible for persisting the returned assets.

    Args:
        name: Human-readable name for the agent
        system_prompt: The system prompt content
        model: LiteLLM model string or Model enum

    Returns:
        A tuple of (agent, assets) where assets is a list of all assets
        that need to be persisted (prompt, conversation, agent).
    """
    # Create assets
    prompt_asset = assets.TextAsset.create(system_prompt)
    conversation = messages.Conversation.create()
    agent, _ = agent_mod.ImmAgent._create(
        name=name,
        system_prompt=prompt_asset,
        model=model,
        conversation=conversation,
    )

    # Collect all assets
    new_assets: list[assets.Asset] = [prompt_asset, conversation, agent]

    # Cache
    cache.cache_all(*new_assets)

    return agent, new_assets


async def save(db: db_mod.Database, *assets_to_save: assets.Asset) -> None:
    """Save assets to the database.

    Args:
        db: Database connection
        *assets_to_save: Assets to persist
    """
    await db.save_all(*assets_to_save)


async def load_agent(db: db_mod.Database, agent_id: UUID) -> agent_mod.ImmAgent | None:
    """Load an agent by ID from cache or database.

    Args:
        db: Database connection
        agent_id: The agent's UUID

    Returns:
        The agent if found, None otherwise
    """
    return await cache.get_agent(db, agent_id)


async def get_messages(
    agent: agent_mod.ImmAgent,
    db: db_mod.Database,
) -> list[messages.Message]:
    """Get all messages in an agent's conversation.

    Args:
        agent: The agent
        db: Database connection

    Returns:
        List of messages in conversation order
    """
    conversation = await cache.get_conversation(db, agent.conversation_id)
    if conversation is None:
        return []
    return await cache.get_messages(db, conversation.message_ids)


async def get_lineage(
    agent: agent_mod.ImmAgent,
    db: db_mod.Database,
) -> list[agent_mod.ImmAgent]:
    """Get the agent's lineage by walking the parent_id chain.

    Returns agents from oldest ancestor to current agent.

    Args:
        agent: The agent
        db: Database connection

    Returns:
        List of agents from oldest to newest (current agent is last)
    """
    lineage: list[agent_mod.ImmAgent] = [agent]
    current = agent

    while current.parent_id is not None:
        parent = await cache.get_agent(db, current.parent_id)
        if parent is None:
            break
        lineage.append(parent)
        current = parent

    lineage.reverse()
    return lineage
