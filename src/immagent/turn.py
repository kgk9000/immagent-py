"""Turn processing - the core agent loop."""

from immagent.agent import ImmAgent
from immagent.cache import cache_all, get_conversation, get_messages, get_text_asset
from immagent.db import Database
from immagent.llm import complete
from immagent.mcp import MCPManager
from immagent.messages import Message


async def process_turn(
    agent: ImmAgent,
    user_input: str,
    db: Database,
    mcp: MCPManager | None = None,
    max_tool_rounds: int = 10,
) -> ImmAgent:
    """Process a single turn of conversation.

    This is the main entry point for interacting with an agent. It:
    1. Loads the conversation history
    2. Adds the user message
    3. Calls the LLM
    4. If tools are requested, executes them and loops
    5. Creates a new agent with the updated conversation
    6. Saves all new assets to the database

    Args:
        agent: The current agent state
        user_input: The user's message
        db: Database connection for persistence
        mcp: Optional MCP manager for tool calling
        max_tool_rounds: Maximum number of tool-use rounds (safety limit)

    Returns:
        A new ImmAgent with updated conversation (new UUID)
    """
    # Load existing conversation and system prompt
    conversation = await get_conversation(db, agent.conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {agent.conversation_id} not found")

    system_prompt = await get_text_asset(db, agent.system_prompt_id)
    if system_prompt is None:
        raise ValueError(f"System prompt {agent.system_prompt_id} not found")

    # Load existing messages
    messages = await get_messages(db, conversation.message_ids)

    # Create user message
    user_message = Message.user(user_input)
    messages.append(user_message)

    # Get tools if MCP is available
    tools = mcp.get_all_tools() if mcp else None

    # New messages created in this turn
    new_messages: list[Message] = [user_message]

    # Tool loop
    for _ in range(max_tool_rounds):
        # Call LLM
        assistant_message = await complete(
            model=agent.model,
            messages=messages,
            system=system_prompt.content,
            tools=tools,
        )
        messages.append(assistant_message)
        new_messages.append(assistant_message)

        # Check for tool calls
        if not assistant_message.tool_calls or not mcp:
            break

        # Execute tool calls
        for tool_call in assistant_message.tool_calls:
            result = await mcp.execute(tool_call.name, tool_call.arguments)
            tool_result_message = Message.tool_result(tool_call.id, result)
            messages.append(tool_result_message)
            new_messages.append(tool_result_message)

    # Create new conversation with all message IDs
    new_conversation = conversation.with_messages(*[m.id for m in new_messages])

    # Create new agent state
    new_agent = agent.evolve(new_conversation)

    # Save all new assets to database
    await db.save_all(*new_messages, new_conversation, new_agent)

    # Cache the new assets
    cache_all(*new_messages, new_conversation, new_agent)

    return new_agent


async def create_agent(
    name: str,
    system_prompt: str,
    model: str,
    db: Database,
) -> ImmAgent:
    """Create a new agent with an empty conversation.

    This is a convenience function for creating a fresh agent.

    Args:
        name: Human-readable name for the agent
        system_prompt: The system prompt content
        model: LiteLLM model string
        db: Database connection for persistence

    Returns:
        A new ImmAgent ready for conversation
    """
    from immagent.assets import TextAsset
    from immagent.messages import Conversation

    # Create assets
    prompt_asset = TextAsset.create(system_prompt)
    conversation = Conversation.create()
    agent, _ = ImmAgent.create(
        name=name,
        system_prompt=prompt_asset,
        model=model,
        conversation=conversation,
    )

    # Save to database
    await db.save_all(prompt_asset, conversation, agent)

    # Cache
    cache_all(prompt_asset, conversation, agent)

    return agent
