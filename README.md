# ImmAgent

An immutable agent architecture for Python. Every state transition creates a new agent with a fresh UUID—the old agent remains unchanged.

## Public API

| Function | Description |
|----------|-------------|
| `immagent.create_agent()` | Create a new agent |
| `immagent.advance()`      | Call LLM and advance state |
| `immagent.save()`         | Persist assets to database |
| `immagent.load_agent()`   | Load agent from cache/database |
| `immagent.get_messages()` | Get conversation messages |
| `immagent.get_lineage()`  | Walk agent's parent chain |
| `immagent.Model`          | Enum of common LLM models |

## Core Concept

```python
import immagent

# Every advance returns a NEW agent with a new ID
new_agent, assets = await immagent.advance(agent, "Hello!", db)
await immagent.save(db, *assets)  # Persist explicitly

assert new_agent.id != agent.id  # Different UUIDs
```

Because everything is immutable:
- **Safe caching** — once loaded, assets never change
- **Full history** — follow `parent_id` to trace the agent's lineage
- **Reproducibility** — given an agent ID, you can reconstruct its exact state

## Installation

```bash
uv add immagent
```

Or for development:

```bash
git clone https://github.com/youruser/immagent-py
cd immagent-py
uv sync --all-extras
```

## Quick Start

```python
import asyncio

import immagent

async def main():
    # Connect to PostgreSQL (auto-closes with context manager)
    async with await immagent.Database.connect("postgresql://user:pass@localhost/immagent") as db:
        await db.init_schema()

        # Create an agent
        agent, assets = immagent.create_agent(
            name="Assistant",
            system_prompt="You are a helpful assistant.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )
        await immagent.save(db, *assets)
        print(f"Created agent: {agent.id}")

        # Advance the agent — returns a NEW agent
        agent, assets = await immagent.advance(agent, "What is 2 + 2?", db)
        await immagent.save(db, *assets)
        print(f"New agent: {agent.id}")

asyncio.run(main())
```

## Architecture

### Assets

Everything is an **Asset** with a UUID and timestamp:

```python
@dataclass(frozen=True)
class Asset:
    id: UUID
    created_at: datetime
```

Asset types:
- `TextAsset` — system prompts and other text content
- `Message` — user, assistant, or tool messages
- `Conversation` — ordered list of message IDs
- `ImmAgent` — the agent itself

### ImmAgent

```python
@dataclass(frozen=True)
class ImmAgent(Asset):
    name: str
    system_prompt_id: UUID      # References a TextAsset
    parent_id: UUID | None      # Previous agent state
    conversation_id: UUID       # References a Conversation
    model: str                  # LiteLLM model string
```

### Advancing

`advance()` is the main entry point:

1. Load conversation history and system prompt
2. Add the user message
3. Call the LLM (via LiteLLM)
4. If tool calls requested, execute via MCP and loop
5. Create new `Conversation` with all messages
6. Create new `ImmAgent` with `parent_id` pointing to the old agent
7. Return the new agent and all new assets

```python
import immagent

agent_v2, assets = await immagent.advance(agent_v1, "Hello", db)
await immagent.save(db, *assets)  # Persist explicitly

# agent_v1 unchanged, agent_v2 is the new state
# agent_v2.parent_id == agent_v1.id
```

### Database

PostgreSQL with separate tables per asset type:

- `text_assets` — system prompts
- `messages` — conversation messages
- `conversations` — ordered message ID lists
- `agents` — agent states

```python
import immagent

db = await immagent.Database.connect("postgresql://...")
await db.init_schema()  # Creates tables if not exist
```

### Loading

Load an agent by ID to resume a conversation:

```python
import immagent

# First call loads from DB, subsequent calls return cached
agent = await immagent.load_agent(db, agent_id)
```

## LLM Providers

Uses [LiteLLM](https://docs.litellm.ai/) for multi-provider support. Use the `Model` enum for common models:

```python
import immagent

# Anthropic
immagent.Model.CLAUDE_3_5_HAIKU
immagent.Model.CLAUDE_SONNET_4
immagent.Model.CLAUDE_OPUS_4

# OpenAI
immagent.Model.GPT_4O
immagent.Model.GPT_4O_MINI
immagent.Model.O1
immagent.Model.O1_MINI
```

Or pass any LiteLLM model string directly:

```python
model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
```

Set the appropriate API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

## MCP Tools

Agents can use tools via [Model Context Protocol](https://modelcontextprotocol.io/):

```python
import immagent

# Connect to MCP servers
mcp = immagent.MCPManager()
await mcp.connect(
    "filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

# Advance with tools available
agent, assets = await immagent.advance(agent, "List files in /tmp", db, mcp=mcp)
await immagent.save(db, *assets)

await mcp.close()
```

The agent will automatically discover and use available tools.

## Development

```bash
# Install dev dependencies
make dev

# Format code
make fmt

# Lint
make lint

# Run tests (requires Docker for PostgreSQL)
make test

# Run with coverage
make test-cov
```

### Running Tests

Tests use [testcontainers](https://testcontainers-python.readthedocs.io/) to spin up PostgreSQL in Docker:

```bash
# Make sure Docker is running
docker ps

# Run all tests
make test
```

For LLM integration tests, set your API key:

```bash
ANTHROPIC_API_KEY=sk-ant-... make test
```

## Project Structure

```
src/immagent/
├── __init__.py     # Public API
├── assets.py       # Asset base class, TextAsset
├── agent.py        # ImmAgent dataclass
├── messages.py     # Message, ToolCall, Conversation
├── db.py           # PostgreSQL persistence
├── cache.py        # In-memory UUID→Asset cache
├── llm.py          # LiteLLM wrapper
├── mcp.py          # MCP client for tools
└── turn.py         # advance(), create_agent(), save(), load_agent()
```
