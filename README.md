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
| `immagent.clear_cache()`  | Clear the in-memory cache |
| `immagent.Model`          | Enum of common LLM models |
| `immagent.Database`       | PostgreSQL connection with pooling |
| `immagent.MCPManager`     | MCP tool server manager |

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

Configuration options:

```python
agent, assets = await immagent.advance(
    agent,
    "Hello",
    db,
    max_retries=3,      # Retry on transient failures (default: 3)
    timeout=120.0,      # Request timeout in seconds (default: 120)
    max_tool_rounds=10, # Max tool-use loops (default: 10)
)
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

Connection pool configuration:

```python
db = await immagent.Database.connect(
    "postgresql://...",
    min_size=2,                          # Min pool connections (default: 2)
    max_size=10,                         # Max pool connections (default: 10)
    max_inactive_connection_lifetime=300, # Idle timeout in seconds (default: 300)
)
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

# Use as async context manager for automatic cleanup
async with immagent.MCPManager() as mcp:
    await mcp.connect(
        "filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    # Advance with tools available
    agent, assets = await immagent.advance(agent, "List files in /tmp", db, mcp=mcp)
    await immagent.save(db, *assets)
```

Or manage the lifecycle manually:

```python
mcp = immagent.MCPManager()
await mcp.connect("filesystem", command="npx", args=[...])
# ... use mcp ...
await mcp.close()
```

The agent will automatically discover and use available tools.

## Error Handling

Custom exceptions for precise error handling:

```python
import immagent

try:
    agent, assets = await immagent.advance(agent, "Hello", db)
except immagent.ConversationNotFoundError as e:
    print(f"Conversation {e.asset_id} not found")
except immagent.SystemPromptNotFoundError as e:
    print(f"System prompt {e.asset_id} not found")
except immagent.ImmAgentError as e:
    print(f"Agent error: {e}")
```

Exception hierarchy:
- `ImmAgentError` — base exception
  - `AssetNotFoundError` — asset lookup failed
    - `ConversationNotFoundError`
    - `SystemPromptNotFoundError`
    - `AgentNotFoundError`
  - `LLMError` — LLM call failed
  - `ToolExecutionError` — MCP tool execution failed

## Logging

Enable logging for debugging and observability:

```python
import logging

# Enable debug logging for immagent
logging.basicConfig(level=logging.DEBUG)

# Or configure specifically
immagent_logger = logging.getLogger("immagent")
immagent_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
immagent_logger.addHandler(handler)
```

Log output includes:
- LLM requests/responses with timing and token usage
- MCP tool connections and executions
- Agent state transitions

## Development

```bash
# Install dev dependencies
make dev

# Format code
make fmt

# Lint
make lint

# Type check
make typecheck

# Run all checks (lint + typecheck)
make check

# Run tests (requires Docker for PostgreSQL)
make test

# Run with coverage
make test-cov
```

### API Keys

For LLM integration tests, create a `.env` file with your API key:

```bash
# Option 1: Create .env directly
echo 'export ANTHROPIC_API_KEY=sk-ant-...' > .env

# Option 2: Symlink to existing env file
ln -s ~/.env/anthropic.env .env
```

See `.env.example` for the expected format. The `.env` file is gitignored.

### Running Tests

Tests use [testcontainers](https://testcontainers-python.readthedocs.io/) to spin up PostgreSQL in Docker:

```bash
# Make sure Docker is running
docker ps

# Run all tests (sources .env automatically)
make test
```

## Project Structure

```
src/immagent/
├── __init__.py     # Public API exports
├── api.py          # advance(), create_agent(), save(), load_agent()
├── agent.py        # ImmAgent dataclass
├── assets.py       # Asset base class, TextAsset
├── cache.py        # In-memory UUID→Asset cache
├── db.py           # PostgreSQL persistence with connection pooling
├── exceptions.py   # Custom exception types
├── llm.py          # LiteLLM wrapper with retries/timeout
├── logging.py      # Logging configuration
├── mcp.py          # MCP client for tools
└── messages.py     # Message, ToolCall, Conversation
```
