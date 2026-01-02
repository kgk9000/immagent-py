# ImmAgent

An immutable agent architecture for Python. Every state transition creates a new agent with a fresh UUID—the old agent remains unchanged.

## Quick Start

```python
import asyncio
import immagent

async def main():
    async with await immagent.Store.connect("postgresql://...") as store:
        await store.init_schema()

        # Create an agent (auto-saved)
        agent = await store.create_agent(
            name="Assistant",
            system_prompt="You are helpful.",
            model=immagent.Model.CLAUDE_3_5_HAIKU,
        )

        # Advance returns a NEW agent with a new ID (auto-saved)
        agent = await agent.advance("Hello!")

        # Get messages
        for msg in await agent.get_messages():
            print(f"{msg.role}: {msg.content}")

asyncio.run(main())
```

## Public API

| Method | Description |
|--------|-------------|
| `Store.connect(dsn)` | Connect to PostgreSQL |
| `store.init_schema()` | Create tables if not exist |
| `store.create_agent()` | Create and save a new agent |
| `store.load_agent(id)` | Load agent by UUID |
| `store.delete(agent)` | Delete an agent |
| `store.gc()` | Remove orphaned assets |
| `store.clear_cache()` | Clear in-memory cache |
| `agent.advance(input)` | Call LLM and return new agent |
| `agent.get_messages()` | Get conversation messages |
| `agent.get_lineage()` | Walk agent's parent chain |
| `agent.copy()` | Copy agent with new ID |
| `immagent.Model` | Enum of common LLM models |
| `immagent.MCPManager` | MCP tool server manager |

## Core Concept

```python
# Every advance returns a NEW agent with a new ID
new_agent = await agent.advance("Hello!")

assert new_agent.id != agent.id  # Different UUIDs
assert new_agent.parent_id == agent.id  # Linked
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

## Architecture

### Store

The `Store` is the main interface. It combines:
- **Database** — PostgreSQL persistence
- **Cache** — Weak reference cache (auto-cleanup when assets are dropped)

```python
async with await immagent.Store.connect("postgresql://...") as store:
    await store.init_schema()
    # ... use store ...
```

Connection pool configuration:

```python
store = await immagent.Store.connect(
    "postgresql://...",
    min_size=2,                          # Min pool connections (default: 2)
    max_size=10,                         # Max pool connections (default: 10)
    max_inactive_connection_lifetime=300, # Idle timeout in seconds (default: 300)
)
```

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

Agents are bound to a Store and have methods for interaction:

```python
# Create via store (auto-saved)
agent = await store.create_agent(
    name="Bot",
    system_prompt="You are helpful.",
    model=immagent.Model.CLAUDE_3_5_HAIKU,
)

# Interact via agent methods (auto-saved)
agent = await agent.advance("Hello!")
messages = await agent.get_messages()
lineage = await agent.get_lineage()
```

### Advancing

`agent.advance()` is the main entry point:

1. Load conversation history and system prompt
2. Add the user message
3. Call the LLM (via LiteLLM)
4. If tool calls requested, execute via MCP and loop
5. Create new `Conversation` with all messages
6. Create new `ImmAgent` with `parent_id` pointing to the old agent
7. Save to database and cache
8. Return the new agent

Configuration options:

```python
agent = await agent.advance(
    "Hello",
    max_retries=3,      # Retry on transient failures (default: 3)
    timeout=120.0,      # Request timeout in seconds (default: 120)
    max_tool_rounds=10, # Max tool-use loops (default: 10)
)
```

### Auto-Save

All mutations are automatically persisted. There's no need to call `save()` manually:

```python
agent = await store.create_agent(...)  # Saved immediately
agent = await agent.advance("Hello")   # Saved immediately
```

The cache uses weak references: items are saved to the database first, then cached. When you drop an asset, it's automatically removed from the cache.

### Token Tracking

Assistant messages include token usage from each LLM call:

```python
messages = await agent.get_messages()
for msg in messages:
    if msg.role == "assistant":
        print(f"Input: {msg.input_tokens}, Output: {msg.output_tokens}")

# Total usage for a conversation
total_input = sum(m.input_tokens or 0 for m in messages)
total_output = sum(m.output_tokens or 0 for m in messages)
```

## LLM Providers

Uses [LiteLLM](https://docs.litellm.ai/) for multi-provider support. Use the `Model` enum for common models:

```python
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
async with immagent.MCPManager() as mcp:
    await mcp.connect(
        "filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    agent = await agent.advance("List files in /tmp", mcp=mcp)
```

The agent will automatically discover and use available tools.

### Writing MCP Servers

You can create custom MCP servers in Python. See `examples/weather_server.py` for a complete example:

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("my-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="my_tool",
            description="Does something useful",
            inputSchema={"type": "object", "properties": {...}},
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # Implement your tool logic here
    return [TextContent(type="text", text="Result")]

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
```

## Error Handling

Custom exceptions for precise error handling:

```python
try:
    agent = await agent.advance("Hello")
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
├── store.py        # Store (main interface - cache + db)
├── agent.py        # ImmAgent dataclass with methods
├── assets.py       # Asset base class, TextAsset
├── exceptions.py   # Custom exception types
├── llm.py          # LiteLLM wrapper with retries/timeout
├── logging.py      # Logging configuration
├── mcp.py          # MCP client for tools
└── messages.py     # Message, ToolCall, Conversation
```
