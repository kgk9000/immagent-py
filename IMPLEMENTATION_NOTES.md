# Implementation Notes

Observations from code review with analysis of whether each item is OK, needs attention, or is a deliberate tradeoff.

---

## 1. Threading Lock with Async Code

**Location:** `store.py:94`
```python
self._lock = threading.RLock()
```

**Observation:** Using `threading.RLock` in async code can block the event loop if held during an await.

**Status:** ✅ Actually OK

**Why:** The lock only protects `WeakValueDictionary` operations (`get`, `__setitem__`, `pop`, `clear`), which are all synchronous and near-instant. The lock is never held across `await` boundaries. The pattern is:
```python
def _get_cached(self, asset_id: UUID) -> assets.Asset | None:
    with self._lock:
        return self._cache.get(asset_id)  # Sync, fast
```

An `asyncio.Lock` would actually be worse here since it would require making these simple getters async.

---

## 2. MemoryStore Uses Strong References

**Location:** `store.py:844-847`
```python
class MemoryStore(Store):
    def __init__(self) -> None:
        super().__init__(pool=None)
        self._cache: dict[UUID, assets.Asset] = {}  # Strong refs
```

**Observation:** MemoryStore overrides the cache with a regular dict, so assets are never evicted.

**Status:** ✅ Actually OK

**Why:** MemoryStore is for manual work and testing; production uses PostgreSQL. LRU eviction is not an option because there's no DB fallback — evicting an asset would lose it forever, leaving dangling UUID references. The strong-ref behavior is correct: assets live until explicitly deleted or the store closes.

---

## 3. Hidden `_store` Field on ImmAgent

**Location:** `agent.py:38`
```python
_store: Store = field(compare=True, hash=False, repr=False)
```

**Observation:** The underscore suggests internal, but it's required for the public API to work. Also `compare=True` means agents from different stores aren't equal.

**Status:** ✅ Actually OK

**Why:**
1. **`compare=True` is irrelevant** — ImmAgent identity is UUID-based. You wouldn't have the same UUID in different stores, so this never matters in practice.
2. **Underscore is intentional** — `_store` is internal plumbing for cache/DB access. Users shouldn't touch it; they interact via `agent.advance()`, `agent.get_messages()`, etc. which delegate to the store.

---

## 4. Model Class is Not an Enum

**Location:** `llm.py:13-29`
```python
class Model:
    CLAUDE_3_5_HAIKU = "anthropic/claude-3-5-haiku-20241022"
    # ...
```

**Observation:** `Model` is just a namespace of string constants, not a real enum.

**Status:** ✅ Actually OK (deliberate)

**Why:** The docstring explains: "These are just string constants — any LiteLLM model string works directly." This allows users to use any model LiteLLM supports without updating the library. A `StrEnum` would restrict to known models only.

---

## 5. clone() Creates a Sibling, Not a Child

**Location:** `agent.py:143-149`, `store.py:768-787`
```python
async def _clone_agent(self, agent: ImmAgent) -> ImmAgent:
    new_agent = ImmAgent(
        # ...
        parent_id=agent.parent_id,  # Same parent as original
        # ...
    )
```

**Observation:** `clone()` preserves the original's parent_id, creating a sibling. `with_metadata()` sets parent_id to the original, creating a child.

**Status:** ✅ Actually OK

**Why:** `clone()` creates a clone at the same point in the tree, allowing you to explore different conversation branches. The new agent has a different UUID to distinguish the branches. This is intentional:
- `clone()` → sibling (branch from same point)
- `with_metadata()` / `advance()` → child (evolve forward)

**Note:** Renamed from `copy()` to `clone()` for clarity.

---

## 6. Conversation Indirection

**Location:** `messages.py:100-124`
```python
@dataclass(frozen=True)
class Conversation(assets.Asset):
    message_ids: tuple[UUID, ...]
```

**Observation:** Every agent points to a `Conversation` which is just a tuple of message UUIDs. Every `advance()` creates a new Conversation.

**Status:** ✅ Actually OK

**Why:** The indirection provides real benefits:
1. **Cache efficiency** — Conversation is cached by UUID. When you `clone()`, both agents share the same `conversation_id` and reuse the cached Conversation.
2. **DB normalization** — Agents store just a UUID reference, not the full message_ids array. Less data duplication.
3. **Cheaper clones** — Clone just references the same conversation_id instead of copying the array.

If message_ids were stored directly on ImmAgent, each agent row would duplicate the full UUID array, and there'd be no cache sharing.

**On efficiency:** Creating a new Conversation each `advance()` is cheap — it's just appending UUIDs to a tuple. The actual Message content is shared via UUID references, never duplicated.

---

## 7. Silent Break on Incomplete Lineage

**Location:** `store.py:817-820`
```python
while current.parent_id is not None:
    parent = await self._get_agent(current.parent_id)
    if parent is None:
        raise exc.AgentNotFoundError(current.parent_id)
```

**Observation:** If a parent is missing (deleted, DB corruption), lineage traversal should fail loudly.

**Status:** ✅ Fixed

**Resolution:** Now raises `AgentNotFoundError` if a non-null `parent_id` references a missing agent. The loop exits normally when `parent_id is None` (root agent).

---

## 8. N+1 Queries in Lineage Traversal

**Location:** `store.py:809-875`

**Observation:** `get_lineage()` was loading parents one at a time in a loop.

**Status:** ✅ Fixed

**Resolution:** Now uses a recursive CTE for single-query traversal:
```sql
WITH RECURSIVE lineage AS (
    SELECT * FROM agents WHERE id = $1
    UNION ALL
    SELECT a.* FROM agents a
    INNER JOIN lineage l ON a.id = l.parent_id
)
SELECT * FROM lineage
```

For depth N, this is now 1 query instead of N. MemoryStore falls back to iterative traversal (no DB). Results are cached for future use.

---

## 9. Asset Base Class

**Location:** `assets.py:18-27`
```python
@dataclass(frozen=True)
class Asset:
    id: UUID
    created_at: datetime
```

**Observation:** Base class just defines `id` and `created_at`. All subclasses are handled differently.

**Status:** ✅ Actually OK

**Why:** The base class provides:
1. Consistent identity semantics (all assets have UUID + timestamp)
2. Type hint for the cache: `WeakValueDictionary[UUID, Asset]`
3. Documentation of the immutability contract

The lack of shared behavior is intentional — assets are structurally similar but operationally distinct.

---

## 10. GC Without Transaction

**Location:** `store.py:614-641`
```python
async with self._pool.acquire() as conn:
    async with conn.transaction():
        # All DELETEs now atomic
        ...
```

**Observation:** GC was running three DELETEs without a transaction wrapper.

**Status:** ✅ Fixed

**Resolution:** Now wrapped in `async with conn.transaction():`. All three DELETEs are atomic — if any fails, the whole operation rolls back. No partial cleanup.

---

## 11. Write-Through Cache Ordering

**Location:** `store.py:496-500`
```python
# Cache first (_save() looks up dependencies in cache)
self._cache_all(prompt_asset, conversation, agent)

# Save to database
await self._save(agent)
```

**Observation:** Must cache before save because `_save()` looks up dependencies in cache.

**Status:** ✅ Actually OK

**Why:** The pattern is simple once you know it: "cache dependencies before saving the thing that references them." The code already has comments explaining this. It's internal code with a consistent pattern in only a few places. Not confusing once documented.

---

## 12. Tool Loop Error Handling

**Location:** `store.py:722-727`
```python
async def execute_one(tc: messages.ToolCall) -> messages.Message:
    try:
        result = await mcp.execute(tc.name, tc.arguments)
    except exc.ToolExecutionError as e:
        result = f"Error: {e}"
    return messages.Message.tool_result(tc.id, result)
```

**Observation:** Tool errors are caught and converted to error messages, not raised.

**Status:** ✅ Actually OK (deliberate)

**Why:** This follows the standard pattern for LLM tool calling:
1. Tool fails → return error message to LLM
2. LLM sees error → can retry or handle gracefully
3. Conversation continues

Raising would abort the entire `advance()`, losing all progress. The LLM is better equipped to handle tool failures.

---

## 13. No Streaming Support

**Observation:** `advance()` blocks until the full response is ready.

**Status:** 📋 Planned feature

**Notes:** Critical for UX in chat applications. Will require:
1. New method like `advance_stream()` returning `AsyncIterator[str]`
2. Changes to `llm.complete()` to support streaming
3. Decision on when to persist (after stream completes? incrementally?)

---

## 14. No Agent Listing/Querying

**Observation:** Can only load agents by exact UUID. No `list_agents()` or `find_agents()`.

**Status:** ✅ Implemented

**Resolution:** Added three search methods to Store:
```python
async def list_agents(limit=100, offset=0, name=None) -> list[ImmAgent]
async def count_agents(name=None) -> int
async def find_by_name(name) -> list[ImmAgent]
```

- `list_agents`: Paginated listing with optional name filter (substring, case-insensitive)
- `count_agents`: Total count with optional name filter
- `find_by_name`: Exact name match (case-sensitive)

More search methods can be added as needed (by model, metadata, date range, etc.).

---

## 15. No Model Configuration

**Observation:** Can't set temperature, max_tokens, top_p, etc.

**Status:** ✅ Implemented

**Resolution:** Added two-level configuration:
1. **Agent defaults:** `create_agent(..., model_config={"temperature": 0.7})`
2. **Per-call overrides:** `advance(input, temperature=0.9, max_tokens=1000, top_p=0.95)`

Per-call values override agent defaults. Config is passed through to LiteLLM.

**Future consideration:** Per-call overrides are ephemeral — not stored. For reproducibility, we may want to store the effective config used for each LLM call (e.g., on the Message alongside input_tokens/output_tokens).

---

## 16. No Multimodal Support

**Location:** `messages.py:31`
```python
content: str | None
```

**Observation:** Message content is string-only, no image support.

**Status:** 📋 Planned feature

**Notes:** LiteLLM supports multimodal. The UUID-based architecture is well-suited for this — images can be stored as separate assets referenced by UUID, avoiding duplication in the message content.

Would require:
```python
content: str | list[ContentBlock] | None

@dataclass
class ContentBlock:
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: dict | None = None
```

---

## 17. Foreign Key ON DELETE SET NULL

**Location:** `store.py:59`
```python
parent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
```

**Observation:** Deleting a parent agent sets children's `parent_id` to NULL.

**Status:** ✅ Actually OK (deliberate)

**Why:** This preserves children when parents are deleted. The alternative `ON DELETE CASCADE` would delete the entire lineage, which is rarely desired. The lineage traversal already handles NULL parent_id as the termination condition.

---

## 18. Index on conversation_id

**Location:** `store.py:67`
```python
CREATE INDEX IF NOT EXISTS idx_agents_conversation_id ON agents(conversation_id);
```

**Observation:** Index exists but no queries currently use `WHERE conversation_id = ?`.

**Status:** ✅ Actually OK (future-proofing)

**Why:** GC uses conversation_id in a subquery. The index may also be useful for:
- Finding all agents sharing a conversation
- Future "list agents by conversation" feature

---

## 19. ValidationError Field Tracking

**Location:** `exceptions.py:12-17`
```python
class ValidationError(ImmAgentError):
    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(f"{field}: {message}")
```

**Observation:** ValidationError tracks which field failed.

**Status:** ✅ Good pattern

**Why:** Enables programmatic error handling:
```python
except ValidationError as e:
    if e.field == "user_input":
        # Handle missing input
```

---

## 20. Tool Call Arguments as JSON String

**Location:** `messages.py:19`
```python
arguments: str  # JSON string of arguments
```

**Observation:** Tool arguments stored as JSON string, not parsed dict.

**Status:** ✅ Actually OK

**Why:** This matches the LLM API format exactly. Parsing/re-serializing would:
1. Risk format changes
2. Add overhead
3. Potentially lose precision on numbers

The JSON string is parsed only when executing the tool (`mcp.py:165`).

---

## Summary Table

| # | Item | Status | Action |
|---|------|--------|--------|
| 1 | Threading lock with async | ✅ OK | None |
| 2 | MemoryStore strong refs | ✅ OK | None (no DB fallback) |
| 3 | Hidden `_store` field | ✅ OK | UUID is identity, underscore intentional |
| 4 | Model not enum | ✅ OK | None |
| 5 | clone() semantics | ✅ OK | Renamed from copy(); sibling for branching |
| 6 | Conversation indirection | ✅ OK | Cache efficiency + DB normalization |
| 7 | Silent lineage break | ✅ Fixed | Now raises AgentNotFoundError |
| 8 | N+1 lineage queries | ✅ Fixed | Now uses recursive CTE |
| 9 | Asset base class | ✅ OK | None |
| 10 | GC without transaction | ✅ Fixed | Now wrapped in transaction |
| 11 | Write-through ordering | ✅ OK | Simple pattern, documented |
| 12 | Tool error handling | ✅ OK | None |
| 13 | No streaming | 📋 Planned | Future feature |
| 14 | No listing/querying | ✅ Implemented | list_agents, count_agents, find_by_name |
| 15 | No model config | ✅ Implemented | model_config + advance() overrides |
| 16 | No multimodal | 📋 Planned | UUID approach enables this |
| 17 | ON DELETE SET NULL | ✅ OK | None |
| 18 | conversation_id index | ✅ OK | None |
| 19 | ValidationError.field | ✅ OK | None |
| 20 | Arguments as JSON string | ✅ OK | None |
