"""Base asset types for the immutable agent system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar, Self
from uuid import UUID, uuid4


def new_id() -> UUID:
    """Generate a new UUID4 for an asset."""
    return uuid4()


def now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class Asset(ABC):
    """Base class for all immutable assets.

    Every asset has a unique UUID and creation timestamp.
    Assets are immutable - any modification creates a new asset with a new ID.

    Subclasses must define:
    - TABLE: The database table name
    - SELECT_SQL: SQL to select by ID (with $1 placeholder)
    - from_row(): Class method to construct from a database row
    - to_insert_params(): Returns (sql, params) for insertion
    """

    id: UUID
    created_at: datetime

    # Subclasses override these
    TABLE: ClassVar[str]
    SELECT_SQL: ClassVar[str]

    @classmethod
    @abstractmethod
    def from_row(cls, row: Any) -> Self:
        """Construct an asset from a database row."""
        ...

    @abstractmethod
    def to_insert_params(self) -> tuple[str, tuple[Any, ...]]:
        """Return (INSERT SQL, parameters) for this asset."""
        ...


@dataclass(frozen=True)
class SystemPrompt(Asset):
    """Immutable system prompt for an agent."""

    content: str

    TABLE: ClassVar[str] = "text_assets"
    SELECT_SQL: ClassVar[str] = "SELECT id, created_at, content FROM text_assets WHERE id = $1"

    @classmethod
    def create(cls, content: str) -> "SystemPrompt":
        """Create a new SystemPrompt with auto-generated ID and timestamp."""
        return cls(id=new_id(), created_at=now(), content=content)

    @classmethod
    def from_row(cls, row: Any) -> "SystemPrompt":
        """Construct from a database row."""
        return cls(
            id=row["id"],
            created_at=row["created_at"],
            content=row["content"],
        )

    def to_insert_params(self) -> tuple[str, tuple[Any, ...]]:
        """Return (INSERT SQL, parameters) for this asset."""
        return (
            """INSERT INTO text_assets (id, created_at, content)
               VALUES ($1, $2, $3) ON CONFLICT (id) DO NOTHING""",
            (self.id, self.created_at, self.content),
        )
