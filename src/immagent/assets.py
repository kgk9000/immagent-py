"""Base asset types for the immutable agent system."""

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4


def new_id() -> UUID:
    """Generate a new UUID4 for an asset."""
    return uuid4()


def now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class Asset:
    """Base class for all immutable assets.

    Every asset has a unique UUID and creation timestamp.
    Assets are immutable - any modification creates a new asset with a new ID.
    """

    id: UUID
    created_at: datetime


@dataclass(frozen=True)
class SystemPrompt(Asset):
    """Immutable system prompt for an agent."""

    content: str

    @classmethod
    def create(cls, content: str) -> "SystemPrompt":
        """Create a new SystemPrompt with auto-generated ID and timestamp."""
        return cls(id=new_id(), created_at=now(), content=content)
