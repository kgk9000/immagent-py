"""Pytest fixtures for immagent tests."""

import pytest
from testcontainers.postgres import PostgresContainer

from immagent import Database
from immagent.cache import clear_cache


@pytest.fixture(scope="session")
def postgres_container():
    """Start a PostgreSQL container for the test session."""
    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


@pytest.fixture(scope="session")
def database_url(postgres_container):
    """Get the database URL from the container."""
    # testcontainers returns postgresql+psycopg2://, asyncpg needs postgresql://
    url = postgres_container.get_connection_url()
    return url.replace("postgresql+psycopg2", "postgresql")


@pytest.fixture
async def db(database_url):
    """Create a database connection and initialize schema.

    This fixture is function-scoped so each test gets a fresh schema.
    Tables are dropped and recreated between tests.
    """
    database = await Database.connect(database_url)
    await database.init_schema()
    yield database
    # Clean up: drop all tables for next test
    async with database._pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS agents CASCADE")
        await conn.execute("DROP TABLE IF EXISTS conversations CASCADE")
        await conn.execute("DROP TABLE IF EXISTS messages CASCADE")
        await conn.execute("DROP TABLE IF EXISTS text_assets CASCADE")
    await database.close()


@pytest.fixture(autouse=True)
def clear_asset_cache():
    """Clear the asset cache before each test."""
    clear_cache()
    yield
    clear_cache()
