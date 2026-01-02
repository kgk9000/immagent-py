"""Shared helpers for examples."""

import os
from contextlib import asynccontextmanager

import immagent
from testcontainers.postgres import PostgresContainer


@asynccontextmanager
async def get_store():
    """Get a Store, using testcontainers if DATABASE_URL not set."""
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        async with await immagent.Store.connect(db_url) as store:
            await store.init_schema()
            yield store
    else:
        with PostgresContainer("postgres:16-alpine") as pg:
            url = pg.get_connection_url().replace("postgresql+psycopg2", "postgresql")
            async with await immagent.Store.connect(url) as store:
                await store.init_schema()
                yield store
