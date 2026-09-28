import json

import asyncpg
from asyncpg import Pool
from pgvector.asyncpg import register_vector

from settings import get_settings


async def _init_conn(conn) -> None:
    await register_vector(conn)
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def create_pool() -> Pool:
    return await asyncpg.create_pool(
        dsn=get_settings().database_url.get_secret_value(),
        min_size=2,
        max_size=10,
        init=_init_conn,
    )
