"""Streamlit scripts are sync, so each loader runs one short-lived async pool."""
import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import asyncpg
from asyncpg import Pool
import pandas as pd

from db.client import create_pool

T = TypeVar("T")


def run(query: Callable[[Pool], Awaitable[T]]) -> T:
    async def go() -> T:
        pool = await create_pool()
        try:
            return await query(pool)
        finally:
            await pool.close()

    return asyncio.run(go())


def frame(rows: list[asyncpg.Record], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([dict(r) for r in rows], columns=columns)
