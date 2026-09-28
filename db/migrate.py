"""Database migration runner.

Usage:
    uv run python -m db.migrate          # apply all pending migrations
    uv run python -m db.migrate up       # same as above
    uv run python -m db.migrate down     # roll back the last applied migration
"""
import argparse
import asyncio
import re
from pathlib import Path

import asyncpg

from settings import get_settings

_MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"


async def _run(direction: str) -> None:
    conn = await asyncpg.connect(get_settings().database_url.get_secret_value())
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version    TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)

        files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
        applied = {row["version"] for row in await conn.fetch("SELECT version FROM schema_migrations")}

        if direction == "up":
            pending = [f for f in files if f.stem not in applied]
            if not pending:
                print("Already up to date.")
                return
            for f in pending:
                up_sql = _section(f.read_text(), "up")
                async with conn.transaction():
                    await conn.execute(up_sql)
                    await conn.execute(
                        "INSERT INTO schema_migrations (version) VALUES ($1)", f.stem
                    )
                print(f"  applied  {f.stem}")
        else:
            applied_sorted = sorted(applied, reverse=True)
            if not applied_sorted:
                print("Nothing to roll back.")
                return
            last = applied_sorted[0]
            f = _MIGRATIONS_DIR / f"{last}.sql"
            down_sql = _section(f.read_text(), "down")
            async with conn.transaction():
                await conn.execute(down_sql)
                await conn.execute(
                    "DELETE FROM schema_migrations WHERE version = $1", last
                )
            print(f"  rolled back  {last}")
    finally:
        await conn.close()


def _section(sql: str, name: str) -> str:
    m = re.search(
        rf"--\s*migrate:{name}\s*\n(.*?)(?=--\s*migrate:|$)",
        sql,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"No '-- migrate:{name}' section found in migration")
    return m.group(1).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument("direction", choices=["up", "down"], nargs="?", default="up")
    asyncio.run(_run(parser.parse_args().direction))
