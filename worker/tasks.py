import logging

from arq.connections import RedisSettings
from crawl4ai import AsyncWebCrawler

from db.client import create_pool
from db.repository import DocumentRepository
from services.ingestion import ingest_url
from settings import get_settings

log = logging.getLogger(__name__)


async def ingest_url_task(ctx: dict, url: str, mode: str = "crawl") -> dict:
    log.info("Worker: starting ingestion for %s (mode=%s)", url, mode)
    summary, pages_scraped = await ingest_url(url, ctx["repo"], ctx["crawler"], mode)
    log.info("Worker: finished ingestion for %s (%d pages)", url, pages_scraped)
    return {"url": url, "pages_scraped": pages_scraped, "summary": summary}


async def startup(ctx: dict) -> None:
    pool = await create_pool()
    crawler = AsyncWebCrawler()
    await crawler.__aenter__()
    ctx["pool"] = pool
    ctx["repo"] = DocumentRepository(pool)
    ctx["crawler"] = crawler


async def shutdown(ctx: dict) -> None:
    await ctx["pool"].close()
    await ctx["crawler"].__aexit__(None, None, None)


class WorkerSettings:
    functions = [ingest_url_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
