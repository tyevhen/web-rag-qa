import logging

from arq import cron
from arq.connections import RedisSettings
from crawl4ai import AsyncWebCrawler

from db.client import create_pool
from db.eval_repository import EvalRepository
from db.query_log_repository import QueryLogRepository
from db.repository import DocumentRepository
from evaluation.online import score_recent
from services.ingestion import ingest_url
from settings import get_settings

log = logging.getLogger(__name__)


async def ingest_url_task(ctx: dict, url: str, mode: str = "crawl") -> dict:
    log.info("Worker: starting ingestion for %s (mode=%s)", url, mode)
    summary, pages_scraped = await ingest_url(url, ctx["documents"], ctx["crawler"], mode)
    log.info("Worker: finished ingestion for %s (%d pages)", url, pages_scraped)
    return {"url": url, "pages_scraped": pages_scraped, "summary": summary}


async def score_recent_task(ctx: dict) -> dict:
    scored = await score_recent(ctx["documents"], ctx["query_log"], ctx["evals"], get_settings().eval_sample_size)
    return {"scored": scored}


async def startup(ctx: dict) -> None:
    pool = await create_pool()
    crawler = AsyncWebCrawler()
    await crawler.__aenter__()
    ctx["pool"] = pool
    ctx["documents"] = DocumentRepository(pool)
    ctx["query_log"] = QueryLogRepository(pool)
    ctx["evals"] = EvalRepository(pool)
    ctx["crawler"] = crawler


async def shutdown(ctx: dict) -> None:
    await ctx["pool"].close()
    await ctx["crawler"].__aexit__(None, None, None)


class WorkerSettings:
    functions = [ingest_url_task, score_recent_task]
    cron_jobs = [cron(score_recent_task, hour=3, minute=0, run_at_startup=False)]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    # arq's default is 300s; crawls run ~1s/page, so a 570-page site was killed mid-crawl
    job_timeout = 3600
