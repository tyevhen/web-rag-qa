import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from crawl4ai import AsyncUrlSeeder, AsyncWebCrawler, CrawlerRunConfig, CrawlResult, SeedingConfig


@dataclass
class RawChunk:
    content: str
    metadata: dict[str, Any]

log = logging.getLogger(__name__)

_CRAWL_CONFIG = CrawlerRunConfig(
    check_robots_txt=False, mean_delay=0.2, max_range=0.3, verbose=False
)


def _item_url(item: object) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("url", "")
    return getattr(item, "url", "")


def _extract_markdown(result: CrawlResult) -> str:
    return str(result.markdown) if result.markdown else ""


_LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")


async def _seed_urls(domain: str) -> list[str]:
    # extract_head=False: fetching every page's <head> exceeded the timeout on ~280-URL sitemaps,
    # and only the URL is used
    raw = await asyncio.wait_for(
        AsyncUrlSeeder().urls(domain, SeedingConfig(source="sitemap", extract_head=False)),
        timeout=120,
    )
    log.info("Seeder returned %d items for %s", len(raw), domain)
    return [_item_url(item) for item in raw]


async def _section_urls(url: str) -> list[str]:
    """URLs for a site section like https://site/en: its own sitemap, else the domain sitemap filtered by path."""
    prefix = url.rstrip("/")
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        resp = await client.get(f"{prefix}/sitemap.xml")
    own = [u for u in _LOC.findall(resp.text) if u.startswith(prefix)] if resp.status_code == 200 else []
    if own:
        log.info("Section sitemap %s/sitemap.xml returned %d URLs", prefix, len(own))
        return own
    filtered = [u for u in await _seed_urls(urlparse(url).netloc) if u.startswith(prefix)]
    log.info("Domain sitemap filtered to %s: %d URLs", prefix, len(filtered))
    return filtered


async def scrape_single(url: str, crawler: AsyncWebCrawler) -> list[RawChunk]:
    domain = urlparse(url).netloc
    result = cast(CrawlResult, await crawler.arun(url, config=_CRAWL_CONFIG))
    if not result.success:
        log.warning("Failed: %s — %s", result.url, result.error_message)
        return []
    md = _extract_markdown(result)
    if not md.strip():
        return []
    return [RawChunk(content=md, metadata={"url": result.url, "source": domain})]


async def scrape_site(url: str, crawler: AsyncWebCrawler) -> list[RawChunk]:
    parsed = urlparse(url)
    domain = parsed.netloc
    # A path (e.g. /en) scopes the crawl to that section; a bare domain crawls the whole sitemap
    target_urls = await (_section_urls(url) if parsed.path.strip("/") else _seed_urls(domain))

    log.info("Crawling %d URLs", len(target_urls))
    results = cast(list[CrawlResult], await crawler.arun_many(target_urls, config=_CRAWL_CONFIG))

    chunks: list[RawChunk] = []

    for result in results:
        if not result.success:
            log.warning("Failed: %s — %s", result.url, result.error_message)
            continue
        md = _extract_markdown(result)
        if not md.strip():
            continue
        chunks.append(RawChunk(content=md, metadata={"url": result.url, "source": domain}))
        log.info("Ingested [%d] %s", len(chunks), result.url)

    log.info("Done: %d pages ingested from %s", len(chunks), domain)
    return chunks
