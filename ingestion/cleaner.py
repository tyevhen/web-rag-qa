import logging
import re
from collections import Counter

from ingestion.scraper import RawChunk

log = logging.getLogger(__name__)

# Markdown escapes brackets inside alt text and titles, e.g. [ ](url "Bag Valve Mask \(BVM\)")
_LINK_PATTERN = r"\[(?:\\.|[^\]\\])*\]\((?:\\.|[^)\\])*\)"
_IMAGE = re.compile("!" + _LINK_PATTERN)
_LINK = re.compile(_LINK_PATTERN)
_MARKERS = re.compile(r"^(?:#+|\d+[.)]|[-*•])\s*")
# Template text (login boxes, share/donate prompts) repeats across a site; real content rarely does
_MIN_PAGES = 10
_MAX_PAGE_SHARE = 0.2


def _is_navigation(line: str) -> bool:
    """Menus, breadcrumbs, share buttons, link-only headings: nothing is left once links are removed."""
    # shortcut: a paragraph written entirely as one link is dropped too; rare in article text
    # Images first: a linked image "[ ![alt](img) ](href)" only becomes a plain link once its image is gone
    return not any(c.isalpha() for c in _MARKERS.sub("", _LINK.sub("", _IMAGE.sub("", line)).strip()))


def _template_lines(pages: list[RawChunk]) -> set[str]:
    if len(pages) < _MIN_PAGES:
        return set()
    counts = Counter(line for p in pages for line in {s.strip() for s in p.content.splitlines()})
    # Headings stay: a section like "Background" legitimately repeats across same-shaped pages
    return {line for line, n in counts.items()
            if line and not line.startswith("#") and n / len(pages) >= _MAX_PAGE_SHARE}


def strip_layout(pages: list[RawChunk]) -> list[RawChunk]:
    """Drop site navigation and template text from scraped pages, before chunking."""
    template = _template_lines(pages)
    cleaned = [
        RawChunk(content="\n".join(line for line in p.content.splitlines()
                                   if line.strip() and not _is_navigation(line) and line.strip() not in template),
                 metadata=p.metadata)
        for p in pages
    ]
    before, after = sum(len(p.content) for p in pages), sum(len(p.content) for p in cleaned)
    log.info("strip_layout: %d template lines; %d → %d chars", len(template), before, after)
    return [p for p in cleaned if p.content]
