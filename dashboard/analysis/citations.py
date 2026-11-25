"""Clickable [n] markers for saved answers. Rows from before citations were objects hold strings and stay plain."""
import re
from typing import Any

_MARKER = re.compile(r"\[(\d+)\]")


def chunk_urls(citations: list[Any], api_url: str) -> dict[str, str]:
    """[n] marker -> the API's page for the passage it cites."""
    objects = [c for c in citations if isinstance(c, dict)]  # old rows: strings, nothing to link to
    return {str(c["n"]): f"{api_url}/chunks/{c['chunk_id']}" for c in objects}


def link_markers(answer: str, urls: dict[str, str]) -> str:
    """Markdown: each cited [n] becomes a link; markers that cite nothing stay as written."""
    def link(m: re.Match[str]) -> str:
        url = urls.get(m.group(1))
        return f"[{m.group(0)}]({url})" if url else m.group(0)

    return _MARKER.sub(link, answer)
