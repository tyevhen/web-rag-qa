import logging
import re
from typing import Any
from urllib.parse import quote

from agent.nodes.screen import check, refuse
from agent.state import AgentState, Chunk

log = logging.getLogger(__name__)

_LINE_MARKER = re.compile(r"^\s*(?:[#>*-]+|\d+[.)])\s+")  # heading, quote and list markers aren't text on the page
_MARKUP = re.compile(r"[*_`\[\]()<>|#]")
_FRAGMENT_WORDS = 6


def text_fragment(content: str) -> str:
    """`#:~:text=` with the passage's first words, so browsers that support text fragments scroll to it.
    Words stop at the first one with markdown in it (the page shows it rendered); fewer than 3 is too vague."""
    words = _LINE_MARKER.sub("", content.split("\n", 1)[0]).split()
    plain: list[str] = []
    for word in words[:_FRAGMENT_WORDS]:
        if _MARKUP.search(word):
            break
        plain.append(word)
    if len(plain) < 3:
        return ""
    return "#:~:text=" + quote(" ".join(plain), safe="").replace("-", "%2D")  # "-" and "," are fragment syntax


def _build_citation(index: int, chunk: Chunk) -> dict[str, Any]:
    """The API adds `url`, the passage at GET /chunks/{chunk_id}; the answer keeps its plain [n] markers."""
    meta = chunk.metadata
    url = meta.get("url", "")
    if "#page=" not in url:  # a PDF page link (ingestion/pdf.py) has its fragment; PDF viewers ignore text ones
        url += text_fragment(chunk.content)
    return {"n": index + 1, "chunk_id": chunk.id, "section": meta.get("section", ""), "source_url": url}


async def format_response(state: AgentState) -> dict[str, Any]:
    chunks = state["graded_chunks"]
    answer = state["answer"]
    if answer and (category := await check(answer)):
        return refuse(f"output check on reason: REFUSE {category}", state["lang"])

    cited: list[int] = []
    for match in re.finditer(r"\[(\d+)\]", answer):
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(chunks) and idx not in cited:
            cited.append(idx)

    if not cited:
        # No guessed sources: showing chunks the answer may not use is worse than showing none
        log.warning("format: answer cites no sources (%d graded chunks)", len(chunks))

    return {"citations": [_build_citation(i, chunks[i]) for i in cited]}
