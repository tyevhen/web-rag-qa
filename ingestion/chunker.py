from typing import Any

from ingestion.scraper import RawChunk

# ~375 tokens: sized for the embedder's context and keeps reasoner prompts well under LLM limits
_MAX_CHARS = 1500


def _section_aware(text: str, meta: dict[str, Any]) -> list[RawChunk]:
    buf: list[str] = []
    header: str = meta.get("section", "")
    result: list[RawChunk] = []

    for line in text.splitlines():
        s = line.strip()
        if s and len(s) <= 80 and s[0].isupper() and s[-1] not in ".,;:" and buf:
            result.append(RawChunk(content="\n".join(buf).strip(), metadata={**meta, "section": header}))
            header = s
            buf = []
        else:
            buf.append(line)

    if buf:
        result.append(RawChunk(content="\n".join(buf).strip(), metadata={**meta, "section": header}))
    return result


def _markdown_headers(text: str, meta: dict[str, Any]) -> list[RawChunk]:
    buf: list[str] = []
    header = ""
    result: list[RawChunk] = []

    for line in text.splitlines():
        if line.startswith("#"):
            if buf:
                result.append(RawChunk(content="\n".join(buf).strip(), metadata={**meta, "section": header}))
            header = line.lstrip("#").strip()
            buf = []
        else:
            buf.append(line)

    if buf:
        result.append(RawChunk(content="\n".join(buf).strip(), metadata={**meta, "section": header}))
    return result


def _hard_split(text: str) -> list[str]:
    """Split an oversized paragraph on word boundaries."""
    parts: list[str] = []
    buf = ""
    words = [w[i : i + _MAX_CHARS] for w in text.split() for i in range(0, len(w), _MAX_CHARS)]
    for word in words:
        if buf and len(buf) + 1 + len(word) > _MAX_CHARS:
            parts.append(buf)
            buf = word
        else:
            buf = f"{buf} {word}" if buf else word
    if buf:
        parts.append(buf)
    return parts


def _cap_size(chunk: RawChunk) -> list[RawChunk]:
    """Pack paragraphs into pieces of at most _MAX_CHARS; sections without headers would otherwise stay whole."""
    if len(chunk.content) <= _MAX_CHARS:
        return [chunk]
    pieces: list[str] = []
    buf = ""
    for para in (p.strip() for p in chunk.content.split("\n\n")):
        if not para:
            continue
        for part in [para] if len(para) <= _MAX_CHARS else _hard_split(para):
            if buf and len(buf) + 2 + len(part) > _MAX_CHARS:
                pieces.append(buf)
                buf = part
            else:
                buf = f"{buf}\n\n{part}" if buf else part
    if buf:
        pieces.append(buf)
    return [RawChunk(content=p, metadata=chunk.metadata) for p in pieces]


def chunk_raw(
    raw: list[RawChunk],
    strategy: str,
    metadata_fields: list[str],
) -> list[RawChunk]:
    result: list[RawChunk] = []
    for rc in raw:
        filtered = {k: v for k, v in rc.metadata.items() if k in metadata_fields}
        if strategy == "markdown_headers":
            result.extend(_markdown_headers(rc.content, filtered))
        else:
            result.extend(_section_aware(rc.content, filtered))
    return [piece for c in result if c.content.strip() for piece in _cap_size(c)]
