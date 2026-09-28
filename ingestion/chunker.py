from typing import Any

from ingestion.scraper import RawChunk


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
    return [c for c in result if c.content.strip()]
