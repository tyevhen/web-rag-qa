"""PDF sources: one marked-up markdown text per PDF, from the local copy described in sources/<slug>/source.json."""
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pymupdf
import pymupdf4llm
from pymupdf4llm.helpers import pymupdf_rag

from ingestion.chunker import fits
from ingestion.evaluator import MIN_WORDS, word_count
from ingestion.scraper import RawChunk

SOURCES_DIR = Path(__file__).resolve().parent.parent / "sources"
_MARKER = re.compile(r"<!--\s*page\s+(\d+)\s*-->\n?")
_PICTURE_TEXT = re.compile(r"<!-- Start of picture text -->.*?<!-- End of picture text -->", re.S)
_RUNNING_HEADER = re.compile(r"^#*\s*\**T E K E\**\s*$", re.M)
# "**10** <u>TEGUTSEMINE ÕNNETUSE KORRAL</u>", "<u>ESMAABI …</u> **<u>35</u>**": legacy mode keeps the page header
_UNDERLINED_HEADER = re.compile(r"^[*\s]*(<u>)?\d*(</u>)?[*\s]*<u>[^<]*</u>[*\s]*(<u>)?\d*(</u>)?[*\s]*$", re.M)
_MARK = re.compile(r"</?mark>")  # highlighted headings
_SOFT_HYPHEN = re.compile(r"(\w+)\xad\s*(\w+)")
_HEADING = re.compile(r"^(#+)\s+(.*)$", re.M)
_CAPTION = re.compile(r"^(Tabel|LISA)\s")
_LIGATURE = re.compile(r"(\w+) \*\*(\w)\*\* (\w+)")  # "morf **i** n" for "morfiin": the fi ligature split out


def find_source(url: str) -> tuple[Path, dict]:
    """The source.json with this url and its folder; the worker never downloads, the PDF must be in the folder."""
    for path in sorted(SOURCES_DIR.glob("*/source.json")):
        spec = json.loads(path.read_text())
        if spec["url"] == url:
            if not (path.parent / spec["file"]).is_file():
                raise RuntimeError(f"{spec['file']} missing in {path.parent}; download it from {url}")
            return path.parent, spec
    raise RuntimeError(f"No sources/*/source.json with url {url!r}")


def _heading(m: re.Match) -> str:
    text = m.group(2).strip()
    plain = text.strip("*_ ")
    if _CAPTION.match(plain):
        return f"**{plain}**"  # a caption stays in the chunk text instead of becoming section metadata
    letters = [c for c in plain if c.isalpha()]
    if sum(c.isupper() for c in letters) > len(letters) / 2:
        plain = plain.upper()  # small-caps headings come out mixed case ("TrAuMA MEHHANisM")
    return f"{m.group(1)} {plain}"  # without bold marks: the heading becomes the chunk's section


def _unsplit(m: re.Match, raw_text: str) -> str:
    """The word the raw text has: the ligature may stand for one letter or two ("mikrofiibrist")."""
    a, b, c = m.groups()
    return next((w for w in (a + b + c, a + b + b + c) if w in raw_text), a + b + c)


def clean_page(md: str, raw_text: str) -> str:
    """Fix what pymupdf4llm gets wrong on a page, using PyMuPDF's raw text of the same page for soft hyphens."""
    for a, b in set(_SOFT_HYPHEN.findall(raw_text)):
        md = re.sub(rf"\b{re.escape(a)}[\s\xad-]+{re.escape(b)}\b", a + b, md)
    md = _PICTURE_TEXT.sub("", md).replace("<br>", " ").replace("\xad", "")
    md = _MARK.sub("", md)
    md = _UNDERLINED_HEADER.sub("", _RUNNING_HEADER.sub("", md))
    md = _LIGATURE.sub(lambda m: _unsplit(m, raw_text), md)
    return _HEADING.sub(_heading, md)


def load_pdf(url: str) -> RawChunk:
    """The whole PDF as one text, each page preceded by a `<!-- page N -->` marker (see assign_pages)."""
    folder, spec = find_source(url)
    path = folder / spec["file"]
    skip = set(spec.get("skip_pages", []))
    raw = pymupdf.open(path)
    parts = []
    if spec.get("columns"):
        # The layout model interleaves two-column pages ("Väldi" over the other column's steps); the legacy
        # reader keeps columns apart but finds no headings. Called directly: use_layout() is process-wide.
        pages = pymupdf_rag.to_markdown(str(path), page_chunks=True, ignore_images=True)
    else:
        pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, header=False, footer=False, ignore_images=True)
    for page in pages:
        n = page["metadata"].get("page_number") or page["metadata"]["page"]
        if n in skip:
            continue
        text = clean_page(page["text"], raw[n - 1].get_text()).strip()
        figure = folder / "figures" / f"p{n:03d}.md"
        if figure.is_file():
            text = f"{text}\n\n{figure.read_text().strip()}"
        parts.append(f"<!-- page {n} -->\n\n{text}")
    # Blank lines around markers: a marker is a paragraph of its own, so the chunker never splits one
    return RawChunk(content="\n\n".join(parts), metadata={"url": url, "source": urlparse(url).netloc})


def assign_pages(chunks: list[RawChunk], url: str) -> list[RawChunk]:
    """Point each chunk at the page it starts on (`url#page=N`) and drop the markers.
    Runs on all chunks in document order, before any are filtered out, so no marker is lost."""
    page = 1
    result = []
    for chunk in chunks:
        content = chunk.content.strip()
        if m := _MARKER.match(content):
            page = int(m.group(1))
        start = page
        for m in _MARKER.finditer(content):
            page = int(m.group(1))
        text = _MARKER.sub("", content).strip()
        result.append(RawChunk(content=text, metadata={**chunk.metadata, "url": f"{url}#page={start}"}))
    return result


def merge_short(chunks: list[RawChunk]) -> list[RawChunk]:
    """Prepend a section too short for the evaluator (often one sentence under a bold heading) to the next chunk,
    its heading as a bold line, instead of losing it; the merged chunk keeps the short one's page link.
    If the next chunk is too full, append it to the previous one."""
    result: list[RawChunk] = []
    carry: RawChunk | None = None
    for chunk in chunks:
        if carry:
            head = f"**{carry.metadata['section']}**\n" if carry.metadata.get("section") else ""
            merged = f"{head}{carry.content}\n\n{chunk.content}"
            if fits(merged):
                chunk = RawChunk(content=merged, metadata={**chunk.metadata, "url": carry.metadata["url"]})
            elif result and fits(appended := f"{result[-1].content}\n\n{head}{carry.content}"):
                result[-1] = RawChunk(content=appended, metadata=result[-1].metadata)
            else:
                result.append(carry)
            carry = None
        if word_count(chunk.content) < MIN_WORDS:
            carry = chunk
        else:
            result.append(chunk)
    return result + ([carry] if carry else [])
