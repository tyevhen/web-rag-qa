import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

GOLDEN_PATH = Path("tests/eval/golden.jsonl")

# answerable = synthetic single-hop; unanswerable = must route to clarify; multi_turn = uses history;
# cross_domain = hand-written, needs both first-aid and survival sources (exercises the decomposer)
CATEGORIES = ("answerable", "unanswerable", "multi_turn", "cross_domain")


@dataclass
class GoldenRow:
    question: str
    category: str
    reference: str = ""
    reference_contexts: list[str] = field(default_factory=list)
    # MD5 of the source chunk content — stable across re-ingestion, unlike chunk UUIDs
    reference_context_hashes: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    synthesizer: str = ""


def content_hash(text: str) -> str:
    return hashlib.md5(" ".join(text.split()).encode()).hexdigest()


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenRow]:
    rows = [GoldenRow(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    unknown = {r.category for r in rows} - set(CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown categories in {path}: {unknown}")
    return rows


def save_golden(rows: list[GoldenRow], path: Path = GOLDEN_PATH) -> None:
    path.write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in rows))
