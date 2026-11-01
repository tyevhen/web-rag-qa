import logging
import re
from typing import Any

from agent.state import AgentState, Chunk

log = logging.getLogger(__name__)


def _build_citation(index: int, chunk: Chunk) -> str:
    parts = [f"[{index + 1}]"] + [f"{k}: {v}" for k, v in chunk.metadata.items()]
    return " | ".join(parts)


async def format_response(state: AgentState) -> dict[str, Any]:
    chunks = state["graded_chunks"]
    answer = state["answer"]

    cited: list[int] = []
    for match in re.finditer(r"\[(\d+)\]", answer):
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(chunks) and idx not in cited:
            cited.append(idx)

    if not cited:
        # No guessed sources: showing chunks the answer may not use is worse than showing none
        log.warning("format: answer cites no sources (%d graded chunks)", len(chunks))

    return {"citations": [_build_citation(i, chunks[i]) for i in cited]}
