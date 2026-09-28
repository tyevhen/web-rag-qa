import re
from typing import Any

from agent.state import AgentState, Chunk


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
        cited = list(range(min(len(chunks), 3)))

    return {"citations": [_build_citation(i, chunks[i]) for i in cited]}
