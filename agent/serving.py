"""Serving numbers per LLM call: time to first token, time per output token and token counts."""
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult


@dataclass
class LlmCall:
    node: str  # the graph node that made the call
    start_ms: int  # from the start of the run, so the user's time to first answer token is reason start + ttft
    ttft_ms: int  # time to first token: reading the prompt (prefill), queueing and any model load
    total_ms: int
    prompt_tokens: int
    output_tokens: int  # as reported; a reasoning model's hidden tokens are included
    tpot_ms: float | None  # time per visible output token after the first (decode speed); None below 2 tokens


def tpot(ttft_ms: int, total_ms: int, visible_tokens: int) -> float | None:
    return (total_ms - ttft_ms) / (visible_tokens - 1) if visible_tokens > 1 else None


class ServingStats(AsyncCallbackHandler):
    """Collects one LlmCall per streamed chat-model call of a run; pass it in the run config's callbacks.

    A call that streamed no token (a cache hit, a failure, an empty reply) has no time to first token and is skipped.
    """

    def __init__(self, t0: float | None = None) -> None:
        self.t0 = t0 if t0 is not None else time.monotonic()
        self.calls: list[LlmCall] = []
        self._open: dict[UUID, tuple[str, float]] = {}
        self._first: dict[UUID, float] = {}

    async def on_chat_model_start(self, serialized: dict[str, Any], messages: list, *, run_id: UUID,
                                  metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._open[run_id] = ((metadata or {}).get("langgraph_node", "?"), time.monotonic())

    async def on_llm_new_token(self, token: str, *, run_id: UUID, **kwargs: Any) -> None:
        if token and run_id in self._open:  # reasoning models stream empty content first
            self._first.setdefault(run_id, time.monotonic())

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        node, start = self._open.pop(run_id, ("?", 0.0))
        first = self._first.pop(run_id, None)
        if first is None:
            return
        generation = response.generations[0][0]
        message = generation.message if isinstance(generation, ChatGeneration) else None
        usage = (message.usage_metadata if isinstance(message, AIMessage) else None) or {}
        output = usage.get("output_tokens", 0)
        reasoning = usage.get("output_token_details", {}).get("reasoning", 0)
        ttft_ms, total_ms = _ms(first - start), _ms(time.monotonic() - start)
        self.calls.append(LlmCall(node, _ms(start - self.t0), ttft_ms, total_ms, usage.get("input_tokens", 0),
                                  output, tpot(ttft_ms, total_ms, output - reasoning)))

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._open.pop(run_id, None)
        self._first.pop(run_id, None)


def _ms(seconds: float) -> int:
    return int(seconds * 1000)
