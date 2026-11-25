import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from agent.graph import log_entry, route_of
from agent.nodes.reasoner import MarkerStream
from agent.serving import ServingStats
from agent.state import initial_state
from db.repositories.query_log import QueryLogRepository

log = logging.getLogger(__name__)
Event = tuple[Literal["token", "status"], str]  # ("token", answer text) or ("status", "held")


@dataclass
class Run:
    """One graph run and what is known about it so far; a failed run keeps its partial state for the log row."""

    run_id: str
    question: str
    model: str | None = None  # one of agent.nodes._llm.models() for every LLM call of the run; None: the default
    conversation_id: str | None = None
    state: dict[str, Any] = field(default_factory=dict)  # no reducers in AgentState, so updates just overwrite keys
    last_node: str | None = None
    t0: float = field(default_factory=time.monotonic)
    logged: bool = False
    sent: str = ""  # answer text already streamed to the client
    serving: ServingStats = field(init=False)

    def __post_init__(self) -> None:
        self.serving = ServingStats(self.t0)

    @property
    def answer(self) -> str:
        """What the user got: streamed text is final (no retraction), even when the output check flags it later."""
        return self.sent or self.state.get("answer", "")

    @property
    def latency_ms(self) -> int:
        return int((time.monotonic() - self.t0) * 1000)

    @property
    def keeps_history(self) -> bool:
        """Blocked turns stay out of the history the next turn sees."""
        return route_of(self.state) in ("reason", "clarify")

    async def record(self, query_log: QueryLogRepository, route: str | None = None, reason: str | None = None) -> None:
        """Write this run's query_log row once; best-effort, since analytics must not fail the user's request."""
        if self.logged:
            return
        self.logged = True
        try:
            await query_log.log_query(log_entry(self.run_id, self.question, self.state, self.latency_ms,
                                                self.conversation_id, self.model, route=route, reason=reason, llm_calls=self.serving.calls,
                                                streamed=self.sent))
        except Exception:
            log.exception("Failed to write query_log for run_id=%s", self.run_id)


class RunError(Exception):
    def __init__(self, run: Run, reason: str) -> None:
        super().__init__(reason)
        self.run = run
        self.reason = reason


class RunTimeout(RunError):
    pass


class RunFailed(RunError):
    pass


async def run_graph(graph: CompiledStateGraph, run: Run, history: list[dict], timeout: float,
                    stream: bool = False) -> AsyncIterator[Event]:
    """Run the graph into run.state; raises RunTimeout past the deadline and RunFailed on any other error.

    With stream, yields the reason node's text as it is written when screen set stream_ok, else one
    ("status", "held") once screen has passed the question. Only the graph's own work counts toward the
    deadline, not the time the caller takes to send each event.
    """
    config: RunnableConfig = {
        "run_id": uuid.UUID(run.run_id),
        "run_name": "web-rag-qa",
        "metadata": {"question": run.question[:200]},
        "tags": ["web-rag-qa"],
        "callbacks": [run.serving],
        "configurable": {"model": run.model},  # read by get_llm in every node
    }
    initial = initial_state(run.question, history)
    run.state.update(initial)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    markers = MarkerStream()
    events = cast(AsyncGenerator, graph.astream(initial, config=config,  # typed as an iterator, is a generator
                                                stream_mode=["updates", "messages"] if stream else ["updates"]))
    try:
        while (item := await _next(events, deadline)) is not None:
            mode, data = item
            if mode == "updates":
                for node, delta in data.items():
                    run.state.update(delta or {})
                    run.last_node = node
                    if stream and node == "screen" and not (run.state["stream_ok"] or run.state["block_reason"]):
                        yield "status", "held"
            elif run.state["stream_ok"] and data[1].get("langgraph_node") == "reason":
                if text := markers.feed(str(data[0].content)):
                    run.sent += text
                    yield "token", text
        if tail := markers.flush():
            run.sent += tail
            yield "token", tail
    except TimeoutError as exc:
        if loop.time() < deadline:  # a node's own TimeoutError (e.g. a slow asyncpg query) is a failure, not the deadline
            raise RunFailed(run, f"TimeoutError: {str(exc)[:300]} ({_where(run)})") from exc
        raise RunTimeout(run, f"timed out after {timeout:.0f} s; last finished node: {run.last_node or 'none'}") from exc
    except Exception as exc:
        raise RunFailed(run, f"{type(exc).__name__}: {str(exc)[:300]} ({_where(run)})") from exc
    finally:
        await events.aclose()


async def _next(events: AsyncIterator[Any], deadline: float) -> Any:
    """The next stream item, or None at the end; past the deadline raises TimeoutError."""
    async with asyncio.timeout_at(deadline):
        return await anext(events, None)


def _where(run: Run) -> str:
    """astream yields finished nodes only, so the failing one is whichever ran after last_node."""
    return f"after {run.last_node}" if run.last_node else "before any node finished"
