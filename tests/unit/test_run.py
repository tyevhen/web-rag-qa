import asyncio
import uuid
from typing import Any, cast

import pytest
from arq.jobs import JobResult, JobStatus
from langchain_core.messages import AIMessageChunk
from langgraph.graph.state import CompiledStateGraph

from agent.run import Run, RunFailed, RunTimeout, run_graph
from agent.state import Chunk
from api.schemas import IngestJobStatus
from db.repositories.query_log import QueryLogRepository


class FakeGraph:
    """Streams `updates` (a dict is an update, a (node, text) pair a token), then raises `error` or stalls."""

    def __init__(self, updates: list[dict | tuple[str, str]], error: Exception | None = None, stall: float = 0) -> None:
        self.updates, self.error, self.stall = updates, error, stall

    async def astream(self, state: Any, config: Any, stream_mode: list[str]):
        for update in self.updates:
            if isinstance(update, dict):
                yield "updates", update
            elif "messages" in stream_mode:
                yield "messages", (AIMessageChunk(content=update[1]), {"langgraph_node": update[0]})
        if self.stall:
            await asyncio.sleep(self.stall)
        if self.error:
            raise self.error


def _run() -> Run:
    return Run(run_id=str(uuid.uuid4()), question="q")


async def _go(graph: FakeGraph, timeout: float = 5) -> Run:
    run = _run()
    async for _ in run_graph(cast(CompiledStateGraph, graph), run, [], timeout):
        pass
    return run


async def _stream(graph: FakeGraph) -> tuple[Run, list[tuple[str, str]]]:
    run = _run()
    return run, [e async for e in run_graph(cast(CompiledStateGraph, graph), run, [], 5, stream=True)]


async def test_an_allowed_question_streams_only_the_reason_text():
    run, events = await _stream(FakeGraph([{"screen": {"stream_ok": True}}, ("decompose", "sub q"),
                                           ("reason", "Boil it 【1"), ("reason", "】."), {"format": {"answer": "x"}}]))
    assert events == [("token", "Boil it "), ("token", "[1].")]  # a wide marker split across tokens is held, then fixed
    assert run.answer == "Boil it [1]."  # the sent text is the answer, not the formatter's rewrite


async def test_a_sensitive_question_is_held_and_streams_nothing():
    run, events = await _stream(FakeGraph([{"screen": {}}, ("reason", "secret"), {"format": {"answer": "whole"}}]))
    assert events == [("status", "held")] and run.answer == "whole"


async def test_a_blocked_question_is_neither_held_nor_streamed():
    _, events = await _stream(FakeGraph([{"screen": {"block_reason": "input check: REFUSE weapons"}}]))
    assert events == []


async def test_a_finished_run_merges_every_update():
    chunk = Chunk(content="c", metadata={}, id="1", score=0.0)
    run = await _go(FakeGraph([{"decompose": {"sub_questions": ["s"]}}, {"grade": {"graded_chunks": [chunk]}},
                               {"format": {"answer": "a"}}]))
    assert run.state["sub_questions"] == ["s"] and run.state["answer"] == "a" and run.state["question"] == "q"
    assert run.last_node == "format" and run.keeps_history


async def test_a_blocked_run_keeps_no_history():
    run = await _go(FakeGraph([{"screen": {"block_reason": "input check: REFUSE weapons", "answer": "no"}}]))
    assert not run.keeps_history


async def test_a_failing_node_names_the_last_finished_one():
    with pytest.raises(RunFailed) as info:
        await _go(FakeGraph([{"decompose": {"sub_questions": ["s"]}}], error=ValueError("boom")))
    assert info.value.reason == "ValueError: boom (after decompose)"
    assert info.value.run.state["sub_questions"] == ["s"]


async def test_the_deadline_raises_a_timeout_with_the_last_node():
    with pytest.raises(RunTimeout) as info:
        await _go(FakeGraph([{"screen": {}}], stall=1), timeout=0.01)
    assert info.value.reason == "timed out after 0 s; last finished node: screen"


async def test_a_nodes_own_timeout_is_a_failure_not_the_deadline():
    with pytest.raises(RunFailed) as info:
        await _go(FakeGraph([{"decompose": {}}], error=TimeoutError("query timed out")))
    assert info.value.reason == "TimeoutError: query timed out (after decompose)"


class BrokenLog:
    calls = 0

    async def log_query(self, entry: Any) -> None:
        self.calls += 1
        raise ConnectionError("db down")


async def test_record_is_best_effort_and_writes_once():
    run, log = _run(), BrokenLog()
    await run.record(cast(QueryLogRepository, log), "error", "x")
    await run.record(cast(QueryLogRepository, log))
    assert run.logged and log.calls == 1


def _job(success: bool, result: Any) -> JobResult:
    return cast(JobResult, type("Info", (), {"success": success, "result": result})())


def test_job_status_mapping():
    done = IngestJobStatus.from_job("j", JobStatus.complete,
                                    _job(True, {"url": "u", "pages_scraped": 2, "summary": "s"}))
    assert done.status == "complete" and done.result and done.result.pages_scraped == 2
    failed = IngestJobStatus.from_job("j", JobStatus.complete, _job(False, RuntimeError("no sitemap")))
    assert failed.status == "failed" and failed.error == "RuntimeError" and failed.result is None
    queued = IngestJobStatus.from_job("j", JobStatus.queued, None)
    assert queued.status == "queued" and queued.result is None and queued.error is None
