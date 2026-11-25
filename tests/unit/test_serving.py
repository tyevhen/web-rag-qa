import uuid

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agent.serving import ServingStats, tpot


def _result(output: int, reasoning: int = 0) -> LLMResult:
    usage = {"input_tokens": 100, "output_tokens": output, "total_tokens": 100 + output,
             "output_token_details": {"reasoning": reasoning}}
    return LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x", usage_metadata=usage))]])


async def _call(stats: ServingStats, tokens: list[str], result: LLMResult) -> None:
    run_id = uuid.uuid4()
    await stats.on_chat_model_start({}, [], run_id=run_id, metadata={"langgraph_node": "reason"})
    for t in tokens:
        await stats.on_llm_new_token(t, run_id=run_id)
    await stats.on_llm_end(result, run_id=run_id)


async def test_a_streamed_call_is_recorded_with_its_node_and_tokens():
    stats = ServingStats()
    await _call(stats, ["", "a", "b"], _result(output=12, reasoning=2))
    [call] = stats.calls
    assert (call.node, call.prompt_tokens, call.output_tokens) == ("reason", 100, 12)
    assert call.ttft_ms <= call.total_ms and call.tpot_ms is not None


async def test_a_call_without_tokens_is_skipped():  # a cache hit streams nothing
    stats = ServingStats()
    await _call(stats, [], _result(output=5))
    assert stats.calls == []


def test_tpot_leaves_out_the_first_token_and_needs_two():
    assert tpot(ttft_ms=100, total_ms=1100, visible_tokens=11) == 100
    assert tpot(ttft_ms=100, total_ms=100, visible_tokens=1) is None
