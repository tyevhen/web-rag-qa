"""Training examples for the answer step: a written question → the real search and grader → the writer's answer.

    uv run python -m tuning.export --lang et --kinds answerable,detail,near_miss --out tuning/data/et.jsonl
    uv run python -m tuning.export --lang en --chunks 450 --kinds answerable,near_miss --out tuning/data/en.jsonl

Each example is the exact prompt the reasoner gets in production (answer_inputs) plus a target answer from
the writer. Writer, judge and generator are three different models, so neither the target nor its score
comes from the model being tuned. Rejected rows go to <out>.rejected.jsonl with the reason, for review.
Rows are written as they finish; a rerun skips the (chunk, kind) pairs already written, except errors.
"""
import argparse
import asyncio
import json
import logging
import math
from pathlib import Path

from ragas import SingleTurnSample

from agent.config import get_agent_config
from agent.nodes.decomposer import decompose
from agent.nodes.grader import grade
from agent.nodes.reasoner import _WIDE_MARKER, ANSWER_PROMPT, answer_inputs
from agent.nodes.retriever import make_retriever
from agent.state import initial_state
from db.client import create_pool
from db.repositories.documents import DocumentRepository
from evaluation.metrics import faithfulness_and_answer_relevancy
from tests.eval.dataset import content_hash, load_golden
from tests.eval.refusal import is_refusal
from tuning.checks import answer_problem, is_pure_refusal, near_duplicate
from tuning.writer import LANG_NAME, write_answer, write_question

log = logging.getLogger(__name__)

_GOLDEN = [Path("tests/eval/golden.jsonl"), Path("tests/eval/golden_et.jsonl")]
_MIN_FAITHFULNESS = 0.8
_CONCURRENCY = 6  # the time is waiting on Mistral, the judge and Ollama; the judge's RPM limit is shared
_ROLE = {"system": "system", "human": "user", "ai": "assistant"}


async def _faithfulness(question: str, answer: str, contexts: list[str]) -> float | None:
    metric = next(m for m in faithfulness_and_answer_relevancy() if m.name == "faithfulness")
    sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=contexts)
    score = float(await metric.single_turn_ascore(sample))
    return None if math.isnan(score) else score


async def _example(chunk: dict, kind: str, lang: str, repo: DocumentRepository) -> dict:
    """One row: {"messages", "meta"}; meta["rejected"] says why when it can't be used."""
    meta: dict = {"kind": kind, "lang": lang, "source_chunk": chunk["id"], "rejected": None}
    question = await write_question(kind, chunk["content"], lang)
    state = initial_state(question, [])
    meta["question"] = question
    if state["lang"] != lang:
        return {"meta": {**meta, "rejected": f"question read as {state['lang']}"}}
    for node in (decompose, make_retriever(repo), grade):
        state.update(await node(state))  # type: ignore[typeddict-item]
    graded = state["graded_chunks"]
    meta.update(sub_questions=state["sub_questions"], graded=len(graded),
                source_graded=any(c.id == chunk["id"] for c in graded))
    if not graded:  # production would retry once, then clarify instead of answering
        return {"meta": {**meta, "rejected": "no passage passed the grader"}}
    prompt = ANSWER_PROMPT.format_messages(**answer_inputs(get_agent_config(), state))
    answer = _WIDE_MARKER.sub(r"[\1]", await write_answer(prompt))
    row = {"messages": [{"role": _ROLE[m.type], "content": m.content} for m in prompt] + [{"role": "assistant", "content": answer}], "meta": meta}
    meta["refusal"] = is_refusal(answer)
    if problem := answer_problem(answer, len(graded), lang):
        meta["rejected"] = problem
    elif not is_pure_refusal(answer):
        meta["faithfulness"] = await _faithfulness(question, answer, [c.content for c in graded])
        if (meta["faithfulness"] or 0) < _MIN_FAITHFULNESS:
            meta["rejected"] = f"faithfulness {meta['faithfulness']}"
    return row


def _done(out: Path) -> set[tuple[str, str]]:
    rows = [json.loads(line) for f in (out, out.with_suffix(".rejected.jsonl")) if f.exists()
            for line in f.read_text().splitlines()]
    return {(r["meta"]["source_chunk"], r["meta"]["kind"]) for r in rows
            if not (r["meta"]["rejected"] or "").startswith("error")}


async def main(lang: str, kinds: list[str], n_chunks: int | None, out: Path) -> None:
    golden = [r for path in _GOLDEN for r in load_golden(path)]
    golden_hashes, golden_questions = {h for r in golden for h in r.reference_context_hashes}, [g.question for g in golden]
    out.parent.mkdir(parents=True, exist_ok=True)
    done, sem = _done(out), asyncio.Semaphore(_CONCURRENCY)
    pool = await create_pool()
    try:
        repo = DocumentRepository(pool)
        sample = [dict(r) for r in await repo.sample_chunks(100_000 if n_chunks is None else n_chunks, lang)]
        jobs = [(c, k) for c in sample if content_hash(c["content"]) not in golden_hashes for k in kinds
                if (c["id"], k) not in done]
        counts = {"kept": 0, "rejected": 0}

        async def one(chunk: dict, kind: str) -> None:
            async with sem:
                try:
                    row = await _example(chunk, kind, lang, repo)
                except Exception as e:  # one failed call shouldn't cost the batch; a rerun retries it
                    log.exception("example %s/%s failed", chunk["id"], kind)
                    row = {"meta": {"kind": kind, "source_chunk": chunk["id"], "rejected": f"error: {e}"}}
            meta = row["meta"]
            if not meta["rejected"] and near_duplicate(meta["question"], golden_questions):
                meta["rejected"] = "near-duplicate of a golden question"  # would leak the eval set into training
            target = out.with_suffix(".rejected.jsonl") if meta["rejected"] else out
            with target.open("a") as f:  # written as it finishes: a crash loses at most the rows in flight
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["rejected" if meta["rejected"] else "kept"] += 1
            log.warning("%d/%d %s → %s", sum(counts.values()), len(jobs), kind, meta["rejected"] or "kept")

        await asyncio.gather(*(one(c, k) for c, k in jobs))
    finally:
        await pool.close()
    print(f"kept {counts['kept']}/{len(jobs)} → {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=sorted(LANG_NAME), default="et")
    parser.add_argument("--kinds", type=lambda s: s.split(","), default=["answerable", "detail", "near_miss"])
    parser.add_argument("--chunks", type=int, default=None, help="random chunks to use; default all in the language")
    parser.add_argument("--out", type=Path, default=Path("tuning/data/et.jsonl"))
    args = parser.parse_args()
    asyncio.run(main(args.lang, args.kinds, args.chunks, args.out))
