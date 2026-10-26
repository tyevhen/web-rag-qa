"""LLM-judged metrics via Ragas. Costs judge API calls — skipped with --skip-llm."""
import asyncio
import math
from collections.abc import Sequence
from typing import Any

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.dataset_schema import EvaluationResult, SingleTurnSampleOrMultiTurnSample
from ragas.metrics import LLMContextRecall
from ragas.metrics.base import Metric

from evaluation.judge import RUN_CONFIG, get_judge_llm
from evaluation.metrics import correctness_metrics, reference_free_metrics, retrieval_metrics
from tests.eval.runner import RunResult

Scores = dict[str, list[float]]


def _evaluate(samples: Sequence[SingleTurnSample], metrics: Sequence[Metric]) -> Scores:
    if not samples:
        return {m.name: [] for m in metrics}
    dataset = EvaluationDataset(samples=list[SingleTurnSampleOrMultiTurnSample](samples))
    result = evaluate(dataset, metrics=metrics, run_config=RUN_CONFIG, show_progress=True)
    if not isinstance(result, EvaluationResult):
        raise TypeError(f"ragas.evaluate returned {type(result).__name__}, expected EvaluationResult")
    df = result.to_pandas()
    # Some metrics suffix their column with parameters, e.g. "factual_correctness(mode=f1)"
    columns = {m.name: next(c for c in df.columns if c == m.name or c.startswith(f"{m.name}(")) for m in metrics}
    return {name: [float(v) for v in df[col]] for name, col in columns.items()}


def _sample(r: RunResult, contexts: list[str], with_response: bool = False) -> SingleTurnSample:
    return SingleTurnSample(
        user_input=r.row.question,
        retrieved_contexts=contexts,
        reference=r.row.reference or None,
        response=r.answer if with_response else None,
    )


def _run_all(results: list[RunResult]) -> dict[str, Scores]:
    ok = [r for r in results if not r.error]
    with_ref = [r for r in ok if r.row.reference and r.retrieved]
    answered = [r for r in ok if r.route == "reason" and r.graded]
    return {
        "retrieval": _evaluate([_sample(r, [c.content for c in r.retrieved]) for r in with_ref], retrieval_metrics()),
        # Same recall metric on the post-grader set; the difference = evidence the grader threw away
        "after_grading": _evaluate(
            [_sample(r, [c.content for c in r.graded]) for r in with_ref if r.graded],
            [LLMContextRecall(llm=get_judge_llm())],
        ),
        "generation": _evaluate(
            [_sample(r, [c.content for c in r.graded], with_response=True) for r in answered],
            reference_free_metrics(),
        ),
        "correctness": _evaluate(
            [_sample(r, [c.content for c in r.graded], with_response=True) for r in answered if r.row.reference],
            correctness_metrics(),
        ),
    }


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if not math.isnan(v)]
    return round(sum(clean) / len(clean), 3) if clean else None


def summarize(raw: dict[str, Scores], results: list[RunResult]) -> dict[str, Any]:
    with_ref = [r for r in results if not r.error and r.row.reference and r.retrieved]
    graded_empty = sum(1 for r in with_ref if not r.graded)
    after = raw["after_grading"]["context_recall"] + [0.0] * graded_empty  # nothing kept → nothing recalled
    before_recall = _mean(raw["retrieval"]["context_recall"])
    after_recall = _mean(after)
    all_scores = [v for group in raw.values() for values in group.values() for v in values]
    return {
        "retrieval": {name: _mean(v) for name, v in raw["retrieval"].items()},
        "grader": {
            "context_recall_after_grading": after_recall,
            "recall_loss": round(before_recall - after_recall, 3)
            if before_recall is not None and after_recall is not None else None,
        },
        "generation": {name: _mean(v) for group in ("generation", "correctness") for name, v in raw[group].items()},
        "judge": {"nan_rate": round(sum(math.isnan(v) for v in all_scores) / len(all_scores), 3) if all_scores else None},
    }


async def run(results: list[RunResult]) -> dict[str, Any]:
    # ragas.evaluate drives its own event loop — keep it off ours
    raw = await asyncio.to_thread(_run_all, results)
    return summarize(raw, results)
