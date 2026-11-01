import logging
import math

import asyncpg
from ragas import SingleTurnSample

from db.eval_repository import EvalRepository
from db.query_log_repository import QueryLogRepository
from db.repository import DocumentRepository
from evaluation.alerts import ALERT_THRESHOLDS
from evaluation.judge import judge_model_name
from evaluation.metrics import reference_free_metrics

log = logging.getLogger(__name__)

_WINDOW_DAYS = 7


async def _score_sample(sample: SingleTurnSample) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for metric in reference_free_metrics():
        try:
            scores[metric.name] = float(await metric.single_turn_ascore(sample))
        except Exception:
            log.exception("online eval: metric %s failed", metric.name)
            scores[metric.name] = None
    return scores


async def score_row(documents: DocumentRepository, evals: EvalRepository, row: asyncpg.Record) -> dict[str, float]:
    """Judge one logged answer against the chunks it was given; saves and returns the scores that came back."""
    chunks = await documents.get_chunks_by_ids(row["graded_chunk_ids"])
    if not chunks:
        log.info("online eval: run %s skipped, its chunks were re-ingested", row["run_id"])
        return {}
    sample = SingleTurnSample(
        user_input=row["question"],
        response=row["answer"],
        retrieved_contexts=[c["content"] for c in chunks],
    )
    # A stored row marks the run as scored, so failed (None/NaN) metrics are left out to be retried next night
    scores = {k: v for k, v in (await _score_sample(sample)).items() if v is not None and not math.isnan(v)}
    if scores:
        await evals.save_eval_scores(row["run_id"], scores, judge_model_name())
    return scores


async def score_recent(
    documents: DocumentRepository, query_log: QueryLogRepository, evals: EvalRepository, limit: int
) -> int:
    """Score a random sample of unscored production answers with reference-free metrics."""
    rows = await query_log.fetch_unscored_queries(limit)
    scored = 0
    for row in rows:
        if await score_row(documents, evals, row):
            scored += 1

    means = await evals.recent_eval_means(_WINDOW_DAYS)
    for metric, threshold in ALERT_THRESHOLDS.items():
        mean = means.get(metric)
        if mean is not None and mean < threshold:
            log.warning("online eval: %s %d-day mean %.3f below %.2f", metric, _WINDOW_DAYS, mean, threshold)
    log.info("online eval: scored %d/%d sampled queries, means=%s", scored, len(rows), means)
    return scored
