from typing import cast

import asyncpg

from evaluation import online
from db.eval_repository import EvalRepository
from db.query_log_repository import QueryLogRepository
from db.repository import DocumentRepository


class Documents:
    async def get_chunks_by_ids(self, ids: list[str]) -> list[dict]:
        return [{"content": "ctx"}]


class QueryLog:
    async def fetch_unscored_queries(self, limit: int) -> list[dict]:
        return [{"run_id": r, "question": "q", "answer": "a", "graded_chunk_ids": ["c"]} for r in ("ok", "failed")]


class Evals:
    def __init__(self) -> None:
        self.saved: dict[str, dict[str, float]] = {}

    async def save_eval_scores(self, run_id: str, scores: dict[str, float], judge_model: str) -> None:
        self.saved[run_id] = scores

    async def recent_eval_means(self, days: int) -> dict[str, float]:
        return {}


async def test_failed_judge_scores_are_not_stored(monkeypatch):
    results = iter([{"faithfulness": 0.9, "answer_relevancy": float("nan")}, {"faithfulness": None, "answer_relevancy": float("nan")}])

    async def fake_score(sample):
        return next(results)

    monkeypatch.setattr(online, "_score_sample", fake_score)
    monkeypatch.setattr(online, "judge_model_name", lambda: "judge")
    evals = Evals()

    scored = await online.score_recent(cast(DocumentRepository, Documents()), cast(QueryLogRepository, QueryLog()),
                                       cast(EvalRepository, evals), 2)
    assert scored == 1
    assert evals.saved == {"ok": {"faithfulness": 0.9}}  # "failed" has no row, so it is picked again next night


async def test_score_row_saves_and_returns_only_real_scores(monkeypatch):
    async def fake_score(sample):
        return {"faithfulness": 0.8, "answer_relevancy": float("nan")}

    monkeypatch.setattr(online, "_score_sample", fake_score)
    monkeypatch.setattr(online, "judge_model_name", lambda: "judge")
    evals = Evals()
    row = {"run_id": "r1", "question": "q", "answer": "a", "graded_chunk_ids": ["c"]}

    assert await online.score_row(cast(DocumentRepository, Documents()), cast(EvalRepository, evals), cast(asyncpg.Record, row)) == {"faithfulness": 0.8}
    assert evals.saved == {"r1": {"faithfulness": 0.8}}
