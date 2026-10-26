from typing import cast

from evaluation import online
from db.repository import DocumentRepository


class Repo:
    def __init__(self) -> None:
        self.saved: dict[str, dict[str, float]] = {}

    async def fetch_unscored_queries(self, limit: int) -> list[dict]:
        return [{"run_id": r, "question": "q", "answer": "a", "graded_chunk_ids": ["c"]} for r in ("ok", "failed")]

    async def get_chunks_by_ids(self, ids: list[str]) -> list[dict]:
        return [{"content": "ctx"}]

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
    repo = Repo()

    assert await online.score_recent(cast(DocumentRepository, repo), 2) == 1
    assert repo.saved == {"ok": {"faithfulness": 0.9}}  # "failed" has no row, so it is picked again next night
