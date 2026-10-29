from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.metrics.base import SingleTurnMetric

from evaluation.judge import get_judge_embeddings, get_judge_llm


def reference_free_metrics() -> list[SingleTurnMetric]:
    """Generation metrics that need no reference answer — usable on live traffic."""
    llm = get_judge_llm()
    return [Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=get_judge_embeddings())]


def retrieval_metrics() -> list[SingleTurnMetric]:
    """Context quality vs a reference answer — offline only."""
    llm = get_judge_llm()
    return [LLMContextRecall(llm=llm), LLMContextPrecisionWithReference(llm=llm)]


def correctness_metrics() -> list[SingleTurnMetric]:
    return [FactualCorrectness(llm=get_judge_llm(), mode="recall")]
