from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.metrics.base import SingleTurnMetric

from evaluation.judge import get_judge_embeddings, get_judge_llm


def faithfulness_and_answer_relevancy() -> list[SingleTurnMetric]:
    """Generation metrics that need no reference answer — usable on live traffic."""
    llm = get_judge_llm()
    return [Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=get_judge_embeddings())]


def context_recall_and_precision() -> list[SingleTurnMetric]:
    """Context quality vs a reference answer — offline only."""
    llm = get_judge_llm()
    return [LLMContextRecall(llm=llm), LLMContextPrecisionWithReference(llm=llm)]


def factual_correctness() -> FactualCorrectness:
    """Claim overlap with the reference answer — offline only."""
    return FactualCorrectness(llm=get_judge_llm(), mode="recall")
