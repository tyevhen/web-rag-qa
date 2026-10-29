"""Free, exact metrics — no LLM calls. Safe to run in CI on every change."""
import re
import statistics

from tests.eval.dataset import content_hash
from tests.eval.refusal import is_refusal
from tests.eval.runner import RunResult

_MARKER = re.compile(r"\[(\d+)\]")
_THRESHOLD_SWEEP = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def _ranking(ranked: list[str], relevant: set[str]) -> tuple[float, float, float]:
    """Returns (hit, recall, reciprocal rank) of relevant hashes in a ranked list."""
    found = [h for h in relevant if h in ranked]
    first = next((i for i, h in enumerate(ranked, 1) if h in relevant), None)
    return float(bool(found)), len(found) / len(relevant), 1.0 / first if first else 0.0


def _with_refs(results: list[RunResult]) -> list[RunResult]:
    return [r for r in results if not r.error and r.row.reference_context_hashes]


def retrieval(results: list[RunResult]) -> dict[str, dict[str, float | None]]:
    scored = _with_refs(results)
    variants: dict[str, list[tuple[float, float, float]]] = {"hybrid_decomposed": []}
    for r in scored:
        relevant = set(r.row.reference_context_hashes)
        variants["hybrid_decomposed"].append(_ranking([content_hash(c.content) for c in r.retrieved], relevant))
        for name, ranked in r.ablation.items():
            variants.setdefault(name, []).append(_ranking(ranked, relevant))
    return {
        name: {
            "hit@6": _mean([v[0] for v in vals]),
            "recall@6": _mean([v[1] for v in vals]),
            "mrr": _mean([v[2] for v in vals]),
        }
        for name, vals in variants.items()
    }


def grader(results: list[RunResult]) -> dict[str, dict[str, float | None]]:
    """Cross-encoder as a binary classifier; reference chunks are the positive labels."""
    labelled = [
        (score, content_hash(c.content) in set(r.row.reference_context_hashes))
        for r in _with_refs(results)
        for c, score in zip(r.retrieved, r.retrieved_ce_scores)
    ]
    sweep: dict[str, dict[str, float | None]] = {}
    for t in _THRESHOLD_SWEEP:
        tp = sum(1 for s, pos in labelled if s >= t and pos)
        fp = sum(1 for s, pos in labelled if s >= t and not pos)
        fn = sum(1 for s, pos in labelled if s < t and pos)
        sweep[f"threshold={t:+.1f}"] = {"precision": _rate(tp, tp + fp), "recall": _rate(tp, tp + fn)}
    return sweep


def citations(results: list[RunResult]) -> dict[str, float | None]:
    answered = [r for r in results if not r.error and r.route == "reason"]
    markers = [(int(m), len(r.graded)) for r in answered for m in _MARKER.findall(r.answer)]
    return {
        # Measured on the raw reasoner output — the formatter's top-3 fallback would mask this
        "answers_with_marker": _rate(sum(1 for r in answered if _MARKER.search(r.answer)), len(answered)),
        "valid_markers": _rate(sum(1 for n, k in markers if 1 <= n <= k), len(markers)),
    }


def routing(results: list[RunResult]) -> dict[str, float | None]:
    ok = [r for r in results if not r.error]
    answerable = [r for r in ok if r.row.category != "unanswerable"]
    unanswerable = [r for r in ok if r.row.category == "unanswerable"]
    retried = [r for r in ok if r.retry_count > 0]
    return {
        "false_refusal_rate": _rate(sum(1 for r in answerable if r.route == "clarify"), len(answerable)),
        "false_answer_rate": _rate(sum(1 for r in unanswerable if r.route == "reason"), len(unanswerable)),
        # Route alone can't tell "the guide doesn't cover this" from a made-up answer; these read the text
        "off_corpus_answered_rate": _rate(
            sum(1 for r in unanswerable if r.route == "reason" and not is_refusal(r.answer)), len(unanswerable)),
        "answerable_refused_in_text_rate": _rate(
            sum(1 for r in answerable if r.route == "reason" and is_refusal(r.answer)), len(answerable)),
        "retry_rate": _rate(len(retried), len(ok)),
        "retry_success_rate": _rate(sum(1 for r in retried if r.route == "reason"), len(retried)),
    }


def operational(results: list[RunResult]) -> dict[str, float | None]:
    latencies = sorted(r.latency_s for r in results if not r.error)
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None
    return {
        "latency_p50_s": statistics.median(latencies) if latencies else None,
        "latency_p95_s": p95,
        "error_rate": _rate(sum(1 for r in results if r.error), len(results)),
    }
