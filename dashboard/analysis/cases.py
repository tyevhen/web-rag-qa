"""Find golden questions worth reading when comparing two eval runs. Pure functions over result-JSON question items."""
import re
from typing import Any

from tests.eval.refusal import is_refusal

_MARKER = re.compile(r"\[(\d+)\]")
_SCORE_GAP = 0.5
_OFF_CORPUS = "unanswerable"


def answered_off_corpus(item: dict[str, Any]) -> bool:
    return item["category"] == _OFF_CORPUS and item["route"] == "reason" and not is_refusal(item["answer"])


def citation_problem(item: dict[str, Any]) -> str | None:
    if item["route"] != "reason":
        return None
    cited = [int(n) for n in _MARKER.findall(item["answer"])]
    if not cited:
        return None if is_refusal(item["answer"]) else "no [n] citation"
    context = item.get("context")
    if isinstance(context, list) and max(cited) > len(context):
        return f"cites [{max(cited)}] but saw {len(context)} chunks"
    return None


def _reasons(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    why = []
    if a["route"] != b["route"]:
        why.append(f"route: A {a['route'] or 'error'}, B {b['route'] or 'error'}")
    for side, item in (("A", a), ("B", b)):
        if answered_off_corpus(item):
            why.append(f"{side} answered an off-corpus question without refusing")
        if item["category"] == "answerable" and item["route"] == "clarify":
            why.append(f"{side} refused an answerable question")
        if problem := citation_problem(item):
            why.append(f"{side}: {problem}")
    if (a["retrieved_ref_hits"] > 0) != (b["retrieved_ref_hits"] > 0):
        why.append(f"gold chunk retrieved only by {'A' if a['retrieved_ref_hits'] else 'B'}")
    sa, sb = a.get("scores") or {}, b.get("scores") or {}
    for metric in sa.keys() & sb.keys():
        if sa[metric] is not None and sb[metric] is not None and abs(sa[metric] - sb[metric]) >= _SCORE_GAP:
            why.append(f"{metric}: A {sa[metric]:.2f}, B {sb[metric]:.2f}")
    return why


def find_cases(a_items: list[dict[str, Any]], b_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per question both runs have and that has at least one reason to look at it."""
    b_by_question = {q["question"]: q for q in b_items}
    rows = []
    for a in a_items:
        b = b_by_question.get(a["question"])
        if b and (why := _reasons(a, b)):
            rows.append({"question": a["question"], "category": a["category"], "why": "; ".join(why)})
    return rows
