"""Cheap filters for a written target answer, run before the judge. Pure functions, no I/O."""
import re
from difflib import SequenceMatcher

from agent.lang import detect_lang
from tests.eval.refusal import is_refusal

_MARKER = re.compile(r"\[(\d+)\]")
# The answer prompt forbids "according to the context"-style phrases
_STYLE = re.compile(r"according to (the )?(context|source|information)|vastavalt (allika|kontekst|teab)"
                    r"|(allika|kontekst)\w* (põhjal|järgi)", re.IGNORECASE)


def answer_problem(answer: str, n_passages: int, lang: str) -> str | None:
    """Why this answer can't be a training target, or None when it can."""
    if not answer.strip():
        return "empty"
    cited = {int(n) for n in _MARKER.findall(answer)}
    if bad := sorted(n for n in cited if not 1 <= n <= n_passages):
        return f"cites {bad}, only {n_passages} passages"
    if m := _STYLE.search(answer):
        return f"style: {m.group(0)!r}"
    if detect_lang(answer) != lang:
        return f"language {detect_lang(answer)}"
    if not cited and not is_refusal(answer):
        return "answer without citations"
    return None


def is_pure_refusal(answer: str) -> bool:
    """A one-sentence "the sources don't cover this": nothing in it for the judge to check.
    A refusal followed by more text may go on from general knowledge, so it still gets judged."""
    return is_refusal(answer) and len(re.split(r"(?<=[.!?])\s+", answer.strip())) == 1


def near_duplicate(question: str, others: list[str], ratio: float = 0.8) -> bool:
    """Same wording up to small edits. Embeddings put any two Estonian questions at ~0.9 cosine, so text it is."""
    q = " ".join(question.lower().split())
    return any(SequenceMatcher(None, q, " ".join(o.lower().split())).ratio() >= ratio for o in others)
