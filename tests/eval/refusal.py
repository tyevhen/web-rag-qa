"""Refusal detection shared by the eval metrics and the dashboard. Dependency-free (no torch)."""
import re

# shortcut: phrase list fitted to gpt-oss and qwen2.5 refusals; a model that refuses differently reads as "answered"
_REFUSAL = re.compile(
    r"(does not|doesn['’]t|do not|don['’]t) (include|contain|cover|provide|have|mention)|no information"
    r"|not (?:directly |explicitly )?(covered|mentioned|provided)|can(?:not|['’]t) (find|answer|provide)",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(answer))
