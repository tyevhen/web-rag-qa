"""Refusal detection shared by the eval metrics and the dashboard. Dependency-free (no torch)."""
import re

# shortcut: phrase list fitted to gpt-oss and qwen2.5 refusals; a model that refuses differently reads as "answered"
_REFUSAL = re.compile(
    r"(does not|doesn['’]t|do not|don['’]t) (include|contain|cover|provide|have|mention)|no information"
    r"|not (?:directly |explicitly )?(covered|mentioned|provided)|can(?:not|['’]t) (find|answer|provide)"
    # Estonian: "does not contain/mention", "no information", "could not find", "cannot answer"
    r"|ei (sisalda|maini|käsitle|kirjelda)|(teave|info|informatsioon)[^.]{0,30} puudub|puudub (teave|info)"
    r"|ei ole (mainitud|kirjeldatud|käsitletud)|ei leidnud|ei (saa|oska) vastata",
    re.IGNORECASE,
)


# Only the opening sentence: answers from general knowledge often end with "...not covered in the documents, but..."
_FIRST_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(_FIRST_SENTENCE.split(answer.strip(), maxsplit=1)[0]))
