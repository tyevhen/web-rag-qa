import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from agent.nodes._llm import cap_queries, invoke_lines
from agent.state import AgentState

log = logging.getLogger(__name__)

_prompt = ChatPromptTemplate.from_messages([
    ("human",
     "These search queries found too little. Rewrite each one in plain, everyday words: "
     "short, concrete, using common synonyms for the key terms and no jargon. "
     "Keep each query in its own language; never translate it. "
     "Return only the rewritten queries, one per line, same count.\n\n"
     "Queries:\n{queries}"),
])


def make_augmenter():
    async def augment(state: AgentState) -> dict[str, Any]:
        original = state["sub_questions"] or [state["question"]]
        log.info("augment: rewriting %d sub_questions with domain vocabulary", len(original))
        rewritten = await invoke_lines(_prompt, max_tokens=200, queries="\n".join(original))
        if len(rewritten) != len(original):
            log.warning("augment: expected %d queries, got %d — falling back to originals", len(original), len(rewritten))
            rewritten = original
        # Count checked before capping, so extra lines still count as a mismatch; combined stays <= 2 x 3 queries
        rewritten = cap_queries(rewritten)
        # Keep the originals: a rewrite widens the search, it should not replace what already matched
        combined = list(dict.fromkeys([*original, *rewritten]))
        log.info("augment: rewritten sub_questions: %s", rewritten)
        return {"sub_questions": combined}

    return augment
