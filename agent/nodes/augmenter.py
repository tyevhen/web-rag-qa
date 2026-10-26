import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.nodes._llm import get_groq_llm
from agent.state import AgentState

log = logging.getLogger(__name__)

_prompt = ChatPromptTemplate.from_messages([
    ("human",
     "These search queries found too little. Rewrite each one in plain, everyday words: "
     "short, concrete, using common synonyms for the key terms and no jargon. "
     "Return only the rewritten queries, one per line, same count.\n\n"
     "Queries:\n{queries}"),
])


def make_augmenter():
    chain = _prompt | get_groq_llm() | StrOutputParser()

    async def augment(state: AgentState) -> dict[str, Any]:
        original = state["sub_questions"] or [state["question"]]
        log.info("augment: rewriting %d sub_questions with domain vocabulary", len(original))
        content = await chain.ainvoke({"queries": "\n".join(original)})
        rewritten = [line.strip() for line in content.strip().splitlines() if line.strip()]
        if len(rewritten) != len(original):
            log.warning("augment: expected %d queries, got %d — falling back to originals", len(original), len(rewritten))
            rewritten = original
        # Keep the originals: a rewrite widens the search, it should not replace what already matched
        combined = list(dict.fromkeys([*original, *rewritten]))
        log.info("augment: rewritten sub_questions: %s", rewritten)
        return {"sub_questions": combined}

    return augment
