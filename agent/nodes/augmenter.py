import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.nodes._llm import get_groq_llm
from agent.state import AgentState

log = logging.getLogger(__name__)

_prompt = ChatPromptTemplate.from_messages([
    ("human",
     "Rewrite these search queries to use broader, more technical vocabulary suitable for "
     "searching a medical and emergency-preparedness document database. "
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
        log.info("augment: rewritten sub_questions: %s", rewritten)
        return {"sub_questions": rewritten}

    return augment
