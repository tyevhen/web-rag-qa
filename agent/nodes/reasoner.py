import logging
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.config import TopicConfig
from agent.nodes._llm import invoke_chain
from agent.nodes.decomposer import _history_messages
from agent.state import AgentState

log = logging.getLogger(__name__)

# gpt-oss often cites as 【1】 or 【2†L9-L12】; the formatter, the UI and the eval read only [n]
_WIDE_MARKER = re.compile(r"【(\d+)[^】]*】")

_prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder("history"),
    ("human",
     "Answer the question based on the information below. "
     "Always respond in English, regardless of the language of the question or context. "
     "Respond directly and confidently — do not say 'according to the context', "
     "'the context states', or any similar phrase. "
     "After each statement, cite the numbered source it comes from, e.g. [1] or [2][3]. "
     "Only cite numbers that appear below. "
     "Base every statement on the information below. If the information is about a different topic "
     "than the question, say it does not cover the question instead of answering from general knowledge.\n\n"
     "Question: {question}\n\n"
     "Information:\n{context}"),
])


def make_reasoner(config: TopicConfig):
    async def reason(state: AgentState) -> dict[str, Any]:
        chunks = state["graded_chunks"]
        log.info("reason: building answer from %d graded chunks", len(chunks))
        for i, chunk in enumerate(chunks):
            log.info("reason: chunk [%d] score=%.4f source=%s | %.80r", i + 1, chunk.score, chunk.metadata.get("source", "?"), chunk.content)
        context = "\n\n---\n\n".join(f"[{i + 1}] {chunk.content}" for i, chunk in enumerate(chunks))
        answer = await invoke_chain(
            _prompt,
            system_prompt=config.system_prompt,
            history=_history_messages(state["history"]),
            question=state["question"],
            context=context,
        )
        answer = _WIDE_MARKER.sub(r"[\1]", answer.strip())
        log.info("reason: answer (%d chars): %.120r", len(answer), answer)
        return {"answer": answer}

    return reason
