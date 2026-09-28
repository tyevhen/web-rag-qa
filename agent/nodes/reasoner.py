import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.config import TopicConfig
from agent.nodes._llm import invoke_chain
from agent.nodes.decomposer import _history_messages
from agent.state import AgentState

log = logging.getLogger(__name__)

_prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder("history"),
    ("human",
     "Answer the question based on the information below. "
     "Always respond in English, regardless of the language of the question or context. "
     "Respond directly and confidently — do not say 'according to the context', "
     "'the context states', or any similar phrase.\n\n"
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
        answer = answer.strip()
        log.info("reason: answer (%d chars): %.120r", len(answer), answer)
        return {"answer": answer}

    return reason
