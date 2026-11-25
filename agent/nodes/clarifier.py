import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.config import TopicConfig
from agent.nodes._llm import invoke_lines
from agent.nodes.decomposer import _history_messages
from agent.nodes.screen import check, refuse
from agent.state import AgentState
from db.repositories.documents import DocumentRepository

log = logging.getLogger(__name__)

# Keyed by AgentState.lang (agent/lang.py: "en" or "et")
_NOT_FOUND = {
    "en": "I wasn't able to find a clear answer. Could you clarify what you mean?",
    "et": "Ma ei leidnud selget vastust. Kas saaksid täpsustada, mida silmas pead?",
}

_prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder("history"),
    ("human",
     "I could not find enough information to answer this question:\n\n"
     "{question}\n\n"
     "Generate 2-3 short first-person statements that represent different things the user might mean. "
     "Each statement should be in the first person (start with 'I' in English) and be something the user could select to clarify their intent "
     "(e.g. 'I want to know about evacuation procedures during a flood.'). "
     "The ingested sources are: {sources}. Each statement must ask about something those sources could answer, "
     "and must not restate the user's goal or request. "
     "Write them in the same language as the question. "
     "Return only the statements, one per line, no numbering or bullets."),
])


def make_clarifier(config: TopicConfig, repo: DocumentRepository):
    async def clarify(state: AgentState) -> dict[str, Any]:
        log.info("clarify: corpus had no relevant content for question=%r, generating clarifications", state["question"])
        options = await invoke_lines(
            _prompt,
            max_tokens=150,
            system_prompt=config.system_prompt,
            history=_history_messages(state["history"]),
            question=state["question"],
            # shortcut: one query per clarify call; cache it if clarify traffic grows
            sources=", ".join(row["filename"] for row in await repo.list_documents()),
        )
        log.info("clarify: generated %d options: %s", len(options), options)
        if options and (category := await check("\n".join(options))):
            return refuse(f"output check on clarify: REFUSE {category}", state["lang"])
        return {
            "answer": _NOT_FOUND[state["lang"]],
            "clarifying_questions": options,
        }

    return clarify
