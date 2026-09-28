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
     "I could not find enough information to answer this question:\n\n"
     "{question}\n\n"
     "Generate 2-3 short first-person statements that represent different things the user might mean. "
     "Each statement should start with 'I' and be something the user could select to clarify their intent "
     "(e.g. 'I want to know about evacuation procedures during a flood.'). "
     "Return only the statements, one per line, no numbering or bullets."),
])


def make_clarifier(config: TopicConfig):
    async def clarify(state: AgentState) -> dict[str, Any]:
        log.info("clarify: corpus had no relevant content for question=%r, generating clarifications", state["question"])
        content = await invoke_chain(
            _prompt,
            system_prompt=config.system_prompt,
            history=_history_messages(state["history"]),
            question=state["question"],
        )
        options = [line.strip() for line in content.strip().splitlines() if line.strip()]
        log.info("clarify: generated %d options: %s", len(options), options)
        return {
            "answer": "I wasn't able to find a clear answer. Could you clarify what you mean?",
            "clarifying_questions": options,
        }

    return clarify
