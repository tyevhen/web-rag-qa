import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.config import TopicConfig
from agent.nodes._llm import invoke_chain
from agent.state import AgentState

log = logging.getLogger(__name__)

_prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder("history"),
    ("human",
     "Break the following question into 2-3 specific sub-questions that "
     "together fully answer it. Return only the sub-questions, one per line.\n\n"
     "Question: {question}"),
])


def _history_messages(history: list[dict]) -> list[BaseMessage]:
    role_map = {"user": HumanMessage, "assistant": AIMessage}
    return [role_map[t["role"]](content=t["content"]) for t in history if t["role"] in role_map]


def make_decomposer(config: TopicConfig):
    async def decompose(state: AgentState) -> dict[str, Any]:
        log.info("decompose: question=%r history_turns=%d", state["question"], len(state["history"]))
        content = await invoke_chain(
            _prompt,
            system_prompt=config.system_prompt,
            history=_history_messages(state["history"]),
            question=state["question"],
        )
        lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
        sub_questions = lines or [state["question"]]
        log.info("decompose: %d sub_questions: %s", len(sub_questions), sub_questions)
        return {"sub_questions": sub_questions}

    return decompose
