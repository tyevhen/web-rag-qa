from __future__ import annotations

from dataclasses import dataclass

from settings import get_settings


@dataclass
class TopicConfig:
    system_prompt: str


# Not part of the persona: the corpus is bilingual (agent/lang.py), so this holds for every topic
_ANSWER_LANGUAGE = "Answer in the same language as the user's question."


def get_agent_config() -> TopicConfig:
    return TopicConfig(system_prompt=f"{get_settings().system_prompt}\n\n{_ANSWER_LANGUAGE}")
