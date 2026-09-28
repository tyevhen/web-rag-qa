from __future__ import annotations

from dataclasses import dataclass

from settings import get_settings


@dataclass
class TopicConfig:
    system_prompt: str


def get_agent_config() -> TopicConfig:
    return TopicConfig(system_prompt=get_settings().system_prompt)
