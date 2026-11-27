"""The writer model and its prompts: it writes the questions and the target answers."""
from functools import lru_cache

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from settings import get_settings

WRITER_MODEL = "mistralai/mistral-large-2512"
WRITER_HOST = "mistral/eu"  # Mistral's own API: the customer owns the output, no ban on training other models
LANG_NAME = {"et": "Estonian", "en": "English"}
# One chunk can feed several kinds; each asks from a different angle so its questions don't repeat
ASK = {
    "answerable": "Write one practical question that a member of the public would ask and that the passage below answers.",
    "detail": "Write one precise question about a number, condition, exception or order of steps that the passage below answers.",
    "near_miss": "Write one question on the same topic as the passage below that the passage does NOT answer: "
                 "a detail, step, number or case a reader would plausibly ask next.",
}
_ASK_RULES = ("Write it in {lang}, as a person would type it. It must stand alone: never mention the passage, "
              "the text, a figure, a table or an earlier conversation. Return only the question.\n\nPassage:\n{passage}")
# Only the writer sees this; the saved example keeps the production prompt, so the tuned model learns
# the stricter behaviour from the plain prompt it will get in production
_STRICT = ("\n\nUse only the information inside <documents>, never your own knowledge, even for well-known facts. "
           "If it does not answer the question, say so in one sentence and stop; if it answers only part, "
           "answer that part and say in one sentence what it does not cover.")


@lru_cache(maxsize=2)
def _writer(temperature: float) -> ChatOpenAI:
    s = get_settings()
    if s.eval_judge_base_url is None or s.eval_judge_api_key is None:
        raise ValueError("The writer uses the judge's OpenRouter endpoint: set EVAL_JUDGE_BASE_URL and EVAL_JUDGE_API_KEY")
    return ChatOpenAI(model=WRITER_MODEL, api_key=s.eval_judge_api_key, base_url=s.eval_judge_base_url,
                      temperature=temperature, max_completion_tokens=1024,
                      max_retries=8,  # Mistral's shared pool on OpenRouter answers 429 in bursts
                      extra_body={"provider": {"order": [WRITER_HOST], "allow_fallbacks": False}})


async def write_question(kind: str, passage: str, lang: str) -> str:
    ask = f"{ASK[kind]} {_ASK_RULES.format(lang=LANG_NAME[lang], passage=passage)}"
    return str((await _writer(0.7).ainvoke(ask)).content).strip().strip('"„“')


async def write_answer(prompt: list[BaseMessage]) -> str:
    strict = [*prompt[:-1], HumanMessage(content=f"{prompt[-1].content}{_STRICT}")]
    return str((await _writer(0).ainvoke(strict)).content).strip()
