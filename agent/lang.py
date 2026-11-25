from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # langdetect is random by default; the same text must always get the same language


def detect_lang(text: str) -> str | None:
    """ISO 639-1 code, or None when undetectable. Finnish counts as Estonian: short Estonian text is often read as it."""
    try:
        lang = detect(text)
    except LangDetectException:
        return None
    return "et" if lang == "fi" else lang


def question_lang(question: str) -> str:
    """The corpus language a question searches: "et" or "en" (any other language reads English sources)."""
    # shortcut: question-only detection; a short or mixed question may be misrouted, upgrade to history-aware detection if logs show it
    return "et" if detect_lang(question) == "et" else "en"
