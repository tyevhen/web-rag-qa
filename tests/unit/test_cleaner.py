from ingestion.cleaner import strip_layout
from ingestion.scraper import RawChunk

_TEMPLATE = "Log in to your personal account\n## Background\n* [ About ](https://x/about)\n[Telegram](https://t.me/share)"


def _site(pages: int) -> list[RawChunk]:
    return [RawChunk(content=f"{_TEMPLATE}\nApply the tourniquet high and tight on limb {i}.", metadata={"url": f"https://x/{i}"})
            for i in range(pages)]


def test_navigation_and_template_lines_are_dropped_content_and_headings_kept():
    cleaned = strip_layout(_site(10))

    assert cleaned[3].content == "## Background\nApply the tourniquet high and tight on limb 3."
    assert cleaned[3].metadata == {"url": "https://x/3"}


def test_small_crawls_keep_repeated_lines():
    # Too few pages to tell template text from content; only link-only lines go
    assert strip_layout(_site(2))[0].content.splitlines()[0] == "Log in to your personal account"


def test_link_only_heading_dropped_inline_links_kept_and_empty_pages_removed():
    pages = [
        RawChunk(content="# [See other formats](https://archive.org/details/x)\nSee [the guide](https://g) for doses.", metadata={}),
        RawChunk(content="1. [ Home ](https://x)\n![logo](https://x/logo.png)\n[ ![Card \\(CUF\\)](https://x/c.jpg) ](https://x/c)",
                 metadata={}),
    ]

    assert [p.content for p in strip_layout(pages)] == ["See [the guide](https://g) for doses."]
