"""
Web search tool (DuckDuckGo, no API key).

Design reasoning
----------------
The agentic router can decide a question needs *fresh public* information the
private corpus cannot contain (news, prices, "latest ..."). This tool provides
that escape hatch. Results are normalized to the same evidence shape as RAG so
the generation layer and citations are uniform regardless of source.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str


def web_search(query: str, max_results: int = 5) -> list[WebResult]:
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        return [
            WebResult(
                title=h.get("title", ""),
                url=h.get("href", "") or h.get("url", ""),
                snippet=h.get("body", "") or h.get("snippet", ""),
            )
            for h in hits
        ]
    except Exception as exc:  # noqa: BLE001
        log.warning("Web search failed: %s", exc)
        return []
