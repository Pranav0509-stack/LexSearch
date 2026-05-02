"""
Sanhita — Web Signal Retrieval for Indian Law.

Fetches live legal news, recent judgments, and legislative updates from
trusted Indian legal sources. These signals augment the 31M corpus with
current developments that haven't been ingested yet.

Sources (all free, no API key required):
  1. Indian Kanoon — latest judgments (RSS feed)
  2. Bar & Bench — legal news
  3. LiveLaw — legal news & analysis
  4. SCC Online Blog — case analysis
  5. Supreme Court Observer — SCI analysis
  6. PRS Legislative Research — bill tracker

Results are cached for 30 minutes to avoid hammering sources.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

# Use certifi if available
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl.create_default_context()

_UA = "Sanhita/1.0 (+https://sanhita.law)"
_TIMEOUT = 8  # seconds per request


# ── Simple HTML text extractor ────────────────────────────────────────────
class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._text = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._text.append(data.strip())

    def get_text(self) -> str:
        return " ".join(t for t in self._text if t)


def _html_to_text(html: str) -> str:
    ex = _TextExtractor()
    ex.feed(html)
    return ex.get_text()


# ── RSS/Atom feed parser (lightweight, no lxml dependency) ────────────────
def _parse_rss_items(xml: str, max_items: int = 10) -> list[dict[str, str]]:
    """Extract items from RSS/Atom XML without external XML libs."""
    items = []
    # Try RSS <item> blocks
    item_re = re.compile(r"<item[^>]*>(.*?)</item>", re.DOTALL | re.IGNORECASE)
    title_re = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
    link_re = re.compile(r"<link[^>]*>(.*?)</link>", re.DOTALL | re.IGNORECASE)
    desc_re = re.compile(r"<description[^>]*>(.*?)</description>", re.DOTALL | re.IGNORECASE)
    date_re = re.compile(r"<pubDate[^>]*>(.*?)</pubDate>", re.DOTALL | re.IGNORECASE)

    matches = item_re.findall(xml)
    if not matches:
        # Try Atom <entry> blocks
        item_re = re.compile(r"<entry[^>]*>(.*?)</entry>", re.DOTALL | re.IGNORECASE)
        matches = item_re.findall(xml)
        link_re = re.compile(r'<link[^>]*href="([^"]*)"', re.IGNORECASE)
        date_re = re.compile(r"<(?:published|updated)[^>]*>(.*?)</(?:published|updated)>", re.DOTALL | re.IGNORECASE)
        desc_re = re.compile(r"<(?:summary|content)[^>]*>(.*?)</(?:summary|content)>", re.DOTALL | re.IGNORECASE)

    for block in matches[:max_items]:
        title_m = title_re.search(block)
        link_m = link_re.search(block)
        desc_m = desc_re.search(block)
        date_m = date_re.search(block)

        title = _html_to_text(title_m.group(1)) if title_m else ""
        link = link_m.group(1).strip() if link_m else ""
        desc = _html_to_text(desc_m.group(1))[:400] if desc_m else ""
        date = date_m.group(1).strip() if date_m else ""

        # Clean CDATA
        title = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", title)
        desc = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", desc)
        link = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", link)

        if title:
            items.append({
                "title": title.strip(),
                "url": link.strip(),
                "excerpt": desc.strip(),
                "date": date.strip(),
            })

    return items


def _fetch_url(url: str) -> Optional[str]:
    """Fetch a URL with timeout, return text or None."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT, context=_SSL_CTX) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.debug("web_signals: failed to fetch %s: %s", url, e)
        return None


# ── Source definitions ────────────────────────────────────────────────────

SOURCES = {
    "indiankanoon": {
        "name": "Indian Kanoon",
        "type": "search",
        "url_template": "https://indiankanoon.org/search/?formInput={query}&pagenum=0",
        "description": "India's largest case law database",
    },
    "barandbench": {
        "name": "Bar & Bench",
        "type": "rss",
        "url": "https://www.barandbench.com/feed",
        "description": "Legal news and analysis",
    },
    "livelaw": {
        "name": "LiveLaw",
        "type": "rss",
        "url": "https://www.livelaw.in/feed",
        "description": "Legal news, judgments, and analysis",
    },
    "scobserver": {
        "name": "Supreme Court Observer",
        "type": "rss",
        "url": "https://www.scobserver.in/feed/",
        "description": "Supreme Court case tracking and analysis",
    },
    "prsindia": {
        "name": "PRS Legislative Research",
        "type": "rss",
        "url": "https://prsindia.org/billtrack/rss",
        "description": "Bill tracker — Parliament legislative updates",
    },
}


# ── Cache ─────────────────────────────────────────────────────────────────
_CACHE: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL = 1800  # 30 minutes


def _get_cached(key: str) -> Optional[list[dict]]:
    if key in _CACHE:
        ts, data = _CACHE[key]
        if time.monotonic() - ts < _CACHE_TTL:
            return data
        del _CACHE[key]
    return None


def _set_cache(key: str, data: list[dict]) -> None:
    _CACHE[key] = (time.monotonic(), data)


# ── Public API ────────────────────────────────────────────────────────────

@dataclass
class WebSignal:
    title: str
    url: str
    source: str
    source_name: str
    excerpt: str = ""
    date: str = ""
    relevance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "source_name": self.source_name,
            "excerpt": self.excerpt,
            "date": self.date,
            "relevance": round(self.relevance, 2),
        }


def _score_relevance(query: str, title: str, excerpt: str) -> float:
    """Simple keyword-overlap relevance score (0..1)."""
    query_terms = set(re.findall(r"\w+", query.lower()))
    text = f"{title} {excerpt}".lower()
    text_terms = set(re.findall(r"\w+", text))
    if not query_terms:
        return 0.0
    overlap = query_terms & text_terms
    return len(overlap) / len(query_terms)


def fetch_legal_news(max_items: int = 15) -> list[WebSignal]:
    """Fetch latest legal news from RSS sources. Cached 30 min."""
    cache_key = "legal_news_all"
    cached = _get_cached(cache_key)
    if cached is not None:
        return [WebSignal(**s) for s in cached]

    signals: list[WebSignal] = []

    for src_id, src in SOURCES.items():
        if src["type"] != "rss":
            continue
        url = src["url"]
        xml = _fetch_url(url)
        if not xml:
            continue
        items = _parse_rss_items(xml, max_items=8)
        for item in items:
            signals.append(WebSignal(
                title=item["title"],
                url=item["url"],
                source=src_id,
                source_name=src["name"],
                excerpt=item["excerpt"][:300],
                date=item["date"],
            ))

    # Sort by date (newest first, rough sort since formats vary)
    signals.sort(key=lambda s: s.date, reverse=True)
    signals = signals[:max_items]

    _set_cache(cache_key, [s.to_dict() for s in signals])
    return signals


def search_web_signals(query: str, max_items: int = 8) -> list[WebSignal]:
    """
    Search for web signals relevant to a legal query.
    Fetches from RSS feeds and filters by relevance to the query.
    """
    cache_key = f"web_search:{query[:100]}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return [WebSignal(**s) for s in cached]

    signals: list[WebSignal] = []

    # Fetch from all RSS sources
    for src_id, src in SOURCES.items():
        if src["type"] != "rss":
            continue
        xml = _fetch_url(src["url"])
        if not xml:
            continue
        items = _parse_rss_items(xml, max_items=15)
        for item in items:
            relevance = _score_relevance(query, item["title"], item["excerpt"])
            if relevance >= 0.15:  # at least some overlap
                signals.append(WebSignal(
                    title=item["title"],
                    url=item["url"],
                    source=src_id,
                    source_name=src["name"],
                    excerpt=item["excerpt"][:300],
                    date=item["date"],
                    relevance=relevance,
                ))

    # Sort by relevance
    signals.sort(key=lambda s: s.relevance, reverse=True)
    signals = signals[:max_items]

    _set_cache(cache_key, [s.to_dict() for s in signals])
    return signals


def get_web_context_for_brief(query: str, max_signals: int = 4) -> str:
    """
    Build a context block from web signals for the Brief assistant.
    This gets appended to the retrieved-cases context so the LLM can
    reference current legal developments.
    """
    signals = search_web_signals(query, max_items=max_signals)
    if not signals:
        return ""

    lines = ["\n--- Recent Legal Developments (from trusted Indian legal news sources) ---\n"]
    for i, s in enumerate(signals, 1):
        lines.append(
            f"[NEWS-{i}] {s.title}\n"
            f"    Source: {s.source_name} | {s.date}\n"
            f"    {s.excerpt[:250]}"
        )
    lines.append(
        "\nNote: You may reference these news items as [NEWS-n] if relevant "
        "to the question, but prioritize corpus judgments [1]-[10] for legal analysis."
    )
    return "\n".join(lines)


# ── DuckDuckGo HTML search ───────────────────────────────────────────────

class _DuckDuckGoParser(HTMLParser):
    """Extract result titles, URLs, and snippets from DuckDuckGo HTML search."""

    def __init__(self):
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_result_link = False
        self._in_snippet = False
        self._current: dict[str, str] = {}
        self._text_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]):
        attr_dict = dict(attrs)
        classes = (attr_dict.get("class") or "").split()

        # Result title link: <a class="result__a" href="...">
        if tag == "a" and "result__a" in classes:
            self._in_result_link = True
            self._text_buf = []
            raw_href = attr_dict.get("href", "")
            # DuckDuckGo wraps URLs in a redirect; extract the actual URL
            url = raw_href
            if "uddg=" in raw_href:
                import urllib.parse
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_href).query)
                url = parsed.get("uddg", [raw_href])[0]
            self._current["url"] = url

        # Snippet: <a class="result__snippet" ...> or <td class="result__snippet">
        if "result__snippet" in classes:
            self._in_snippet = True
            self._text_buf = []

    def handle_endtag(self, tag: str):
        if self._in_result_link and tag == "a":
            self._in_result_link = False
            self._current["title"] = " ".join(self._text_buf).strip()
            self._text_buf = []

        if self._in_snippet and tag in ("a", "td"):
            self._in_snippet = False
            self._current["snippet"] = " ".join(self._text_buf).strip()
            self._text_buf = []
            # A snippet closes a full result
            if self._current.get("title") and self._current.get("url"):
                self.results.append(self._current)
            self._current = {}

    def handle_data(self, data: str):
        if self._in_result_link or self._in_snippet:
            self._text_buf.append(data.strip())


def search_duckduckgo(query: str, max_results: int = 8) -> list[WebSignal]:
    """
    Search DuckDuckGo HTML for Indian law results.
    Returns up to max_results WebSignal objects. Cached 30 minutes.
    """
    cache_key = f"ddg:{query[:100]}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return [WebSignal(**s) for s in cached]

    import urllib.parse
    search_query = f"{query} India law"
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(search_query)}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; Sanhita/1.0; +https://sanhita.law)",
    })

    signals: list[WebSignal] = []
    try:
        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        parser = _DuckDuckGoParser()
        parser.feed(html)

        for item in parser.results[:max_results]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            result_url = item.get("url", "")
            if not title or not result_url:
                continue
            relevance = _score_relevance(query, title, snippet)
            signals.append(WebSignal(
                title=title,
                url=result_url,
                source="duckduckgo",
                source_name="DuckDuckGo",
                excerpt=snippet[:300],
                date="",
                relevance=relevance,
            ))
    except Exception as e:
        logger.warning("search_duckduckgo failed: %s", e)

    _set_cache(cache_key, [s.to_dict() for s in signals])
    return signals


def available_sources() -> list[dict[str, str]]:
    """List configured web signal sources."""
    return [
        {"id": k, "name": v["name"], "description": v["description"], "type": v["type"]}
        for k, v in SOURCES.items()
    ]
