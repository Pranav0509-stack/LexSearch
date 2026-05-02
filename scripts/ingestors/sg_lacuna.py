"""Singapore case-law ingestor — `hueyy/lacuna-db`.

Lacuna-DB ships free Singapore legal corpora as JSON dumps under
`data/`. We pull four of them:

  • fc-judgments.json     — Family Court judgments
  • stc-judgments.json    — State Courts (Magistrates / District / etc.)
  • pdpc-decisions.json   — Personal Data Protection Commission rulings
  • lss-dt-reports.json   — Law Society Disciplinary Tribunal reports

Each row arrives with this shape:
  {tags, date, court, case-number, title, citation, url, counsel,
   timestamp, coram, html}

We strip the embedded HTML to plain text for BM25 indexing, derive the
SG tier (HC, CA, CFI, FC, MC, …) from the citation (`[YYYY] SG<TIER>`),
and emit Documents with the standard hit shape.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Iterator

from retrieval_pkg import Document
from . import _common as C

logger = logging.getLogger(__name__)

SOURCE = "sg_lacuna"

# Per-source path on the lacuna-db raw mirror plus a defaulted court
# label when the JSON row is sparse. Order matters only for log
# readability — rows are deduped by case_id downstream.
SOURCES: list[tuple[str, str, str]] = [
    # (filename,                 default_court,                tier_default)
    ("fc-judgments.json",        "Family Court",               "FC"),
    ("stc-judgments.json",       "State Courts",               ""),
    ("pdpc-decisions.json",      "Personal Data Protection",   "PDPC"),
    ("lss-dt-reports.json",      "Law Society Disciplinary",   "LSS-DT"),
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SG_TIER_RE = re.compile(r"\[(\d{4})\]\s*SG([A-Z()]{2,8})", re.IGNORECASE)


def _html_to_text(html: str | None) -> str:
    if not html:
        return ""
    # Lightweight strip — good enough for BM25 tokens. We don't need
    # paragraph structure here (the chunker handles that downstream).
    text = _HTML_TAG_RE.sub(" ", html)
    text = (text
            .replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&#39;", "'")
            .replace("\xa0", " "))
    return _WS_RE.sub(" ", text).strip()


def _tier_from_citation(citation: str, default: str) -> str:
    m = _SG_TIER_RE.search(citation or "")
    return (m.group(2).upper() if m else default).strip("()")


def _ingest_one(filename: str, default_court: str, default_tier: str,
                limit_remaining: int | None) -> Iterator[Document]:
    url = C.gh_raw("hueyy", "lacuna-db", f"data/{filename}", ref="main")
    logger.info("[%s] downloading %s", SOURCE, url)
    raw = C.http_get(url)
    rows = json.loads(raw.decode("utf-8"))
    if not isinstance(rows, list):
        logger.warning("[%s] %s: expected list, got %s; skipping", SOURCE, filename, type(rows).__name__)
        return
    now = int(time.time())
    yielded = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if limit_remaining is not None and yielded >= limit_remaining:
            break
        title = (row.get("title") or "").strip()
        citation = (row.get("citation") or "").strip()
        case_number = (row.get("case-number") or "").strip()
        court = (row.get("court") or default_court).strip()
        date = (row.get("date") or "").strip()
        coram = row.get("coram") or ""
        if isinstance(coram, list):
            coram = "; ".join(str(c) for c in coram)
        counsel = row.get("counsel") or []
        if isinstance(counsel, list):
            counsel_str = "; ".join(str(c) for c in counsel)
        else:
            counsel_str = str(counsel)
        url_field = (row.get("url") or "").strip()
        # Body text — prefer stripped HTML, fall back to tags joined.
        body = _html_to_text(row.get("html"))
        if not body:
            body = " ".join(row.get("tags") or [])
        if not body and not title:
            continue
        year = C.extract_year(date, citation)
        tier = _tier_from_citation(citation, default_tier)
        case_id = C.stable_case_id("sg", year or "x", tier or "x",
                                   case_number or title[:30])
        text_parts = []
        if row.get("tags"):
            text_parts.append("Topics: " + " | ".join(row["tags"]))
        if body:
            text_parts.append(body)
        text = "\n\n".join(text_parts)

        yield Document(
            case_id=case_id,
            title=title or case_number or "Untitled SG case",
            text=text,
            court=court,
            year=year,
            citation=citation,
            jurisdiction="SG",
            tier=tier,
            url=url_field,
            source=SOURCE,
            added_at=now,
            extra={
                "case_number": case_number,
                "date": date,
                "coram": coram,
                "counsel": counsel_str,
                "tags": row.get("tags") or [],
                "lacuna_file": filename,
            },
        )
        yielded += 1
    logger.info("[%s] %s: yielded %d docs", SOURCE, filename, yielded)


def ingest(*, limit: int | None = None) -> Iterator[Document]:
    """Yield Documents from every lacuna-db dump, in order. `limit`
    caps the *combined* output across all four files."""
    total = 0
    for filename, default_court, default_tier in SOURCES:
        remaining = None if limit is None else max(0, limit - total)
        if remaining == 0:
            return
        try:
            for doc in _ingest_one(filename, default_court, default_tier, remaining):
                yield doc
                total += 1
                if limit is not None and total >= limit:
                    return
        except Exception as e:  # noqa: BLE001
            # One bad JSON shouldn't kill the rest of the run.
            logger.warning("[%s] %s failed: %s", SOURCE, filename, e)
            continue
    logger.info("[%s] total yielded across all files: %d", SOURCE, total)
