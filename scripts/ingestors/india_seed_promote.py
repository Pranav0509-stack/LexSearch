"""Indian case-law ingestor — promote `seed_corpus.SEED_CORPUS` into BM25.

The vanga/openjustice-in datasets either live on AWS S3 (1.1 TB) or
require running a scraper, neither of which is appropriate for a
zero-config first-run. Until the bigger pipeline lands, this ingestor
takes the 30 hand-curated landmark Indian Supreme Court / High Court
judgments already shipped in `seed_corpus.py` and writes them into the
shared BM25 index, so the Court Search "India" filter returns real
results from minute one.

Idempotent — re-running just no-ops thanks to BM25Index.add()'s
case_id-based dedup.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

from retrieval_pkg import Document

logger = logging.getLogger(__name__)

SOURCE = "india_seed_promote"


def ingest(*, limit: int | None = None) -> Iterator[Document]:
    try:
        import seed_corpus  # type: ignore
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] seed_corpus import failed: %s", SOURCE, e)
        return
    rows = getattr(seed_corpus, "SEED_CORPUS", []) or []
    now = int(time.time())
    yielded = 0
    for row in rows:
        if limit is not None and yielded >= limit:
            break
        if not isinstance(row, dict):
            continue
        # Only promote IN entries — seed_corpus may contain non-IN test
        # rows in future. Today every entry is IN.
        juris = (row.get("jurisdiction") or "IN").upper()
        if juris != "IN":
            continue
        case_id = (row.get("case_id") or "").strip()
        if not case_id:
            continue
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        if not text and not title:
            continue
        yield Document(
            case_id=case_id,
            title=title,
            text=text,
            court=row.get("court") or "Supreme Court of India",
            year=row.get("year"),
            citation=row.get("citation") or "",
            jurisdiction="IN",
            tier=row.get("tier") or "SC",
            url=row.get("url") or "",
            source=SOURCE,
            added_at=now,
            extra={"promoted_from": "seed_corpus"},
        )
        yielded += 1
    logger.info("[%s] yielded %d docs", SOURCE, yielded)
