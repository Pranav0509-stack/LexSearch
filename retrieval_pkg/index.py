"""
Sanhita — BM25 case-law index.

The product layer (`server.py`, `connectors.py`) imports four symbols
from this package: `BM25Index`, `Document`, `build_index`,
`doc_to_retrieve_hit`. Until this file existed the import raised and the
whole retrieval path silently fell back to the 27-doc seed corpus on
every request — visible in production logs as
`hybrid retrieve for X: N hits from {'seed': N}`.

The on-disk format is a plain pickle (`bm25.pkl`, configurable via the
`LEXSEARCH_BM25_PATH` env var). We use `rank_bm25.BM25Okapi` for
ranking — pure Python, ~5ms/query at 100K docs, no native deps.

Hit shape returned by `doc_to_retrieve_hit` mirrors the existing
`seed_corpus.query` output so the rest of the pipeline (ranking, citation
chips, source-card rail) needs no changes.
"""

from __future__ import annotations

import logging
import pickle
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# Token list mirrors `seed_corpus._tok` so query/index tokenisation match
# end-to-end. Lifted verbatim — keeping a second copy here so this module
# has no dependency on seed_corpus (which is itself a fallback path).
_STOPWORDS: set[str] = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by",
    "is", "are", "was", "were", "be", "been", "with", "as", "from",
    "that", "this", "these", "those", "it", "its", "under", "vs", "v",
    "vs.", "case", "law", "court", "what", "how", "do", "does", "did",
    "any", "who", "whom", "which", "can", "may", "shall", "will",
}


def tokenize(s: str) -> list[str]:
    """Lowercase word-character tokenization with stopword filter.

    Returns a list (not set) — `rank_bm25` needs the term-frequency
    counts from a list, not a deduped set."""
    return [
        w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
        if w not in _STOPWORDS and len(w) > 2
    ]


@dataclass
class Document:
    """A single indexable case-law unit. Fields mirror the standard hit
    shape used everywhere else in the pipeline."""
    case_id: str
    title: str = ""
    text: str = ""               # full body or headnote — what we index
    court: str = ""
    year: int | None = None
    citation: str = ""
    jurisdiction: str = ""       # "IN" | "SG" | "HK"
    tier: str = ""               # "SC" | "HC" | "CFA" | "CA" | ...
    url: str = ""
    source: str = ""             # ingestor name, e.g. "hk_cuthchow_csv"
    added_at: int = 0            # unix seconds — powers the "Latest" tab
    extra: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────
# BM25Index — append-only token store + rank_bm25 wrapper.
#
# We keep `_tokens` (list[list[str]]) and `_docs` (list[Document]) in
# parallel arrays. The BM25Okapi instance is rebuilt lazily after any
# `add()` call — re-fitting takes ~3-4s at 100K docs which is fine for
# nightly cron rebuilds. Query path never rebuilds.
# ─────────────────────────────────────────────────────────────────────────


class BM25Index:
    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._dirty: bool = False
        # Dedup map so re-running an ingestor doesn't add the same case
        # twice. Keyed by case_id which is mandatory.
        self._by_id: dict[str, int] = {}

    # ── Mutation ────────────────────────────────────────────────────────

    def add(self, docs: Iterable[Document]) -> int:
        """Append documents. Returns the count actually added (after
        dedup against existing case_ids)."""
        added = 0
        for d in docs:
            if not d.case_id:
                continue
            if d.case_id in self._by_id:
                continue
            indexable = " ".join(filter(None, [d.title, d.text]))
            tokens = tokenize(indexable)
            if not tokens:
                continue
            self._by_id[d.case_id] = len(self._docs)
            self._docs.append(d)
            self._tokens.append(tokens)
            added += 1
        if added:
            self._dirty = True
        return added

    def _refit(self) -> None:
        """Re-fit BM25 over the current corpus. Called lazily before the
        first query after an add()."""
        if not self._tokens:
            self._bm25 = None
        else:
            self._bm25 = BM25Okapi(self._tokens)
        self._dirty = False

    # ── Read paths ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._docs)

    def query(
        self,
        q: str,
        k: int = 6,
        *,
        tier: str | None = None,
        jurisdiction: str | None = None,
    ) -> list[tuple[Document, float]]:
        """Top-k BM25 hits with optional tier/jurisdiction filter."""
        if self._dirty or self._bm25 is None:
            self._refit()
        if self._bm25 is None:
            return []
        q_tokens = tokenize(q)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        # Filter+sort. We score the whole corpus then drop non-matching
        # juris/tier — cheaper than rebuilding sub-indexes per request.
        idxs = list(range(len(scores)))
        if jurisdiction:
            j = jurisdiction.upper()
            idxs = [i for i in idxs if (self._docs[i].jurisdiction or "").upper() == j]
        if tier:
            t = tier.upper()
            idxs = [i for i in idxs if (self._docs[i].tier or "").upper() == t]
        idxs.sort(key=lambda i: scores[i], reverse=True)
        out: list[tuple[Document, float]] = []
        for i in idxs[:k]:
            s = float(scores[i])
            if s <= 0:
                break
            out.append((self._docs[i], s))
        return out

    def latest(
        self,
        *,
        jurisdiction: str | None = None,
        k: int = 20,
    ) -> list[Document]:
        """Newest-first by `(added_at, year)` — powers the Court Search
        Latest tab. No BM25 work involved."""
        pool: list[Document] = self._docs
        if jurisdiction:
            j = jurisdiction.upper()
            pool = [d for d in pool if (d.jurisdiction or "").upper() == j]
        pool = sorted(
            pool,
            key=lambda d: (d.added_at or 0, d.year or 0),
            reverse=True,
        )
        return pool[:k]

    def get(self, case_id: str) -> Document | None:
        i = self._by_id.get(case_id)
        return self._docs[i] if i is not None else None

    def stats(self) -> dict[str, Any]:
        """Counts by jurisdiction / source — for the UI's "X cases
        indexed" chip."""
        by_juris: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for d in self._docs:
            j = (d.jurisdiction or "?").upper()
            by_juris[j] = by_juris.get(j, 0) + 1
            s = d.source or "?"
            by_source[s] = by_source.get(s, 0) + 1
        return {
            "total": len(self._docs),
            "by_jurisdiction": by_juris,
            "by_source": by_source,
        }

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Atomically write to disk. Skips re-fitting BM25 if dirty —
        load() will re-fit on first query post-load."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        # We persist tokens + docs; BM25Okapi itself is cheap to rebuild
        # and keeping it out keeps the pickle smaller and forward-compat.
        payload = {
            "version": 1,
            "docs": [asdict(d) for d in self._docs],
            "tokens": self._tokens,
        }
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)
        logger.info("BM25Index saved: %d docs -> %s", len(self._docs), p)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        """Read pickled corpus from disk. Returns an empty index if the
        file is missing — caller should still call save() after the next
        ingest run."""
        idx = cls()
        p = Path(path)
        if not p.exists():
            logger.info("BM25Index.load: %s missing, returning empty index", p)
            return idx
        t0 = time.monotonic()
        with p.open("rb") as fh:
            payload = pickle.load(fh)
        for raw in payload.get("docs", []):
            extra = raw.get("extra") or {}
            doc = Document(
                case_id=raw["case_id"],
                title=raw.get("title", ""),
                text=raw.get("text", ""),
                court=raw.get("court", ""),
                year=raw.get("year"),
                citation=raw.get("citation", ""),
                jurisdiction=raw.get("jurisdiction", ""),
                tier=raw.get("tier", ""),
                url=raw.get("url", ""),
                source=raw.get("source", ""),
                added_at=raw.get("added_at", 0) or 0,
                extra=extra if isinstance(extra, dict) else {},
            )
            idx._by_id[doc.case_id] = len(idx._docs)
            idx._docs.append(doc)
        idx._tokens = list(payload.get("tokens", []))
        idx._dirty = True  # rebuild BM25 lazily on first query
        logger.info(
            "BM25Index loaded: %d docs from %s in %.2fs",
            len(idx._docs), p, time.monotonic() - t0,
        )
        return idx


# ─────────────────────────────────────────────────────────────────────────
# Convenience builders + adapter to the legacy hit shape.
# ─────────────────────────────────────────────────────────────────────────


def build_index(docs: Iterable[Document]) -> BM25Index:
    idx = BM25Index()
    idx.add(docs)
    return idx


def doc_to_retrieve_hit(doc: Document, score: float, query: str) -> dict[str, Any]:
    """Adapter: Document -> the dict shape the rest of the pipeline
    speaks. Mirrors `seed_corpus.query` so render code is identical."""
    _ = query  # accepted for parity; not used (no per-query highlighting yet)
    return {
        "case_id": doc.case_id,
        "title": doc.title,
        "citation": doc.citation,
        "court": doc.court,
        "year": doc.year,
        "tier": doc.tier,
        "excerpt": (doc.text or "")[:600],
        "score": round(float(score), 4),
        "jurisdiction": doc.jurisdiction,
        "source": doc.source or "bm25",
        "url": doc.url,
    }
