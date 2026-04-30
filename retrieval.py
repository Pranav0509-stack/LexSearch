"""
LexSearch retrieval layer.

Provides:
 - BM25Index: an in-memory BM25 index over Indian Supreme Court + top 6 High
   Court judgments, built from the S3 parquet metadata the rest of LexSearch
   already reads. The index is tokenised, case-folded, and indian-legal-stop-
   worded.
 - build_index(): loads parquet partitions within a year window and returns a
   populated BM25Index. Intended for startup, nightly rebuild, and test use.
 - extract_excerpt(): given a case row and a query, returns a <= 400-char
   window centred on query term hits. Falls back to the first 400 chars of
   title + headnote when no term overlap exists.
 - classify_tier(): SC / HC / DC label from the `court` field.

This module is deliberately dependency-light (rank_bm25 + pandas + s3fs) so
it is safe to import from both server.py (API path) and ingest/*.py (batch
jobs).
"""

from __future__ import annotations

import logging
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import s3fs
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config — which partitions to index for the MVP BM25 layer.
# ---------------------------------------------------------------------------
# Supreme Court: last 5 years. High Courts: top 6 by volume (Delhi, Bombay,
# Madras, Karnataka, Calcutta, Allahabad) — covers ~60% of all HC queries
# our helpline sees. Everything else stays reachable via the existing
# substring /search endpoint.
HC_BUCKET = "indian-high-court-judgments"
SC_BUCKET = "indian-supreme-court-judgments"

TOP_HC_PARTITIONS: list[tuple[str, str]] = [
    ("7_26", "dhcdb"),                       # Delhi
    ("27_1", "newas"),                       # Bombay (Appellate)
    ("27_1", "newos"),                       # Bombay (Original)
    ("33_10", "hc_cis_mas"),                 # Madras
    ("29_3", "karnataka_bng_old"),           # Karnataka
    ("19_16", "calcutta_appellate_side"),    # Calcutta
    ("9_13", "cishclko"),                    # Allahabad (Lucknow)
]

DEFAULT_WINDOW_YEARS = 5

# Tokeniser: keep alphanumerics + section markers, drop punctuation.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# A deliberately small stop-word set — we keep "v", "vs", "section",
# "article" because they are load-bearing in legal queries.
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "by",
    "is", "was", "be", "been", "with", "as", "at", "that", "this", "it",
    "from", "are", "were", "has", "have", "had",
}


def tokenise(text: str) -> list[str]:
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------
def classify_tier(row: dict) -> str:
    court = (row.get("court") or "").lower()
    if row.get("type") == "sc" or "supreme" in court or court == "sc":
        return "SC"
    if "high court" in court or row.get("type") == "hc":
        return "HC"
    return "DC"


# ---------------------------------------------------------------------------
# Excerpt extraction
# ---------------------------------------------------------------------------
def extract_excerpt(row: dict, query: str, max_chars: int = 400) -> str:
    """Return a <=max_chars window of text that best overlaps `query`.

    Prefers, in order: headnote, description, title+petitioner+respondent.
    Windowed around the earliest query-term match; otherwise head of text.
    """
    candidates = []
    for key in ("headnote", "description", "judgment_text", "first_page_text"):
        v = row.get(key)
        if v:
            candidates.append(str(v))
    fallback_parts = [str(row.get(k, "")) for k in ("title", "petitioner", "respondent", "citation")]
    fallback = " — ".join(p for p in fallback_parts if p)
    body = " ".join(candidates) or fallback

    if not body:
        return ""

    body = re.sub(r"\s+", " ", body).strip()
    if len(body) <= max_chars:
        return body

    q_tokens = [t for t in tokenise(query) if len(t) > 2]
    lower = body.lower()
    best_idx = -1
    for tok in q_tokens:
        idx = lower.find(tok)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx = idx

    if best_idx == -1:
        return body[:max_chars].rstrip() + "…"

    half = max_chars // 2
    start = max(0, best_idx - half)
    end = min(len(body), start + max_chars)
    snippet = body[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(body):
        snippet = snippet + "…"
    return snippet


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------
@dataclass
class IndexedDoc:
    case_id: str
    court: str
    bench: str
    year: int
    date: str
    title: str
    citation: str
    tier: str
    s3_key: str
    source_row: dict = field(default_factory=dict)


@dataclass
class BM25Index:
    """In-memory BM25Okapi over tokenised doc bodies."""
    docs: list[IndexedDoc]
    bm25: BM25Okapi
    built_at: float

    def query(self, q: str, k: int = 5, tier: Optional[str] = None) -> list[tuple[IndexedDoc, float]]:
        tokens = tokenise(q)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # If tier requested, mask out non-matching tiers by -inf.
        if tier:
            tier_up = tier.upper()
            for i, d in enumerate(self.docs):
                if d.tier != tier_up:
                    scores[i] = float("-inf")
        # top-k
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out = []
        for i in top:
            if scores[i] <= 0 or scores[i] == float("-inf"):
                continue
            out.append((self.docs[i], float(scores[i])))
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)  # atomic on POSIX
        logger.info("BM25 saved: %s (%d docs)", path, len(self.docs))

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------
def _iter_sc_rows(fs: s3fs.S3FileSystem, years: Iterable[int]) -> Iterable[tuple[dict, int]]:
    for yr in years:
        path = f"s3://{SC_BUCKET}/metadata/parquet/year={yr}/metadata.parquet"
        try:
            with fs.open(path, "rb") as f:
                df = pd.read_parquet(f)
        except Exception as e:
            logger.debug("SC skip year=%d: %s", yr, e)
            continue
        logger.info("SC year=%d loaded %d rows", yr, len(df))
        for _, row in df.iterrows():
            yield row.to_dict(), yr


def _iter_hc_rows(
    fs: s3fs.S3FileSystem,
    years: Iterable[int],
    partitions: list[tuple[str, str]],
) -> Iterable[tuple[dict, str, str, int]]:
    for yr in years:
        for court, bench in partitions:
            path = f"s3://{HC_BUCKET}/metadata/parquet/year={yr}/court={court}/bench={bench}/metadata.parquet"
            try:
                with fs.open(path, "rb") as f:
                    df = pd.read_parquet(f)
            except Exception as e:
                logger.debug("HC skip %s: %s", path, e)
                continue
            logger.info("HC year=%d court=%s bench=%s loaded %d rows", yr, court, bench, len(df))
            for _, row in df.iterrows():
                yield row.to_dict(), court, bench, yr


def _row_body_sc(row: dict) -> str:
    parts = [
        str(row.get("title", "")),
        str(row.get("petitioner", "")),
        str(row.get("respondent", "")),
        str(row.get("citation", "")),
        str(row.get("headnote", "")),
        str(row.get("description", "")),
        str(row.get("judge", "")),
    ]
    return " ".join(p for p in parts if p and p != "nan")


def _row_body_hc(row: dict) -> str:
    parts = [
        str(row.get("title", "")),
        str(row.get("judge", "")),
        str(row.get("description", "")),
        str(row.get("disposal_nature", "")),
    ]
    return " ".join(p for p in parts if p and p != "nan")


def build_index(
    *,
    years: Optional[list[int]] = None,
    hc_partitions: Optional[list[tuple[str, str]]] = None,
    max_docs: Optional[int] = None,
    fs: Optional[s3fs.S3FileSystem] = None,
) -> BM25Index:
    """Build a fresh BM25 index from S3 parquet.

    Defaults: last 5 years SC + top 6 HC partitions. Pass `max_docs` to cap
    for dev runs (the full build is ~2M rows and takes ~5 minutes).
    """
    if fs is None:
        fs = s3fs.S3FileSystem(anon=True)
    if years is None:
        current = pd.Timestamp.now().year
        years = list(range(current, current - DEFAULT_WINDOW_YEARS, -1))
    if hc_partitions is None:
        hc_partitions = TOP_HC_PARTITIONS

    t0 = time.time()
    docs: list[IndexedDoc] = []
    corpus_tokens: list[list[str]] = []

    for row, yr in _iter_sc_rows(fs, years):
        body = _row_body_sc(row)
        tokens = tokenise(body)
        if not tokens:
            continue
        case_id = str(row.get("case_id") or row.get("cnr") or "")
        if not case_id:
            continue
        doc = IndexedDoc(
            case_id=f"SC-{case_id}",
            court="Supreme Court of India",
            bench="",
            year=yr,
            date=str(row.get("decision_date", "") or yr),
            title=str(row.get("title", "") or f"{row.get('petitioner','')} vs {row.get('respondent','')}"),
            citation=str(row.get("citation", "") or ""),
            tier="SC",
            s3_key="",
            source_row=row,
        )
        docs.append(doc)
        corpus_tokens.append(tokens)
        if max_docs and len(docs) >= max_docs:
            break

    if not (max_docs and len(docs) >= max_docs):
        for row, court, bench, yr in _iter_hc_rows(fs, years, hc_partitions):
            body = _row_body_hc(row)
            tokens = tokenise(body)
            if not tokens:
                continue
            case_id = str(row.get("cnr") or row.get("case_id") or "")
            if not case_id:
                continue
            doc = IndexedDoc(
                case_id=f"HC-{court}-{bench}-{case_id}",
                court=str(row.get("court", "") or f"HC {court}"),
                bench=bench,
                year=yr,
                date=str(row.get("decision_date", "") or yr),
                title=str(row.get("title", "") or case_id),
                citation="",
                tier="HC",
                s3_key="",
                source_row=row,
            )
            docs.append(doc)
            corpus_tokens.append(tokens)
            if max_docs and len(docs) >= max_docs:
                break

    if not docs:
        raise RuntimeError("No documents indexed — check S3 connectivity and years.")

    logger.info("Building BM25 over %d docs...", len(docs))
    bm25 = BM25Okapi(corpus_tokens)
    elapsed = time.time() - t0
    logger.info("Index built in %.1fs (%d docs)", elapsed, len(docs))
    return BM25Index(docs=docs, bm25=bm25, built_at=time.time())


# ---------------------------------------------------------------------------
# Public contract for /retrieve
# ---------------------------------------------------------------------------
def doc_to_retrieve_hit(doc: IndexedDoc, score: float, query: str) -> dict:
    """Shape the BM25 hit into NyayaSathi's expected payload."""
    return {
        "case_id": doc.case_id,
        "court": doc.court,
        "bench": doc.bench,
        "year": doc.year,
        "date": doc.date,
        "title": doc.title,
        "citation": doc.citation,
        "tier": doc.tier,
        "excerpt": extract_excerpt(doc.source_row, query),
        "score": round(score, 4),
    }
