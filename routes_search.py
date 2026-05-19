"""routes_search.py — Smart court search routes (semantic + hybrid).

Adds three endpoints on top of the existing legacy /api/cases/search:

  POST  /api/cases/smart-search   Hybrid BM25 + semantic via HybridSearchEngine
  GET   /api/cases/document/{id}  Full-text document for in-app viewer
  GET   /api/cases/suggest        Type-ahead suggestions

The legacy /api/cases/search is preserved so the existing UI doesn't break.
The redesigned court-search-pane calls smart-search for richer results.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Make scripts importable
_CORPUS = Path("/Users/pranav/Desktop/india-judgments-corpus")
if str(_CORPUS) not in sys.path:
    sys.path.insert(0, str(_CORPUS))

try:
    from scripts.search.engine import HybridSearchEngine, SearchFilters
    _ENGINE_AVAILABLE = True
except Exception as e:
    _ENGINE_AVAILABLE = False
    _ENGINE_ERR = str(e)

DB_PATH = _CORPUS / "india_courts.db"

router = APIRouter(prefix="/api/cases", tags=["search"])


# Singleton engine — built lazily, kept warm
_engine: Optional["HybridSearchEngine"] = None


def get_engine() -> "HybridSearchEngine":
    global _engine
    if _engine is None and _ENGINE_AVAILABLE:
        _engine = HybridSearchEngine(db_path=str(DB_PATH))
    if _engine is None:
        raise HTTPException(503, f"search engine unavailable: {_ENGINE_ERR if not _ENGINE_AVAILABLE else 'init failed'}")
    return _engine


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH), timeout=30.0)
    c.execute("PRAGMA busy_timeout=30000")
    c.row_factory = sqlite3.Row
    return c


# ── Smart search ────────────────────────────────────────────────────────

class SmartSearchRequest(BaseModel):
    q: str
    mode: str = Field(default="hybrid", pattern="^(keyword|semantic|hybrid)$")
    sort: str = Field(default="relevance", pattern="^(relevance|latest|oldest|most_cited)$")
    court: Optional[str] = None
    state: Optional[str] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    verdict: Optional[str] = None
    judge: Optional[str] = None
    act: Optional[str] = None
    doc_type: Optional[str] = None
    source_table: Optional[str] = None
    limit: int = Field(default=30, ge=1, le=100)


@router.post("/smart-search")
def smart_search(req: SmartSearchRequest):
    if not req.q.strip():
        return {"hits": [], "total": 0, "mode": req.mode, "engine": "noop"}
    engine = get_engine()
    bm25_mode = {"keyword": "bm25", "semantic": "semantic", "hybrid": "hybrid"}[req.mode]
    filters = SearchFilters(
        court=req.court, state=req.state,
        year_from=req.year_from, year_to=req.year_to,
        verdict=req.verdict, judge=req.judge, act=req.act,
        doc_type=req.doc_type, source_table=req.source_table,
    )
    # Over-fetch so we can sort
    raw = engine.search(req.q, filters=filters, mode=bm25_mode, limit=req.limit * 2)
    def _hit(r):
        title = (getattr(r, "title", "") or "").strip()
        pet = (getattr(r, "petitioner", "") or "").strip()
        res = (getattr(r, "respondent", "") or "").strip()
        court = (getattr(r, "court", "") or "").strip()
        year = getattr(r, "year", None)
        # Synthesize a readable title from parties when none stored (pipeline_docs)
        if not title and (pet or res):
            title = f"{pet} versus {res}".strip(" v.")
        if not title:
            doc_id = getattr(r, "doc_id", "") or ""
            parts = [court, f"({year})" if year else "", doc_id[-12:] if doc_id else ""]
            title = " · ".join(p for p in parts if p) or "Case record"
        # Strip JSON brackets from judge field if list
        judge = (getattr(r, "judge", "") or "").strip()
        if judge.startswith("[") and judge.endswith("]"):
            try:
                import json as _j
                arr = _j.loads(judge)
                if isinstance(arr, list):
                    judge = ", ".join(str(x) for x in arr if x)
            except Exception:
                pass
        snippet = (getattr(r, "snippet", None)
                   or (getattr(r, "summary", "") or "")[:280])
        # Annotate IPC / CrPC / IEA section refs with their BNS / BNSS / BSA
        # equivalents so the lawyer immediately sees both numbers.
        try:
            from legal_code_mapping import annotate_text   # type: ignore
            snippet = annotate_text(snippet) if snippet else snippet
            title_annotated = annotate_text(title) if title else title
        except ImportError:
            title_annotated = title
        return {
            "doc_id":       getattr(r, "doc_id", None),
            "case_id":      getattr(r, "doc_id", None),  # alias for legacy frontend
            "source_table": getattr(r, "source_table", None),
            "title":        title_annotated or "(untitled)",
            "petitioner":   pet or None,
            "respondent":   res or None,
            "court":        getattr(r, "court", None),
            "year":         getattr(r, "year", None),
            "date":         getattr(r, "date_decided", None),
            "judge":        judge or None,
            "citation":     getattr(r, "citation", None),
            "verdict":      getattr(r, "verdict", None),
            "doc_type":     getattr(r, "doc_type", None),
            "acts_cited":   getattr(r, "acts_cited", None),
            "snippet":      snippet,
            "summary":      getattr(r, "summary", None),
            "score":        round(float(getattr(r, "score", 0.0)), 4),
            "bm25_score":   round(float(getattr(r, "bm25_score", 0.0)), 4),
            "sem_score":    round(float(getattr(r, "sem_score", 0.0)), 4),
        }
    hits = [_hit(r) for r in raw]
    # Dedupe by (title, court, year) — pipeline_docs has overlapping records
    seen = set()
    deduped: list[dict] = []
    for h in hits:
        key = (h.get("title", "")[:120], h.get("court", "") or "", h.get("year") or 0, h.get("source_table"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)
    hits = deduped
    # Sort
    if req.sort == "latest":
        hits.sort(key=lambda h: (h.get("year") or 0, h.get("date") or ""), reverse=True)
    elif req.sort == "oldest":
        hits.sort(key=lambda h: (h.get("year") or 9999, h.get("date") or ""))
    elif req.sort == "most_cited":
        # No citator stats hooked yet → fall back to relevance
        pass
    return {
        "hits":   hits[:req.limit],
        "total":  len(hits),
        "mode":   req.mode,
        "sort":   req.sort,
        "engine": "hybrid",
        "facets": {},   # populated by /facets in v2
    }


# ── Document view (full text + structured metadata) ─────────────────────

@router.get("/document/{case_id:path}")
def case_document(case_id: str):
    """Return full document body + cited acts / sections / paras for the in-app viewer.
    Handles pipeline_docs, judgments, legal_docs, statutes, documents.
    """
    # Strip well-known prefixes
    bare = case_id
    for prefix in ("doc_", "statute_", "qa_", "judg_", "pd_"):
        if bare.startswith(prefix):
            bare = bare[len(prefix):]
            break

    with _conn() as c:
        # Try pipeline_docs — match by doc_id OR by cnr_number (HC/SC PDFs are
        # keyed by cnr_number in pipeline_docs; legacy callers may pass cnr).
        row = c.execute(
            """SELECT doc_id, source, court_type, court_name, bench, state, district,
                      case_number, cnr_number, case_type, case_year, petitioner, respondent,
                      petitioner_advocate, respondent_advocate, judge_names,
                      filing_date, judgment_date, status, outcome, disposal_nature,
                      title, full_text, summary, word_count, language,
                      acts_cited, section_refs, cases_cited, doc_type,
                      pdf_url, source_url, has_pdf
               FROM pipeline_docs
               WHERE doc_id = ? OR doc_id = ? OR cnr_number = ? OR cnr_number = ?
               ORDER BY CASE WHEN pdf_url != '' AND has_pdf = 1 THEN 0 ELSE 1 END
               LIMIT 1""",
            (case_id, bare, case_id, bare),
        ).fetchone()
        if row:
            d = dict(row)
            return {
                "id":           d["doc_id"],
                "source_table": "pipeline_docs",
                "title":        d.get("title"),
                "parties":      {"petitioner": d.get("petitioner"), "respondent": d.get("respondent")},
                "advocates":    {"petitioner": d.get("petitioner_advocate"), "respondent": d.get("respondent_advocate")},
                "court":        d.get("court_name") or d.get("court_type"),
                "bench":        d.get("bench"),
                "judges":       _try_json(d.get("judge_names")),
                "state":        d.get("state"),
                "district":     d.get("district"),
                "case_number":  d.get("case_number"),
                "cnr":          d.get("cnr_number"),
                "year":         d.get("case_year"),
                "judgment_date": d.get("judgment_date"),
                "filing_date":   d.get("filing_date"),
                "outcome":      d.get("outcome") or d.get("disposal_nature") or d.get("status"),
                "doc_type":     d.get("doc_type"),
                "acts_cited":   _try_json(d.get("acts_cited")),
                "section_refs": _try_json(d.get("section_refs")),
                "cases_cited":  _try_json(d.get("cases_cited")),
                "full_text":    d.get("full_text") or "",
                "word_count":   d.get("word_count"),
                "summary":      d.get("summary"),
                "pdf_url":      d.get("pdf_url"),
                "source_url":   d.get("source_url"),
                "has_pdf":      bool(d.get("has_pdf")),
            }
        # Try legal_docs
        row = c.execute(
            """SELECT doc_id, title, court, judge, citation, date_decided, year,
                      verdict, summary, full_text, acts_cited, doc_type, source
               FROM legal_docs WHERE doc_id = ? LIMIT 1""", (bare,)
        ).fetchone()
        if row:
            d = dict(row)
            # Cross-lookup pipeline_docs by cnr_number (legal_docs.doc_id is
            # often the same CNR string for HC/SC judgments).
            pdf_url = None
            has_pdf = False
            try:
                pd_row = c.execute(
                    """SELECT pdf_url, has_pdf FROM pipeline_docs
                       WHERE cnr_number = ? AND source IN ('aws_s3_hc','huggingface_debkanchan')
                       LIMIT 1""", (d["doc_id"],)
                ).fetchone()
                if pd_row:
                    pd = dict(pd_row)
                    if pd.get("pdf_url"):
                        pdf_url = pd["pdf_url"]
                        has_pdf = bool(pd.get("has_pdf", 1))
            except sqlite3.Error:
                pass
            return {
                "id":           d["doc_id"],
                "source_table": "legal_docs",
                "title":        d.get("title"),
                "court":        d.get("court"),
                "judges":       [d.get("judge")] if d.get("judge") else [],
                "citation":     d.get("citation"),
                "year":         d.get("year"),
                "judgment_date": d.get("date_decided"),
                "outcome":      d.get("verdict"),
                "doc_type":     d.get("doc_type"),
                "acts_cited":   _try_json(d.get("acts_cited")),
                "full_text":    d.get("full_text") or "",
                "summary":      d.get("summary"),
                "source":       d.get("source"),
                "pdf_url":      pdf_url,
                "has_pdf":      has_pdf,
            }
        # Try judgments
        row = c.execute(
            """SELECT cnr, title, court, court_code, bench, judge, date_decided, year,
                      disposal, description, pdf_link, pdf_available, verdict, full_text
               FROM judgments WHERE cnr = ? LIMIT 1""", (bare,)
        ).fetchone()
        if row:
            d = dict(row)
            # Cross-lookup: same CNR may exist in pipeline_docs (aws_s3_hc)
            # which DOES have a working pdf_url. Inherit it.
            pdf_url = d.get("pdf_link")
            has_pdf = bool(d.get("pdf_available"))
            try:
                pd_row = c.execute(
                    """SELECT pdf_url, has_pdf, full_text
                       FROM pipeline_docs
                       WHERE cnr_number = ? AND source = 'aws_s3_hc'
                       LIMIT 1""", (d["cnr"],)
                ).fetchone()
                if pd_row:
                    pd = dict(pd_row)
                    if pd.get("pdf_url"):
                        pdf_url = pd["pdf_url"]
                        has_pdf = bool(pd.get("has_pdf", 1))
                    if pd.get("full_text") and not d.get("full_text"):
                        d["full_text"] = pd["full_text"]
            except sqlite3.Error:
                pass
            return {
                "id":           d["cnr"],
                "source_table": "judgments",
                "title":        d.get("title"),
                "court":        d.get("court"),
                "court_code":   d.get("court_code"),
                "bench":        d.get("bench"),
                "judges":       [d.get("judge")] if d.get("judge") else [],
                "year":         d.get("year"),
                "judgment_date": d.get("date_decided"),
                "outcome":      d.get("verdict") or d.get("disposal"),
                "full_text":    d.get("full_text") or d.get("description") or "",
                "pdf_url":      pdf_url,
                "has_pdf":      has_pdf,
            }
    raise HTTPException(404, f"document {case_id} not found")


def _try_json(s):
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
        return [v]
    except Exception:
        # Comma-separated fallback
        if isinstance(s, str) and "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]
        return [s] if s else []


# ── Suggest (typeahead) ─────────────────────────────────────────────────

@router.get("/suggest")
def suggest(q: str = Query(...), limit: int = Query(default=8, le=20)):
    if len(q) < 2:
        return {"suggestions": []}
    try:
        engine = get_engine()
        return {"suggestions": engine.suggest(q, limit=limit)}
    except HTTPException:
        # No engine — return empty
        return {"suggestions": []}


@router.get("/engine-status")
def engine_status():
    try:
        engine = get_engine()
        return {
            "engine_available":  True,
            "semantic_available": engine.semantic_available,
            "tables": ["legal_docs", "judgments", "pipeline_docs", "statutes", "legal_qa", "documents"],
        }
    except HTTPException as e:
        return {"engine_available": False, "error": str(e.detail)}
