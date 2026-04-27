"""
LexSearch – Indian Court Judgments (High Courts + Supreme Court)
FastAPI backend: search API + PDF proxy from public AWS S3.
Run: uvicorn server:app --reload --port 8080
"""

import io
import json
import logging
import os
import tarfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd
import s3fs
from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

# Phase 2 — auth + Brief assistant. Both modules degrade gracefully when
# optional deps (google-generativeai) aren't installed.
import auth
from brief_service import answer_question, answer_open, answer_web, answer_agent, answer_chitchat, serialize_citations, generate_followups
from validators import input_guards
import vault_service
import workflows
from fastapi import File, UploadFile, Form

# Optional BM25 retrieval layer. Imports are lazy-friendly: if rank_bm25 or
# the retrieval module is missing, /retrieve returns 503 and the rest of
# the API keeps working.
try:
    from retrieval import (
        BM25Index,
        build_index,
        doc_to_retrieve_hit,
    )
    _RETRIEVAL_AVAILABLE = True
except Exception as _retrieval_err:  # pragma: no cover
    BM25Index = None  # type: ignore[assignment]
    build_index = None  # type: ignore[assignment]
    doc_to_retrieve_hit = None  # type: ignore[assignment]
    _RETRIEVAL_AVAILABLE = False
    _RETRIEVAL_IMPORT_ERROR: Optional[BaseException] = _retrieval_err

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LexSearch")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "HEAD", "POST"],
    allow_headers=["*"],
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Sensible defaults: deny framing, strict referrer, no MIME sniffing."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault(
            "Referrer-Policy", "strict-origin-when-cross-origin"
        )
        return response


app.add_middleware(SecurityHeadersMiddleware)

# ---------------------------------------------------------------------------
# S3 config
# ---------------------------------------------------------------------------
HC_BUCKET = "indian-high-court-judgments"
SC_BUCKET = "indian-supreme-court-judgments"
HC_S3 = f"s3://{HC_BUCKET}"
SC_S3 = f"s3://{SC_BUCKET}"
HC_HTTP = f"https://{HC_BUCKET}.s3.amazonaws.com"
SC_HTTP = f"https://{SC_BUCKET}.s3.ap-south-1.amazonaws.com"

_fs: Optional[s3fs.S3FileSystem] = None


def get_fs() -> s3fs.S3FileSystem:
    global _fs
    if _fs is None:
        _fs = s3fs.S3FileSystem(anon=True)
    return _fs


# ---------------------------------------------------------------------------
# High Court mapping
# ---------------------------------------------------------------------------
HC_COURTS = [
    {"s3_code": "9_13", "name": "Allahabad High Court", "benches": [
        {"code": "cishclko", "name": "Lucknow Bench"},
        {"code": "cisdb_16012018", "name": "Allahabad"},
    ]},
    {"s3_code": "28_2", "name": "Andhra Pradesh High Court", "benches": [
        {"code": "aphc", "name": "Amaravati"},
    ]},
    {"s3_code": "27_1", "name": "Bombay High Court", "benches": [
        {"code": "newas", "name": "Appellate Side"},
        {"code": "newos", "name": "Original Side"},
        {"code": "newos_spl", "name": "Original Side (Special)"},
        {"code": "hcaurdb", "name": "Aurangabad Bench"},
        {"code": "kolhcdb", "name": "Nagpur Bench"},
        {"code": "hcbgoa", "name": "Goa Bench"},
    ]},
    {"s3_code": "19_16", "name": "Calcutta High Court", "benches": [
        {"code": "calcutta_appellate_side", "name": "Appellate Side"},
        {"code": "calcutta_original_side", "name": "Original Side"},
        {"code": "calcutta_circuit_bench_at_jalpaiguri", "name": "Jalpaiguri Circuit"},
        {"code": "calcutta_circuit_bench_at_port_blair", "name": "Port Blair Circuit"},
    ]},
    {"s3_code": "22_18", "name": "Chhattisgarh High Court", "benches": [
        {"code": "cghccisdb", "name": "Bilaspur"},
    ]},
    {"s3_code": "7_26", "name": "Delhi High Court", "benches": [
        {"code": "dhcdb", "name": "New Delhi"},
    ]},
    {"s3_code": "18_6", "name": "Gauhati High Court", "benches": [
        {"code": "asghccis", "name": "Guwahati"},
        {"code": "azghccis", "name": "Aizawl Bench"},
        {"code": "arghccis", "name": "Itanagar Bench"},
        {"code": "nlghccis", "name": "Kohima Bench"},
    ]},
    {"s3_code": "24_17", "name": "Gujarat High Court", "benches": [
        {"code": "gujarathc", "name": "Ahmedabad"},
    ]},
    {"s3_code": "2_5", "name": "Himachal Pradesh High Court", "benches": [
        {"code": "cmis", "name": "Shimla"},
    ]},
    {"s3_code": "1_12", "name": "Jammu & Kashmir High Court", "benches": [
        {"code": "jammuhc", "name": "Jammu"},
        {"code": "kashmirhc", "name": "Srinagar"},
    ]},
    {"s3_code": "20_7", "name": "Jharkhand High Court", "benches": [
        {"code": "jhar_pg", "name": "Ranchi"},
    ]},
    {"s3_code": "29_3", "name": "Karnataka High Court", "benches": [
        {"code": "karnataka_bng_old", "name": "Bengaluru"},
        {"code": "karhcdharwad", "name": "Dharwad Bench"},
        {"code": "karhckalaburagi", "name": "Kalaburagi Bench"},
    ]},
    {"s3_code": "32_4", "name": "Kerala High Court", "benches": [
        {"code": "highcourtofkerala", "name": "Ernakulam"},
    ]},
    {"s3_code": "23_23", "name": "Madhya Pradesh High Court", "benches": [
        {"code": "mphc_db_jbp", "name": "Jabalpur"},
        {"code": "mphc_db_gwl", "name": "Gwalior Bench"},
        {"code": "mphc_db_ind", "name": "Indore Bench"},
    ]},
    {"s3_code": "33_10", "name": "Madras High Court", "benches": [
        {"code": "hc_cis_mas", "name": "Chennai"},
        {"code": "mdubench", "name": "Madurai Bench"},
    ]},
    {"s3_code": "14_25", "name": "Manipur High Court", "benches": [
        {"code": "manipurhc_pg", "name": "Imphal"},
    ]},
    {"s3_code": "17_21", "name": "Meghalaya High Court", "benches": [
        {"code": "meghalaya", "name": "Shillong"},
    ]},
    {"s3_code": "21_11", "name": "Orissa High Court", "benches": [
        {"code": "cisnc", "name": "Cuttack"},
    ]},
    {"s3_code": "10_8", "name": "Patna High Court", "benches": [
        {"code": "patnahcucisdb94", "name": "Patna"},
    ]},
    {"s3_code": "3_22", "name": "Punjab & Haryana High Court", "benches": [
        {"code": "phhc", "name": "Chandigarh"},
    ]},
    {"s3_code": "8_9", "name": "Rajasthan High Court", "benches": [
        {"code": "rhcjodh240618", "name": "Jodhpur"},
        {"code": "jaipur", "name": "Jaipur Bench"},
    ]},
    {"s3_code": "11_24", "name": "Sikkim High Court", "benches": [
        {"code": "sikkimhc_pg", "name": "Gangtok"},
    ]},
    {"s3_code": "16_20", "name": "Telangana High Court", "benches": [
        {"code": "thcnc", "name": "Hyderabad"},
    ]},
    {"s3_code": "36_29", "name": "Andhra Pradesh High Court (Kurnool)", "benches": [
        {"code": "taphc", "name": "Kurnool"},
    ]},
    {"s3_code": "5_15", "name": "Uttarakhand High Court", "benches": [
        {"code": "ukhcucis_pg", "name": "Nainital"},
    ]},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _extract_pdf_filename(pdf_link: str) -> str:
    if not pdf_link:
        return ""
    return pdf_link.rstrip("/").split("/")[-1]


# ── High Court helpers ────────────────────────────────────────────────────

def _hc_parquet_path(year: int, court: str, bench: str) -> str:
    return f"{HC_S3}/metadata/parquet/year={year}/court={court}/bench={bench}/metadata.parquet"


def _hc_df_to_results(df: pd.DataFrame, court: str, bench: str, year: int) -> list[dict]:
    results = []
    for _, row in df.iterrows():
        pdf_link = _safe_str(row.get("pdf_link", ""))
        pdf_filename = _extract_pdf_filename(pdf_link)
        if not pdf_filename:
            continue
        s3_key = f"data/pdf/year={year}/court={court}/bench={bench}/{pdf_filename}"
        results.append({
            "type": "hc",
            "case_number": _safe_str(row.get("cnr", "")) or pdf_filename.replace(".pdf", ""),
            "title": _safe_str(row.get("title", "")) or pdf_filename.replace(".pdf", ""),
            "court_name": _safe_str(row.get("court", "")),
            "court": court,
            "bench": bench,
            "year": year,
            "judge": _safe_str(row.get("judge", "")),
            "date": _safe_str(row.get("decision_date", "")) or str(year),
            "disposal": _safe_str(row.get("disposal_nature", "")),
            "s3_key": urllib.parse.quote(s3_key, safe=""),
        })
    return results


# ── Supreme Court helpers ─────────────────────────────────────────────────

def _sc_parquet_path(year: int) -> str:
    return f"{SC_S3}/metadata/parquet/year={year}/metadata.parquet"


def _sc_df_to_results(df: pd.DataFrame, year: int) -> list[dict]:
    results = []
    for _, row in df.iterrows():
        path = _safe_str(row.get("path", ""))
        if not path:
            continue
        pdf_name = f"{path}_EN.pdf"
        title = _safe_str(row.get("title", ""))
        petitioner = _safe_str(row.get("petitioner", ""))
        respondent = _safe_str(row.get("respondent", ""))
        citation = _safe_str(row.get("citation", ""))
        case_id = _safe_str(row.get("case_id", ""))
        # Build a tar-based key: sc_tar/{year}/{pdf_name}
        tar_key = f"sc_tar/{year}/{pdf_name}"
        results.append({
            "type": "sc",
            "case_number": case_id or _safe_str(row.get("cnr", "")),
            "title": title or f"{petitioner} vs {respondent}",
            "petitioner": petitioner,
            "respondent": respondent,
            "citation": citation,
            "court_name": "Supreme Court of India",
            "court": "sc",
            "bench": "",
            "year": year,
            "judge": _safe_str(row.get("judge", "")),
            "author_judge": _safe_str(row.get("author_judge", "")),
            "date": _safe_str(row.get("decision_date", "")) or str(year),
            "disposal": _safe_str(row.get("disposal_nature", "")),
            "s3_key": urllib.parse.quote(tar_key, safe=""),
        })
    return results


def _apply_filters(df: pd.DataFrame, q: str, cnr: str, judge: str,
                   case_type: str, disposal: str, petitioner: str,
                   respondent: str, citation: str) -> pd.DataFrame:
    """Apply common filters to a DataFrame."""
    if cnr:
        col = "cnr" if "cnr" in df.columns else "case_id"
        if col in df.columns:
            df = df[df[col].astype(str).str.lower().str.contains(cnr.lower(), na=False)]
    if judge:
        df = df[df["judge"].astype(str).str.lower().str.contains(judge.lower(), na=False)]
    if case_type:
        df = df[df["title"].astype(str).str.upper().str.startswith(case_type.upper(), na=False)]
    if disposal:
        df = df[df["disposal_nature"].astype(str).str.lower().str.contains(disposal.lower(), na=False)]
    if petitioner and "petitioner" in df.columns:
        df = df[df["petitioner"].astype(str).str.lower().str.contains(petitioner.lower(), na=False)]
    if respondent and "respondent" in df.columns:
        df = df[df["respondent"].astype(str).str.lower().str.contains(respondent.lower(), na=False)]
    if citation and "citation" in df.columns:
        df = df[df["citation"].astype(str).str.lower().str.contains(citation.lower(), na=False)]
    if q:
        q_lower = q.lower()
        mask = pd.Series(False, index=df.index)
        for col_name in ["title", "description", "petitioner", "respondent"]:
            if col_name in df.columns:
                mask |= df[col_name].astype(str).str.lower().str.contains(q_lower, na=False)
        df = df[mask]
    return df


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/courts")
def list_courts():
    return JSONResponse(HC_COURTS)


@app.get("/search")
def search(
    mode: str = Query(default="hc", description="hc or sc"),
    court: str = Query(default=""),
    bench: str = Query(default=""),
    year: int = Query(default=0),
    q: str = Query(default=""),
    cnr: str = Query(default=""),
    judge: str = Query(default=""),
    case_type: str = Query(default=""),
    disposal: str = Query(default=""),
    petitioner: str = Query(default=""),
    respondent: str = Query(default=""),
    citation: str = Query(default=""),
    page: int = Query(default=1, ge=1),
):
    PAGE_SIZE = 50
    fs = get_fs()
    results: list[dict] = []

    years = [year] if year else list(range(2024, 2019, -1))

    if mode == "sc":
        # ── Supreme Court search ──
        has_filter = q or cnr or judge or case_type or disposal or petitioner or respondent or citation or year
        if not has_filter:
            raise HTTPException(400, "Provide at least one filter (year recommended).")

        for yr in years:
            path = _sc_parquet_path(yr)
            try:
                with fs.open(path, "rb") as f:
                    df = pd.read_parquet(f)
                logger.info(f"SC: Loaded {len(df)} rows from year={yr}")
            except Exception as e:
                logger.debug(f"SC: Skipping year={yr}: {e}")
                continue

            df = _apply_filters(df, q, cnr, judge, case_type, disposal, petitioner, respondent, citation)
            results.extend(_sc_df_to_results(df, yr))

            if len(results) >= PAGE_SIZE * page + PAGE_SIZE:
                break
    else:
        # ── High Court search ──
        has_filter = court or q or cnr or judge or case_type or year
        if not has_filter:
            raise HTTPException(400, "Provide at least one filter.")

        courts_to_scan: list[tuple[str, str]] = []
        if court and bench:
            courts_to_scan = [(court, bench)]
        elif court:
            for c in HC_COURTS:
                if c["s3_code"] == court:
                    courts_to_scan = [(court, b["code"]) for b in c["benches"]]
                    break
            if not courts_to_scan:
                courts_to_scan = [(court, court)]
        else:
            courts_to_scan = [
                ("7_26", "dhcdb"), ("27_1", "newas"),
                ("33_10", "hc_cis_mas"), ("19_16", "calcutta_appellate_side"),
                ("9_13", "cishclko"),
            ]

        for yr in years:
            for ct, bn in courts_to_scan:
                path = _hc_parquet_path(yr, ct, bn)
                try:
                    with fs.open(path, "rb") as f:
                        df = pd.read_parquet(f)
                    logger.info(f"HC: Loaded {len(df)} rows from year={yr}/court={ct}/bench={bn}")
                except Exception as e:
                    logger.debug(f"HC: Skipping {path}: {e}")
                    continue

                df = _apply_filters(df, q, cnr, judge, case_type, disposal, "", "", "")
                results.extend(_hc_df_to_results(df, ct, bn, yr))

                if len(results) >= PAGE_SIZE * page + PAGE_SIZE:
                    break
            if len(results) >= PAGE_SIZE * page + PAGE_SIZE:
                break

    start = (page - 1) * PAGE_SIZE
    return JSONResponse({
        "total": len(results),
        "page": page,
        "page_size": PAGE_SIZE,
        "results": results[start:start + PAGE_SIZE],
    })


# ---------------------------------------------------------------------------
# PDF endpoints
# ---------------------------------------------------------------------------

@app.get("/pdf/{s3_key:path}")
async def proxy_pdf(s3_key: str, download: bool = False):
    """Proxy HC PDF from S3.

    Opens the upstream stream and inspects status BEFORE returning a
    response — so a missing PDF turns into a JSON 404, not a half-streamed
    error the browser can't recover from.
    """
    decoded = urllib.parse.unquote(s3_key)
    url = f"{HC_HTTP}/{decoded}"
    fname = decoded.split("/")[-1] or "judgment.pdf"
    disp = f'attachment; filename="{fname}"' if download else f'inline; filename="{fname}"'

    client = httpx.AsyncClient(timeout=60)
    try:
        req = client.build_request("GET", url)
        resp = await client.send(req, stream=True)
    except httpx.RequestError as exc:
        await client.aclose()
        logger.warning("HC PDF upstream error for %s: %s", decoded, exc)
        raise HTTPException(502, "Court archive is unreachable. Try again in a minute.")

    if resp.status_code != 200:
        await resp.aclose()
        await client.aclose()
        raise HTTPException(404, "PDF not available in the public archive.")

    async def stream():
        try:
            async for chunk in resp.aiter_bytes(65536):
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        stream(),
        media_type="application/pdf",
        headers={"Content-Disposition": disp},
    )


@app.head("/pdf/{s3_key:path}")
async def probe_pdf(s3_key: str):
    """Lightweight probe so the UI can detect a dead PDF link without
    downloading bytes. Returns 200 if the object exists, 404 otherwise."""
    decoded = urllib.parse.unquote(s3_key)
    url = f"{HC_HTTP}/{decoded}"
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.head(url)
    except httpx.RequestError:
        raise HTTPException(502, "Court archive is unreachable.")
    if resp.status_code != 200:
        raise HTTPException(404, "PDF not available.")
    return Response(status_code=200)


# ---------------------------------------------------------------------------
# Retrieval layer — BM25 index for NyayaSathi grounding
# ---------------------------------------------------------------------------
BM25_PATH = Path(os.environ.get("LEXSEARCH_BM25_PATH", str(Path(__file__).parent / "bm25.pkl")))
BM25_ENABLED = os.environ.get("LEXSEARCH_BM25_ENABLED", "true").lower() == "true"
BM25_MAX_DOCS = int(os.environ.get("LEXSEARCH_BM25_MAX_DOCS", "0")) or None
BM25_LAZY = os.environ.get("LEXSEARCH_BM25_LAZY", "true").lower() == "true"

_bm25_lock = threading.Lock()
_bm25_index: Optional["BM25Index"] = None  # type: ignore[name-defined]
_bm25_loading = False
_bm25_load_error: Optional[str] = None
_bm25_loaded_at: float = 0.0


def _load_bm25_blocking() -> None:
    """Load a pre-built BM25 pickle from disk, or build on-the-fly if missing."""
    global _bm25_index, _bm25_loading, _bm25_load_error, _bm25_loaded_at
    if not _RETRIEVAL_AVAILABLE:
        _bm25_load_error = "retrieval module unavailable"
        return
    try:
        if BM25_PATH.exists():
            logger.info("Loading BM25 index from %s", BM25_PATH)
            _bm25_index = BM25Index.load(BM25_PATH)  # type: ignore[union-attr]
        else:
            logger.warning(
                "BM25 pickle not found at %s — building from S3 (this takes minutes). "
                "In production pre-build via `python ingest/rebuild_bm25.py`.",
                BM25_PATH,
            )
            _bm25_index = build_index(max_docs=BM25_MAX_DOCS)  # type: ignore[misc]
            try:
                _bm25_index.save(BM25_PATH)  # type: ignore[union-attr]
            except Exception as save_err:
                logger.warning("Could not persist BM25 index: %s", save_err)
        _bm25_loaded_at = time.time()
        _bm25_load_error = None
        logger.info("BM25 ready: %d docs", len(_bm25_index))  # type: ignore[arg-type]
    except Exception as e:
        _bm25_load_error = f"{type(e).__name__}: {e}"
        logger.error("BM25 load failed: %s", _bm25_load_error)
    finally:
        _bm25_loading = False


def _ensure_bm25() -> Optional["BM25Index"]:
    """Return the loaded index, kicking off a background load on first call."""
    global _bm25_loading
    if not BM25_ENABLED or not _RETRIEVAL_AVAILABLE:
        return None
    if _bm25_index is not None:
        return _bm25_index
    with _bm25_lock:
        if _bm25_index is not None:
            return _bm25_index
        if not _bm25_loading:
            _bm25_loading = True
            if BM25_LAZY:
                t = threading.Thread(target=_load_bm25_blocking, daemon=True)
                t.start()
            else:
                _load_bm25_blocking()
    return _bm25_index


@app.on_event("startup")
def _startup_load_index() -> None:
    # Phase 2: ensure the SQLite schema exists before any request lands.
    try:
        auth.init_db()
        logger.info("auth db ready at %s", auth.DB_PATH)
    except Exception as e:
        logger.error("auth.init_db failed: %s", e)

    # Phase 3: dashboard schema (dash_activity, dash_settings) lives on
    # whichever DB DATABASE_URL points at — Postgres in prod, SQLite
    # otherwise. Best-effort: a missing psycopg/Postgres can't block boot.
    try:
        import db_adapter  # type: ignore
        db_adapter.init_dashboard_schema()
    except Exception as e:  # noqa: BLE001
        logger.warning("dashboard schema init failed (non-fatal): %s", e)

    if BM25_ENABLED and not BM25_LAZY:
        _ensure_bm25()
    elif BM25_ENABLED:
        # kick off background warm-up so the first /retrieve isn't cold
        _ensure_bm25()


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    k: int = Field(default=5, ge=1, le=20)
    tier: Optional[str] = Field(default=None, description="SC / HC / DC filter")


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    """Dense retrieval endpoint consumed by NyayaSathi for grounding.

    Returns an ordered list of hits. Empty list with 200 means the index is
    loaded but no relevant docs were found — the caller should fall back to
    its local corpus. A 503 means the index isn't ready yet.
    """
    if not _RETRIEVAL_AVAILABLE:
        raise HTTPException(503, "retrieval module not installed on this deployment")
    idx = _ensure_bm25()
    if idx is None:
        raise HTTPException(503, f"BM25 index not ready ({_bm25_load_error or 'loading'})")

    hits = idx.query(req.query, k=req.k, tier=req.tier)
    return JSONResponse(
        {
            "query": req.query,
            "k": req.k,
            "tier": req.tier,
            "count": len(hits),
            "hits": [doc_to_retrieve_hit(d, s, req.query) for d, s in hits],  # type: ignore[misc]
        }
    )


@app.get("/judgment/{case_id}")
def get_judgment(case_id: str):
    """Return the indexed row for a single case_id (title, citation, body).

    NyayaSathi uses this when the caller asks a follow-up like "tell me more
    about that judgment". Text, not PDF — use /pdf/* or /sc-pdf/* for files.
    """
    idx = _ensure_bm25()
    if idx is None:
        raise HTTPException(503, "BM25 index not ready")
    d = idx.get(case_id)
    if d is None:
        raise HTTPException(404, f"case_id {case_id} not in index")
    return JSONResponse(
        {
            "case_id": d.case_id,
            "court": d.court,
            "year": d.year,
            "title": d.title,
            "citation": d.citation,
            "tier": d.tier,
            "jurisdiction": d.jurisdiction,
            "url": d.url,
            "source": d.source,
            "text": d.text[:8000],
            "extra": d.extra,
        }
    )


@app.post("/admin/reload")
def admin_reload(authorization: Optional[str] = Header(default=None)):
    """Hot-reload the BM25 pickle after `rebuild_bm25.py` writes a new one.
    Auth: Bearer token via LEXSEARCH_ADMIN_TOKEN env. If unset, endpoint is
    effectively disabled (403)."""
    global _bm25_index, _bm25_loading, _bm25_load_error
    expected = os.environ.get("LEXSEARCH_ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(403, "admin reload disabled (set LEXSEARCH_ADMIN_TOKEN)")
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(401, "bad token")
    with _bm25_lock:
        _bm25_index = None
        _bm25_loading = False
        _bm25_load_error = None
    _ensure_bm25()
    return {"status": "reloading"}


@app.get("/health")
def health():
    """Liveness + BM25 readiness. NyayaSathi pings this before enabling
    LexSearch-backed grounding on a call."""
    idx = _bm25_index
    return {
        "status": "ok",
        "retrieval_available": _RETRIEVAL_AVAILABLE,
        "bm25_enabled": BM25_ENABLED,
        "bm25_loaded": idx is not None,
        "bm25_loading": _bm25_loading,
        "bm25_doc_count": (len(idx) if idx is not None else 0),
        "bm25_loaded_at": _bm25_loaded_at,
        "bm25_load_error": _bm25_load_error,
    }


@app.get("/api/connectors")
def api_connectors():
    """
    List available data-source connectors + which have API keys wired up.
    The UI uses this to render the source-picker dropdown with live/greyed
    states, so users can see exactly which Asian databases Sanhita is
    hitting for grounding.
    """
    try:
        import connectors as _c
        flags = _c.available_connectors()
    except Exception as e:
        logger.warning("connectors.available_connectors failed: %s", e)
        flags = {}
    return {
        "connectors": flags,
        "default_chain": {
            "IN": ["indian_kanoon", "ecourts", "web", "seed"],
            "JP": ["egov_japan", "web", "seed"],
            "*":  ["web", "seed"],
        },
    }


@app.get("/sc-pdf/{year}/{pdf_name}")
def serve_sc_pdf(year: int, pdf_name: str, download: bool = False):
    """Extract a single PDF from the Supreme Court tar archive on S3.

    Failure modes are normalised to JSON 404 so the UI can show a clean
    "PDF unavailable" state instead of a 500 spinner that never resolves.
    """
    fs = get_fs()
    tar_path = f"{SC_S3}/data/tar/year={year}/english/english.tar"

    try:
        with fs.open(tar_path, "rb") as f:
            with tarfile.open(fileobj=f, mode="r|") as tf:
                for member in tf:
                    if member.name == pdf_name or member.name.endswith(pdf_name):
                        extracted = tf.extractfile(member)
                        if extracted:
                            pdf_bytes = extracted.read()
                            disp = f'attachment; filename="{pdf_name}"' if download else f'inline; filename="{pdf_name}"'
                            return Response(
                                content=pdf_bytes,
                                media_type="application/pdf",
                                headers={"Content-Disposition": disp},
                            )
        raise HTTPException(404, "PDF not available in the archive.")
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(404, "Archive for that year is not available.")
    except Exception as e:
        logger.error("SC PDF extraction error for %s/%s: %s", year, pdf_name, e)
        raise HTTPException(404, "PDF could not be extracted.")


@app.head("/sc-pdf/{year}/{pdf_name}")
def probe_sc_pdf(year: int, pdf_name: str):
    """HEAD probe for SC PDFs. Confirms the tar archive for the year exists;
    we don't scan for the filename (would mean reading the whole tar) — the
    UI treats a 200 here as "try it, might still 404 on GET" but at least
    blocks links when the year's archive is missing entirely."""
    fs = get_fs()
    tar_path = f"{SC_S3}/data/tar/year={year}/english/english.tar"
    try:
        if fs.exists(tar_path):
            return Response(status_code=200)
    except Exception as e:
        logger.debug("SC archive probe failed for %s: %s", year, e)
    raise HTTPException(404, "SC archive not available for that year.")


# ---------------------------------------------------------------------------
# Phase 2 — access requests, login, Brief chat, admin
# ---------------------------------------------------------------------------

ADMIN_TOKEN = os.environ.get("LEXSEARCH_ADMIN_TOKEN", "")


def _client_ip(request: Request) -> str:
    # Trust X-Forwarded-For if present (behind Render / Nginx), else peer IP.
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else ""


def _require_admin(authorization: Optional[str]) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(403, "admin disabled (set LEXSEARCH_ADMIN_TOKEN)")
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not token or token != ADMIN_TOKEN:
        raise HTTPException(401, "bad token")


def _require_user(session: Optional[str]) -> dict:
    """Dependency-style helper. Returns the authenticated user row or 401."""
    uid = auth.verify_session_token(session)
    if uid is None:
        raise HTTPException(401, "not signed in")
    user = auth.get_user(uid)
    if not user:
        raise HTTPException(401, "session expired")
    return user


# ── public: access request ─────────────────────────────────────────────────

class AccessRequestBody(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    email: str = Field(..., min_length=5, max_length=160)
    role: str = Field(default="", max_length=80)
    firm: str = Field(default="", max_length=160)
    bar_no: str = Field(default="", max_length=60)
    note: str = Field(default="", max_length=1000)


@app.post("/api/access-request")
def api_access_request(body: AccessRequestBody, request: Request):
    ip = _client_ip(request)
    if not auth.rate_limit("access_request", ip, max_hits=5, window_s=3600):
        raise HTTPException(429, "Too many requests from this IP. Try later.")
    # Minimal email shape check — do not run a full RFC 5321.
    if "@" not in body.email or "." not in body.email.split("@")[-1]:
        raise HTTPException(400, "Please provide a valid email address.")
    rid = auth.create_access_request(
        body.name, body.email, body.role, body.firm, body.bar_no, body.note, ip
    )
    return {"ok": True, "request_id": rid}


# ── login / logout ────────────────────────────────────────────────────────

class LoginBody(BaseModel):
    code: str = Field(..., min_length=6, max_length=32)


@app.post("/api/login")
def api_login(body: LoginBody, request: Request):
    ip = _client_ip(request)
    if not auth.rate_limit("login", ip, max_hits=10, window_s=600):
        raise HTTPException(429, "Too many login attempts. Slow down.")
    user = auth.validate_code(body.code)
    if not user:
        raise HTTPException(401, "Invalid or revoked access code.")
    token = auth.make_session_token(user["id"])
    resp = JSONResponse({"ok": True, "user": {"email": user["email"], "name": user["name"]}})
    resp.set_cookie(
        auth.SESSION_COOKIE,
        token,
        max_age=auth.SESSION_TTL_S,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )
    return resp


@app.get("/api/logout")
def api_logout():
    resp = RedirectResponse(url="/", status_code=302)
    resp.delete_cookie(auth.SESSION_COOKIE, path="/")
    return resp


@app.get("/api/languages")
def api_languages():
    """Public — front-end builds the language picker from this. Returns
    every code in brief_service.LANGUAGES with both English label and
    native script label. Order is meaningful (English first, then Indian
    languages, then pan-Asian) — the dict is insertion-ordered."""
    import brief_service
    out = []
    for code, info in brief_service.LANGUAGES.items():
        out.append({
            "code": code,
            "label": info["label"],
            "native": info["native"],
            "family": info["family"],
            "rtl": info["family"] == "Arabic",
        })
    return {"languages": out}


@app.get("/api/me")
def api_me(ls_session: Optional[str] = Cookie(default=None)):
    try:
        user = _require_user(ls_session)
    except HTTPException:
        return JSONResponse({"authenticated": False}, status_code=200)
    return {"authenticated": True, "user": {"email": user["email"], "name": user["name"]}}


# ── brief: threads + chat ─────────────────────────────────────────────────

class NewThreadBody(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)


@app.get("/api/brief/threads")
def api_list_threads(ls_session: Optional[str] = Cookie(default=None)):
    user = _require_user(ls_session)
    threads = auth.list_user_threads(user["id"], limit=30)
    return {
        "user": {"email": user["email"], "name": user["name"]},
        "threads": threads,
    }


@app.post("/api/brief/threads")
def api_create_thread(
    body: NewThreadBody = NewThreadBody(),
    ls_session: Optional[str] = Cookie(default=None),
):
    user = _require_user(ls_session)
    tid = auth.create_thread(user["id"], title=(body.title or "New matter"))
    return {
        "thread": {
            "id": tid,
            "title": body.title or "New matter",
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
        }
    }


@app.get("/api/brief/threads/{thread_id}")
def api_get_thread(
    thread_id: int,
    ls_session: Optional[str] = Cookie(default=None),
):
    user = _require_user(ls_session)
    msgs = auth.get_thread_messages(thread_id, user["id"])
    if msgs is None:
        raise HTTPException(404, "Thread not found.")
    return {"thread_id": thread_id, "messages": msgs}


class ChatBody(BaseModel):
    thread_id: int = Field(..., ge=1)
    question: str = Field(..., min_length=2, max_length=2000)
    jurisdiction: Optional[str] = Field(default=None, max_length=8)
    # ISO 639-1 (or BCP-47 for zh-CN/zh-TW). Validated against
    # brief_service.LANGUAGES; unknown codes silently fall back to English.
    language: Optional[str] = Field(default=None, max_length=8)
    sources: Optional[list[str]] = Field(default=None)
    # Reasoning-model preference. One of "gemini" | "anthropic" | "groq" |
    # "cloudflare" — passed straight into llm.router.generate(prefer=…).
    # Unknown / empty / None = router default (Gemini at the head of the
    # chain). The router is defensive: a `prefer` for a provider with no
    # API key just falls through to the next in the chain.
    prefer: Optional[str] = Field(default=None, max_length=16)


# ── Intent classifier ──────────────────────────────────────────────────────
# Cheap string heuristic that routes the chat endpoint to the right mode
# *without* requiring the user to flip the Canvas/Search toggles. If a query
# starts with a drafting verb ("draft me a letter…"), retrieval is the wrong
# frame — we send it straight to `answer_open` instead of running the 6-gate
# validator and refusing for "insufficient grounding."

_DRAFT_TRIGGERS = (
    "draft", "write a", "write me", "write us", "compose", "prepare a",
    "prepare me", "outline", "letter to", "letter for", "notice to",
    "notice for", "complaint to", "complaint against", "petition", "memo",
    "memorandum", "agreement", "contract for", "nda", "mou ", " mou,", "sha ",
    "share purchase agreement", "shareholders agreement", "demand notice",
    "section 9 application", "bail application", "anticipatory bail application",
    "writ petition", "draft an", "draft me", "draft us",
)
_TRANSLATE_TRIGGERS = ("translate", "translation of", "in hindi", "in english", "into english", "into hindi")
_REVIEW_TRIGGERS = ("review this", "check this", "redline this", "look at this clause", "is this clause", "vet this")
_SUMMARIZE_TRIGGERS = ("summarize", "summarise", "summary of", "tldr", "tl;dr", "in plain english")
_EXPLAIN_TRIGGERS = ("explain", "what is the meaning of", "what does", "how does")

# Conversational filler — greetings, thanks, capability checks. These get a
# warm one-liner from the LLM with no retrieval, no guards, no validation.
# Without this branch, "hi" hits MIN_QUESTION_CHARS=4 in input_guards and the
# user sees "Your question is too short" — a terrible first impression.
_CHITCHAT_EXACT = {
    "hi", "hii", "hiii", "hello", "hey", "hola", "namaste", "namaskar",
    "yo", "sup", "thanks", "thank you", "thx", "ty", "ok", "okay", "cool",
    "great", "nice", "got it", "ack", "good morning", "good evening",
    "good afternoon", "morning", "evening",
}
_CHITCHAT_TRIGGERS = (
    "who are you", "what are you", "what can you do", "what can you help",
    "how do you work", "how does this work", "what is sanhita", "who built",
    "are you a real lawyer", "are you human", "are you an ai",
    "tell me about yourself", "introduce yourself",
)

# Multi-step intent — if a query asks for *both* research AND action
# ("find … and draft …", "compare … then summarize …"), the agent loop is
# the right frame even if the user didn't toggle the green wand. We
# regex-detect on the question and re-route to /api/brief/agent.
_AGENT_TRIGGER_PATTERNS = (
    r"\bfind\b.{1,80}\b(and|then)\b.{1,80}\b(draft|write|prepare|compose|summari[sz]e|redline|translate)\b",
    r"\b(search|look up|research)\b.{1,80}\b(and|then)\b.{1,80}\b(draft|write|prepare|email|send|notify)\b",
    r"\b(compare|contrast)\b.{1,80}\b(and|then)\b.{1,120}\b(summari[sz]e|redline|translate|draft)\b",
    r"\bredline\b.{1,80}\b(and|then)\b.{1,80}\b(translate|draft|email|send)\b",
    r"\b(read|review)\b.{1,80}\b(and|then)\b.{1,80}\b(draft|email|send|summari[sz]e)\b",
    r"\b(retrieve|fetch|pull up)\b.{1,80}\b(cases|judgments|statutes|sections)\b.{1,120}\b(and|then)\b",
    # Google Workspace verbs always imply agent mode (they need tool calls).
    r"\b(save|create|put|push)\b.{1,40}\b(google doc|gdoc|google sheet|gsheet|sheets|drive)\b",
    r"\b(email|send|gmail|mail it)\b.{1,80}\b(to|at)\b\s+\S+@\S+",
    r"\bsend.{0,20}\b(it|this|the draft|the notice|the letter)\b.{0,40}\b(to|via email|over)\b",
)


def _is_chitchat(question: str) -> bool:
    q = (question or "").lower().strip().rstrip("!?.,")
    if not q:
        return False
    if q in _CHITCHAT_EXACT:
        return True
    head = q[:80]
    for t in _CHITCHAT_TRIGGERS:
        if t in head:
            return True
    return False


def _looks_like_agent_task(question: str) -> bool:
    import re
    q = (question or "").lower()
    if len(q) < 20:  # too short to be a multi-step ask
        return False
    for pat in _AGENT_TRIGGER_PATTERNS:
        if re.search(pat, q):
            return True
    return False


def _detect_intent(question: str) -> str:
    """Return one of: 'draft' | 'translate' | 'review' | 'summarize' | 'research'.

    Greedy first-match within the first 100 chars of the (lowered, trimmed)
    question. Order matters: more-specific buckets are checked first so
    "draft a Section 9 application" wins over "section 9 application".
    """
    q = (question or "").lower().strip()
    head = q[:120]
    for t in _DRAFT_TRIGGERS:
        if t in head:
            return "draft"
    for t in _TRANSLATE_TRIGGERS:
        if t in head:
            return "translate"
    for t in _REVIEW_TRIGGERS:
        if t in head:
            return "review"
    for t in _SUMMARIZE_TRIGGERS:
        if t in head:
            return "summarize"
    # Pure "explain X" without a draft verb → research mode (we want the
    # cited authority for an explainer, not free-form prose).
    return "research"


def _last_assistant_citations(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pull the citations attached to the most recent assistant turn so the
    next turn (especially drafting turns like "letter for this similar
    case") can use them as factual scaffolding."""
    if not history:
        return []
    for m in reversed(history):
        if m.get("role") != "assistant":
            continue
        raw = m.get("citations")
        if not raw:
            continue
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, list) and parsed:
                return parsed
        except Exception:
            continue
    return []


def _with_followups(result: dict, question: str, language: Optional[str]) -> dict:
    """Decorate a chat result with 3 follow-up question suggestions.
    Skipped for refusals (suggesting more refusal-prone questions is
    pointless) and for very short answers. Best-effort; never raises."""
    try:
        if result.get("refused"):
            return result
        ans = result.get("answer_markdown") or ""
        if len(ans.strip()) < 80:
            return result
        followups = generate_followups(question, ans, language=language)
        if followups:
            result["followups"] = followups
    except Exception as e:
        logger.warning("followups decoration failed: %s", e)
    return result


@app.post("/api/brief/chat")
def api_brief_chat(
    body: ChatBody,
    request: Request,
    ls_session: Optional[str] = Cookie(default=None),
):
    user = _require_user(ls_session)
    ip = _client_ip(request)
    if not auth.rate_limit("chat", ip, max_hits=30, window_s=60):
        raise HTTPException(429, "You're asking very quickly. Breathe.")

    # Ownership check + history pull in one shot
    history = auth.get_thread_messages(body.thread_id, user["id"])
    if history is None:
        raise HTTPException(404, "Thread not found.")

    # ── Chitchat fast-path (BEFORE input_guards) ─────────────────────────
    # "hi", "thanks", "who are you", etc. should never refuse for length.
    # Bypass guards, retrieval, validation. Just a warm one-liner.
    if _is_chitchat(body.question):
        prior_cites = _last_assistant_citations(history or [])
        result = answer_chitchat(body.question, history or [], prior_citations=prior_cites, language=body.language, prefer=body.prefer)
        result["intent"] = "chitchat"
        auth.append_message(body.thread_id, "user", body.question, None)
        auth.append_message(
            body.thread_id,
            "assistant",
            result["answer_markdown"],
            serialize_citations(result.get("citations") or []),
        )
        return JSONResponse(result)

    # ── Auto-agent re-route ──────────────────────────────────────────────
    # Multi-step asks ("find X and draft Y", "send the draft to foo@bar")
    # belong on the agent loop even if the user didn't toggle the wand.
    if _looks_like_agent_task(body.question):
        prior_cites = _last_assistant_citations(history or [])
        juris = (body.jurisdiction or "").strip().upper() or None
        result = answer_agent(
            body.question,
            history or [],
            jurisdiction=juris,
            prior_citations=prior_cites,
            user_id=user.get("id"),
            language=body.language,
            prefer=body.prefer,
        )
        result["intent"] = "agent"
        result["auto_routed"] = True
        auth.append_message(body.thread_id, "user", body.question, None)
        auth.append_message(
            body.thread_id,
            "assistant",
            result["answer_markdown"],
            serialize_citations(result.get("citations") or []),
        )
        return JSONResponse(_with_followups(result, body.question, body.language))

    # ── Input guardrails: length / injection / scope / PII redaction
    guard = input_guards.check(body.question, history_len=len(history or []))
    if not guard.allow:
        # Persist the user turn so the refusal is visible in the thread,
        # then return a structured refusal envelope (no LLM call, no retrieval).
        auth.append_message(body.thread_id, "user", body.question, None)
        auth.append_message(
            body.thread_id,
            "assistant",
            guard.refusal_message,
            serialize_citations([]),
        )
        return JSONResponse({
            "answer_markdown": guard.refusal_message,
            "citations": [],
            "llm": {"provider": "guard", "model": "input_guards", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": False, "confidence": 0.0, "reasons": [guard.reason]},
            "refused": True,
            "guard": guard.to_dict(),
        })

    safe_question = guard.redacted_question or body.question

    # ── Intent routing (Harvey-style) ────────────────────────────────────
    # Drafting / translation / review / summarization requests don't need
    # retrieval gates. We detect them up-front and route to `answer_open`
    # so "draft me a letter to police for this similar case" gets a real
    # police-complaint draft instead of a 67%-grounded REFUSED.
    intent = _detect_intent(safe_question)
    if intent in ("draft", "translate", "review", "summarize"):
        prior_cites = _last_assistant_citations(history or [])
        result = answer_open(
            safe_question,
            history or [],
            prior_citations=prior_cites,
            language=body.language,
            prefer=body.prefer,
        )
        result["intent"] = intent
        if guard.notes:
            result["guard"] = guard.to_dict()
        auth.append_message(body.thread_id, "user", safe_question, None)
        auth.append_message(
            body.thread_id,
            "assistant",
            result["answer_markdown"],
            serialize_citations(result.get("citations") or []),
        )
        return JSONResponse(result)

    # Retrieve grounding hits from the BM25 index if available, else fall back
    # to the in-memory seed corpus so preview/dev environments can still answer.
    hits: list[dict] = []
    idx = _ensure_bm25()
    if idx is not None:
        try:
            results = idx.query(safe_question, k=6, tier=None)
            hits = [doc_to_retrieve_hit(d, s, safe_question) for d, s in results]  # type: ignore[misc]
        except Exception as e:
            logger.warning("BM25 query failed in /api/brief/chat: %s", e)
    if not hits:
        # Hybrid live retrieval: Indian Kanoon + eCourts + e-Gov Japan + web +
        # seed_corpus (always-on safety net). Connectors without API keys
        # silently no-op so the chain degrades gracefully.
        try:
            import connectors
            juris = (body.jurisdiction or "").strip().upper() or None
            srcs = body.sources if body.sources else None
            hits = connectors.retrieve_hybrid(
                safe_question,
                jurisdiction=juris,
                sources=srcs,
                k=6,
            )
            if hits:
                by_src: dict[str, int] = {}
                for h in hits:
                    by_src[h.get("source", "?")] = by_src.get(h.get("source", "?"), 0) + 1
                logger.info(
                    "hybrid retrieve for %r: %d hits from %s",
                    safe_question[:60], len(hits), by_src,
                )
        except Exception as e:
            logger.warning("connectors.retrieve_hybrid failed: %s", e)
            # Last-resort: seed_corpus alone
            try:
                import seed_corpus
                juris = (body.jurisdiction or "").strip().upper() or None
                hits = seed_corpus.query(safe_question, k=6, jurisdiction=juris)
            except Exception as ee:
                logger.warning("seed_corpus last-resort failed: %s", ee)

    # Compose the grounded answer (LLM or fallback).
    result = answer_question(safe_question, hits, history or [], language=body.language, prefer=body.prefer)
    if guard.notes:
        result["guard"] = guard.to_dict()

    # Persist user + assistant turns. Store the REDACTED question — never
    # write Aadhaar/PAN/etc. to the DB.
    auth.append_message(body.thread_id, "user", safe_question, None)
    auth.append_message(
        body.thread_id,
        "assistant",
        result["answer_markdown"],
        serialize_citations(result.get("citations") or []),
    )

    return JSONResponse(_with_followups(result, safe_question, body.language))


# ─────────────────────────────────────────────────────────────────────────
# BRIEF — drafting + web modes (Harvey-style action toggles in the UI)
# ─────────────────────────────────────────────────────────────────────────

@app.post("/api/brief/draft")
def api_brief_draft(
    body: ChatBody,
    request: Request,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Open-drafting mode. No retrieval, no citation gates — Gemini-first
    direct generation. Used for `Canvas` toggle in the prompt box: contracts,
    translations, plain-English explanations, memo skeletons.
    """
    user = _require_user(ls_session)
    ip = _client_ip(request)
    if not auth.rate_limit("draft", ip, max_hits=20, window_s=60):
        raise HTTPException(429, "Drafting too quickly. Pace yourself.")

    history = auth.get_thread_messages(body.thread_id, user["id"])
    if history is None:
        raise HTTPException(404, "Thread not found.")

    guard = input_guards.check(body.question, history_len=len(history or []))
    if not guard.allow:
        auth.append_message(body.thread_id, "user", body.question, None)
        auth.append_message(body.thread_id, "assistant", guard.refusal_message, serialize_citations([]))
        return JSONResponse({
            "answer_markdown": guard.refusal_message,
            "citations": [],
            "llm": {"provider": "guard", "model": "input_guards", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": False, "confidence": 0.0, "reasons": [guard.reason]},
            "refused": True,
            "mode": "draft",
            "guard": guard.to_dict(),
        })

    safe_question = guard.redacted_question or body.question
    result = answer_open(safe_question, history or [], language=body.language, prefer=body.prefer)
    if guard.notes:
        result["guard"] = guard.to_dict()

    auth.append_message(body.thread_id, "user", safe_question, None)
    auth.append_message(
        body.thread_id,
        "assistant",
        result["answer_markdown"],
        serialize_citations(result.get("citations") or []),
    )
    return JSONResponse(_with_followups(result, safe_question, body.language))


@app.post("/api/brief/web")
def api_brief_web(
    body: ChatBody,
    request: Request,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Web-search mode. Pulls fresh snippets via Serper → Tavily → DDG and
    answers strictly from those snippets with G1+G2+G3 validation.
    Used for `Search` toggle in the prompt box."""
    user = _require_user(ls_session)
    ip = _client_ip(request)
    if not auth.rate_limit("web", ip, max_hits=15, window_s=60):
        raise HTTPException(429, "Web research too quickly. Pace yourself.")

    history = auth.get_thread_messages(body.thread_id, user["id"])
    if history is None:
        raise HTTPException(404, "Thread not found.")

    guard = input_guards.check(body.question, history_len=len(history or []))
    if not guard.allow:
        auth.append_message(body.thread_id, "user", body.question, None)
        auth.append_message(body.thread_id, "assistant", guard.refusal_message, serialize_citations([]))
        return JSONResponse({
            "answer_markdown": guard.refusal_message,
            "citations": [],
            "llm": {"provider": "guard", "model": "input_guards", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": False, "confidence": 0.0, "reasons": [guard.reason]},
            "refused": True,
            "mode": "web",
            "guard": guard.to_dict(),
        })

    safe_question = guard.redacted_question or body.question

    snippets: list[dict] = []
    try:
        import connectors
        snippets = connectors.web_search_snippets(safe_question, k=6)
    except Exception as e:
        logger.warning("web_search_snippets failed: %s", e)

    result = answer_web(safe_question, snippets, history or [], language=body.language, prefer=body.prefer)
    if guard.notes:
        result["guard"] = guard.to_dict()

    auth.append_message(body.thread_id, "user", safe_question, None)
    auth.append_message(
        body.thread_id,
        "assistant",
        result["answer_markdown"],
        serialize_citations(result.get("citations") or []),
    )
    return JSONResponse(_with_followups(result, safe_question, body.language))


@app.post("/api/brief/agent")
def api_brief_agent(
    body: ChatBody,
    request: Request,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Tool-using agent mode (Harvey-style). Gemini drives a multi-turn loop
    over `retrieve_cases`, `retrieve_statutes`, `web_search`,
    `redline_contract`, and `translate`, then composes a final answer with
    accumulated citations and a tool-call trace the UI renders as
    breadcrumbs.
    """
    user = _require_user(ls_session)
    ip = _client_ip(request)
    if not auth.rate_limit("agent", ip, max_hits=10, window_s=60):
        raise HTTPException(429, "Agent calls too quickly. Pace yourself.")

    history = auth.get_thread_messages(body.thread_id, user["id"])
    if history is None:
        raise HTTPException(404, "Thread not found.")

    guard = input_guards.check(body.question, history_len=len(history or []))
    if not guard.allow:
        auth.append_message(body.thread_id, "user", body.question, None)
        auth.append_message(body.thread_id, "assistant", guard.refusal_message, serialize_citations([]))
        return JSONResponse({
            "answer_markdown": guard.refusal_message,
            "citations": [],
            "llm": {"provider": "guard", "model": "input_guards", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": False, "confidence": 0.0, "reasons": [guard.reason]},
            "trace": [],
            "refused": True,
            "mode": "agent",
            "guard": guard.to_dict(),
        })

    safe_question = guard.redacted_question or body.question
    juris = (body.jurisdiction or "").strip().upper() or None
    prior_cites = _last_assistant_citations(history or [])

    result = answer_agent(
        safe_question,
        history or [],
        jurisdiction=juris,
        prior_citations=prior_cites,
        user_id=user.get("id"),
        language=body.language,
        prefer=body.prefer,
    )
    if guard.notes:
        result["guard"] = guard.to_dict()

    auth.append_message(body.thread_id, "user", safe_question, None)
    auth.append_message(
        body.thread_id,
        "assistant",
        result["answer_markdown"],
        serialize_citations(result.get("citations") or []),
    )
    return JSONResponse(_with_followups(result, safe_question, body.language))


# ─────────────────────────────────────────────────────────────────────────
# COURT SEARCH — public BM25-backed case browser. Powers the new
# `/app` Court Search pane (Search + Latest tabs). All three endpoints
# are session-gated; the index itself is loaded once at startup via
# `_ensure_bm25()` and shared across requests.
# ─────────────────────────────────────────────────────────────────────────


@app.get("/api/cases/search")
def api_cases_search(
    q: str = Query(default="", max_length=300),
    jurisdiction: Optional[str] = Query(default=None, max_length=4),
    tier: Optional[str] = Query(default=None, max_length=8),
    k: int = Query(default=20, ge=1, le=50),
    ls_session: Optional[str] = Cookie(default=None),
):
    _require_user(ls_session)
    idx = _ensure_bm25()
    if idx is None:
        raise HTTPException(503, f"Case index not ready ({_bm25_load_error or 'loading'})")
    if not q.strip():
        return JSONResponse({"hits": [], "total_index_size": len(idx), "stats": idx.stats()})
    juris = (jurisdiction or "").strip().upper() or None
    tier_norm = (tier or "").strip().upper() or None
    hits = idx.query(q, k=k, jurisdiction=juris, tier=tier_norm)
    return JSONResponse({
        "hits": [doc_to_retrieve_hit(d, s, q) for d, s in hits],  # type: ignore[misc]
        "total_index_size": len(idx),
        "stats": idx.stats(),
    })


@app.get("/api/cases/latest")
def api_cases_latest(
    jurisdiction: Optional[str] = Query(default=None, max_length=4),
    k: int = Query(default=20, ge=1, le=50),
    ls_session: Optional[str] = Cookie(default=None),
):
    _require_user(ls_session)
    idx = _ensure_bm25()
    if idx is None:
        raise HTTPException(503, f"Case index not ready ({_bm25_load_error or 'loading'})")
    juris = (jurisdiction or "").strip().upper() or None
    docs = idx.latest(jurisdiction=juris, k=k)
    return JSONResponse({
        "hits": [doc_to_retrieve_hit(d, 0.0, "") for d in docs],  # type: ignore[misc]
        "total_index_size": len(idx),
        "stats": idx.stats(),
    })


@app.get("/api/cases/{case_id}")
def api_cases_detail(
    case_id: str,
    ls_session: Optional[str] = Cookie(default=None),
):
    _require_user(ls_session)
    idx = _ensure_bm25()
    if idx is None:
        raise HTTPException(503, f"Case index not ready ({_bm25_load_error or 'loading'})")
    d = idx.get(case_id)
    if d is None:
        raise HTTPException(404, f"case_id {case_id} not in index")
    return JSONResponse({
        "case_id": d.case_id,
        "title": d.title,
        "court": d.court,
        "year": d.year,
        "citation": d.citation,
        "tier": d.tier,
        "jurisdiction": d.jurisdiction,
        "url": d.url,
        "source": d.source,
        "added_at": d.added_at,
        "text": d.text[:8000],
        "extra": d.extra or {},
    })


@app.get("/api/brief/threads/search")
def api_threads_search(
    q: str = Query(default="", max_length=200),
    ls_session: Optional[str] = Cookie(default=None),
):
    """Search across the current user's message history. Empty q → recent."""
    user = _require_user(ls_session)
    rows = auth.search_user_messages(user["id"], q, limit=30)
    return {"q": q, "results": rows}


# ─────────────────────────────────────────────────────────────────────────
# SETTINGS — per-connector API keys (DB-backed)
# ─────────────────────────────────────────────────────────────────────────

# Stable list shown in the Settings pane. Order matters — IN/JP/SG first
# (most demanded), then web providers, then long-tail Asia.
_SETTINGS_KEY_CATALOG = [
    {"name": "indian_kanoon", "label": "Indian Kanoon",      "country": "IN", "country_label": "India",       "kind": "case",  "free": False},
    {"name": "ecourts",       "label": "eCourts India",      "country": "IN", "country_label": "India",       "kind": "case",  "free": False},
    {"name": "lawnet_sg",     "label": "LawNet (SAL)",       "country": "SG", "country_label": "Singapore",   "kind": "case",  "free": False},
    {"name": "dubai_pulse",   "label": "Dubai Pulse",        "country": "AE", "country_label": "UAE",          "kind": "statute", "free": False},
    {"name": "clj",           "label": "CLJ Law",            "country": "MY", "country_label": "Malaysia",    "kind": "case",  "free": False},
    {"name": "serper",        "label": "Serper.dev (Google)", "country": "*",  "country_label": "Web search",  "kind": "web",   "free": False},
    {"name": "tavily",        "label": "Tavily",             "country": "*",  "country_label": "Web search",  "kind": "web",   "free": False},
    {"name": "sarvam",        "label": "Sarvam AI (translate)", "country": "IN", "country_label": "India",     "kind": "translate", "free": True},
]


class SetKeyBody(BaseModel):
    name: str = Field(..., min_length=2, max_length=64)
    key: str = Field(..., min_length=4, max_length=512)
    note: Optional[str] = Field(default="", max_length=200)


@app.get("/api/settings/keys")
def api_settings_keys(ls_session: Optional[str] = Cookie(default=None)):
    """Return the catalog with current state per connector. Never returns plaintext."""
    user = _require_user(ls_session)
    stored = {row["name"]: row for row in auth.list_connector_keys()}
    out = []
    for entry in _SETTINGS_KEY_CATALOG:
        s = stored.get(entry["name"])
        out.append({
            **entry,
            "has_key": bool(s and s["has_key"]),
            "masked_tail": (s["masked_tail"] if s else ""),
            "set_at": (s["set_at"] if s else None),
        })
    return {"user": {"email": user["email"]}, "keys": out}


@app.post("/api/settings/keys")
def api_settings_set_key(
    body: SetKeyBody,
    ls_session: Optional[str] = Cookie(default=None),
):
    user = _require_user(ls_session)
    valid_names = {e["name"] for e in _SETTINGS_KEY_CATALOG}
    if body.name not in valid_names:
        raise HTTPException(400, f"Unknown connector: {body.name}")
    auth.set_connector_key(body.name, body.key, user_id=user["id"], note=body.note or "")
    _dash_log(user.get("email"), "set_key", target=body.name)
    return {"ok": True, "name": body.name}


@app.delete("/api/settings/keys/{name}")
def api_settings_delete_key(
    name: str,
    ls_session: Optional[str] = Cookie(default=None),
):
    actor = _require_user(ls_session)
    ok = auth.delete_connector_key(name)
    if ok:
        _dash_log(actor.get("email"), "delete_key", target=name)
    return {"ok": ok, "name": name}


# ─────────────────────────────────────────────────────────────────────────
# DASHBOARD — in-house admin pane. Lives at /app inside the same shell
# as Assistant / Court Search etc. Endpoints below back the React
# `DashboardPane`. Every mutation also fans out via Socket.io so two
# admins watching the same screen stay in sync.
# ─────────────────────────────────────────────────────────────────────────


def _dash_log(actor: str | None, action: str, target: str | None = None,
              payload: Optional[dict[str, Any]] = None) -> None:
    """Append to dash_activity and broadcast over socketio. Best-effort —
    a logging failure must never break the underlying mutation."""
    import json as _json
    try:
        import db_adapter  # type: ignore
        now = int(time.time())
        row = {
            "actor": actor or "system",
            "action": action,
            "target": target,
            "payload": _json.dumps(payload or {}, ensure_ascii=False),
            "created_at": now,
        }
        with db_adapter.db() as conn:
            db_adapter.execute(
                conn,
                "INSERT INTO dash_activity (actor, action, target, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (row["actor"], row["action"], row["target"], row["payload"], row["created_at"]),
            )
        try:
            import realtime  # type: ignore
            realtime.broadcast("activity:append", row)
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        logger.warning("_dash_log failed: %s", e)


@app.get("/api/dashboard/stats")
def api_dashboard_stats(ls_session: Optional[str] = Cookie(default=None)):
    """High-level numbers shown on the dashboard's Stats widget — total
    users, threads, messages, library docs, BM25 corpus size, breakdown
    by jurisdiction. Cheap aggregate queries, no scans."""
    _require_user(ls_session)
    out: dict[str, Any] = {"users": 0, "threads": 0, "messages": 0, "library": 0}
    try:
        with auth.db() as conn:
            for col, sql in (
                ("users",    "SELECT COUNT(*) FROM access_codes WHERE revoked_at IS NULL"),
                ("threads",  "SELECT COUNT(*) FROM threads"),
                ("messages", "SELECT COUNT(*) FROM messages"),
                ("library",  "SELECT COUNT(*) FROM library_docs"),
            ):
                try:
                    out[col] = int(conn.execute(sql).fetchone()[0])
                except Exception:  # noqa: BLE001  (table may not exist yet)
                    out[col] = 0
    except Exception as e:  # noqa: BLE001
        logger.warning("dashboard.stats: auth.db failed: %s", e)

    # BM25 / corpus stats — read straight off the loaded index.
    idx = _ensure_bm25() if _RETRIEVAL_AVAILABLE else None
    if idx is not None:
        out["bm25"] = idx.stats()
    else:
        out["bm25"] = {"total": 0, "by_jurisdiction": {}, "by_source": {}}
    return out


@app.get("/api/dashboard/users")
def api_dashboard_users(ls_session: Optional[str] = Cookie(default=None)):
    """User list — name, email, when they were issued an access code,
    last-seen via most-recent message authored. No plaintext code here."""
    _require_user(ls_session)
    rows: list[dict[str, Any]] = []
    try:
        with auth.db() as conn:
            cur = conn.execute(
                """SELECT
                       ac.id, ac.email, ac.name, ac.created_at, ac.revoked_at,
                       (SELECT MAX(m.created_at) FROM messages m
                          JOIN threads t ON t.id = m.thread_id
                         WHERE t.user_id = ac.id) AS last_seen,
                       (SELECT COUNT(*) FROM threads t WHERE t.user_id = ac.id) AS thread_count
                     FROM access_codes ac
                    ORDER BY ac.created_at DESC
                    LIMIT 200"""
            )
            for r in cur.fetchall():
                rows.append({
                    "id": r["id"],
                    "email": r["email"],
                    "name": r["name"],
                    "created_at": r["created_at"],
                    "revoked_at": r["revoked_at"],
                    "last_seen": r["last_seen"],
                    "thread_count": r["thread_count"] or 0,
                })
    except Exception as e:  # noqa: BLE001
        logger.warning("dashboard.users failed: %s", e)
    return {"users": rows}


@app.post("/api/dashboard/users/{user_id}/revoke")
def api_dashboard_revoke_user(user_id: int, ls_session: Optional[str] = Cookie(default=None)):
    actor = _require_user(ls_session)
    now = int(time.time())
    with auth.db() as conn:
        conn.execute(
            "UPDATE access_codes SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
            (now, user_id),
        )
    _dash_log(actor.get("email"), "revoke_access", target=str(user_id),
              payload={"user_id": user_id})
    return {"ok": True, "user_id": user_id, "revoked_at": now}


@app.get("/api/dashboard/activity")
def api_dashboard_activity(
    limit: int = Query(default=50, ge=1, le=200),
    ls_session: Optional[str] = Cookie(default=None),
):
    _require_user(ls_session)
    rows: list[dict[str, Any]] = []
    try:
        import db_adapter  # type: ignore
        with db_adapter.db() as conn:
            rows = db_adapter.fetch_all(
                conn,
                "SELECT id, actor, action, target, payload, created_at "
                "FROM dash_activity ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("dashboard.activity failed: %s", e)
    return {"activity": rows}


@app.get("/api/dashboard/system")
def api_dashboard_system(ls_session: Optional[str] = Cookie(default=None)):
    """One-page health snapshot: DB mode, BM25 status, LLM router state,
    web search reachability. Used for the System widget."""
    _require_user(ls_session)
    out: dict[str, Any] = {}

    # DB
    try:
        import db_adapter  # type: ignore
        out["db"] = db_adapter.status()
    except Exception as e:  # noqa: BLE001
        out["db"] = {"ok": False, "error": str(e)}

    # BM25
    idx = _ensure_bm25() if _RETRIEVAL_AVAILABLE else None
    out["bm25"] = {
        "available": _RETRIEVAL_AVAILABLE,
        "enabled": BM25_ENABLED,
        "loaded": idx is not None,
        "loading": _bm25_loading,
        "doc_count": (len(idx) if idx is not None else 0),
        "load_error": _bm25_load_error,
    }

    # LLM router — providers with creds set
    out["llm"] = {
        "gemini":     bool((os.environ.get("GEMINI_API_KEY") or "").strip()),
        "anthropic":  bool((os.environ.get("ANTHROPIC_API_KEY") or "").strip()),
        "groq":       bool((os.environ.get("GROQ_API_KEY") or "").strip()),
        "cloudflare": bool((os.environ.get("CF_API_TOKEN") or "").strip()),
    }

    # Web search providers — keys present?
    out["web"] = {
        "tavily": bool((os.environ.get("TAVILY_API_KEY") or "").strip()),
        "serper": bool((os.environ.get("SERPER_API_KEY") or "").strip()),
        "ddg":    True,
    }

    return out


# ─────────────────────────────────────────────────────────────────────────
# GOOGLE WORKSPACE — OAuth + Docs/Gmail/Sheets/Drive
# ─────────────────────────────────────────────────────────────────────────

# In-memory CSRF nonce store. Map state → user_id (kept tiny; expires on
# restart; OAuth flow is seconds long so this is fine for v1).
_GOOGLE_OAUTH_STATES: dict[str, dict[str, Any]] = {}


@app.get("/api/google/status")
def api_google_status(ls_session: Optional[str] = Cookie(default=None)):
    user = _require_user(ls_session)
    import google_service
    return google_service.status_for_user(user["id"])


@app.get("/api/google/oauth/start")
def api_google_oauth_start(
    request: Request,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Issue a CSRF-state, redirect the user to Google's consent screen."""
    user = _require_user(ls_session)
    import google_service
    if not google_service.is_configured():
        raise HTTPException(503, "Google OAuth isn't configured on the server. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.")
    import secrets as _secrets
    state = _secrets.token_urlsafe(24)
    _GOOGLE_OAUTH_STATES[state] = {"user_id": user["id"], "ts": time.time()}
    # Garbage-collect old states (> 10 min).
    cutoff = time.time() - 600
    for k in [k for k, v in _GOOGLE_OAUTH_STATES.items() if v.get("ts", 0) < cutoff]:
        _GOOGLE_OAUTH_STATES.pop(k, None)
    url = google_service.oauth_authorize_url(state)
    return RedirectResponse(url, status_code=302)


@app.get("/api/google/oauth/callback")
def api_google_oauth_callback(
    code: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None),
    error: Optional[str] = Query(default=None),
):
    """Google redirects back here. Exchange code → tokens, persist, then
    bounce the user back to /app/settings with a success/failure marker.
    """
    if error:
        return RedirectResponse(f"/app?google=denied&reason={urllib.parse.quote(error)}", status_code=302)
    if not code or not state:
        raise HTTPException(400, "Missing code or state.")
    entry = _GOOGLE_OAUTH_STATES.pop(state, None)
    if not entry:
        raise HTTPException(400, "Unknown or expired state. Restart the connect flow.")
    user_id = entry["user_id"]

    import google_service
    try:
        token = google_service.oauth_exchange_code(code)
    except Exception as e:
        logger.error("google oauth exchange failed: %s", e)
        return RedirectResponse(f"/app?google=error&reason={urllib.parse.quote(str(e)[:200])}", status_code=302)

    expiry = int(time.time()) + int(token.get("expires_in", 3600))
    auth.google_save_tokens(
        user_id=user_id,
        access_token=token["access_token"],
        refresh_token=token.get("refresh_token"),
        expiry=expiry,
        scopes=token.get("scope") or " ".join(google_service.SCOPES),
        google_email=token.get("_email"),
    )
    return RedirectResponse("/app?google=connected", status_code=302)


@app.post("/api/google/disconnect")
def api_google_disconnect(ls_session: Optional[str] = Cookie(default=None)):
    user = _require_user(ls_session)
    import google_service
    ok = google_service.revoke(user["id"])
    return {"ok": ok}


# ─────────────────────────────────────────────────────────────────────────
# NYAYASATHI CLIENTS — inbound consumer leads from WhatsApp / voice / web
# ─────────────────────────────────────────────────────────────────────────
# Public intake (token-protected) lets the NyayaSathi consumer surface push
# leads in. Lawyer-side endpoints (session-gated) let the practitioner
# triage the inbox, assign, take notes, and "Open as thread" — which spins
# up a chat thread pre-loaded with the intake summary so the lawyer starts
# at minute 1 instead of minute 0.

NYAYA_INTAKE_TOKEN = os.environ.get("NYAYA_INTAKE_TOKEN", "")


class NyayaIntakeBody(BaseModel):
    source: str = Field(..., max_length=20)         # 'whatsapp' | 'voice' | 'web' | 'manual'
    name: Optional[str] = Field(default=None, max_length=160)
    phone: Optional[str] = Field(default=None, max_length=40)
    email: Optional[str] = Field(default=None, max_length=200)
    language: Optional[str] = Field(default=None, max_length=8)
    jurisdiction: Optional[str] = Field(default=None, max_length=8)
    intake_summary: Optional[str] = Field(default=None, max_length=4000)
    intake_transcript: Optional[str] = Field(default=None, max_length=40000)
    notes: Optional[str] = Field(default=None, max_length=2000)


@app.post("/api/nyaya/intake")
def api_nyaya_intake(
    body: NyayaIntakeBody,
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    """Inbound webhook from the NyayaSathi consumer surface. Auth: bearer
    token via NYAYA_INTAKE_TOKEN env. If unset, the endpoint is closed —
    no anonymous lead-spam. Rate-limited per-IP."""
    if not NYAYA_INTAKE_TOKEN:
        raise HTTPException(403, "intake disabled (set NYAYA_INTAKE_TOKEN)")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    if authorization.split(" ", 1)[1].strip() != NYAYA_INTAKE_TOKEN:
        raise HTTPException(403, "invalid intake token")
    ip = _client_ip(request)
    if not auth.rate_limit("nyaya_intake", ip, max_hits=120, window_s=60):
        raise HTTPException(429, "intake rate limit")
    cid = auth.nyaya_create_client(
        source=body.source,
        name=body.name,
        phone=body.phone,
        email=body.email,
        language=body.language,
        jurisdiction=body.jurisdiction,
        intake_summary=body.intake_summary,
        intake_transcript=body.intake_transcript,
        notes=body.notes,
    )
    logger.info("nyaya intake: client_id=%d source=%s juris=%s", cid, body.source, body.jurisdiction)
    return {"ok": True, "client_id": cid}


@app.get("/api/clients")
def api_list_clients(
    status: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    ls_session: Optional[str] = Cookie(default=None),
):
    """Lawyer's inbox view. Filters: status (new|in_progress|closed),
    q (free-text on name/phone/email/summary). Always scoped to either
    leads explicitly assigned to this user OR unassigned leads (so a
    solo practitioner sees everything by default)."""
    user = _require_user(ls_session)
    rows = auth.nyaya_list_clients(
        status=status,
        assigned_user_id=user["id"],
        q=q,
    )
    counts = auth.nyaya_count_by_status(assigned_user_id=user["id"])
    return {"clients": rows, "counts": counts}


@app.get("/api/clients/{client_id}")
def api_get_client(
    client_id: int,
    ls_session: Optional[str] = Cookie(default=None),
):
    user = _require_user(ls_session)
    row = auth.nyaya_get_client(client_id)
    if not row:
        raise HTTPException(404, "Client not found.")
    # Scope check: must be either unassigned or assigned to this user.
    if row.get("assigned_user_id") and row["assigned_user_id"] != user["id"]:
        raise HTTPException(403, "Not your client.")
    return row


class ClientPatchBody(BaseModel):
    status: Optional[str] = Field(default=None, max_length=20)
    notes: Optional[str] = Field(default=None, max_length=4000)
    assign_to_me: Optional[bool] = Field(default=None)


@app.patch("/api/clients/{client_id}")
def api_patch_client(
    client_id: int,
    body: ClientPatchBody,
    ls_session: Optional[str] = Cookie(default=None),
):
    user = _require_user(ls_session)
    row = auth.nyaya_get_client(client_id)
    if not row:
        raise HTTPException(404, "Client not found.")
    if row.get("assigned_user_id") and row["assigned_user_id"] != user["id"]:
        raise HTTPException(403, "Not your client.")
    auth.nyaya_update_client(
        client_id,
        status=body.status,
        notes=body.notes,
        assigned_user_id=user["id"] if body.assign_to_me else None,
    )
    return auth.nyaya_get_client(client_id)


@app.post("/api/clients/{client_id}/open")
def api_open_client_thread(
    client_id: int,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Spin up a fresh chat thread pre-loaded with the intake summary as
    the first user message, so the lawyer drops into context immediately.
    Marks the client `in_progress` and stores the new thread_id on the
    client row for back-navigation. Idempotent: if a thread is already
    bound, return that one."""
    user = _require_user(ls_session)
    row = auth.nyaya_get_client(client_id)
    if not row:
        raise HTTPException(404, "Client not found.")
    if row.get("assigned_user_id") and row["assigned_user_id"] != user["id"]:
        raise HTTPException(403, "Not your client.")
    if row.get("thread_id"):
        return {"thread_id": row["thread_id"], "reused": True}

    name = (row.get("name") or "Client").strip() or "Client"
    juris = (row.get("jurisdiction") or "").strip() or "?"
    title = f"{name} · {juris} · NyayaSathi intake"
    thread_id = auth.create_thread(user["id"], title=title[:200])

    intake_lines = [f"# Intake from NyayaSathi ({row.get('source','manual')})"]
    if row.get("name"):    intake_lines.append(f"**Client:** {row['name']}")
    if row.get("phone"):   intake_lines.append(f"**Phone:** {row['phone']}")
    if row.get("email"):   intake_lines.append(f"**Email:** {row['email']}")
    if row.get("language"):intake_lines.append(f"**Preferred language:** {row['language']}")
    if row.get("jurisdiction"): intake_lines.append(f"**Jurisdiction:** {row['jurisdiction']}")
    if row.get("intake_summary"):
        intake_lines.append("")
        intake_lines.append("## Summary")
        intake_lines.append(row["intake_summary"])
    if row.get("intake_transcript"):
        intake_lines.append("")
        intake_lines.append("## Transcript")
        intake_lines.append(row["intake_transcript"][:6000])
    auth.append_message(thread_id, "user", "\n".join(intake_lines), None)

    auth.nyaya_update_client(
        client_id,
        status="in_progress",
        assigned_user_id=user["id"],
        thread_id=thread_id,
    )
    return {"thread_id": thread_id, "reused": False}


# ─────────────────────────────────────────────────────────────────────────
# LIBRARY — curated statutes, contract templates, pleadings
# ─────────────────────────────────────────────────────────────────────────

def _maybe_seed_library() -> None:
    """Idempotent seed loader. Reads library_seed.json on first call when DB empty."""
    try:
        if auth.library_count() > 0:
            return
        seed_path = Path(__file__).parent / "library_seed.json"
        if not seed_path.exists():
            return
        import json as _json
        rows = _json.loads(seed_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            return
        for r in rows:
            try:
                auth.library_insert(
                    jurisdiction=r.get("jurisdiction", ""),
                    kind=r.get("kind", ""),
                    title=r.get("title", ""),
                    body_md=r.get("body_md", ""),
                    source_url=r.get("source_url", ""),
                )
            except Exception as ie:
                logger.warning("library seed row failed: %s", ie)
        logger.info("Library seeded: %d rows", len(rows))
    except Exception as e:
        logger.warning("library seed failed: %s", e)


@app.get("/api/library")
def api_library_list(
    jurisdiction: Optional[str] = None,
    kind: Optional[str] = None,
    ls_session: Optional[str] = Cookie(default=None),
):
    _require_user(ls_session)
    _maybe_seed_library()
    return {
        "items": auth.library_list(jurisdiction, kind),
        "count": auth.library_count(),
    }


@app.get("/api/library/{doc_id}")
def api_library_get(
    doc_id: int,
    ls_session: Optional[str] = Cookie(default=None),
):
    _require_user(ls_session)
    row = auth.library_get(doc_id)
    if not row:
        raise HTTPException(404, "Library document not found.")
    return row


# ─────────────────────────────────────────────────────────────────────────
# VAULT — per-user document upload + multi-doc Q&A
# ─────────────────────────────────────────────────────────────────────────

@app.get("/api/vault/docs")
def api_vault_list(ls_session: Optional[str] = Cookie(default=None)):
    user = _require_user(ls_session)
    return {"docs": auth.vault_list_docs(user["id"])}


@app.post("/api/vault/upload")
async def api_vault_upload(
    request: Request,
    file: UploadFile = File(...),
    ls_session: Optional[str] = Cookie(default=None),
):
    user = _require_user(ls_session)
    ip = _client_ip(request)
    if not auth.rate_limit("vault_upload", ip, max_hits=20, window_s=600):
        raise HTTPException(429, "Too many uploads. Try again in a few minutes.")

    # Size check
    blob = await file.read()
    if len(blob) > vault_service.UPLOAD_MAX_BYTES:
        raise HTTPException(413, f"File exceeds {vault_service.UPLOAD_MAX_BYTES // (1024*1024)} MB limit.")

    fname = (file.filename or "upload").lower()
    if not any(fname.endswith(ext) for ext in vault_service.ALLOWED_EXTS):
        raise HTTPException(415, f"Allowed types: {', '.join(sorted(vault_service.ALLOWED_EXTS))}")

    text = vault_service.extract_text(file.filename or fname, blob)
    if not text.strip():
        raise HTTPException(422, "Could not extract text from this file.")

    doc_id = auth.vault_create_doc(user["id"], file.filename or fname, file.content_type or "", len(blob))
    chunks = vault_service.chunk_document(doc_id, file.filename or fname, text)

    # Embed chunks with Gemini text-embedding-004 so we can do semantic
    # retrieval (cosine sim) on top of BM25. Embedding failures are
    # non-fatal — the chunks still get saved without vectors and search
    # falls back to BM25-only for this doc.
    embedded_count = 0
    try:
        from llm import router as _llm
        if _llm.GEMINI_API_KEY and chunks:
            texts = [(c.get("text") or "")[:4000] for c in chunks]
            vecs = _llm.embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
            for c, v in zip(chunks, vecs):
                c["embedding"] = v
            embedded_count = len(vecs)
    except Exception as e:
        # Log and continue — semantic search just degrades to BM25.
        import logging as _logging
        _logging.getLogger(__name__).warning("vault embed failed for doc %s: %s", doc_id, e)

    n = auth.vault_save_chunks(doc_id, user["id"], chunks)
    return {
        "ok": True,
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks": n,
        "embedded": embedded_count,
    }


@app.delete("/api/vault/docs/{doc_id}")
def api_vault_delete(doc_id: int, ls_session: Optional[str] = Cookie(default=None)):
    user = _require_user(ls_session)
    ok = auth.vault_delete_doc(user["id"], doc_id)
    if not ok:
        raise HTTPException(404, "Document not found.")
    return {"ok": True, "doc_id": doc_id}


class VaultChatBody(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    doc_ids: Optional[list[int]] = None  # None = all docs


@app.post("/api/vault/chat")
def api_vault_chat(
    body: VaultChatBody,
    request: Request,
    ls_session: Optional[str] = Cookie(default=None),
):
    user = _require_user(ls_session)
    ip = _client_ip(request)
    if not auth.rate_limit("vault_chat", ip, max_hits=30, window_s=60):
        raise HTTPException(429, "You're asking very quickly. Breathe.")

    guard = input_guards.check(body.question, history_len=1)  # treat as follow-up (don't gate scope)
    if not guard.allow:
        return JSONResponse({"answer_markdown": guard.refusal_message, "refused": True,
                             "citations": [], "guard": guard.to_dict()})

    chunks = auth.vault_load_chunks(user["id"], body.doc_ids)
    if not chunks:
        return JSONResponse({
            "answer_markdown": "Your vault is empty. Upload a document first (PDF, DOCX, or TXT).",
            "citations": [], "refused": True,
        })

    hits = vault_service.rank_chunks(guard.redacted_question, chunks, k=8)
    result = vault_service.answer_over_vault(guard.redacted_question, hits, [])
    return JSONResponse(result)


# ── Adalat-style doc intelligence ─────────────────────────────────────────
# One-click analysis of a vaulted document: parties, key dates, obligations,
# risks, governing law, and a 3-bullet TL;DR. Single LLM call over the first
# ~12k chars of the doc. Output is structured Markdown so the React side can
# render it as cards or just dump it inline.

class VaultAnalyseBody(BaseModel):
    doc_id: int = Field(..., ge=1)
    language: Optional[str] = Field(default=None, max_length=8)


_VAULT_ANALYSE_PROMPT = """You are Sanhita, a senior associate doing a
fast first-pass review of a document a junior just dropped on your desk.
Output strict Markdown with these exact section headers, in this order.
Skip a section only if the document genuinely doesn't contain that
information — never invent.

## TL;DR
3 bullets. What it is, who's bound, what it does.

## Parties
Each party on its own line as `**Role:** Name (jurisdiction if stated)`.

## Key dates & deadlines
Bullet list. Format `YYYY-MM-DD — what happens`. If only relative
("within 30 days of signing") show it as such.

## Obligations
Numbered list. One sentence per obligation. Tag each as **[party name]**
at the start.

## Risks & red flags
Bullets. Anything unusual: uncapped indemnity, broad IP assignment,
unilateral termination, governing law mismatch, missing arbitration seat,
penal clauses, etc. If clean, write "No material red flags spotted on
first pass — full review recommended for execution."

## Governing law / dispute resolution
One short paragraph. State seat, language of arbitration if specified,
exclusive vs non-exclusive jurisdiction.

## Suggested next steps
3 short bullets, lawyer-to-lawyer voice.

Rules: no preamble, no closing apologies, no "as an AI". If a section is
genuinely empty, write "—" under the header and move on."""


@app.post("/api/vault/analyse")
def api_vault_analyse(
    body: VaultAnalyseBody,
    request: Request,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Adalat-style "Analyse this document" one-click. Returns a structured
    review (parties / dates / obligations / risks / governing law / next
    steps) over the entire stored doc, optionally translated."""
    user = _require_user(ls_session)
    ip = _client_ip(request)
    if not auth.rate_limit("vault_analyse", ip, max_hits=20, window_s=60):
        raise HTTPException(429, "Analysing too quickly.")

    chunks = auth.vault_load_chunks(user["id"], [body.doc_id])
    if not chunks:
        raise HTTPException(404, "Doc not found in your vault.")

    # Concatenate up to ~12k chars from the doc. Leaves room for the system
    # prompt + headers in a 32k context model.
    body_text = ""
    for ch in chunks:
        body_text += (ch.get("content") or "") + "\n\n"
        if len(body_text) > 12_000:
            body_text = body_text[:12_000]
            break

    from llm import router
    import brief_service
    sys_prompt = _VAULT_ANALYSE_PROMPT + brief_service._lang_directive(body.language)
    user_prompt = f"Document content:\n\n{body_text}\n\nAnalyse it now."
    try:
        resp = router.generate(sys_prompt, user_prompt, temperature=0.2, max_tokens=1400)
        return {
            "analysis_markdown": resp.text,
            "doc_id": body.doc_id,
            "llm": resp.to_dict(),
            "char_count": len(body_text),
        }
    except Exception as e:
        logger.error("vault analyse failed: %s", e)
        raise HTTPException(502, f"Analysis failed: {e}")


# ─────────────────────────────────────────────────────────────────────────
# DRAFT — 4 Indian-law templates
# ─────────────────────────────────────────────────────────────────────────

@app.get("/api/draft/templates")
def api_draft_templates(ls_session: Optional[str] = Cookie(default=None)):
    _require_user(ls_session)
    return {"templates": workflows.list_draft_templates()}


class DraftBody(BaseModel):
    template: str = Field(..., min_length=2, max_length=64)
    facts: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/draft")
def api_draft(
    body: DraftBody,
    request: Request,
    ls_session: Optional[str] = Cookie(default=None),
):
    _require_user(ls_session)
    ip = _client_ip(request)
    if not auth.rate_limit("draft", ip, max_hits=20, window_s=300):
        raise HTTPException(429, "Too many draft requests. Wait a few minutes.")
    try:
        return workflows.generate_draft(body.template, body.facts)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("draft failed: %s", e)
        raise HTTPException(500, "Draft generation failed.")


# ─────────────────────────────────────────────────────────────────────────
# REVIEW — contract clause review
# ─────────────────────────────────────────────────────────────────────────

class ReviewBody(BaseModel):
    clauses: list[str] = Field(..., min_length=1, max_length=40)


@app.post("/api/review")
def api_review(
    body: ReviewBody,
    request: Request,
    ls_session: Optional[str] = Cookie(default=None),
):
    _require_user(ls_session)
    ip = _client_ip(request)
    if not auth.rate_limit("review", ip, max_hits=20, window_s=300):
        raise HTTPException(429, "Too many review requests.")
    return workflows.review_contract(body.clauses)


# ─────────────────────────────────────────────────────────────────────────
# TRANSLATE — EN ↔ HI
# ─────────────────────────────────────────────────────────────────────────

class TranslateBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    direction: str = Field(default="en->hi")


@app.post("/api/translate")
def api_translate(
    body: TranslateBody,
    request: Request,
    ls_session: Optional[str] = Cookie(default=None),
):
    _require_user(ls_session)
    ip = _client_ip(request)
    if not auth.rate_limit("translate", ip, max_hits=30, window_s=300):
        raise HTTPException(429, "Too many translation requests.")
    try:
        return workflows.translate(body.text, direction=body.direction)
    except ValueError as e:
        raise HTTPException(400, str(e))


# ─────────────────────────────────────────────────────────────────────────
# CITATOR — judicial history + related cases
# ─────────────────────────────────────────────────────────────────────────

class CitatorBody(BaseModel):
    case_title: str = Field(..., min_length=2, max_length=400)
    excerpt: str = Field(..., min_length=10, max_length=20000)
    holdings: str = Field(default="", max_length=8000)


@app.post("/api/citator")
def api_citator(
    body: CitatorBody,
    request: Request,
    ls_session: Optional[str] = Cookie(default=None),
):
    _require_user(ls_session)
    ip = _client_ip(request)
    if not auth.rate_limit("citator", ip, max_hits=30, window_s=300):
        raise HTTPException(429, "Too many citator requests.")
    return workflows.citator_summary(body.case_title, body.excerpt, body.holdings)


# ─────────────────────────────────────────────────────────────────────────
# REDLINE + GENERIC WORKFLOWS (Harvey-parity for India)
# ─────────────────────────────────────────────────────────────────────────

class RedlineBody(BaseModel):
    text: str = Field(..., min_length=50, max_length=40000)


@app.post("/api/redline")
def api_redline(
    body: RedlineBody,
    request: Request,
    ls_session: Optional[str] = Cookie(default=None),
):
    _require_user(ls_session)
    ip = _client_ip(request)
    if not auth.rate_limit("redline", ip, max_hits=15, window_s=300):
        raise HTTPException(429, "Too many redline requests.")
    return workflows.redline_contract(body.text)


class GenericBody(BaseModel):
    key: str = Field(..., min_length=2, max_length=64)
    text: str = Field(..., min_length=10, max_length=40000)


@app.get("/api/workflows")
def api_workflows_list():
    return {"workflows": workflows.list_generic_workflows()}


@app.post("/api/workflows/run")
def api_workflows_run(
    body: GenericBody,
    request: Request,
    ls_session: Optional[str] = Cookie(default=None),
):
    _require_user(ls_session)
    ip = _client_ip(request)
    if not auth.rate_limit("wf", ip, max_hits=30, window_s=300):
        raise HTTPException(429, "Too many workflow runs.")
    try:
        return workflows.run_generic(body.key, body.text)
    except ValueError as e:
        raise HTTPException(400, str(e))


# ── admin endpoints (bearer-token auth) ───────────────────────────────────

@app.get("/api/admin/requests")
def api_admin_list_requests(
    status: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None),
):
    _require_admin(authorization)
    return {"requests": auth.list_access_requests(status=status)}


@app.post("/api/admin/requests/{request_id}/approve")
def api_admin_approve(
    request_id: int,
    authorization: Optional[str] = Header(default=None),
):
    _require_admin(authorization)
    approved = auth.approve_request(request_id)
    if not approved:
        raise HTTPException(404, "No pending request with that id.")
    # access_code is plaintext — returned ONCE.
    return approved


@app.post("/api/admin/requests/{request_id}/reject")
def api_admin_reject(
    request_id: int,
    authorization: Optional[str] = Header(default=None),
):
    _require_admin(authorization)
    ok = auth.reject_request(request_id)
    if not ok:
        raise HTTPException(404, "No pending request with that id.")
    return {"ok": True, "request_id": request_id, "status": "rejected"}


# ── HTML page routes for Phase 2 ──────────────────────────────────────────

@app.get("/login")
async def serve_login():
    return FileResponse(STATIC_DIR / "login.html", media_type="text/html")


@app.get("/brief")
async def serve_brief(ls_session: Optional[str] = Cookie(default=None)):
    # Gate the Brief page — bounce unauthenticated users to /login.
    uid = auth.verify_session_token(ls_session)
    if uid is None or auth.get_user(uid) is None:
        return RedirectResponse(url="/login", status_code=302)
    return FileResponse(STATIC_DIR / "brief.html", media_type="text/html")


@app.get("/brief.js")
async def serve_brief_js():
    return FileResponse(STATIC_DIR / "brief.js", media_type="application/javascript")


@app.get("/admin")
async def serve_admin():
    # Admin page itself is static; the data endpoints are token-gated.
    return FileResponse(STATIC_DIR / "admin.html", media_type="text/html")


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent


@app.get("/viewer.html")
async def serve_viewer():
    return FileResponse(STATIC_DIR / "viewer.html", media_type="text/html")


@app.get("/style.css")
async def serve_css():
    return FileResponse(STATIC_DIR / "style.css", media_type="text/css")


@app.get("/app.js")
async def serve_js():
    return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")


@app.get("/workspaces.js")
async def serve_workspaces_js():
    return FileResponse(STATIC_DIR / "workspaces.js", media_type="application/javascript")


@app.get("/app-shell.js")
async def serve_app_shell_js():
    return FileResponse(STATIC_DIR / "app-shell.js", media_type="application/javascript")


@app.get("/app")
async def serve_app(ls_session: Optional[str] = Cookie(default=None)):
    uid = auth.verify_session_token(ls_session)
    if uid is None or auth.get_user(uid) is None:
        return RedirectResponse(url="/login", status_code=302)
    return FileResponse(STATIC_DIR / "brief.html", media_type="text/html")


@app.get("/assets/{path:path}")
async def serve_asset(path: str):
    """Serve static assets (logo, firm logos, etc.) from ./assets/."""
    safe_path = (STATIC_DIR / "assets" / path).resolve()
    assets_root = (STATIC_DIR / "assets").resolve()
    # Prevent path traversal
    if assets_root not in safe_path.parents and safe_path != assets_root:
        raise HTTPException(404, "Not found")
    if not safe_path.is_file():
        raise HTTPException(404, "Asset not found")
    ext = safe_path.suffix.lower()
    media = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".ico": "image/x-icon",
    }.get(ext, "application/octet-stream")
    return FileResponse(safe_path, media_type=media)


@app.get("/")
async def serve_index(ls_session: Optional[str] = Cookie(default=None)):
    # Marketing site removed — go straight to the product.
    uid = auth.verify_session_token(ls_session)
    if uid is None or auth.get_user(uid) is None:
        return RedirectResponse(url="/login", status_code=302)
    return RedirectResponse(url="/app", status_code=302)


# ─────────────────────────────────────────────────────────────────────────
# ASGI surface — wrap FastAPI with the Socket.io app so /socket.io/* and
# every other route are served by the same uvicorn process. Launch
# entrypoint becomes `server:socketio_app` instead of `server:app`.
# A bare `server:app` import path still works for code that imports
# the FastAPI instance directly (tests, scripts).
# ─────────────────────────────────────────────────────────────────────────
try:
    import realtime  # type: ignore
    socketio_app = realtime.asgi_app(app)
except Exception as _rt_err:  # noqa: BLE001
    logger.warning("realtime mount failed; running without socket.io: %s", _rt_err)
    socketio_app = app
