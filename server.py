"""
LexSearch – Indian Court Judgments (High Courts + Supreme Court)
FastAPI backend: search API + PDF proxy from public AWS S3.
Run: uvicorn server:app --reload --port 8080
"""

import io
import logging
import os
import re
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
from brief_service import answer_question, serialize_citations, generate_followups
from validators import input_guards
import vault_service
import workflows
import web_signals
import doc_editor
from fastapi import File, UploadFile, Form

# ── Retrieval layer: FTS5 (primary) or BM25 (legacy fallback) ──────────
# The FTS5 adapter queries india_courts.db (22M+ records, <50ms) directly.
# Falls back to the legacy S3-based BM25 if FTS5 is not configured.
_RETRIEVAL_AVAILABLE = False
_FTS5_AVAILABLE = False
BM25Index = None  # type: ignore[assignment]
build_index = None  # type: ignore[assignment]
doc_to_retrieve_hit = None  # type: ignore[assignment]

_SERVER_DIR = Path(__file__).resolve().parent
# DB path resolution: env var > local symlink > sibling repo > /data (Docker)
_DB_CANDIDATES = [
    os.environ.get("INDIA_COURTS_DB", ""),
    str(_SERVER_DIR / "india_courts.db"),  # local symlink
    str(_SERVER_DIR.parent / "india-judgments-corpus" / "india_courts.db"),
    "/data/india_courts.db",  # Railway/Docker volume
]
INDIA_COURTS_DB = next((p for p in _DB_CANDIDATES if p and Path(p).exists()), _DB_CANDIDATES[2])

try:
    import sys as _sys
    _adapter_dir = str(_SERVER_DIR.parent / "india-judgments-corpus" / "scripts")
    if _adapter_dir not in _sys.path:
        _sys.path.insert(0, _adapter_dir)
    from sanhita_adapter import FTS5Index
    _FTS5_AVAILABLE = True
    _RETRIEVAL_AVAILABLE = True
except Exception as _fts5_err:
    FTS5Index = None  # type: ignore[assignment]

# Legacy BM25 fallback (only used if FTS5 adapter not found)
if not _FTS5_AVAILABLE:
    try:
        from retrieval import (
            BM25Index,
            build_index,
            doc_to_retrieve_hit,
        )
        _RETRIEVAL_AVAILABLE = True
    except Exception as _retrieval_err:  # pragma: no cover
        pass

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
# Retrieval layer — FTS5 (primary) or BM25 (legacy fallback)
# ---------------------------------------------------------------------------
BM25_PATH = Path(os.environ.get("LEXSEARCH_BM25_PATH", str(Path(__file__).parent / "bm25.pkl")))
BM25_ENABLED = os.environ.get("LEXSEARCH_BM25_ENABLED", "true").lower() == "true"
BM25_MAX_DOCS = int(os.environ.get("LEXSEARCH_BM25_MAX_DOCS", "0")) or None
BM25_LAZY = os.environ.get("LEXSEARCH_BM25_LAZY", "true").lower() == "true"

_bm25_lock = threading.Lock()
_bm25_index = None  # FTS5Index or BM25Index
_bm25_loading = False
_bm25_load_error: Optional[str] = None
_bm25_loaded_at: float = 0.0


def _load_bm25_blocking() -> None:
    """Load retrieval index: FTS5 (instant, preferred) or BM25 (legacy fallback)."""
    global _bm25_index, _bm25_loading, _bm25_load_error, _bm25_loaded_at

    # ── FTS5 path: instant startup, 22M+ docs, <50ms queries ──
    if _FTS5_AVAILABLE and Path(INDIA_COURTS_DB).exists():
        try:
            _bm25_index = FTS5Index(INDIA_COURTS_DB)
            _bm25_loaded_at = time.time()
            _bm25_load_error = None
            logger.info(
                "FTS5 ready: %d docs from %s (instant startup, multi-table search)",
                len(_bm25_index), INDIA_COURTS_DB,
            )
            return
        except Exception as e:
            logger.warning("FTS5 init failed, falling back to BM25: %s", e)

    # ── Legacy BM25 path: S3-based pickle ──
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
        if hasattr(_bm25_index, 'docs'):
            logger.info("BM25 ready: %d docs", len(_bm25_index.docs))
        else:
            logger.info("BM25 ready")
    except Exception as e:
        _bm25_load_error = f"{type(e).__name__}: {e}"
        logger.error("BM25 load failed: %s", _bm25_load_error)
    finally:
        _bm25_loading = False


def _ensure_bm25():
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

    if _FTS5_AVAILABLE and isinstance(idx, FTS5Index):
        result_hits = idx.search(req.query, limit=req.k)
    else:
        raw = idx.query(req.query, k=req.k, tier=req.tier)
        result_hits = [doc_to_retrieve_hit(d, s, req.query) for d, s in raw]  # type: ignore[misc]
    return JSONResponse(
        {
            "query": req.query,
            "k": req.k,
            "tier": req.tier,
            "count": len(result_hits),
            "hits": result_hits,
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
        raise HTTPException(503, "Index not ready")
    if _FTS5_AVAILABLE and isinstance(idx, FTS5Index):
        case = idx.get(case_id)
        if case:
            return JSONResponse(case)
        raise HTTPException(404, f"case_id {case_id} not found")
    # Legacy BM25 path
    for d in idx.docs:
        if d.case_id == case_id:
            return JSONResponse(
                {
                    "case_id": d.case_id,
                    "court": d.court,
                    "bench": d.bench,
                    "year": d.year,
                    "date": d.date,
                    "title": d.title,
                    "citation": d.citation,
                    "tier": d.tier,
                    "text": {k: _safe_str(v) for k, v in d.source_row.items()
                             if k in ("title", "headnote", "description", "judge",
                                      "petitioner", "respondent", "citation",
                                      "disposal_nature")},
                }
            )
    raise HTTPException(404, f"case_id {case_id} not in index")


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


# ---------------------------------------------------------------------------
# Court Search + Analytics API (powered by india_courts.db FTS5)
# ---------------------------------------------------------------------------

# ── Document-type → optimized query rewrite map ────────────────────────────
# Each doc type gets a keyword prefix prepended to the user query so the
# FTS5 search returns the most relevant case law for that document type.
_DOC_TYPE_QUERY_PREFIX: dict[str, str] = {
    "bail_application":    "bail Section 437 439",
    "anticipatory_bail":   "anticipatory bail Section 438",
    "writ_petition":       "writ petition Article 226 mandamus certiorari",
    "legal_notice":        "notice Section 138 dishonour demand",
    "plaint":              "plaint civil suit decree",
    "written_statement":   "written statement reply defence",
    "consumer_complaint":  "consumer complaint deficiency service",
    "affidavit":           "affidavit sworn statement verification",
    "memo_of_appeal":      "appeal judgment set aside",
    "vakalatnama":         "",
}


@app.get("/api/cases/search")
def api_cases_search(
    q: str = "",
    court_code: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    verdict: Optional[str] = None,
    doc_type: Optional[str] = None,   # ← NEW: filter/boost by document type
    k: int = 20,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Full-text search across 31M+ Indian court records.
    Pass doc_type to automatically boost results relevant to that document type.
    """
    if not ls_session:
        raise HTTPException(401)
    if not auth.verify_session_token(ls_session):
        raise HTTPException(401)
    idx = _ensure_bm25()
    if idx is None or not q:
        return {"hits": [], "total": 0, "engine": "none"}

    # Boost query with doc-type prefix so relevant case law surfaces first
    search_q = q
    if doc_type and doc_type in _DOC_TYPE_QUERY_PREFIX:
        prefix = _DOC_TYPE_QUERY_PREFIX[doc_type]
        if prefix:
            search_q = f"{prefix} {q}"

    if _FTS5_AVAILABLE and isinstance(idx, FTS5Index):
        hits = idx.search(
            search_q, court_code=court_code,
            year_from=year_from, year_to=year_to,
            verdict=verdict, limit=k,
        )
        return {"hits": hits, "total": len(idx), "engine": "fts5", "effective_query": search_q}
    return {"hits": [], "total": 0, "engine": "bm25_no_search"}


@app.get("/api/cases/latest")
def api_cases_latest(
    jurisdiction: str = "IN",
    k: int = 20,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Newest cases in the corpus."""
    if not ls_session:
        raise HTTPException(401)
    if not auth.verify_session_token(ls_session):
        raise HTTPException(401)
    idx = _ensure_bm25()
    if idx is None:
        return {"hits": [], "total": 0}
    if _FTS5_AVAILABLE and isinstance(idx, FTS5Index):
        return {"hits": idx.latest(limit=k), "total": len(idx)}
    return {"hits": [], "total": 0}


@app.get("/api/cases/courts")
def api_cases_courts(ls_session: Optional[str] = Cookie(default=None)):
    """List all courts with case counts for filter dropdowns."""
    if not ls_session:
        raise HTTPException(401)
    if not auth.verify_session_token(ls_session):
        raise HTTPException(401)
    idx = _ensure_bm25()
    if idx is None or not (_FTS5_AVAILABLE and isinstance(idx, FTS5Index)):
        return {"courts": []}
    return {"courts": idx.courts()}


@app.get("/api/cases/verdicts")
def api_cases_verdicts(ls_session: Optional[str] = Cookie(default=None)):
    """List top verdicts with counts for filter dropdowns."""
    if not ls_session:
        raise HTTPException(401)
    if not auth.verify_session_token(ls_session):
        raise HTTPException(401)
    idx = _ensure_bm25()
    if idx is None or not (_FTS5_AVAILABLE and isinstance(idx, FTS5Index)):
        return {"verdicts": []}
    return {"verdicts": idx.verdicts()}


@app.get("/api/cases/related/{case_id}")
def api_related_cases(
    case_id: str,
    limit: int = 10,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Get cases related to a given case via citation graph (1-hop neighbors)."""
    _require_user(ls_session)
    idx = _ensure_bm25()
    if idx is None or not (_FTS5_AVAILABLE and isinstance(idx, FTS5Index)):
        return {"related": []}
    return {"related": idx.related_cases(case_id, limit=min(limit, 50))}


@app.get("/api/cases/doc-types")
def api_doc_types(ls_session: Optional[str] = Cookie(default=None)):
    """Document type distribution (from classifier)."""
    _require_user(ls_session)
    idx = _ensure_bm25()
    if idx is None or not (_FTS5_AVAILABLE and isinstance(idx, FTS5Index)):
        return {"doc_types": []}
    return {"doc_types": idx.doc_type_distribution()}


@app.get("/api/templates")
def api_templates_list(ls_session: Optional[str] = Cookie(default=None)):
    """List all draft templates from database."""
    _require_user(ls_session)
    idx = _ensure_bm25()
    if idx is None or not (_FTS5_AVAILABLE and isinstance(idx, FTS5Index)):
        return {"templates": []}
    return {"templates": idx.draft_templates()}


@app.get("/api/templates/{template_id}")
def api_template_detail(
    template_id: str,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Get a single template with full body text."""
    _require_user(ls_session)
    idx = _ensure_bm25()
    if idx is None or not (_FTS5_AVAILABLE and isinstance(idx, FTS5Index)):
        raise HTTPException(503)
    tmpl = idx.draft_template(template_id)
    if not tmpl:
        raise HTTPException(404, "Template not found")
    return tmpl


@app.get("/api/cases/{case_id}")
def api_case_detail(
    case_id: str,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Single case detail."""
    if not ls_session:
        raise HTTPException(401)
    if not auth.verify_session_token(ls_session):
        raise HTTPException(401)
    idx = _ensure_bm25()
    if idx is None:
        raise HTTPException(503)
    if _FTS5_AVAILABLE and isinstance(idx, FTS5Index):
        case = idx.get(case_id)
        if case:
            return case
    raise HTTPException(404)


@app.get("/api/qa/search")
def api_qa_search(
    q: str = "",
    category: Optional[str] = None,
    k: int = 20,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Fast Legal QA search — queries legal_qa_fts directly (2ms vs 8.8s for full search).

    Bypasses the multi-table FTS5Index.search() and queries only legal_qa_fts
    for instant Q&A results. Returns questions, answers, category, acts cited.
    """
    _require_user(ls_session)
    idx = _ensure_bm25()
    if idx is None or not q:
        return {"hits": [], "total": 0, "engine": "none"}

    if not (_FTS5_AVAILABLE and isinstance(idx, FTS5Index)):
        return {"hits": [], "total": 0, "engine": "unavailable"}

    safe_q = re.sub(r"[^\w\s]", " ", q)
    safe_q = re.sub(r"\s+", " ", safe_q).strip()
    if not safe_q:
        return {"hits": [], "total": 0, "engine": "none"}

    try:
        params = [safe_q, k * 2]
        cat_clause = ""
        if category:
            cat_clause = "AND q.category = ?"
            params.insert(1, category)

        rows = idx.conn.execute(
            f"""SELECT q.qa_id, q.question, q.answer, q.category, q.acts_cited,
                       q.context, bm25(legal_qa_fts) AS score
               FROM legal_qa_fts f
               JOIN legal_qa q ON q.rowid = f.rowid
               WHERE legal_qa_fts MATCH ? {cat_clause}
               ORDER BY score
               LIMIT ?""",
            params,
        ).fetchall()

        hits = []
        for r in rows:
            hits.append({
                "qa_id": r[0] or "",
                "question": (r[1] or "")[:300],
                "answer": (r[2] or "")[:1200],
                "category": r[3] or "",
                "acts_cited": r[4] or "",
                "context": (r[5] or "")[:400],
                "score": abs(r[6]) if r[6] else 0,
                "source": "legal_qa_fts5",
            })

        # Get approximate total
        try:
            total = idx.conn.execute("SELECT MAX(rowid) FROM legal_qa").fetchone()[0] or 0
        except Exception:
            total = len(hits)

        return {"hits": hits[:k], "total": total, "engine": "legal_qa_fts5"}
    except Exception as e:
        logger.warning("QA search error: %s", e)
        return {"hits": [], "total": 0, "engine": "error", "error": str(e)}


@app.get("/api/analytics/judge-profile")
def api_judge_profile(
    judge: str = Query(..., min_length=2, description="Judge name (partial match OK)"),
    court_code: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 50,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Judge-level analytics: verdict breakdown, bail rates, case volume over time.

    Queries the 16M+ judgments table for cases decided by the given judge.
    Partial name matching — 'Chandrachud' matches 'Justice D.Y. Chandrachud'.
    """
    _require_user(ls_session)
    idx = _ensure_bm25()
    if idx is None or not (_FTS5_AVAILABLE and isinstance(idx, FTS5Index)):
        return {"judge": judge, "cases": [], "verdict_breakdown": {}, "total": 0}

    try:
        clauses = ["LOWER(j.judge) LIKE ?"]
        params: list = [f"%{judge.lower()}%"]

        if court_code:
            clauses.append("j.court_code = ?")
            params.append(court_code)
        if year_from:
            clauses.append("j.year >= ?")
            params.append(year_from)
        if year_to:
            clauses.append("j.year <= ?")
            params.append(year_to)

        where = " AND ".join(clauses)
        params.append(min(limit, 200))

        rows = idx.conn.execute(
            f"""SELECT j.cnr, j.title, j.court, j.year, j.verdict, j.judge,
                       j.date_decided, j.description
               FROM judgments j
               WHERE {where}
               ORDER BY j.year DESC, j.id DESC
               LIMIT ?""",
            params,
        ).fetchall()

        cases = []
        verdict_counts: dict[str, int] = {}
        court_counts: dict[str, int] = {}
        year_counts: dict[int, int] = {}

        for r in rows:
            verdict = (r[4] or "").strip()
            court = (r[2] or "")
            year = r[3]
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            court_counts[court] = court_counts.get(court, 0) + 1
            if year:
                year_counts[int(year)] = year_counts.get(int(year), 0) + 1
            cases.append({
                "cnr": r[0] or "",
                "title": r[1] or (r[7] or "")[:120] or r[0] or "",
                "court": court,
                "year": year,
                "verdict": verdict,
                "judge": r[5] or "",
                "date_decided": r[6] or "",
                "citation": r[0] or "",
            })

        # Count total (may be larger than limit)
        count_params = [f"%{judge.lower()}%"]
        count_clauses = ["LOWER(j.judge) LIKE ?"]
        if court_code:
            count_clauses.append("j.court_code = ?")
            count_params.append(court_code)
        where_count = " AND ".join(count_clauses)
        total_row = idx.conn.execute(
            f"SELECT COUNT(*) FROM judgments j WHERE {where_count}", count_params
        ).fetchone()
        total = total_row[0] if total_row else len(cases)

        # Find the actual judge names matched (for display)
        name_rows = idx.conn.execute(
            "SELECT DISTINCT judge FROM judgments WHERE LOWER(judge) LIKE ? LIMIT 10",
            [f"%{judge.lower()}%"],
        ).fetchall()
        matched_names = [r[0] for r in name_rows if r[0]]

        return {
            "query": judge,
            "matched_names": matched_names,
            "total": total,
            "cases": cases,
            "verdict_breakdown": verdict_counts,
            "court_breakdown": court_counts,
            "yearly_volume": dict(sorted(year_counts.items())),
        }
    except Exception as e:
        logger.warning("Judge profile error: %s", e)
        return {"judge": judge, "cases": [], "verdict_breakdown": {}, "total": 0, "error": str(e)}


@app.get("/api/analytics/corpus-stats")
def api_corpus_stats(ls_session: Optional[str] = Cookie(default=None)):
    """Corpus overview: total records, courts, year range."""
    if not ls_session:
        raise HTTPException(401)
    if not auth.verify_session_token(ls_session):
        raise HTTPException(401)
    idx = _ensure_bm25()
    if idx is None:
        return {}
    if _FTS5_AVAILABLE and isinstance(idx, FTS5Index):
        return idx.stats()
    return {"total": len(idx.docs) if hasattr(idx, 'docs') else 0}


@app.get("/api/languages")
def api_languages():
    """Available languages for chat responses."""
    from brief_service import LANGUAGES
    return {"languages": [
        {"code": k, "label": v.split("(")[0].strip(),
         "native": v.split("(")[1].rstrip(")") if "(" in v else v}
        for k, v in LANGUAGES.items()
    ]}


# ─────────────────────────────────────────────────────────────────────────
# WEB SIGNALS — live legal news & developments
# ─────────────────────────────────────────────────────────────────────────

@app.get("/api/news")
def api_news():
    """Latest legal news from trusted Indian law sources."""
    signals = web_signals.fetch_legal_news(max_items=20)
    return {"signals": [s.to_dict() for s in signals], "sources": web_signals.available_sources()}


@app.get("/api/news/search")
def api_news_search(q: str = Query(default="", description="Search query")):
    """Search legal news for a specific topic."""
    if not q.strip():
        return api_news()
    signals = web_signals.search_web_signals(q.strip(), max_items=12)
    return {"query": q, "signals": [s.to_dict() for s in signals]}


@app.get("/api/news/sources")
def api_news_sources():
    """List available news signal sources."""
    return {"sources": web_signals.available_sources()}


# ─────────────────────────────────────────────────────────────────────────
# LEGAL DOCUMENT EDITOR  — Google-Docs-style drafting with AI + citations
# ─────────────────────────────────────────────────────────────────────────

class DocCreateBody(BaseModel):
    title: str = "Untitled Document"
    doc_type: str = "general"
    content: str = ""

class DocSaveBody(BaseModel):
    title: Optional[str] = None
    content: str
    citations: str = "[]"

class AiCompleteBody(BaseModel):
    content: str
    cursor_text: str = ""
    doc_type: str = ""

class AiImproveBody(BaseModel):
    selected_text: str
    doc_type: str = ""

class AiWriteSectionBody(BaseModel):
    instruction: str
    doc_type: str = ""
    context: str = ""

class AiSuggestCasesBody(BaseModel):
    argument: str


@app.get("/api/editor/doc-types")
def api_editor_doc_types():
    """List all legal document types with templates."""
    return {"doc_types": doc_editor.list_doc_types()}


@app.get("/api/editor/template/{doc_type}")
def api_editor_template(doc_type: str):
    """Get starter template for a document type."""
    tmpl = doc_editor.get_template(doc_type)
    if not tmpl:
        raise HTTPException(404, f"Unknown doc_type: {doc_type}")
    return {"doc_type": doc_type, "template": tmpl}


@app.get("/api/editor/docs")
def api_editor_list(ls_session: Optional[str] = Cookie(default=None)):
    """List user's saved documents."""
    user = _require_user(ls_session)
    return {"documents": auth.doc_list(user["id"])}


@app.post("/api/editor/docs")
def api_editor_create(body: DocCreateBody, ls_session: Optional[str] = Cookie(default=None)):
    """Create a new document (optionally with template content)."""
    import time as _time
    user = _require_user(ls_session)
    content = body.content
    if not content and body.doc_type != "general":
        content = doc_editor.get_template(body.doc_type)
    doc_id = auth.doc_create(user["id"], body.title, body.doc_type, content)
    now = int(_time.time())
    return {"doc": {"id": doc_id, "title": body.title, "doc_type": body.doc_type,
                    "content": content, "word_count": len(content.split()), "updated_at": now}}


@app.get("/api/editor/docs/{doc_id}")
def api_editor_get(doc_id: int, ls_session: Optional[str] = Cookie(default=None)):
    """Get a document by ID."""
    user = _require_user(ls_session)
    doc = auth.doc_get(doc_id, user["id"])
    if not doc:
        raise HTTPException(404, "Document not found")
    return {"doc": doc}


@app.put("/api/editor/docs/{doc_id}")
def api_editor_save(doc_id: int, body: DocSaveBody, ls_session: Optional[str] = Cookie(default=None)):
    """Save/auto-save a document."""
    import time as _time
    user = _require_user(ls_session)
    # Fetch existing title if not provided
    title = body.title
    if title is None:
        existing = auth.doc_get(doc_id, user["id"])
        title = existing["title"] if existing else "Untitled Document"
    ok = auth.doc_save(doc_id, user["id"], title, body.content, body.citations)
    if not ok:
        raise HTTPException(404, "Document not found")
    wc = len(body.content.split())
    now = int(_time.time())
    return {"doc": {"id": doc_id, "title": title, "word_count": wc, "updated_at": now}}


@app.delete("/api/editor/docs/{doc_id}")
def api_editor_delete(doc_id: int, ls_session: Optional[str] = Cookie(default=None)):
    """Delete a document."""
    user = _require_user(ls_session)
    ok = auth.doc_delete(doc_id, user["id"])
    if not ok:
        raise HTTPException(404, "Document not found")
    return {"deleted": True}


@app.get("/api/editor/docs/{doc_id}/versions")
def api_editor_versions(doc_id: int, ls_session: Optional[str] = Cookie(default=None)):
    """Get version history for a document."""
    user = _require_user(ls_session)
    versions = auth.doc_versions(doc_id, user["id"])
    return {"versions": versions}


@app.post("/api/editor/docs/{doc_id}/restore/{version_id}")
def api_editor_restore(doc_id: int, version_id: int, ls_session: Optional[str] = Cookie(default=None)):
    """Restore a previous version."""
    user = _require_user(ls_session)
    content = auth.doc_restore_version(doc_id, version_id, user["id"])
    if content is None:
        raise HTTPException(404, "Version not found")
    return {"content": content}


@app.post("/api/editor/ai/complete")
def api_editor_ai_complete(body: AiCompleteBody, ls_session: Optional[str] = Cookie(default=None)):
    """AI: continue writing from cursor position."""
    _require_user(ls_session)
    completion = doc_editor.ai_complete(body.content, body.doc_type, body.cursor_text)
    return {"completion": completion}


@app.post("/api/editor/ai/improve")
def api_editor_ai_improve(body: AiImproveBody, ls_session: Optional[str] = Cookie(default=None)):
    """AI: improve selected text."""
    _require_user(ls_session)
    improved = doc_editor.ai_improve(body.selected_text, body.doc_type)
    return {"improved": improved}


@app.post("/api/editor/ai/write-section")
def api_editor_ai_write(body: AiWriteSectionBody, ls_session: Optional[str] = Cookie(default=None)):
    """AI: write a complete section based on instruction."""
    _require_user(ls_session)
    text = doc_editor.ai_write_section(body.instruction, body.doc_type, body.context)
    return {"text": text}


@app.post("/api/editor/ai/suggest-cases")
def api_editor_suggest_cases(
    body: AiSuggestCasesBody,
    ls_session: Optional[str] = Cookie(default=None),
):
    """AI: suggest what case law to search for a given argument. Then search."""
    _require_user(ls_session)
    suggestion = doc_editor.ai_suggest_case_search(body.argument)
    # Actually run the search
    hits: list[dict] = []
    idx = _ensure_bm25()
    if idx is not None and suggestion.get("search_query"):
        try:
            if _FTS5_AVAILABLE and isinstance(idx, FTS5Index):
                hits = idx.search(suggestion["search_query"], limit=8)
        except Exception as e:
            logger.warning("editor case search failed: %s", e)
    return {
        "search_query": suggestion.get("search_query", ""),
        "explanation": suggestion.get("explanation", ""),
        "cases": hits[:8],
    }


@app.post("/api/editor/ai/insert-citation")
def api_editor_insert_citation(
    body: dict,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Format a case as a citation string for insertion."""
    _require_user(ls_session)
    citation_str = doc_editor.ai_insert_citation(body)
    return {"citation": citation_str}


@app.get("/api/analytics/court-efficiency")
def api_court_efficiency(ls_session: Optional[str] = Cookie(default=None)):
    """Court efficiency: avg days to dispose, disposal rate per court."""
    if not ls_session:
        raise HTTPException(401)
    if not auth.verify_session_token(ls_session):
        raise HTTPException(401)
    idx = _ensure_bm25()
    if idx is None or not (_FTS5_AVAILABLE and isinstance(idx, FTS5Index)):
        return {"courts": []}
    return {"courts": idx.court_efficiency()}


@app.get("/api/analytics/bail-intelligence")
def api_bail_intelligence(
    court: Optional[str] = None,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Bail grant/rejection rates by court."""
    if not ls_session:
        raise HTTPException(401)
    if not auth.verify_session_token(ls_session):
        raise HTTPException(401)
    idx = _ensure_bm25()
    if idx is None or not (_FTS5_AVAILABLE and isinstance(idx, FTS5Index)):
        return {}
    return idx.bail_intelligence(court_code=court)


@app.get("/api/analytics/verdict-patterns")
def api_verdict_patterns(
    court: Optional[str] = None,
    year_from: Optional[int] = None,
    ls_session: Optional[str] = Cookie(default=None),
):
    """Verdict distribution patterns."""
    if not ls_session:
        raise HTTPException(401)
    if not auth.verify_session_token(ls_session):
        raise HTTPException(401)
    idx = _ensure_bm25()
    if idx is None or not (_FTS5_AVAILABLE and isinstance(idx, FTS5Index)):
        return {}
    return idx.verdict_patterns(court_code=court, year_from=year_from)


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
    lang: str = Field(default="en", max_length=5)
    language: Optional[str] = Field(default=None, max_length=5)  # frontend sends this
    jurisdiction: Optional[str] = None
    sources: Optional[list[str]] = None
    prefer: Optional[str] = None


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

    # Retrieve grounding hits — FTS5 (primary) or BM25 (legacy).
    # Get up to 10 hits for better grounding coverage.
    hits: list[dict] = []
    idx = _ensure_bm25()
    if idx is not None:
        try:
            if _FTS5_AVAILABLE and isinstance(idx, FTS5Index):
                # FTS5 adapter returns Sanhita-compatible dicts directly
                hits = idx.search(safe_question, limit=12)
                logger.info("FTS5 retrieve for '%s': %d hits", safe_question[:50], len(hits))
            else:
                # Legacy BM25 path
                results = idx.query(safe_question, k=10, tier=None)
                hits = [doc_to_retrieve_hit(d, s, safe_question) for d, s in results]  # type: ignore[misc]
        except Exception as e:
            logger.warning("Retrieval failed in /api/brief/chat: %s", e)

    # Compose the grounded answer (LLM or fallback).
    lang = body.language or body.lang or "en"
    result = answer_question(safe_question, hits, history or [], lang=lang)
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

    return JSONResponse(result)


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
    n = auth.vault_save_chunks(doc_id, user["id"], chunks)
    return {"ok": True, "doc_id": doc_id, "filename": file.filename, "chunks": n}


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
        # Try DB-backed template first
        db_template = None
        idx = _ensure_bm25()
        if idx and _FTS5_AVAILABLE and isinstance(idx, FTS5Index):
            db_template = idx.draft_template(body.template)
        return workflows.generate_draft(body.template, body.facts, db_template=db_template)
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
