"""
LexSearch – Indian High Court Judgments
FastAPI backend: serves search API + proxies PDFs from public AWS S3.
Run: uvicorn server:app --reload --port 8080
"""

import logging
import os
import re
import urllib.parse
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import s3fs
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LexSearch – Indian High Court Judgments")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# S3 configuration (public, no credentials needed)
# ---------------------------------------------------------------------------
BUCKET = "indian-high-court-judgments"
S3_BASE = f"s3://{BUCKET}"
HTTP_BASE = f"https://{BUCKET}.s3.amazonaws.com"

_fs: Optional[s3fs.S3FileSystem] = None


def get_fs() -> s3fs.S3FileSystem:
    global _fs
    if _fs is None:
        _fs = s3fs.S3FileSystem(anon=True)
    return _fs


# ---------------------------------------------------------------------------
# Court / bench mapping  (S3 paths use _ separator, internal codes use ~)
# Mapping: s3_court_code -> { name, benches: [{code, name}] }
# Built from actual S3 data exploration.
# ---------------------------------------------------------------------------
COURTS = [
    {
        "s3_code": "9_13",
        "name": "Allahabad High Court",
        "benches": [
            {"code": "cishclko", "name": "Lucknow Bench"},
            {"code": "cisdb_16012018", "name": "Allahabad"},
        ],
    },
    {
        "s3_code": "28_2",
        "name": "Andhra Pradesh High Court",
        "benches": [{"code": "aphc", "name": "Amaravati"}],
    },
    {
        "s3_code": "27_1",
        "name": "Bombay High Court",
        "benches": [
            {"code": "newas", "name": "Appellate Side"},
            {"code": "newos", "name": "Original Side"},
            {"code": "newos_spl", "name": "Original Side (Special)"},
            {"code": "hcaurdb", "name": "Aurangabad Bench"},
            {"code": "kolhcdb", "name": "Nagpur Bench"},
            {"code": "hcbgoa", "name": "Goa Bench"},
        ],
    },
    {
        "s3_code": "19_16",
        "name": "Calcutta High Court",
        "benches": [
            {"code": "calcutta_appellate_side", "name": "Appellate Side"},
            {"code": "calcutta_original_side", "name": "Original Side"},
            {"code": "calcutta_circuit_bench_at_jalpaiguri", "name": "Jalpaiguri Circuit Bench"},
            {"code": "calcutta_circuit_bench_at_port_blair", "name": "Port Blair Circuit Bench"},
        ],
    },
    {
        "s3_code": "22_18",
        "name": "Chhattisgarh High Court",
        "benches": [{"code": "cghccisdb", "name": "Bilaspur"}],
    },
    {
        "s3_code": "7_26",
        "name": "Delhi High Court",
        "benches": [{"code": "dhcdb", "name": "New Delhi"}],
    },
    {
        "s3_code": "18_6",
        "name": "Gauhati High Court",
        "benches": [
            {"code": "asghccis", "name": "Guwahati"},
            {"code": "azghccis", "name": "Aizawl Bench"},
            {"code": "arghccis", "name": "Itanagar Bench"},
            {"code": "nlghccis", "name": "Kohima Bench"},
        ],
    },
    {
        "s3_code": "24_17",
        "name": "Gujarat High Court",
        "benches": [{"code": "gujarathc", "name": "Ahmedabad"}],
    },
    {
        "s3_code": "2_5",
        "name": "Himachal Pradesh High Court",
        "benches": [{"code": "cmis", "name": "Shimla"}],
    },
    {
        "s3_code": "1_12",
        "name": "Jammu & Kashmir High Court",
        "benches": [
            {"code": "jammuhc", "name": "Jammu"},
            {"code": "kashmirhc", "name": "Srinagar"},
        ],
    },
    {
        "s3_code": "20_7",
        "name": "Jharkhand High Court",
        "benches": [{"code": "jhar_pg", "name": "Ranchi"}],
    },
    {
        "s3_code": "29_3",
        "name": "Karnataka High Court",
        "benches": [
            {"code": "karnataka_bng_old", "name": "Bengaluru"},
            {"code": "karhcdharwad", "name": "Dharwad Bench"},
            {"code": "karhckalaburagi", "name": "Kalaburagi Bench"},
        ],
    },
    {
        "s3_code": "32_4",
        "name": "Kerala High Court",
        "benches": [{"code": "highcourtofkerala", "name": "Ernakulam"}],
    },
    {
        "s3_code": "23_23",
        "name": "Madhya Pradesh High Court",
        "benches": [
            {"code": "mphc_db_jbp", "name": "Jabalpur"},
            {"code": "mphc_db_gwl", "name": "Gwalior Bench"},
            {"code": "mphc_db_ind", "name": "Indore Bench"},
        ],
    },
    {
        "s3_code": "33_10",
        "name": "Madras High Court",
        "benches": [
            {"code": "hc_cis_mas", "name": "Chennai"},
            {"code": "mdubench", "name": "Madurai Bench"},
        ],
    },
    {
        "s3_code": "14_25",
        "name": "Manipur High Court",
        "benches": [{"code": "manipurhc_pg", "name": "Imphal"}],
    },
    {
        "s3_code": "17_21",
        "name": "Meghalaya High Court",
        "benches": [{"code": "meghalaya", "name": "Shillong"}],
    },
    {
        "s3_code": "21_11",
        "name": "Orissa High Court",
        "benches": [{"code": "cisnc", "name": "Cuttack"}],
    },
    {
        "s3_code": "10_8",
        "name": "Patna High Court",
        "benches": [{"code": "patnahcucisdb94", "name": "Patna"}],
    },
    {
        "s3_code": "3_22",
        "name": "Punjab & Haryana High Court",
        "benches": [{"code": "phhc", "name": "Chandigarh"}],
    },
    {
        "s3_code": "8_9",
        "name": "Rajasthan High Court",
        "benches": [
            {"code": "rhcjodh240618", "name": "Jodhpur"},
            {"code": "jaipur", "name": "Jaipur Bench"},
        ],
    },
    {
        "s3_code": "11_24",
        "name": "Sikkim High Court",
        "benches": [{"code": "sikkimhc_pg", "name": "Gangtok"}],
    },
    {
        "s3_code": "16_20",
        "name": "Telangana High Court",
        "benches": [{"code": "thcnc", "name": "Hyderabad"}],
    },
    {
        "s3_code": "36_29",
        "name": "Andhra Pradesh High Court (Kurnool)",
        "benches": [{"code": "taphc", "name": "Kurnool"}],
    },
    {
        "s3_code": "5_15",
        "name": "Uttarakhand High Court",
        "benches": [{"code": "ukhcucis_pg", "name": "Nainital"}],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parquet_path(year: int, s3_court: str, bench: str) -> str:
    return f"{S3_BASE}/metadata/parquet/year={year}/court={s3_court}/bench={bench}/metadata.parquet"


def _extract_pdf_filename(pdf_link: str) -> str:
    """Extract just the filename from the pdf_link path."""
    if not pdf_link:
        return ""
    return pdf_link.rstrip("/").split("/")[-1]


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _df_to_results(df: pd.DataFrame, s3_court: str, bench: str, year: int) -> list[dict]:
    """Convert a metadata DataFrame to a list of result dicts."""
    results = []

    for _, row in df.iterrows():
        pdf_link = _safe_str(row.get("pdf_link", ""))
        pdf_filename = _extract_pdf_filename(pdf_link)
        title = _safe_str(row.get("title", ""))
        description = _safe_str(row.get("description", ""))
        cnr = _safe_str(row.get("cnr", ""))
        judge = _safe_str(row.get("judge", ""))
        decision_date = _safe_str(row.get("decision_date", ""))
        court_name = _safe_str(row.get("court", ""))
        disposal = _safe_str(row.get("disposal_nature", ""))
        if not pdf_filename:
            continue

        # Build the S3 key for the PDF
        s3_key = f"data/pdf/year={year}/court={s3_court}/bench={bench}/{pdf_filename}"

        results.append({
            "case_number": cnr or pdf_filename.replace(".pdf", ""),
            "title": title or cnr or pdf_filename.replace(".pdf", ""),
            "description": description[:300] if description else "",
            "court": s3_court,
            "court_name": court_name,
            "bench": bench,
            "year": year,
            "judge": judge,
            "date": decision_date or str(year),
            "disposal": disposal,
            "filename": pdf_filename,
            "s3_key": urllib.parse.quote(s3_key, safe=""),
        })
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/courts")
def list_courts():
    """Return available courts and their benches."""
    return JSONResponse(COURTS)


@app.get("/search")
def search(
    court: str = Query(default="", description="S3 court code e.g. 7_26"),
    bench: str = Query(default="", description="Bench code e.g. dhcdb"),
    year: int = Query(default=0, description="Year (0 = recent years)"),
    q: str = Query(default="", description="Keyword in title/description"),
    cnr: str = Query(default="", description="CNR number"),
    judge: str = Query(default="", description="Judge name"),
    case_type: str = Query(default="", description="Case type prefix e.g. CWP, CRL"),
    disposal: str = Query(default="", description="Disposal nature"),
    page: int = Query(default=1, ge=1),
):
    """
    Search judgments with advanced filters.
    Provide court + year for fastest results.
    """
    PAGE_SIZE = 50

    has_filter = court or q or cnr or judge or case_type
    if not has_filter:
        raise HTTPException(status_code=400, detail="Provide at least one filter.")

    fs = get_fs()
    results: list[dict] = []

    # Determine years to scan
    if year:
        years_to_scan = [year]
    else:
        years_to_scan = list(range(2024, 2019, -1))

    # Determine courts/benches to scan
    courts_to_scan: list[tuple[str, str]] = []
    if court and bench:
        courts_to_scan = [(court, bench)]
    elif court:
        for c in COURTS:
            if c["s3_code"] == court:
                courts_to_scan = [(court, b["code"]) for b in c["benches"]]
                break
        if not courts_to_scan:
            courts_to_scan = [(court, court)]
    else:
        # No court specified: scan major courts
        courts_to_scan = [
            ("7_26", "dhcdb"),
            ("27_1", "newas"),
            ("33_10", "hc_cis_mas"),
            ("19_16", "calcutta_appellate_side"),
            ("9_13", "cishclko"),
        ]

    for yr in years_to_scan:
        for ct, bn in courts_to_scan:
            path = _parquet_path(yr, ct, bn)
            try:
                with fs.open(path, "rb") as f:
                    df = pd.read_parquet(f)
                logger.info(f"Loaded {len(df)} rows from year={yr}/court={ct}/bench={bn}")
            except Exception as e:
                logger.debug(f"Skipping {path}: {e}")
                continue

            # Apply filters
            if cnr:
                df = df[df["cnr"].astype(str).str.lower().str.contains(cnr.lower(), na=False)]

            if judge:
                df = df[df["judge"].astype(str).str.lower().str.contains(judge.lower(), na=False)]

            if case_type:
                df = df[df["title"].astype(str).str.upper().str.startswith(case_type.upper(), na=False)]

            if disposal:
                df = df[df["disposal_nature"].astype(str).str.lower().str.contains(disposal.lower(), na=False)]

            if q:
                q_lower = q.lower()
                mask = pd.Series(False, index=df.index)
                for col in ["title", "description"]:
                    if col in df.columns:
                        mask |= df[col].astype(str).str.lower().str.contains(q_lower, na=False)
                df = df[mask]

            chunk = _df_to_results(df, ct, bn, yr)
            results.extend(chunk)

            if len(results) >= PAGE_SIZE * page + PAGE_SIZE:
                break
        if len(results) >= PAGE_SIZE * page + PAGE_SIZE:
            break

    # Paginate
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE

    return JSONResponse({
        "total": len(results),
        "page": page,
        "page_size": PAGE_SIZE,
        "results": results[start:end],
    })


@app.get("/pdf/{s3_key:path}")
async def proxy_pdf(s3_key: str, download: bool = False):
    """
    Stream a PDF from S3.
    Pass ?download=true to get Content-Disposition: attachment.
    """
    decoded_key = urllib.parse.unquote(s3_key)
    pdf_url = f"{HTTP_BASE}/{decoded_key}"

    async def stream():
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("GET", pdf_url) as resp:
                if resp.status_code != 200:
                    raise HTTPException(status_code=404, detail="PDF not found on S3.")
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    yield chunk

    filename = decoded_key.split("/")[-1] or "judgment.pdf"
    disposition = f'attachment; filename="{filename}"' if download else f'inline; filename="{filename}"'

    return StreamingResponse(
        stream(),
        media_type="application/pdf",
        headers={"Content-Disposition": disposition},
    )


# ---------------------------------------------------------------------------
# Serve static frontend
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


@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")
