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
import urllib.parse
from functools import lru_cache
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import pdfplumber
import s3fs
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LexSearch")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

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
    """Proxy HC PDF from S3."""
    decoded = urllib.parse.unquote(s3_key)
    url = f"{HC_HTTP}/{decoded}"

    async def stream():
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    raise HTTPException(404, "PDF not found.")
                async for chunk in resp.aiter_bytes(65536):
                    yield chunk

    fname = decoded.split("/")[-1] or "judgment.pdf"
    disp = f'attachment; filename="{fname}"' if download else f'inline; filename="{fname}"'
    return StreamingResponse(stream(), media_type="application/pdf",
                             headers={"Content-Disposition": disp})


@app.get("/sc-pdf/{year}/{pdf_name}")
def serve_sc_pdf(year: int, pdf_name: str, download: bool = False):
    """
    Extract a single PDF from the Supreme Court tar archive on S3.
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
        raise HTTPException(404, f"PDF {pdf_name} not found in archive.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SC PDF extraction error: {e}")
        raise HTTPException(500, f"Error extracting PDF: {str(e)}")


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------

def _load_hc_data(year: int, court: str = "", bench: str = "") -> pd.DataFrame:
    """Load HC parquet data for analytics. Returns combined DataFrame."""
    fs = get_fs()
    frames = []
    courts_to_scan: list[tuple[str, str]] = []

    if court and bench:
        courts_to_scan = [(court, bench)]
    elif court:
        for c in HC_COURTS:
            if c["s3_code"] == court:
                courts_to_scan = [(court, b["code"]) for b in c["benches"]]
                break
    else:
        for c in HC_COURTS:
            for b in c["benches"]:
                courts_to_scan.append((c["s3_code"], b["code"]))

    for ct, bn in courts_to_scan:
        path = _hc_parquet_path(year, ct, bn)
        try:
            with fs.open(path, "rb") as f:
                df = pd.read_parquet(f)
            df["_court_code"] = ct
            df["_bench_code"] = bn
            # Map court name
            court_name = ct
            for c in HC_COURTS:
                if c["s3_code"] == ct:
                    court_name = c["name"]
                    break
            df["_court_name"] = court_name
            df["_year"] = year
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_sc_data(year: int) -> pd.DataFrame:
    """Load SC parquet data for analytics."""
    fs = get_fs()
    path = _sc_parquet_path(year)
    try:
        with fs.open(path, "rb") as f:
            df = pd.read_parquet(f)
        df["_court_name"] = "Supreme Court of India"
        df["_court_code"] = "sc"
        df["_bench_code"] = ""
        df["_year"] = year
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Analytics endpoints
# ---------------------------------------------------------------------------

@app.get("/analytics/judge")
def analytics_judge(
    name: str = Query(..., description="Judge name to search"),
    court: str = Query(default=""),
    year: int = Query(default=0),
    mode: str = Query(default="hc"),
):
    """Judge profile analytics."""
    if not name or len(name) < 2:
        raise HTTPException(400, "Provide a judge name (min 2 chars).")

    name_lower = name.lower()
    years = [year] if year else list(range(2024, 2017, -1))
    all_frames = []

    for yr in years:
        if mode == "sc":
            df = _load_sc_data(yr)
        else:
            df = _load_hc_data(yr, court)
        if df.empty:
            continue
        if "judge" in df.columns:
            mask = df["judge"].astype(str).str.lower().str.contains(name_lower, na=False)
            filtered = df[mask]
            if not filtered.empty:
                all_frames.append(filtered)

    if not all_frames:
        return JSONResponse({"total_cases": 0, "disposal_breakdown": {}, "cases_per_year": {}, "courts": [], "recent_cases": []})

    combined = pd.concat(all_frames, ignore_index=True)

    # Disposal breakdown
    disposal_col = "disposal_nature" if "disposal_nature" in combined.columns else None
    disposal_breakdown = {}
    if disposal_col:
        counts = combined[disposal_col].astype(str).value_counts().head(10)
        disposal_breakdown = {k: int(v) for k, v in counts.items() if k and k != "nan"}

    # Cases per year
    cases_per_year = {}
    if "_year" in combined.columns:
        yc = combined["_year"].value_counts().sort_index()
        cases_per_year = {str(k): int(v) for k, v in yc.items()}

    # Courts served
    courts_list = []
    if "_court_name" in combined.columns:
        courts_list = combined["_court_name"].unique().tolist()

    # Recent cases (last 20)
    recent = combined.head(20)
    recent_cases = []
    for _, row in recent.iterrows():
        recent_cases.append({
            "title": _safe_str(row.get("title", "")),
            "court": _safe_str(row.get("_court_name", "")),
            "year": int(row.get("_year", 0)),
            "disposal": _safe_str(row.get("disposal_nature", "")),
            "date": _safe_str(row.get("decision_date", "")),
        })

    return JSONResponse({
        "total_cases": len(combined),
        "disposal_breakdown": disposal_breakdown,
        "cases_per_year": cases_per_year,
        "courts": courts_list,
        "recent_cases": recent_cases,
    })


@app.get("/analytics/court")
def analytics_court(
    court: str = Query(..., description="Court s3_code"),
    year: int = Query(default=0),
):
    """Court-level analytics."""
    years = [year] if year else list(range(2024, 2017, -1))
    all_frames = []

    for yr in years:
        if court == "sc":
            df = _load_sc_data(yr)
        else:
            df = _load_hc_data(yr, court)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        return JSONResponse({"total_cases": 0, "top_judges": [], "disposal_breakdown": {}, "yearly_volume": {}})

    combined = pd.concat(all_frames, ignore_index=True)

    # Top judges
    top_judges = []
    if "judge" in combined.columns:
        jc = combined["judge"].astype(str).value_counts().head(15)
        for judge_name, count in jc.items():
            if judge_name and judge_name != "nan":
                judge_cases = combined[combined["judge"].astype(str) == judge_name]
                disposal_data = {}
                if "disposal_nature" in judge_cases.columns:
                    dc = judge_cases["disposal_nature"].astype(str).value_counts()
                    disposal_data = {k: int(v) for k, v in dc.items() if k and k != "nan"}
                top_judges.append({"name": judge_name, "total": int(count), "disposal": disposal_data})

    # Disposal breakdown
    disposal_breakdown = {}
    if "disposal_nature" in combined.columns:
        counts = combined["disposal_nature"].astype(str).value_counts().head(10)
        disposal_breakdown = {k: int(v) for k, v in counts.items() if k and k != "nan"}

    # Yearly volume
    yearly_volume = {}
    if "_year" in combined.columns:
        yc = combined["_year"].value_counts().sort_index()
        yearly_volume = {str(k): int(v) for k, v in yc.items()}

    return JSONResponse({
        "total_cases": len(combined),
        "top_judges": top_judges,
        "disposal_breakdown": disposal_breakdown,
        "yearly_volume": yearly_volume,
    })


@app.get("/analytics/trends")
def analytics_trends(
    court: str = Query(default=""),
    mode: str = Query(default="hc"),
):
    """Case volume and disposal trends over years."""
    years = list(range(2024, 2017, -1))
    yearly_volumes = []
    disposal_trends = []

    for yr in years:
        if mode == "sc":
            df = _load_sc_data(yr)
        else:
            df = _load_hc_data(yr, court)
        if df.empty:
            continue

        yearly_volumes.append({"year": yr, "count": len(df)})

        if "disposal_nature" in df.columns:
            dc = df["disposal_nature"].astype(str).value_counts()
            trend = {"year": yr}
            for k, v in dc.items():
                if k and k != "nan":
                    trend[k] = int(v)
            disposal_trends.append(trend)

    yearly_volumes.sort(key=lambda x: x["year"])
    disposal_trends.sort(key=lambda x: x["year"])

    return JSONResponse({
        "yearly_volumes": yearly_volumes,
        "disposal_trends": disposal_trends,
    })


@app.get("/analytics/judges/top")
def analytics_top_judges(
    court: str = Query(default=""),
    year: int = Query(default=2023),
    mode: str = Query(default="hc"),
    limit: int = Query(default=20, le=50),
):
    """Top judges by case count."""
    if mode == "sc":
        df = _load_sc_data(year)
    else:
        df = _load_hc_data(year, court)

    if df.empty:
        return JSONResponse([])

    if "judge" not in df.columns:
        return JSONResponse([])

    jc = df["judge"].astype(str).value_counts().head(limit)
    result = []
    for judge_name, count in jc.items():
        if not judge_name or judge_name == "nan":
            continue
        judge_df = df[df["judge"].astype(str) == judge_name]
        total = len(judge_df)
        allowed = 0
        dismissed = 0
        if "disposal_nature" in judge_df.columns:
            disp = judge_df["disposal_nature"].astype(str).str.lower()
            allowed = int((disp.str.contains("allowed", na=False)).sum())
            dismissed = int((disp.str.contains("dismissed", na=False)).sum())
        result.append({
            "judge": judge_name,
            "total_cases": total,
            "allowed": allowed,
            "dismissed": dismissed,
            "allowed_pct": round(allowed / total * 100, 1) if total else 0,
            "dismissed_pct": round(dismissed / total * 100, 1) if total else 0,
        })

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# AI Summarization (Groq free tier — Llama 3.1 70B)
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Simple in-memory cache for summaries
_summary_cache: dict[str, dict] = {}


def _extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 30) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                break
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


async def _fetch_hc_pdf_bytes(s3_key: str) -> bytes:
    """Fetch HC PDF bytes from S3."""
    decoded = urllib.parse.unquote(s3_key)
    url = f"{HC_HTTP}/{decoded}"
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            raise HTTPException(404, "PDF not found.")
        return resp.content


def _fetch_sc_pdf_bytes(year: int, pdf_name: str) -> bytes:
    """Fetch SC PDF bytes from tar on S3."""
    fs = get_fs()
    tar_path = f"{SC_S3}/data/tar/year={year}/english/english.tar"
    with fs.open(tar_path, "rb") as f:
        with tarfile.open(fileobj=f, mode="r|") as tf:
            for member in tf:
                if member.name == pdf_name or member.name.endswith(pdf_name):
                    extracted = tf.extractfile(member)
                    if extracted:
                        return extracted.read()
    raise HTTPException(404, f"PDF {pdf_name} not found in archive.")


async def _call_groq(system_prompt: str, user_prompt: str) -> str:
    """Call Groq API."""
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY not configured.")
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 2000,
            },
        )
        if resp.status_code != 200:
            logger.error(f"Groq error: {resp.status_code} {resp.text}")
            raise HTTPException(502, "AI service error.")
        data = resp.json()
        return data["choices"][0]["message"]["content"]


SUMMARIZE_SYSTEM = """You are an expert Indian legal analyst. Given the text of a court judgment, produce a structured summary in JSON format with these keys:
- "facts": Brief summary of the factual background (2-3 sentences)
- "issues": Key legal issues considered by the court (list of strings)
- "arguments": Brief summary of arguments by both sides (2-3 sentences)
- "held": What the court decided/held (2-3 sentences)
- "ratio": The ratio decidendi — the legal principle established (1-2 sentences)
- "statutes": List of statutes/sections cited (list of strings)
- "result": One word — "Allowed", "Dismissed", "Disposed", or "Other"

Return ONLY valid JSON, no markdown formatting."""

QA_SYSTEM = """You are an expert Indian legal analyst. Answer the user's question about the court judgment based on the provided text. Be concise and accurate. If the answer is not in the text, say "Not found in this judgment." """

TRANSLATE_SYSTEM = """You are a legal translator. Translate the following legal text from English to Hindi. Maintain legal terminology accuracy. Return only the Hindi translation."""


@app.post("/ai/summarize")
async def ai_summarize(body: dict = Body(...)):
    """Summarize a judgment PDF using AI."""
    s3_key = body.get("s3_key", "")
    sc = body.get("sc", False)
    year = body.get("year", 0)
    pdf_name = body.get("pdf_name", "")

    cache_key = f"{s3_key}_{pdf_name}_{year}"
    if cache_key in _summary_cache:
        return JSONResponse(_summary_cache[cache_key])

    # Fetch PDF
    try:
        if sc and year and pdf_name:
            pdf_bytes = _fetch_sc_pdf_bytes(int(year), pdf_name)
        elif s3_key:
            pdf_bytes = await _fetch_hc_pdf_bytes(s3_key)
        else:
            raise HTTPException(400, "Provide s3_key or sc+year+pdf_name.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error fetching PDF: {e}")

    # Extract text
    text = _extract_text_from_pdf_bytes(pdf_bytes)
    if not text or len(text) < 100:
        raise HTTPException(422, "Could not extract sufficient text from this PDF. It may be a scanned image.")

    # Truncate to ~6000 words for Groq context
    words = text.split()
    if len(words) > 6000:
        text = " ".join(words[:6000])

    # Call Groq
    result_text = await _call_groq(SUMMARIZE_SYSTEM, text)

    # Parse JSON from response
    try:
        # Strip markdown code fences if present
        cleaned = result_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        summary = json.loads(cleaned)
    except json.JSONDecodeError:
        summary = {"raw_summary": result_text}

    _summary_cache[cache_key] = summary
    return JSONResponse(summary)


@app.post("/ai/ask")
async def ai_ask(body: dict = Body(...)):
    """Ask a question about a judgment."""
    question = body.get("question", "")
    s3_key = body.get("s3_key", "")
    sc = body.get("sc", False)
    year = body.get("year", 0)
    pdf_name = body.get("pdf_name", "")

    if not question:
        raise HTTPException(400, "Provide a question.")

    # Fetch & extract text
    try:
        if sc and year and pdf_name:
            pdf_bytes = _fetch_sc_pdf_bytes(int(year), pdf_name)
        elif s3_key:
            pdf_bytes = await _fetch_hc_pdf_bytes(s3_key)
        else:
            raise HTTPException(400, "Provide s3_key or sc+year+pdf_name.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error fetching PDF: {e}")

    text = _extract_text_from_pdf_bytes(pdf_bytes)
    if not text or len(text) < 100:
        raise HTTPException(422, "Could not extract text from PDF.")

    words = text.split()
    if len(words) > 6000:
        text = " ".join(words[:6000])

    prompt = f"Judgment text:\n{text}\n\nQuestion: {question}"
    answer = await _call_groq(QA_SYSTEM, prompt)
    return JSONResponse({"answer": answer})


@app.post("/ai/translate")
async def ai_translate(body: dict = Body(...)):
    """Translate text to Hindi."""
    text = body.get("text", "")
    if not text:
        raise HTTPException(400, "Provide text to translate.")

    # Truncate if too long
    if len(text) > 3000:
        text = text[:3000]

    translation = await _call_groq(TRANSLATE_SYSTEM, text)
    return JSONResponse({"translation": translation})


# ---------------------------------------------------------------------------
# ML Case Predictor
# ---------------------------------------------------------------------------
import re
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Model state
_model_trained = False
_outcome_model = None
_le_court = LabelEncoder()
_le_casetype = LabelEncoder()
_le_outcome = LabelEncoder()
_duration_stats: dict = {}  # {(court, case_type): {median_days, mean_days, count}}
_cost_map = {
    "WP": {"min": 15000, "max": 80000, "label": "Writ Petition"},
    "CRL": {"min": 20000, "max": 150000, "label": "Criminal"},
    "BAIL": {"min": 10000, "max": 50000, "label": "Bail Application"},
    "BA": {"min": 10000, "max": 50000, "label": "Bail Application"},
    "ABA": {"min": 15000, "max": 60000, "label": "Anticipatory Bail"},
    "CS": {"min": 25000, "max": 200000, "label": "Civil Suit"},
    "APL": {"min": 20000, "max": 100000, "label": "Appeal"},
    "FA": {"min": 20000, "max": 100000, "label": "First Appeal"},
    "ITA": {"min": 30000, "max": 150000, "label": "Income Tax Appeal"},
    "SA": {"min": 25000, "max": 120000, "label": "Second Appeal"},
    "CRA": {"min": 25000, "max": 150000, "label": "Criminal Appeal"},
    "IA": {"min": 5000, "max": 30000, "label": "Interim Application"},
    "REVN": {"min": 20000, "max": 80000, "label": "Revision"},
    "CP": {"min": 30000, "max": 200000, "label": "Company Petition"},
    "MCA": {"min": 25000, "max": 120000, "label": "Misc Civil Application"},
    "MAC": {"min": 15000, "max": 80000, "label": "Motor Accident"},
    "LPA": {"min": 25000, "max": 120000, "label": "Letters Patent Appeal"},
    "OMP": {"min": 30000, "max": 200000, "label": "Arbitration"},
}


def _extract_case_type(title: str) -> str:
    """Extract case type prefix from title."""
    m = re.match(r'^([A-Z\.()]+)', str(title).strip())
    if m:
        ct = m.group(1).rstrip("./()").replace(".", "").replace("(", "").replace(")", "")
        return ct
    return "OTHER"


def _simplify_outcome(disp: str) -> str:
    """Simplify disposal to 3 categories."""
    d = str(disp).lower().strip()
    if "allowed" in d and "withdrawn" not in d:
        return "Allowed"
    if "dismissed" in d or "rejected" in d:
        return "Dismissed"
    return "Disposed"


def _parse_date(d) -> Optional[datetime]:
    """Parse various date formats."""
    if pd.isna(d) or not d:
        return None
    if isinstance(d, pd.Timestamp):
        return d.to_pydatetime()
    s = str(d).strip()
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s.split(" ")[0], fmt)
            if dt.year < 1950 or dt.year > 2030:
                return None
            return dt
        except ValueError:
            continue
    return None


def _train_model():
    """Train the prediction model from Bombay HC data (has best disposal coverage)."""
    global _model_trained, _outcome_model, _duration_stats

    fs = get_fs()
    all_frames = []

    # Load data from courts with good disposal coverage
    train_courts = [
        ("27_1", "newas", "Bombay HC"),
        ("27_1", "hcaurdb", "Bombay HC Aurangabad"),
        ("29_3", "karnataka_bng_old", "Karnataka HC"),
        ("32_4", "highcourtofkerala", "Kerala HC"),
    ]

    for court_code, bench, name in train_courts:
        for year in [2022, 2023]:
            path = f"s3://indian-high-court-judgments/metadata/parquet/year={year}/court={court_code}/bench={bench}/metadata.parquet"
            try:
                with fs.open(path, "rb") as f:
                    df = pd.read_parquet(f)
                df["_court"] = court_code
                df["_year"] = year
                all_frames.append(df)
                logger.info(f"ML: Loaded {len(df)} rows from {name} {year}")
            except Exception as e:
                logger.debug(f"ML: Skip {path}: {e}")

    if not all_frames:
        logger.warning("ML: No training data found.")
        return

    data = pd.concat(all_frames, ignore_index=True)

    # Filter rows with valid disposal
    data = data[data["disposal_nature"].astype(str).str.strip() != ""]
    data = data[data["disposal_nature"].notna()]
    data["outcome"] = data["disposal_nature"].apply(_simplify_outcome)
    data["case_type"] = data["title"].apply(_extract_case_type)

    # Parse dates for duration
    data["reg_date"] = data["date_of_registration"].apply(_parse_date)
    data["dec_date"] = data["decision_date"].apply(_parse_date)
    valid_dates = data.dropna(subset=["reg_date", "dec_date"])
    valid_dates = valid_dates.copy()
    try:
        valid_dates["duration_days"] = (pd.to_datetime(valid_dates["dec_date"]) - pd.to_datetime(valid_dates["reg_date"])).dt.days
    except Exception as e:
        logger.warning(f"ML: Duration calc error: {e}, skipping duration stats")
        valid_dates = pd.DataFrame()
    valid_dates = valid_dates[valid_dates["duration_days"] > 0]
    valid_dates = valid_dates[valid_dates["duration_days"] < 10000]  # filter outliers

    # Duration stats by court + case_type
    for (court, ct), grp in valid_dates.groupby(["_court", "case_type"]):
        if len(grp) >= 5:
            _duration_stats[(court, ct)] = {
                "median_days": int(grp["duration_days"].median()),
                "mean_days": int(grp["duration_days"].mean()),
                "count": len(grp),
            }

    # Also store aggregate by case_type only
    for ct, grp in valid_dates.groupby("case_type"):
        if len(grp) >= 10:
            _duration_stats[("all", ct)] = {
                "median_days": int(grp["duration_days"].median()),
                "mean_days": int(grp["duration_days"].mean()),
                "count": len(grp),
            }

    # Train outcome classifier
    train_data = data[["_court", "case_type", "outcome"]].dropna()
    if len(train_data) < 100:
        logger.warning("ML: Not enough training data.")
        return

    _le_court.fit(train_data["_court"])
    _le_casetype.fit(train_data["case_type"])
    _le_outcome.fit(train_data["outcome"])

    X = pd.DataFrame({
        "court": _le_court.transform(train_data["_court"]),
        "case_type": _le_casetype.transform(train_data["case_type"]),
    })
    y = _le_outcome.transform(train_data["outcome"])

    _outcome_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    _outcome_model.fit(X, y)
    _model_trained = True

    logger.info(f"ML: Model trained on {len(train_data)} cases. Duration stats for {len(_duration_stats)} groups.")


@app.on_event("startup")
async def startup_train_model():
    """Train ML model in background on startup."""
    import threading
    t = threading.Thread(target=_train_model, daemon=True)
    t.start()


@app.get("/predict")
def predict_case(
    court: str = Query(default="27_1"),
    case_type: str = Query(default="WP"),
):
    """Predict case outcome, duration, and estimated cost."""
    if not _model_trained:
        raise HTTPException(503, "Model is still training. Please try again in ~30 seconds.")

    # Outcome prediction
    outcome_probs = {}
    try:
        ct_encoded = _le_court.transform([court])[0] if court in _le_court.classes_ else 0
        case_encoded = _le_casetype.transform([case_type])[0] if case_type in _le_casetype.classes_ else 0
        X_pred = pd.DataFrame({"court": [ct_encoded], "case_type": [case_encoded]})
        probs = _outcome_model.predict_proba(X_pred)[0]
        classes = _le_outcome.classes_
        outcome_probs = {classes[i]: round(float(probs[i]) * 100, 1) for i in range(len(classes))}
    except Exception as e:
        logger.error(f"Predict error: {e}")
        outcome_probs = {"Allowed": 33.3, "Dismissed": 33.3, "Disposed": 33.4}

    # Duration estimate
    dur = _duration_stats.get((court, case_type)) or _duration_stats.get(("all", case_type))
    duration = None
    if dur:
        duration = {
            "median_days": dur["median_days"],
            "median_months": round(dur["median_days"] / 30, 1),
            "mean_days": dur["mean_days"],
            "mean_months": round(dur["mean_days"] / 30, 1),
            "based_on": dur["count"],
        }

    # Cost estimate
    cost = None
    # Normalize case type for cost lookup
    ct_upper = case_type.upper().replace(".", "").replace("(", "").replace(")", "").replace("/", "")
    for key in _cost_map:
        if ct_upper.startswith(key):
            base = _cost_map[key]
            multiplier = 1.0
            if duration:
                months = duration["median_months"]
                if months > 24:
                    multiplier = 2.0
                elif months > 12:
                    multiplier = 1.5
                elif months > 6:
                    multiplier = 1.2
            cost = {
                "min_inr": int(base["min"] * multiplier),
                "max_inr": int(base["max"] * multiplier),
                "label": base["label"],
                "note": "Estimated legal fees (lawyer + court). Actual costs vary.",
            }
            break

    if not cost:
        cost = {"min_inr": 15000, "max_inr": 100000, "label": case_type, "note": "General estimate."}

    # Court name
    court_name = court
    for c in HC_COURTS:
        if c["s3_code"] == court:
            court_name = c["name"]
            break

    return JSONResponse({
        "court": court_name,
        "case_type": case_type,
        "outcome_probability": outcome_probs,
        "duration": duration,
        "cost_estimate": cost,
    })


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent



@app.get("/predict.html")
async def serve_predict():
    return FileResponse(STATIC_DIR / "predict.html", media_type="text/html")


@app.get("/predict.js")
async def serve_predict_js():
    return FileResponse(STATIC_DIR / "predict.js", media_type="application/javascript")


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
