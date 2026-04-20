"""
Legacy LexSearch routes — Indian HC + SC judgment search, PDF proxy, static UI.

Moved from server.py unchanged. These stay public (search tool + a free top-of-funnel
for NyayaSathi) and are also consumed internally as the RAG source via
app.rag.judgment_tool.
"""

import io
import logging
import tarfile
import urllib.parse
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import s3fs
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)
router = APIRouter()

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


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _extract_pdf_filename(pdf_link: str) -> str:
    if not pdf_link:
        return ""
    return pdf_link.rstrip("/").split("/")[-1]


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


@router.get("/courts")
def list_courts():
    return JSONResponse(HC_COURTS)


@router.get("/search")
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


@router.get("/pdf/{s3_key:path}")
async def proxy_pdf(s3_key: str, download: bool = False):
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


@router.get("/sc-pdf/{year}/{pdf_name}")
def serve_sc_pdf(year: int, pdf_name: str, download: bool = False):
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


STATIC_DIR = Path(__file__).parent.parent.parent  # repo root


@router.get("/viewer.html")
async def serve_viewer():
    return FileResponse(STATIC_DIR / "viewer.html", media_type="text/html")


@router.get("/style.css")
async def serve_css():
    return FileResponse(STATIC_DIR / "style.css", media_type="text/css")


@router.get("/app.js")
async def serve_js():
    return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")


@router.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")
