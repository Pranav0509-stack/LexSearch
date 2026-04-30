"""
Supreme Court of India — daily judgment ingester.

Pulls the day's judgments from main.sci.gov.in, extracts text from each PDF,
and writes a parquet row per judgment into:

    s3://indian-supreme-court-judgments/metadata/parquet/year={Y}/daily/{YYYY-MM-DD}.parquet

The schema matches the existing `year={Y}/metadata.parquet` columns the rest
of LexSearch already knows how to read, so no server.py changes are needed
to pick up daily deltas — `rebuild_bm25.py` simply globs the `daily/`
partition when it rebuilds.

Usage:
    python ingest/sc_daily.py                # today in IST
    python ingest/sc_daily.py --date 2026-04-17
    python ingest/sc_daily.py --dry-run      # print rows, don't write
    python ingest/sc_daily.py --out ./out    # write locally instead of S3

Env vars:
    SC_JUDGMENTS_URL   — override default cause-list URL (for testing / proxy)
    AWS_PROFILE        — if writing to S3
    LEXSEARCH_S3_WRITE — set to "false" to short-circuit all writes

This intentionally avoids any JS execution; the SC site serves judgment
links as plain HTML + query params.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import logging
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import pdfplumber  # type: ignore
except ImportError:  # pragma: no cover
    pdfplumber = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("sc_daily")

IST = ZoneInfo("Asia/Kolkata")

SC_BASE = "https://main.sci.gov.in"
SC_JUDGMENTS_URL = os.environ.get(
    "SC_JUDGMENTS_URL",
    f"{SC_BASE}/php/case_status/get_judgements.php",
)
USER_AGENT = (
    "NyayaSathiIngest/1.0 (+https://nyaysathi-website.vercel.app; "
    "contact=pranavpandey.pr@gmail.com) httpx"
)

# Output schema — columns server.py's _sc_df_to_results expects.
SC_COLUMNS = [
    "case_id", "cnr", "diary_number", "title", "petitioner", "respondent",
    "citation", "judge", "author_judge", "decision_date", "disposal_nature",
    "path", "pdf_url", "headnote", "description", "source", "ingested_at",
]


@dataclass
class SCJudgment:
    case_id: str
    cnr: str
    diary_number: str
    title: str
    petitioner: str
    respondent: str
    citation: str
    judge: str
    author_judge: str
    decision_date: str          # ISO YYYY-MM-DD
    disposal_nature: str
    path: str                   # filename stem, matches server.py tar convention
    pdf_url: str
    headnote: str
    description: str            # extracted judgment text, first 4KB
    source: str = "sci.gov.in"
    ingested_at: str = ""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def _http_get(url: str, *, client: httpx.Client, params: Optional[dict] = None) -> httpx.Response:
    r = client.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------
def fetch_listing(date: dt.date, client: httpx.Client) -> list[dict]:
    """Return raw judgment records for a given date.

    The SCI endpoint returns JSON-ish HTML. We normalise into dicts that
    downstream parsing can consume. For dates with no judgments (public
    holidays), returns an empty list.
    """
    params = {"from_date": date.isoformat(), "to_date": date.isoformat()}
    try:
        r = _http_get(SC_JUDGMENTS_URL, client=client, params=params)
    except Exception as e:
        logger.error("Listing fetch failed for %s: %s", date, e)
        return []

    text = r.text
    # SCI payloads tend to be table-row HTML. Extract <a href="..pdf"> links
    # with nearby metadata. This parser is intentionally tolerant.
    records = []
    row_re = re.compile(
        r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE,
    )
    cell_re = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL | re.IGNORECASE)
    tag_re = re.compile(r'<[^>]+>')
    href_re = re.compile(r'href=["\']([^"\']+\.pdf)["\']', re.IGNORECASE)
    for row_match in row_re.finditer(text):
        row_html = row_match.group(1)
        href = href_re.search(row_html)
        if not href:
            continue
        cells = [tag_re.sub("", c).strip() for c in cell_re.findall(row_html)]
        pdf_url = href.group(1)
        if not pdf_url.startswith("http"):
            pdf_url = f"{SC_BASE}/{pdf_url.lstrip('/')}"
        records.append({
            "cells": cells,
            "pdf_url": pdf_url,
        })
    logger.info("Listing %s returned %d rows", date, len(records))
    return records


# ---------------------------------------------------------------------------
# Parse a single record
# ---------------------------------------------------------------------------
_DIARY_RE = re.compile(r'(\d{1,8}/\d{4})')
_CITATION_RE = re.compile(r'(\(\d{4}\)\s*\d+\s*SCC\s*\d+|AIR\s*\d{4}\s*SC\s*\d+)')
_VS_RE = re.compile(r'\s+v(?:s|ersus|\.)?\s+', re.IGNORECASE)


def parse_listing_row(rec: dict, date: dt.date) -> Optional[SCJudgment]:
    cells = rec["cells"]
    pdf_url = rec["pdf_url"]
    blob = " | ".join(cells)
    diary = ""
    m = _DIARY_RE.search(blob)
    if m:
        diary = m.group(1)
    citation = ""
    m = _CITATION_RE.search(blob)
    if m:
        citation = m.group(1)
    title = next((c for c in cells if _VS_RE.search(c)), "") or (cells[0] if cells else "")
    petitioner, _, respondent = _VS_RE.split(title, 1)[0], None, None
    parts = _VS_RE.split(title, maxsplit=1)
    if len(parts) == 2:
        petitioner, respondent = parts[0].strip(), parts[1].strip()
    else:
        petitioner, respondent = title.strip(), ""

    path = Path(pdf_url).stem.replace("_EN", "")
    case_id = f"SC-{diary.replace('/', '-')}" if diary else f"SC-{path}"

    return SCJudgment(
        case_id=case_id,
        cnr="",
        diary_number=diary,
        title=title,
        petitioner=petitioner,
        respondent=respondent,
        citation=citation,
        judge="",
        author_judge="",
        decision_date=date.isoformat(),
        disposal_nature="",
        path=path,
        pdf_url=pdf_url,
        headnote="",
        description="",
        ingested_at=dt.datetime.now(IST).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=10))
def fetch_pdf_bytes(url: str, client: httpx.Client) -> bytes:
    r = client.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def extract_pdf_text(pdf_bytes: bytes, max_chars: int = 4096) -> str:
    if pdfplumber is None:
        logger.warning("pdfplumber not installed — skipping text extraction")
        return ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            chunks: list[str] = []
            total = 0
            for page in pdf.pages:
                txt = page.extract_text() or ""
                chunks.append(txt)
                total += len(txt)
                if total >= max_chars:
                    break
            return (" ".join(chunks))[:max_chars]
    except Exception as e:
        logger.warning("PDF extract failed: %s", e)
        return ""


def enrich_with_text(j: SCJudgment, client: httpx.Client) -> SCJudgment:
    if not j.pdf_url:
        return j
    try:
        body = fetch_pdf_bytes(j.pdf_url, client)
    except Exception as e:
        logger.warning("PDF fetch failed for %s: %s", j.case_id, e)
        return j
    text = extract_pdf_text(body)
    if text:
        j.description = text
        j.headnote = text[:800]
    return j


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def write_parquet(rows: list[SCJudgment], date: dt.date, *, out_dir: Optional[Path], dry_run: bool) -> Optional[str]:
    if not rows:
        logger.info("No rows to write for %s", date)
        return None
    df = pd.DataFrame([{c: getattr(r, c, "") for c in SC_COLUMNS} for r in rows])
    rel = f"metadata/parquet/year={date.year}/daily/{date.isoformat()}.parquet"

    if dry_run:
        logger.info("[dry-run] would write %d rows to %s", len(df), rel)
        print(df.head(3).to_string())
        return None

    write_disabled = os.environ.get("LEXSEARCH_S3_WRITE", "true").lower() == "false"
    if out_dir or write_disabled:
        target_dir = out_dir or Path("./out_sc")
        dst = target_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        logger.info("Wrote %d rows → %s", len(df), dst)
        return str(dst)

    import s3fs  # local import so dry-run doesn't require creds
    s3 = s3fs.S3FileSystem()
    uri = f"s3://indian-supreme-court-judgments/{rel}"
    with s3.open(uri, "wb") as f:
        df.to_parquet(f, index=False)
    logger.info("Wrote %d rows → %s", len(df), uri)
    return uri


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="ISO YYYY-MM-DD (default: today IST)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", type=Path, help="Local output dir (skip S3)")
    p.add_argument("--max", type=int, default=0, help="Cap rows for testing")
    p.add_argument("--no-pdf", action="store_true", help="Skip PDF text extraction")
    args = p.parse_args()

    target_date = dt.date.fromisoformat(args.date) if args.date else dt.datetime.now(IST).date()
    logger.info("SC daily ingest — date=%s dry_run=%s", target_date, args.dry_run)

    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        records = fetch_listing(target_date, client)
        if args.max:
            records = records[: args.max]

        parsed: list[SCJudgment] = []
        for rec in records:
            j = parse_listing_row(rec, target_date)
            if j:
                parsed.append(j)

        if not args.no_pdf:
            for j in parsed:
                enrich_with_text(j, client)

    write_parquet(parsed, target_date, out_dir=args.out, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
