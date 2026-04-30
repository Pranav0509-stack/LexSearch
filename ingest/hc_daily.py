"""
High Court — daily judgment ingester (top 6 courts).

Pulls same-day "reportable judgments" / order sheets from each court's
public eCourts portal and writes parquet to:

    s3://indian-high-court-judgments/metadata/parquet/year={Y}/court={C}/bench={B}/daily/{YYYY-MM-DD}.parquet

We ship with adapters for the 6 highest-volume courts. Adding another HC is
~30 lines (URL template + row parser).

Respect rules:
    - User-Agent identifies NyayaSathi + contact email
    - 2s gap between requests per court (see `RATE_LIMIT_SECONDS`)
    - Honour the project SCRAPING_POLICY.md — do not hit a court more than
      once per day for the same date window

Usage:
    python ingest/hc_daily.py                             # all courts, today IST
    python ingest/hc_daily.py --court delhi --date 2026-04-17
    python ingest/hc_daily.py --dry-run
    python ingest/hc_daily.py --out ./out_hc
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("hc_daily")

IST = ZoneInfo("Asia/Kolkata")
RATE_LIMIT_SECONDS = float(os.environ.get("HC_DAILY_RATE_LIMIT_S", "2.0"))

USER_AGENT = (
    "NyayaSathiIngest/1.0 (+https://nyaysathi-website.vercel.app; "
    "contact=pranavpandey.pr@gmail.com) httpx"
)

HC_COLUMNS = [
    "cnr", "title", "court", "bench", "judge", "decision_date",
    "disposal_nature", "pdf_link", "description", "source", "ingested_at",
]


@dataclass
class HCJudgment:
    cnr: str
    title: str
    court: str           # human-readable name, matches server.py
    bench: str           # s3 bench code
    judge: str
    decision_date: str
    disposal_nature: str
    pdf_link: str
    description: str = ""
    source: str = ""
    ingested_at: str = ""


# ---------------------------------------------------------------------------
# Court adapter registry
# ---------------------------------------------------------------------------
@dataclass
class HCAdapter:
    key: str                               # CLI name
    court_code: str                        # s3 court code (matches server.py HC_COURTS)
    bench_code: str                        # s3 bench code
    human_name: str
    fetch: Callable[[dt.date, httpx.Client], list[HCJudgment]]


def _rate_sleep() -> None:
    time.sleep(RATE_LIMIT_SECONDS)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
def _http_get(url: str, client: httpx.Client, **kw) -> httpx.Response:
    r = client.get(url, timeout=30, **kw)
    r.raise_for_status()
    return r


# ---- Delhi HC ---------------------------------------------------------------
def fetch_delhi(date: dt.date, client: httpx.Client) -> list[HCJudgment]:
    # Delhi HC judgment-information-system exposes a JSON-ish list at
    # https://dhcmisc.nic.in/djudge/judjson.aspx?from=YYYY-MM-DD&to=YYYY-MM-DD
    # (Exact URL may evolve; adapter is the single replacement point.)
    url = "https://dhcmisc.nic.in/djudge/judjson.aspx"
    r = _http_get(url, client, params={"from": date.isoformat(), "to": date.isoformat()})
    _rate_sleep()
    # Tolerant parse: look for pdf links + titles
    out: list[HCJudgment] = []
    for m in re.finditer(r'href=["\']([^"\']+\.pdf)["\'][^>]*>([^<]+)</a>', r.text, re.IGNORECASE):
        pdf, title = m.group(1), m.group(2).strip()
        out.append(HCJudgment(
            cnr="",
            title=title,
            court="Delhi High Court",
            bench="dhcdb",
            judge="",
            decision_date=date.isoformat(),
            disposal_nature="",
            pdf_link=pdf if pdf.startswith("http") else f"https://dhcmisc.nic.in{pdf}",
            source="dhcmisc.nic.in",
            ingested_at=dt.datetime.now(IST).isoformat(timespec="seconds"),
        ))
    return out


# ---- Bombay HC --------------------------------------------------------------
def fetch_bombay(date: dt.date, client: httpx.Client) -> list[HCJudgment]:
    url = "https://bombayhighcourt.nic.in/generatenew.php"
    r = _http_get(url, client, params={"bhcpar": "judg", "date": date.strftime("%d/%m/%Y")})
    _rate_sleep()
    out = []
    for m in re.finditer(r'href=["\']([^"\']+\.pdf)["\'][^>]*>([^<]{5,200})</a>', r.text, re.IGNORECASE):
        pdf, title = m.group(1), re.sub(r'\s+', ' ', m.group(2)).strip()
        out.append(HCJudgment(
            cnr="", title=title, court="Bombay High Court", bench="newas",
            judge="", decision_date=date.isoformat(), disposal_nature="",
            pdf_link=pdf if pdf.startswith("http") else f"https://bombayhighcourt.nic.in/{pdf.lstrip('/')}",
            source="bombayhighcourt.nic.in",
            ingested_at=dt.datetime.now(IST).isoformat(timespec="seconds"),
        ))
    return out


# ---- Madras HC --------------------------------------------------------------
def fetch_madras(date: dt.date, client: httpx.Client) -> list[HCJudgment]:
    url = "https://www.mhc.tn.gov.in/judis/"
    r = _http_get(url, client, params={"date": date.isoformat()})
    _rate_sleep()
    out = []
    for m in re.finditer(r'href=["\']([^"\']+\.pdf)["\'][^>]*>([^<]{5,200})</a>', r.text, re.IGNORECASE):
        pdf, title = m.group(1), re.sub(r'\s+', ' ', m.group(2)).strip()
        out.append(HCJudgment(
            cnr="", title=title, court="Madras High Court", bench="hc_cis_mas",
            judge="", decision_date=date.isoformat(), disposal_nature="",
            pdf_link=pdf if pdf.startswith("http") else f"https://www.mhc.tn.gov.in/{pdf.lstrip('/')}",
            source="mhc.tn.gov.in",
            ingested_at=dt.datetime.now(IST).isoformat(timespec="seconds"),
        ))
    return out


# ---- Karnataka HC -----------------------------------------------------------
def fetch_karnataka(date: dt.date, client: httpx.Client) -> list[HCJudgment]:
    url = "https://karnatakajudiciary.kar.nic.in/hckfor/daily.asp"
    r = _http_get(url, client, params={"dt": date.strftime("%d/%m/%Y")})
    _rate_sleep()
    out = []
    for m in re.finditer(r'href=["\']([^"\']+\.pdf)["\'][^>]*>([^<]{5,200})</a>', r.text, re.IGNORECASE):
        pdf, title = m.group(1), re.sub(r'\s+', ' ', m.group(2)).strip()
        out.append(HCJudgment(
            cnr="", title=title, court="Karnataka High Court", bench="karnataka_bng_old",
            judge="", decision_date=date.isoformat(), disposal_nature="",
            pdf_link=pdf if pdf.startswith("http") else f"https://karnatakajudiciary.kar.nic.in/{pdf.lstrip('/')}",
            source="karnatakajudiciary.kar.nic.in",
            ingested_at=dt.datetime.now(IST).isoformat(timespec="seconds"),
        ))
    return out


# ---- Calcutta HC ------------------------------------------------------------
def fetch_calcutta(date: dt.date, client: httpx.Client) -> list[HCJudgment]:
    url = "https://www.calcuttahighcourt.gov.in/Judgement-Orders"
    r = _http_get(url, client, params={"judgmentDate": date.isoformat()})
    _rate_sleep()
    out = []
    for m in re.finditer(r'href=["\']([^"\']+\.pdf)["\'][^>]*>([^<]{5,200})</a>', r.text, re.IGNORECASE):
        pdf, title = m.group(1), re.sub(r'\s+', ' ', m.group(2)).strip()
        out.append(HCJudgment(
            cnr="", title=title, court="Calcutta High Court", bench="calcutta_appellate_side",
            judge="", decision_date=date.isoformat(), disposal_nature="",
            pdf_link=pdf if pdf.startswith("http") else f"https://www.calcuttahighcourt.gov.in/{pdf.lstrip('/')}",
            source="calcuttahighcourt.gov.in",
            ingested_at=dt.datetime.now(IST).isoformat(timespec="seconds"),
        ))
    return out


# ---- Allahabad HC -----------------------------------------------------------
def fetch_allahabad(date: dt.date, client: httpx.Client) -> list[HCJudgment]:
    url = "https://www.allahabadhighcourt.in/caseinfo/ordersforjudge.jsp"
    r = _http_get(url, client, params={"date": date.strftime("%d-%m-%Y")})
    _rate_sleep()
    out = []
    for m in re.finditer(r'href=["\']([^"\']+\.pdf)["\'][^>]*>([^<]{5,200})</a>', r.text, re.IGNORECASE):
        pdf, title = m.group(1), re.sub(r'\s+', ' ', m.group(2)).strip()
        out.append(HCJudgment(
            cnr="", title=title, court="Allahabad High Court", bench="cishclko",
            judge="", decision_date=date.isoformat(), disposal_nature="",
            pdf_link=pdf if pdf.startswith("http") else f"https://www.allahabadhighcourt.in/{pdf.lstrip('/')}",
            source="allahabadhighcourt.in",
            ingested_at=dt.datetime.now(IST).isoformat(timespec="seconds"),
        ))
    return out


ADAPTERS: list[HCAdapter] = [
    HCAdapter("delhi", "7_26", "dhcdb", "Delhi High Court", fetch_delhi),
    HCAdapter("bombay", "27_1", "newas", "Bombay High Court", fetch_bombay),
    HCAdapter("madras", "33_10", "hc_cis_mas", "Madras High Court", fetch_madras),
    HCAdapter("karnataka", "29_3", "karnataka_bng_old", "Karnataka High Court", fetch_karnataka),
    HCAdapter("calcutta", "19_16", "calcutta_appellate_side", "Calcutta High Court", fetch_calcutta),
    HCAdapter("allahabad", "9_13", "cishclko", "Allahabad High Court", fetch_allahabad),
]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def write_parquet(
    rows: list[HCJudgment], adapter: HCAdapter, date: dt.date,
    *, out_dir: Optional[Path], dry_run: bool,
) -> Optional[str]:
    if not rows:
        logger.info("No rows for %s/%s on %s", adapter.key, adapter.court_code, date)
        return None
    df = pd.DataFrame([{c: getattr(r, c, "") for c in HC_COLUMNS} for r in rows])
    rel = (
        f"metadata/parquet/year={date.year}/court={adapter.court_code}/"
        f"bench={adapter.bench_code}/daily/{date.isoformat()}.parquet"
    )
    if dry_run:
        logger.info("[dry-run] would write %d rows → %s", len(df), rel)
        print(df.head(3).to_string())
        return None

    if out_dir or os.environ.get("LEXSEARCH_S3_WRITE", "true").lower() == "false":
        target_dir = out_dir or Path("./out_hc")
        dst = target_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        logger.info("Wrote %d rows → %s", len(df), dst)
        return str(dst)

    import s3fs
    s3 = s3fs.S3FileSystem()
    uri = f"s3://indian-high-court-judgments/{rel}"
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
    p.add_argument("--court", choices=[a.key for a in ADAPTERS], help="Run only one court")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    target = dt.date.fromisoformat(args.date) if args.date else dt.datetime.now(IST).date()
    selected = [a for a in ADAPTERS if (args.court is None or a.key == args.court)]
    logger.info("HC daily ingest — date=%s courts=%s", target, [a.key for a in selected])

    ok, failed = 0, 0
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        for adapter in selected:
            try:
                rows = adapter.fetch(target, client)
                write_parquet(rows, adapter, target, out_dir=args.out, dry_run=args.dry_run)
                ok += 1
            except Exception as e:
                logger.exception("Adapter %s failed: %s", adapter.key, e)
                failed += 1
    logger.info("Done. ok=%d failed=%d", ok, failed)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
