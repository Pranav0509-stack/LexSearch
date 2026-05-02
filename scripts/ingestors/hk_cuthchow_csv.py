"""HK case-law ingestor — `cuthchow/Hong-Kong-Courts/case_df.csv`.

The CSV is a single ~930KB file with 6,091 Hong Kong court judgments.
Columns: case_name, d_rep, decision, judge, judge_title, link, p_rep.

`case_name` packs title + citation + filing number + decision date in
one string, e.g.:
    MENFOND ELECTRONIC … [2013] HKCFI 7; [2013] 2 HKC 259; HCA 293/2011 (3 January 2013)

We split that into a clean title + the most authoritative citation,
extract the year and tier from `[YYYY] HK<TIER>`, and use the full
`decision` column as the indexable text.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import time
from typing import Iterator

from retrieval_pkg import Document
from . import _common as C

logger = logging.getLogger(__name__)

CSV_URL = "https://raw.githubusercontent.com/cuthchow/Hong-Kong-Courts/master/case_df.csv"
SOURCE = "hk_cuthchow_csv"

# Pull citation + year + tier out of the trailing bracketed parts of the
# case_name string. Anchored to find the FIRST `[YYYY] HK<XX>` token.
_CITATION_RE = re.compile(r"\[(\d{4})\]\s*HK([A-Z]{2,4})\s*\d*")


def _split_title(case_name: str) -> tuple[str, str]:
    """Return (clean_title, full_citation_block)."""
    if not case_name:
        return "", ""
    cn = case_name.strip()
    m = _CITATION_RE.search(cn)
    if not m:
        return cn[:200], ""
    # Title is everything before the first citation token.
    title = cn[: m.start()].strip().rstrip(",;: ")
    # Citation block is everything from the citation onwards (often
    # "[2013] HKCFI 7; [2013] 2 HKC 259; HCA 293/2011 (3 January 2013)").
    citation_block = cn[m.start():].strip()
    return title or cn[:200], citation_block


def ingest(*, limit: int | None = None) -> Iterator[Document]:
    logger.info("[%s] downloading %s", SOURCE, CSV_URL)
    raw = C.http_get(CSV_URL).decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    now = int(time.time())
    yielded = 0
    for row in reader:
        if limit is not None and yielded >= limit:
            break
        case_name = (row.get("case_name") or "").strip()
        decision = C.clean_text(row.get("decision"), max_chars=10000)
        if not case_name and not decision:
            continue
        title, citation = _split_title(case_name)
        year = C.extract_year(citation, case_name)
        tier = C.hk_tier_from_citation(citation)
        # Court name guessed from tier suffix.
        court_map = {
            "CFA": "Court of Final Appeal",
            "CA":  "Court of Appeal",
            "CFI": "Court of First Instance",
            "DC":  "District Court",
            "MC":  "Magistrates' Court",
            "FC":  "Family Court",
            "LdT": "Lands Tribunal",
            "LBT": "Labour Tribunal",
        }
        court = court_map.get(tier, "Hong Kong Courts")

        case_id = C.stable_case_id("hk", year or "x", tier or "x", title[:40])
        link = (row.get("link") or "").strip()

        doc = Document(
            case_id=case_id,
            title=title,
            text=decision or title,
            court=court,
            year=year,
            citation=citation,
            jurisdiction="HK",
            tier=tier,
            url=link,
            source=SOURCE,
            added_at=now,
            extra={
                "judge": (row.get("judge") or "").strip(),
                "judge_title": (row.get("judge_title") or "").strip(),
                "p_rep": (row.get("p_rep") or "").strip(),
                "d_rep": (row.get("d_rep") or "").strip(),
            },
        )
        yielded += 1
        yield doc
    logger.info("[%s] yielded %d docs", SOURCE, yielded)
