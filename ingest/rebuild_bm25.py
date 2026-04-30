"""
Nightly BM25 index rebuild.

Loads the last N days of parquet (both quarterly partitions and daily/
deltas from sc_daily.py + hc_daily.py), rebuilds the in-memory BM25Okapi
index, and atomically swaps the pickle file served by server.py.

Usage:
    python ingest/rebuild_bm25.py                    # full rebuild (~5 min)
    python ingest/rebuild_bm25.py --window-days 90   # last 90 days only
    python ingest/rebuild_bm25.py --max 50000        # cap for dev
    python ingest/rebuild_bm25.py --out ./bm25.pkl

After success, optionally POST /admin/reload to the running LexSearch
server to hot-reload without a restart (env: LEXSEARCH_ADMIN_URL /
LEXSEARCH_ADMIN_TOKEN).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import s3fs

# Allow `python ingest/rebuild_bm25.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import build_index  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("rebuild_bm25")

IST = ZoneInfo("Asia/Kolkata")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("bm25.pkl"))
    p.add_argument("--window-days", type=int, default=0,
                   help="If >0, only index docs from the last N days (based on decision_date).")
    p.add_argument("--years", type=int, nargs="*",
                   help="Explicit year list (overrides window-days). Default: last 5 years.")
    p.add_argument("--max", type=int, default=0, help="Cap total doc count (dev/testing)")
    args = p.parse_args()

    if args.years:
        years = args.years
    elif args.window_days > 0:
        today = dt.datetime.now(IST).date()
        years = sorted({(today - dt.timedelta(days=d)).year for d in range(args.window_days + 1)}, reverse=True)
    else:
        y = dt.datetime.now(IST).year
        years = [y, y - 1, y - 2, y - 3, y - 4]

    logger.info("Rebuilding BM25 — years=%s max=%s out=%s", years, args.max or "∞", args.out)
    fs = s3fs.S3FileSystem(anon=True)
    idx = build_index(years=years, fs=fs, max_docs=args.max or None)
    idx.save(args.out)
    logger.info("Wrote %s (%d docs)", args.out, len(idx.docs))

    admin_url = os.environ.get("LEXSEARCH_ADMIN_URL")
    admin_token = os.environ.get("LEXSEARCH_ADMIN_TOKEN")
    if admin_url:
        try:
            r = httpx.post(
                admin_url.rstrip("/") + "/admin/reload",
                headers={"Authorization": f"Bearer {admin_token}"} if admin_token else {},
                timeout=15,
            )
            logger.info("Hot-reload ping %s → %d", admin_url, r.status_code)
        except Exception as e:
            logger.warning("Hot-reload ping failed: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
