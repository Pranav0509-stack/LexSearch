#!/usr/bin/env python3
"""Sanhita — GitHub case-law ingestion driver.

Pulls cases from each registered ingestor and appends them to the
shared `bm25.pkl` index. Designed to be safe to re-run: dedup by
case_id is built into BM25Index.add().

Usage:
    python scripts/ingest_github_data.py --all
    python scripts/ingest_github_data.py --source hk_cuthchow_csv
    python scripts/ingest_github_data.py --source hk_cuthchow_csv --limit 100
    python scripts/ingest_github_data.py --all --top-up   # only new since last
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time
from pathlib import Path

# Make the repo root importable when this script is run directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval_pkg import BM25Index  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")

# Registry — order matters only for log readability.
INGESTORS: dict[str, dict[str, int | str | None]] = {
    "hk_cuthchow_csv":     {"limit": None,  "module": "scripts.ingestors.hk_cuthchow_csv"},
    "hk_ylchan_list":      {"limit": 1000,  "module": "scripts.ingestors.hk_ylchan_list"},
    "sg_codelah":          {"limit": 1000,  "module": "scripts.ingestors.sg_codelah"},
    "sg_lacuna":           {"limit": 8000,  "module": "scripts.ingestors.sg_lacuna"},
    "india_seed_promote":  {"limit": None,  "module": "scripts.ingestors.india_seed_promote"},
    "india_openjustice":   {"limit": 15000, "module": "scripts.ingestors.india_openjustice"},
    "india_vanga_hc":      {"limit": 80000, "module": "scripts.ingestors.india_vanga_hc"},
}

DEFAULT_BM25_PATH = ROOT / "bm25.pkl"


def _load_index() -> BM25Index:
    p = Path(os.environ.get("LEXSEARCH_BM25_PATH", str(DEFAULT_BM25_PATH)))
    return BM25Index.load(p)


def _save_index(idx: BM25Index) -> None:
    p = Path(os.environ.get("LEXSEARCH_BM25_PATH", str(DEFAULT_BM25_PATH)))
    idx.save(p)


def _run_one(name: str, idx: BM25Index, *, limit: int | None) -> int:
    cfg = INGESTORS.get(name)
    if cfg is None:
        logger.error("unknown ingestor: %s", name)
        return 0
    try:
        mod = importlib.import_module(str(cfg["module"]))
    except ImportError as e:
        logger.warning("[%s] not yet implemented (%s) — skipping", name, e)
        return 0
    eff_limit = limit if limit is not None else cfg["limit"]
    t0 = time.monotonic()
    added = idx.add(mod.ingest(limit=eff_limit))
    dt = time.monotonic() - t0
    logger.info("[%s] +%d docs in %.1fs (total index: %d)", name, added, dt, len(idx))
    return added


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="Run a single ingestor by name")
    ap.add_argument("--all", action="store_true", help="Run every registered ingestor")
    ap.add_argument("--limit", type=int, help="Override per-source limit")
    ap.add_argument("--top-up", action="store_true",
                    help="(reserved) Only fetch items newer than current index — currently relies on case_id dedup")
    ap.add_argument("--list", action="store_true", help="List registered ingestors and exit")
    args = ap.parse_args()

    if args.list:
        for k, v in INGESTORS.items():
            print(f"  {k:24s}  default_limit={v['limit']}")
        return 0

    if not args.source and not args.all:
        ap.error("specify --source NAME or --all")

    idx = _load_index()
    logger.info("loaded index: %d existing docs", len(idx))

    total_added = 0
    if args.source:
        total_added += _run_one(args.source, idx, limit=args.limit)
    else:
        for name in INGESTORS.keys():
            total_added += _run_one(name, idx, limit=args.limit)

    if total_added > 0:
        _save_index(idx)
        logger.info("saved index: %d docs (+%d this run)", len(idx), total_added)
    else:
        logger.info("no new docs ingested; not re-saving")

    stats = idx.stats()
    logger.info("stats — by jurisdiction: %s", stats["by_jurisdiction"])
    logger.info("stats — by source:       %s", stats["by_source"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
