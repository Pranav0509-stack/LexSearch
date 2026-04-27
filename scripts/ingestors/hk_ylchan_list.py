"""HK case-list ingestor — `ylchan87/HKCourtList`.

Note: the upstream repo is scraper code (no static JSON dumps and no
release artefacts), so this ingestor is a no-op until we either run
the upstream scraper ourselves or the repo starts publishing
releases. Left as a stub so the driver's `--all` mode keeps working.
"""

from __future__ import annotations

import logging
from typing import Iterator

from retrieval_pkg import Document

logger = logging.getLogger(__name__)
SOURCE = "hk_ylchan_list"


def ingest(*, limit: int | None = None) -> Iterator[Document]:
    _ = limit
    logger.info(
        "[%s] no static data available upstream; skipping. "
        "Run the upstream scraper to produce JSON, then port here.",
        SOURCE,
    )
    return
    yield  # pragma: no cover  (makes the function a generator)
