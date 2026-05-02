"""IN ingestor stub — `openjustice-in/ecourts`.

The upstream repo is a *scraper* library, not a pre-built dataset, so
ingestion via static raw URLs returns nothing useful. Left as a stub.

To turn this into a real ingestor: install the upstream library
(`pip install ecourts`), run it against eCourts.gov.in for a list of
court codes, persist the output JSON, then read those files here.
That's a full overnight scrape job — out of scope for the v1 sample.
"""

from __future__ import annotations

import logging
from typing import Iterator

from retrieval_pkg import Document

logger = logging.getLogger(__name__)
SOURCE = "india_openjustice"


def ingest(*, limit: int | None = None) -> Iterator[Document]:
    _ = limit
    logger.info(
        "[%s] upstream is scraper-only; skipping. "
        "See AWS S3 sync at registry.opendata.aws/indian-high-court-judgments/ "
        "for the full IN HC dataset (1.1TB).",
        SOURCE,
    )
    return
    yield  # pragma: no cover
