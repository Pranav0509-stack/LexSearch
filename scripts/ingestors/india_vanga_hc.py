"""IN ingestor stub — `vanga/indian-high-court-judgments`.

The full dataset is 1.1 TB across 25 Indian High Courts and lives on
AWS Open Data Registry (`s3://indian-high-court-judgments/`), not on
GitHub raw. Pulling even a 80K-doc sample requires authenticated S3
access, AWS CLI, and several GB of staging space — well beyond a
zero-config first run.

Left as a stub so the driver's `--all` doesn't crash. The cheap path
to real Indian results today is `india_seed_promote` (30 landmark SC
cases) plus `connectors.indian_kanoon_search` (live API; needs a key).
"""

from __future__ import annotations

import logging
from typing import Iterator

from retrieval_pkg import Document

logger = logging.getLogger(__name__)
SOURCE = "india_vanga_hc"


def ingest(*, limit: int | None = None) -> Iterator[Document]:
    _ = limit
    logger.info(
        "[%s] full dataset on AWS S3 (1.1TB) — skipped. "
        "Use `aws s3 sync s3://indian-high-court-judgments/ ...` to "
        "stage parquet partitions, then read them here via pyarrow.",
        SOURCE,
    )
    return
    yield  # pragma: no cover
