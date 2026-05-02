"""SG ingestor stub — `codelah/singapore`.

The upstream repo is a Python package wrapping Singapore government
APIs (data.gov.sg, NEA, OneMap) — those are great for civic data but
contain no court-judgment dumps. Left as a stub so the driver's
`--all` keeps working; revisit if codelah ever ships a legal-data
sub-module.
"""

from __future__ import annotations

import logging
from typing import Iterator

from retrieval_pkg import Document

logger = logging.getLogger(__name__)
SOURCE = "sg_codelah"


def ingest(*, limit: int | None = None) -> Iterator[Document]:
    _ = limit
    logger.info("[%s] no legal-judgment data in this repo; skipping.", SOURCE)
    return
    yield  # pragma: no cover
