"""
Judgment RAG tool — wraps the legacy LexSearch BM25/parquet pipeline for
in-dialog citation (plan §2.3).
"""

import hashlib
import json
from typing import Any, Optional

from app.deps import get_redis

CACHE_TTL = 7 * 24 * 3600  # 7 days


def _cache_key(query: str, top_k: int, filters: Optional[dict]) -> str:
    blob = json.dumps({"q": query, "k": top_k, "f": filters or {}}, sort_keys=True)
    return f"jmt:{hashlib.sha1(blob.encode()).hexdigest()}"


async def judgment_search(
    query: str,
    top_k: int = 3,
    filters: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Returns:
      {"citations": [{cnr, title, court, year, snippet}], "cached": bool}
    """
    r = get_redis()
    key = _cache_key(query, top_k, filters)
    cached = await r.get(key)
    if cached:
        return {"citations": json.loads(cached), "cached": True}

    # TODO: real BM25 lookup. For now, call the existing /search endpoint
    # in-process (via app.api.lexsearch) once a programmatic `search()` is
    # exposed there. See plan §13 repo-audit note.
    citations: list[dict] = []

    await r.setex(key, CACHE_TTL, json.dumps(citations))
    return {"citations": citations, "cached": False}
