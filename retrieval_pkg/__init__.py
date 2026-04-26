"""Sanhita retrieval package.

`server.py` imports `from retrieval import …` (legacy module name); we
provide a top-level `retrieval.py` shim that re-exports from here so
both spellings work.
"""

from .index import (
    BM25Index,
    Document,
    build_index,
    doc_to_retrieve_hit,
    tokenize,
)

__all__ = [
    "BM25Index",
    "Document",
    "build_index",
    "doc_to_retrieve_hit",
    "tokenize",
]
