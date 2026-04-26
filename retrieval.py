"""Compatibility shim — `server.py` does `from retrieval import …`.

The actual code lives in `retrieval_pkg/`. Keeping this file thin so we
don't have to touch every import site in the existing codebase.
"""

from retrieval_pkg import (
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
