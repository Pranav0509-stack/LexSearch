"""
Sanhita Vault — multi-document Q&A.

Advocates upload PDFs / DOCX / TXT (briefs, contracts, statements,
judgment copies) and ask questions across them. Inspired by Harvey Vault
but India-native: supports multi-language text, runs through the same
six-gate validator, and stores everything per-user with DPDP-compliant
retention defaults (90 days, delete-on-demand).

Pipeline:
  upload → extract text → chunk (paragraph chunker) → store chunks in SQLite
         → on query: BM25-rank user's chunks → Claude Sonnet 4.5 over top-k
         → validator → cite as [doc.pdf ¶17]

We don't run vector search on uploaded docs (yet) — BM25 over chunks is
fast, deterministic, and sufficient for typical brief volumes
(<100 docs / advocate).
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from llm import router
from validators import answer_gates
from retrieval_pkg.chunker import chunk_judgment

logger = logging.getLogger(__name__)

UPLOAD_MAX_BYTES = 25 * 1024 * 1024  # 25 MB / file
ALLOWED_MIMES = {"application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"}
ALLOWED_EXTS = {".pdf", ".docx", ".txt"}


SYSTEM_PROMPT_VAULT = """You are Sanhita Vault, an AI research assistant
that answers questions strictly from documents the advocate has uploaded
to their private chamber. Rules — non-negotiable:

1. Cite every substantive sentence as [doc.NN] where NN is the chunk index
   shown in the retrieved context. Do not invent chunk numbers.
2. NEVER pull facts from outside the uploaded documents. If the answer
   is not in the uploaded record, say so plainly in one sentence and stop.
3. Do not write "Based on the documents…" — just answer.
4. Indian-law vocabulary: "advocate", "chamber", "matter".
5. Concise. 200-400 words. End with one line prefixed "Practice note:"
   when there is a practical takeaway in the record.

Output plain markdown. No preamble."""


# ── Text extraction ───────────────────────────────────────────────────────
def extract_text(filename: str, blob: bytes) -> str:
    """Best-effort text extraction. Returns empty string on failure."""
    name = filename.lower()
    try:
        if name.endswith(".pdf"):
            try:
                import pdfplumber  # type: ignore
            except ImportError:
                logger.warning("pdfplumber not installed; PDF extraction disabled")
                return ""
            text_parts = []
            with pdfplumber.open(io.BytesIO(blob)) as pdf:
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n\n".join(text_parts)
        elif name.endswith(".docx"):
            try:
                from docx import Document  # type: ignore
            except ImportError:
                logger.warning("python-docx not installed; DOCX extraction disabled")
                return ""
            doc = Document(io.BytesIO(blob))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif name.endswith(".txt"):
            return blob.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("extract_text failed for %s: %s", filename, e)
    return ""


def chunk_document(doc_id: int, filename: str, text: str) -> list[dict]:
    """Chunk an uploaded document. Reuses the judgment chunker; non-judgment
    text just gets sentence-split."""
    if not text.strip():
        return []
    chunks = chunk_judgment(case_id=f"doc{doc_id}", body=text, title=filename)
    # If the chunker returned nothing (no numbered paragraphs), force a
    # single chunk so the doc is at least indexable.
    if not chunks:
        chunks = [{
            "case_id": f"doc{doc_id}", "court": "", "year": None,
            "citation": "", "title": filename,
            "chunk_id": f"doc{doc_id}#full", "para_no": None,
            "para_label": "Full", "text": text[:8000],
            "n_tokens": len(text.split()), "is_holding": False,
        }]
    return chunks


# ── BM25 over user chunks ──────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def rank_chunks(query: str, chunks: list[dict], k: int = 8) -> list[dict]:
    """
    Tiny in-memory BM25 across the user's chunks. We keep this dependency-
    free (no rank_bm25 import) because chunk volumes per advocate are small
    (typically <2k chunks). Scoring: tf-idf with length normalization.
    """
    if not chunks:
        return []
    q_terms = _tokenize(query)
    if not q_terms:
        return []

    # IDF
    N = len(chunks)
    df: dict[str, int] = {}
    chunk_terms: list[list[str]] = []
    for c in chunks:
        terms = _tokenize(c.get("text", ""))
        chunk_terms.append(terms)
        for t in set(terms):
            df[t] = df.get(t, 0) + 1
    import math
    idf = {t: math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5)) for t in q_terms}

    avgdl = sum(len(t) for t in chunk_terms) / max(1, N)
    k1, b = 1.5, 0.75

    scored: list[tuple[float, dict]] = []
    for c, terms in zip(chunks, chunk_terms):
        if not terms:
            continue
        dl = len(terms)
        tf: dict[str, int] = {}
        for t in terms:
            if t in idf:
                tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t, f in tf.items():
            denom = f + k1 * (1 - b + b * dl / avgdl)
            score += idf[t] * (f * (k1 + 1) / denom)
        if score > 0:
            scored.append((score, c))

    scored.sort(key=lambda x: -x[0])
    return [{**c, "score": float(s)} for s, c in scored[:k]]


# ── Q&A entrypoint ─────────────────────────────────────────────────────────
def _build_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(
            f"[{i}] {h.get('title','doc')} ¶ {h.get('para_label','?')}\n"
            f"    {(h.get('text') or '')[:900]}"
        )
    return "\n\n".join(blocks) if blocks else "(no relevant excerpts in your uploaded documents)"


def answer_over_vault(question: str, hits: list[dict], history: list[dict]) -> dict[str, Any]:
    citations = [{
        "n": i + 1,
        "doc_title": h.get("title", "doc"),
        "para_label": h.get("para_label", ""),
        "excerpt": (h.get("text") or "")[:400],
        "score": h.get("score"),
    } for i, h in enumerate(hits)]

    if not router.available_providers():
        return {
            "answer_markdown": answer_gates.refusal_payload(question, hits, ["no LLM provider configured"]),
            "citations": citations, "refused": True,
            "llm": {"provider": "none", "model": "", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": False, "confidence": 0.0, "reasons": ["no LLM"]},
        }

    history_block = ""
    if history:
        lines = []
        for m in history[-6:]:
            role = "User" if m.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {m.get('content','').strip()}")
        history_block = "\n\n".join(lines)

    user_prompt = (
        (f"Prior conversation:\n{history_block}\n\n" if history_block else "")
        + f"Uploaded excerpts:\n{_build_context(hits)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the uploaded excerpts. Cite each claim with [n]."
    )

    try:
        resp = router.generate(SYSTEM_PROMPT_VAULT, user_prompt, temperature=0.1, max_tokens=900)
    except Exception as e:
        logger.error("vault router failed: %s", e)
        return {
            "answer_markdown": "I couldn't reach the language model. Try again in a moment.",
            "citations": citations, "refused": True,
            "llm": {"provider": "error", "model": "", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": False, "confidence": 0.0, "reasons": [str(e)]},
        }

    verdict = answer_gates.validate(resp.text, hits, question=question)
    refused = not (verdict.gates.get("cite_present") and verdict.gates.get("cite_resolves"))
    return {
        "answer_markdown": resp.text if not refused else answer_gates.refusal_payload(question, hits, verdict.reasons),
        "citations": citations,
        "refused": refused,
        "llm": resp.to_dict(),
        "validation": verdict.to_dict(),
    }
