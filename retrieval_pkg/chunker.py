"""
Sanhita — Paragraph chunker for Indian judgments.

Indian judgments have a distinctive shape: numbered paragraphs ("1.", "2.",
"15."), often with sub-paras ("(a)", "(i)"), heavy use of citations inline,
and a clear holding-before-discussion convention. Whole-document BM25 ranks
the *case* but loses the *paragraph that actually answers the question* —
which is exactly what Brief needs to cite.

This chunker:

  • Preserves the original paragraph numbering as `para_no` so the UI can
    render "see ¶ 17" and the validator can verify cite spans.
  • Targets ~500 tokens per chunk with 100-token overlap (rough word count;
    we don't tokenize — that's Qdrant's job in Phase B).
  • Merges sub-paras into their parent when the parent is short.
  • Splits monster paragraphs at sentence boundaries.
  • Returns chunks with enough metadata that BM25 / Qdrant / re-rank all
    work off the same shape.

Output shape (one dict per chunk):
  {
    case_id, court, year, citation, title,    # case-level
    chunk_id,        # f"{case_id}#p{para_no}" or f"{case_id}#p{para_no}-{i}"
    para_no,         # int or None for preamble
    para_label,      # "1", "15(a)", "Preamble"
    text,            # the chunk text
    n_tokens,        # rough word count
    is_holding,      # bool — heuristic: contains "we hold" / "it is held"
  }
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator, Optional

# Match a paragraph head like "1.", "1)", "(1)", "15.", followed by space.
# Anchored to line start.
PARA_HEAD_RE = re.compile(r"^\s*(?:\(?(\d{1,3})\)?[.)])\s+", re.MULTILINE)

# Sub-para markers — kept attached to their parent unless the parent is huge.
SUBPARA_RE = re.compile(r"^\s*\(([a-z]|[ivx]{1,4})\)\s+", re.MULTILINE)

# Sentence split — conservative; keeps abbreviations like "Hon'ble" intact.
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\(\[])")

HOLDING_RE = re.compile(
    r"\b(we hold|it is held|we are of the (?:considered )?opinion|"
    r"we direct|the appeal is (?:allowed|dismissed)|"
    r"the petition is (?:allowed|dismissed)|the writ is (?:allowed|dismissed)|"
    r"in (?:our|the) view|we therefore hold)\b",
    re.IGNORECASE,
)

DEFAULT_TARGET_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 100
HARD_MAX_TOKENS = 900           # split anything bigger
MIN_CHUNK_TOKENS = 12           # drop chunks smaller than this (preamble noise)


def _tokens(text: str) -> int:
    return len(text.split())


def _split_paragraphs(body: str) -> list[tuple[Optional[str], str]]:
    """
    Split a judgment body into (para_label, text) pairs.

    Anything before the first numbered paragraph is returned with label None
    (preamble — case caption, judges, counsel).
    """
    if not body:
        return []

    matches = list(PARA_HEAD_RE.finditer(body))
    if not matches:
        return [(None, body.strip())]

    out: list[tuple[Optional[str], str]] = []

    # Preamble (before paragraph 1)
    first = matches[0]
    pre = body[: first.start()].strip()
    if pre:
        out.append((None, pre))

    for i, m in enumerate(matches):
        label = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if not text:
            continue
        out.append((label, text))

    return out


def _split_long_paragraph(text: str, target: int, overlap: int) -> list[str]:
    """
    Split an oversized paragraph at sentence boundaries with overlap.
    """
    if _tokens(text) <= HARD_MAX_TOKENS:
        return [text]

    sentences = SENT_SPLIT_RE.split(text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for sent in sentences:
        st = _tokens(sent)
        if buf and buf_tokens + st > target:
            chunks.append(" ".join(buf).strip())
            # carry over the last `overlap` tokens worth of sentences
            carry: list[str] = []
            carry_tokens = 0
            for prev in reversed(buf):
                pt = _tokens(prev)
                if carry_tokens + pt > overlap:
                    break
                carry.insert(0, prev)
                carry_tokens += pt
            buf = carry
            buf_tokens = carry_tokens
        buf.append(sent)
        buf_tokens += st

    if buf:
        chunks.append(" ".join(buf).strip())

    return [c for c in chunks if c]


def chunk_judgment(
    *,
    case_id: str,
    body: str,
    court: str = "",
    year: Optional[int] = None,
    citation: str = "",
    title: str = "",
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[dict]:
    """
    Chunk one judgment. Pure function — no I/O.
    """
    paras = _split_paragraphs(body)
    chunks: list[dict] = []

    for label, text in paras:
        n = _tokens(text)
        if n < MIN_CHUNK_TOKENS:
            continue

        if n <= HARD_MAX_TOKENS:
            sub_chunks = [text]
        else:
            sub_chunks = _split_long_paragraph(text, target_tokens, overlap_tokens)

        for i, sub in enumerate(sub_chunks):
            para_label = label or "Preamble"
            if len(sub_chunks) > 1:
                chunk_id = f"{case_id}#p{label or 'pre'}-{i+1}"
                para_label = f"{para_label} (cont. {i+1}/{len(sub_chunks)})"
            else:
                chunk_id = f"{case_id}#p{label or 'pre'}"

            chunks.append({
                "case_id": case_id,
                "court": court,
                "year": year,
                "citation": citation,
                "title": title,
                "chunk_id": chunk_id,
                "para_no": int(label) if label and label.isdigit() else None,
                "para_label": para_label,
                "text": sub,
                "n_tokens": _tokens(sub),
                "is_holding": bool(HOLDING_RE.search(sub)),
            })

    return chunks


def chunk_iter(judgments: Iterable[dict]) -> Iterator[dict]:
    """Stream-friendly wrapper. Each input dict needs `case_id` + `body`."""
    for j in judgments:
        for c in chunk_judgment(
            case_id=j["case_id"],
            body=j.get("body") or j.get("text") or "",
            court=j.get("court", ""),
            year=j.get("year"),
            citation=j.get("citation", ""),
            title=j.get("title", ""),
        ):
            yield c
