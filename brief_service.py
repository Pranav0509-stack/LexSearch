"""
Sanhita Brief — the Assistant.

Grounded legal-research chat. Every answer:
  1. Composed by the LLM router (Groq → Cloudflare → Gemini fallback).
  2. Run through the six-gate validator.
  3. Returned with provider metadata + per-gate confidence so the UI can
     show an AI-disclosure chip and a confidence pill.
  4. If the validator hard-fails after one rewrite attempt, we refuse:
     return the closest 3 cases with no prose. Better silence than slop.

The earlier "LLM not configured" dead-end is gone — if no provider has
keys, we now refuse with the closest cases instead of pretending.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
from typing import Any

from llm import router
from validators import answer_gates

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are Sanhita Brief, an AI research assistant for Indian
advocates. You answer legal questions strictly from the retrieved judgments
provided below.

Rules — non-negotiable:

1. Ground EVERY substantive sentence in a numbered citation like [1], [2],
   matching the index of the retrieved case. A sentence without a [n] is a
   bug.
2. NEVER invent case names, citations, statute sections, or holdings. If the
   retrieved cases do not cover the question, say so plainly in one
   sentence and stop.
3. Do not write "Based on the retrieved cases…" or "I think…" or "In my
   opinion…". Just answer.
4. Indian-law vocabulary: "advocate", "chamber", "matter", "reportable".
   When a user cites a superseded provision (IPC / CrPC / IEA), note the BNS
   / BNSS / BSA equivalent in parentheses if the retrieved record supports
   the mapping.
5. Be concise. 200-400 words. End with one line prefixed "Practice note:"
   when a practical takeaway exists in the record.

Output plain markdown. No preamble."""


REWRITE_NUDGE = """Your previous answer failed grounding checks. Rewrite it
following the rules strictly: every sentence must end with a [n] marker
referring to one of the retrieved cases below. Do NOT introduce any case
name or section number that isn't in the retrieved excerpts. If the record
truly doesn't cover the question, write one sentence saying so and stop."""


def _build_context(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "(no relevant cases found in the corpus)"
    blocks = []
    for i, h in enumerate(hits, 1):
        title = h.get("title") or h.get("case_id") or "Untitled"
        citation = h.get("citation") or "(no citation)"
        court = h.get("court") or ""
        year = h.get("year") or ""
        excerpt = h.get("excerpt") or ""
        blocks.append(
            f"[{i}] {title}\n"
            f"    Citation: {citation}\n"
            f"    Court/Year: {court} / {year}\n"
            f"    Excerpt: {excerpt[:700]}"
        )
    return "\n\n".join(blocks)


def _history_for_prompt(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = []
    for m in history[-8:]:
        role = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {m.get('content','').strip()}")
    return "\n\n".join(lines)


def _citation_payload(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, h in enumerate(hits, 1):
        case_id = h.get("case_id") or ""
        pdf_url = ""
        tier = (h.get("tier") or "").upper()
        if tier == "HC" and h.get("s3_key"):
            pdf_url = f"/pdf/{urllib.parse.quote(h['s3_key'], safe='')}"
        elif tier == "SC" and h.get("pdf_name") and h.get("year"):
            pdf_url = f"/sc-pdf/{h['year']}/{urllib.parse.quote(h['pdf_name'])}"

        out.append({
            "n": i,
            "case_id": case_id,
            "title": h.get("title") or case_id or "Untitled",
            "citation": h.get("citation") or "",
            "court": h.get("court") or "",
            "year": h.get("year") or "",
            "excerpt": (h.get("excerpt") or "")[:400],
            "tier": tier,
            "pdf_url": pdf_url,
            "score": h.get("score"),
        })
    return out


def _build_user_prompt(question: str, context: str, history_block: str, *, rewrite: bool = False) -> str:
    parts = []
    if history_block:
        parts.append(f"Prior conversation:\n{history_block}")
    parts.append(f"Retrieved cases:\n{context}")
    parts.append(f"Question: {question}")
    if rewrite:
        parts.append(REWRITE_NUDGE)
    else:
        parts.append("Answer using only the retrieved cases. Cite each claim with [n].")
    return "\n\n".join(parts)


def answer_question(
    question: str,
    hits: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compose a grounded, validated answer.

    Returns:
      {
        answer_markdown: str,
        citations: [...],
        llm: { provider, model, latency_ms, fallback_chain },
        validation: { passed, confidence, gates, reasons, ... },
        refused: bool,
      }
    """
    citations = _citation_payload(hits)
    available = router.available_providers()

    # No provider configured → refuse cleanly (no more "LLM not configured" cop-out).
    if not available:
        return {
            "answer_markdown": answer_gates.refusal_payload(question, hits, ["no LLM provider configured"]),
            "citations": citations,
            "llm": {"provider": "none", "model": "", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": False, "confidence": 0.0, "reasons": ["no LLM provider configured"]},
            "refused": True,
        }

    context = _build_context(hits)
    history_block = _history_for_prompt(history)
    user_prompt = _build_user_prompt(question, context, history_block)

    # ── First pass (fast lane: 600 tokens is enough for 350-400 word answers)
    try:
        resp = router.generate(SYSTEM_PROMPT, user_prompt, temperature=0.2, max_tokens=600)
    except Exception as e:
        logger.error("router.generate failed: %s", e)
        return {
            "answer_markdown": answer_gates.refusal_payload(question, hits, [str(e)]),
            "citations": citations,
            "llm": {"provider": "error", "model": "", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": False, "confidence": 0.0, "reasons": [str(e)]},
            "refused": True,
        }

    verdict = answer_gates.validate(resp.text, hits, question=question)

    # ── If soft-failed, try one rewrite with stricter prompt
    if not verdict.passed and verdict.gates.get("cite_present") and verdict.gates.get("scope_check"):
        rewrite_prompt = _build_user_prompt(question, context, history_block, rewrite=True)
        try:
            resp2 = router.generate(SYSTEM_PROMPT, rewrite_prompt, temperature=0.1, max_tokens=600)
            verdict2 = answer_gates.validate(resp2.text, hits, question=question)
            if verdict2.confidence > verdict.confidence:
                resp, verdict = resp2, verdict2
        except Exception as e:
            logger.warning("rewrite pass failed: %s", e)

    # ── Hard refusal: surface closest cases, no fabricated prose
    if not verdict.passed and not (verdict.gates.get("cite_present") and verdict.gates.get("cite_resolves")):
        return {
            "answer_markdown": answer_gates.refusal_payload(question, hits, verdict.reasons),
            "citations": citations,
            "llm": resp.to_dict(),
            "validation": verdict.to_dict(),
            "refused": True,
        }

    return {
        "answer_markdown": resp.text,
        "citations": citations,
        "llm": resp.to_dict(),
        "validation": verdict.to_dict(),
        "refused": False,
    }


def serialize_citations(citations: list[dict[str, Any]]) -> str:
    return json.dumps(citations, ensure_ascii=False)
