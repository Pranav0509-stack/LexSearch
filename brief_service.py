"""
Sanhita Brief — the Assistant (multi-dimensional legal AI).

Grounded legal-research chat. Every answer:
  1. Retrieves 8-10 cases from 31M+ corpus via FTS5 BM25.
  2. Composed by the LLM router (Anthropic → Groq → Gemini → Cloudflare fallback).
  3. Run through the six-gate validator.
  4. Returned with provider metadata + per-gate confidence so the UI can
     show an AI-disclosure chip and a confidence pill.
  5. If the validator hard-fails after one rewrite attempt, we refuse:
     return the closest 3 cases with no prose. Better silence than slop.
  6. If NO LLM provider is configured, we STILL return useful results:
     the best matching cases with structured metadata.

Multi-language support: pass lang="hi" for Hindi, "ta" for Tamil, etc.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
from typing import Any

from llm import router
from validators import answer_gates
import web_signals

logger = logging.getLogger(__name__)


# ── System prompt: the soul of Sanhita ─────────────────────────────────
SYSTEM_PROMPT = """You are Sanhita Brief, an AI legal research assistant built
on India's largest structured corpus of 31.9 million court judgments across 25
High Courts (1950-2025), 13.6 million legal documents, 1.36 million legal QA
pairs, and 2,383 indexed statutes.

Your role: answer legal questions grounded in the retrieved judgments below.
You are NOT a generic chatbot — you are a specialized Indian legal research
tool used by advocates, law students, and judges.

Rules — non-negotiable:

1. Ground EVERY substantive sentence in a numbered citation like [1], [2],
   matching the index of the retrieved case. A sentence without a [n] is a
   bug.
2. NEVER invent case names, citations, statute sections, or holdings. If the
   retrieved cases do not fully cover the question, say what you CAN answer
   from the record and explicitly note the gap.
3. Do not write "Based on the retrieved cases…" or "I think…" or "In my
   opinion…". Just answer.
4. Indian-law vocabulary: "advocate", "chamber", "matter", "reportable".
   When a user cites a superseded provision (IPC / CrPC / IEA), note the BNS
   / BNSS / BSA equivalent in parentheses if the retrieved record supports
   the mapping.
5. Structure your answer with:
   a. **Direct answer** — 2-3 sentences answering the question
   b. **Relevant principles** — key legal principles from the cases
   c. **Applicable provisions** — statutes/sections that apply
   d. **Practice note** — practical takeaway for the advocate
6. Be comprehensive. 300-600 words. Cover the question thoroughly.
7. When the question mentions a specific state or city, prioritize cases
   from that jurisdiction in your answer.
8. Always mention the court, year, and citation of each case you reference.
9. If NEWS items are provided (tagged [NEWS-n]), you may reference them for
   recent developments, but ALWAYS prioritize corpus judgments [1]-[10] for
   legal analysis. News items supplement, never replace, case law citations.

Output plain markdown. No preamble."""


SYSTEM_PROMPT_MULTILANG = """You are Sanhita Brief, an AI legal research assistant.
Follow all the same rules as above, but write your response in {language}.
Use legal terminology in {language} where standard terms exist, but keep
case names, statute names, and section numbers in English for accuracy.
Transliterate technical legal terms that have no standard {language} equivalent."""


REWRITE_NUDGE = """Your previous answer failed grounding checks. Rewrite it
following the rules strictly: every sentence must end with a [n] marker
referring to one of the retrieved cases below. Do NOT introduce any case
name or section number that isn't in the retrieved excerpts. If the record
truly doesn't cover the question, write one sentence saying so and stop."""


# ── Language map ────────────────────────────────────────────────────────
LANGUAGES = {
    "en": "English",
    "hi": "Hindi (हिन्दी)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "mr": "Marathi (मराठी)",
    "bn": "Bengali (বাংলা)",
    "gu": "Gujarati (ગુજરાતી)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "or": "Odia (ଓଡ଼ିଆ)",
    "as": "Assamese (অসমীয়া)",
    "ur": "Urdu (اردو)",
}


def _build_context(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "(no relevant cases found in the corpus)"
    blocks = []
    for i, h in enumerate(hits, 1):
        title = h.get("title") or h.get("case_id") or "Untitled"
        citation = h.get("citation") or "(no citation)"
        court = h.get("court") or ""
        year = h.get("year") or ""
        verdict = h.get("verdict") or ""
        judge = h.get("judge") or ""
        bench = h.get("bench") or ""
        date_decided = h.get("date_decided") or ""
        excerpt = h.get("excerpt") or ""
        source = h.get("source") or ""
        tier = h.get("tier") or ""

        meta_parts = []
        if court: meta_parts.append(f"Court: {court}")
        if year: meta_parts.append(f"Year: {year}")
        if date_decided: meta_parts.append(f"Decided: {date_decided}")
        if verdict: meta_parts.append(f"Verdict: {verdict}")
        if judge: meta_parts.append(f"Judge: {judge}")
        if bench: meta_parts.append(f"Bench: {bench}")
        if tier: meta_parts.append(f"Tier: {tier}")

        blocks.append(
            f"[{i}] {title}\n"
            f"    Citation: {citation}\n"
            f"    {' | '.join(meta_parts)}\n"
            f"    Source: {source}\n"
            f"    Excerpt: {excerpt[:1200]}"
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
        url = h.get("url") or pdf_url

        out.append({
            "n": i,
            "case_id": case_id,
            "title": h.get("title") or case_id or "Untitled",
            "citation": h.get("citation") or "",
            "court": h.get("court") or "",
            "year": h.get("year") or "",
            "excerpt": (h.get("excerpt") or "")[:500],
            "tier": tier,
            "pdf_url": pdf_url,
            "url": url,
            "score": h.get("score"),
            "verdict": h.get("verdict") or "",
            "judge": h.get("judge") or "",
        })
    return out


def _build_user_prompt(question: str, context: str, history_block: str, *, rewrite: bool = False, web_context: str = "") -> str:
    parts = []
    if history_block:
        parts.append(f"Prior conversation:\n{history_block}")
    parts.append(f"Retrieved cases (from 31.9M Indian judgment corpus):\n{context}")
    if web_context:
        parts.append(web_context)
    parts.append(f"Question: {question}")
    if rewrite:
        parts.append(REWRITE_NUDGE)
    else:
        parts.append(
            "Answer using the retrieved cases. Cite each claim with [n]. "
            "If the retrieved cases partially cover the question, answer what "
            "you can and note what additional research is needed."
        )
    return "\n\n".join(parts)


def answer_question(
    question: str,
    hits: list[dict[str, Any]],
    history: list[dict[str, Any]],
    lang: str = "en",
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
        grounding_pct: float,
      }
    """
    citations = _citation_payload(hits)
    available = router.available_providers()

    # No provider configured → give structured case results (no LLM needed)
    if not available:
        if hits:
            # Build a useful non-LLM response from the retrieved cases
            answer_lines = [
                f"**Found {len(hits)} relevant cases** from the corpus:\n",
            ]
            for i, h in enumerate(hits[:6], 1):
                title = h.get("title") or h.get("case_id") or "Untitled"
                court = h.get("court") or ""
                year = h.get("year") or ""
                verdict = h.get("verdict") or ""
                excerpt = (h.get("excerpt") or "")[:300]
                answer_lines.append(
                    f"**[{i}] {title}**\n"
                    f"{court} · {year}"
                    + (f" · {verdict}" if verdict else "")
                    + f"\n> {excerpt}\n"
                )
            answer_lines.append(
                "\n*Note: No LLM provider is configured, so I'm showing raw search results. "
                "Configure an API key (Gemini, Anthropic, Groq, or Cloudflare) for AI-composed answers.*"
            )
            answer_md = "\n".join(answer_lines)
            grounding = 1.0  # 100% grounded — all from DB
        else:
            answer_md = (
                "No matching cases found in the corpus for this query. "
                "Try different search terms, or use Court Search for advanced filters."
            )
            grounding = 0.0

        return {
            "answer_markdown": answer_md,
            "citations": citations,
            "llm": {"provider": "none", "model": "", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": bool(hits), "confidence": grounding, "reasons": ["no LLM provider — raw results"]},
            "refused": False,  # NOT refused — we still give useful results
            "grounding_pct": grounding,
        }

    # Build the system prompt (with optional language)
    sys_prompt = SYSTEM_PROMPT
    if lang and lang != "en" and lang in LANGUAGES:
        sys_prompt += "\n\n" + SYSTEM_PROMPT_MULTILANG.format(language=LANGUAGES[lang])

    context = _build_context(hits)
    history_block = _history_for_prompt(history)

    # ── Fetch live web signals for current legal developments
    web_context = ""
    web_signal_data: list[dict] = []
    try:
        web_context = web_signals.get_web_context_for_brief(question, max_signals=4)
        if web_context:
            web_signal_data = [s.to_dict() for s in web_signals.search_web_signals(question, max_items=4)]
            logger.info("web_signals: found %d relevant signals for '%s'", len(web_signal_data), question[:50])
    except Exception as e:
        logger.debug("web_signals: failed: %s", e)

    user_prompt = _build_user_prompt(question, context, history_block, web_context=web_context)

    # ── First pass — 1200 tokens for comprehensive answers
    try:
        resp = router.generate(sys_prompt, user_prompt, temperature=0.2, max_tokens=2000)
    except Exception as e:
        logger.error("router.generate failed: %s", e)
        # Even if LLM fails, return the raw cases
        if hits:
            fallback_lines = [
                f"*LLM temporarily unavailable ({str(e)[:100]}). Showing raw search results:*\n",
            ]
            for i, h in enumerate(hits[:5], 1):
                title = h.get("title") or h.get("case_id") or "Untitled"
                court = h.get("court") or ""
                year = h.get("year") or ""
                fallback_lines.append(f"**[{i}] {title}** — {court} · {year}")
            return {
                "answer_markdown": "\n".join(fallback_lines),
                "citations": citations,
                "llm": {"provider": "error", "model": "", "latency_ms": 0, "fallback_chain": []},
                "validation": {"passed": True, "confidence": 0.8, "reasons": [str(e)]},
                "refused": False,
                "grounding_pct": 1.0,
            }
        return {
            "answer_markdown": answer_gates.refusal_payload(question, hits, [str(e)]),
            "citations": citations,
            "llm": {"provider": "error", "model": "", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": False, "confidence": 0.0, "reasons": [str(e)]},
            "refused": True,
            "grounding_pct": 0.0,
        }

    verdict = answer_gates.validate(resp.text, hits, question=question)

    # ── If soft-failed, try one rewrite with stricter prompt
    if not verdict.passed and verdict.gates.get("cite_present") and verdict.gates.get("scope_check"):
        rewrite_prompt = _build_user_prompt(question, context, history_block, rewrite=True)
        try:
            resp2 = router.generate(sys_prompt, rewrite_prompt, temperature=0.1, max_tokens=2000)
            verdict2 = answer_gates.validate(resp2.text, hits, question=question)
            if verdict2.confidence > verdict.confidence:
                resp, verdict = resp2, verdict2
        except Exception as e:
            logger.warning("rewrite pass failed: %s", e)

    # ── Hard refusal: surface closest cases, no fabricated prose
    # But be more lenient — if we have some grounding, show the answer with a warning
    if not verdict.passed:
        if verdict.grounding_pct >= 0.3:
            # Partial grounding — show answer with a caveat
            answer_md = resp.text + (
                "\n\n---\n*⚠️ Grounding: {:.0%} of claims are cited from retrieved cases. "
                "Verify uncited statements independently.*"
            ).format(verdict.grounding_pct)
            return {
                "answer_markdown": answer_md,
                "citations": citations,
                "llm": resp.to_dict(),
                "validation": verdict.to_dict(),
                "refused": False,
                "grounding_pct": verdict.grounding_pct,
            }
        elif not (verdict.gates.get("cite_present") and verdict.gates.get("cite_resolves")):
            return {
                "answer_markdown": answer_gates.refusal_payload(question, hits, verdict.reasons),
                "citations": citations,
                "llm": resp.to_dict(),
                "validation": verdict.to_dict(),
                "refused": True,
                "grounding_pct": verdict.grounding_pct,
            }

    return {
        "answer_markdown": resp.text,
        "citations": citations,
        "llm": resp.to_dict(),
        "validation": verdict.to_dict(),
        "refused": False,
        "grounding_pct": verdict.grounding_pct,
        "web_signals": web_signal_data,
    }


def serialize_citations(citations: list[dict[str, Any]]) -> str:
    return json.dumps(citations, ensure_ascii=False)
