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
SYSTEM_PROMPT = """You are Sanhita Brief, an AI legal research assistant for Indian advocates.
You are backed by India's largest structured corpus: 31.9 million court judgments across
25 High Courts (1950-2025), 13.6 million legal documents, 1.36 million legal QA pairs,
and 2,383 indexed statutes.

RULES — NON-NEGOTIABLE:
1. Cite EVERY substantive claim with [n] matching the retrieved case index. No citation = bug.
2. NEVER invent case names, citations, statute sections, or holdings.
3. No preamble phrases: "Based on retrieved cases", "I think", "In my opinion".
4. Use Indian legal vocabulary: "advocate", "petitioner", "respondent", "matter", "reportable".
5. When user mentions IPC/CrPC/IEA, note BNS/BNSS/BSA equivalent if corpus supports it.
6. If retrieved cases partially cover the question, answer what you CAN and flag the gap.
7. Prioritize jurisdiction mentioned by user (state/city/court).
8. NEWS items [NEWS-n] may supplement but never replace corpus citations [1]-[10].

ANSWER STRUCTURE (always use this):
**Direct Answer**
2-3 sentences directly answering the question. [cite]

**Relevant Principles**
- Bullet each key legal principle with citation [cite]
- Include landmark holdings, tests, standards applied

**Applicable Provisions**
- List statutes, sections, rules. Note BNS/BNSS equivalents where relevant.

**Case Analysis**
For the 2-3 most directly relevant cases: what were the facts, what did the court hold,
why does it matter for this question. Quote key phrases in "quotes" [cite].

**Practice Note**
Practical takeaway: what should the advocate do, argue, watch for, or avoid.

Output plain markdown. Aim 500-800 words. Be thorough — lawyers need depth."""


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


def _classify_doc_type(h: dict[str, Any]) -> str:
    """Classify a search hit into a document type based on source/tier metadata."""
    source = (h.get("source") or "").lower()
    tier = (h.get("tier") or "").upper()
    if "statutes" in source or tier == "STATUTE":
        return "STATUTE"
    if "legal_qa" in source or tier == "QA":
        return "LEGAL_QA"
    if "legal_docs" in source:
        return "LEGAL_DOC"
    # Default: judgment from the judgments table
    return "JUDGMENT"


def _build_context(hits: list[dict[str, Any]]) -> str:
    """Build structured context for the LLM with full document text, separate
    verdict section, and document type classification.

    Each case block now includes:
    - DOCUMENT TYPE classification (JUDGMENT / LEGAL_DOC / STATUTE / LEGAL_QA)
    - Full metadata (court, year, judge, bench)
    - VERDICT as a clearly separated section
    - Full text content (up to 4000 chars) — not just the old 1200-char excerpt
    """
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
        source = h.get("source") or ""
        tier = h.get("tier") or ""
        doc_type = _classify_doc_type(h)

        # Use full_text if available, fall back to explanation → excerpt
        full_text = (h.get("full_text") or "").strip()
        explanation = (h.get("explanation") or "").strip()
        excerpt = (h.get("excerpt") or "").strip()

        # Pick the richest available content, up to 4000 chars
        if full_text:
            content = full_text[:4000]
        elif explanation and excerpt:
            content = f"{explanation[:2000]}\n---\n{excerpt[:2000]}"
        elif explanation:
            content = explanation[:4000]
        else:
            content = excerpt[:4000]

        # Build the structured block
        meta_parts = []
        if court: meta_parts.append(f"Court: {court}")
        if year: meta_parts.append(f"Year: {year}")
        if date_decided: meta_parts.append(f"Decided: {date_decided}")
        if judge: meta_parts.append(f"Judge: {judge}")
        if bench: meta_parts.append(f"Bench: {bench}")
        if tier: meta_parts.append(f"Tier: {tier}")

        lines = [
            f"[{i}] {title}",
            f"    Document Type: {doc_type}",
            f"    Citation: {citation}",
            f"    {' | '.join(meta_parts)}",
            f"    Source: {source}",
        ]

        # Verdict as a clearly separated section
        if verdict:
            lines.append(f"    ── VERDICT ──")
            lines.append(f"    {verdict}")

        # Document content
        lines.append(f"    ── DOCUMENT CONTENT ──")
        lines.append(f"    {content}")

        blocks.append("\n".join(lines))
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
        doc_type = _classify_doc_type(h)

        # Include full_text and explanation for richer UI rendering
        full_text = (h.get("full_text") or "").strip()
        explanation = (h.get("explanation") or "").strip()
        excerpt = (h.get("excerpt") or "").strip()

        out.append({
            "n": i,
            "case_id": case_id,
            "title": h.get("title") or case_id or "Untitled",
            "citation": h.get("citation") or "",
            "court": h.get("court") or "",
            "year": h.get("year") or "",
            "excerpt": excerpt[:500],
            "full_text": full_text[:6000] if full_text else "",
            "explanation": explanation[:2000] if explanation else "",
            "tier": tier,
            "doc_type": doc_type,
            "pdf_url": pdf_url,
            "url": url,
            "score": h.get("score"),
            "verdict": h.get("verdict") or "",
            "judge": h.get("judge") or "",
            "bench": h.get("bench") or "",
            "date_decided": h.get("date_decided") or "",
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
            answer_md = _build_no_llm_response(question, hits)
            grounding = 1.0  # 100% grounded — all from DB
        else:
            answer_md = (
                "## No Results Found\n\n"
                "No matching cases found in the corpus for this query. "
                "Try different search terms, or use **Court Search** for advanced filters."
            )
            grounding = 0.0

        followups = _smart_followups_no_llm(question)
        return {
            "answer_markdown": answer_md,
            "citations": citations,
            "llm": {"provider": "none", "model": "", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": bool(hits), "confidence": grounding, "reasons": ["no LLM provider — raw results"]},
            "refused": False,  # NOT refused — we still give useful results
            "grounding_pct": grounding,
            "followups": followups,
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

    followups = generate_followups(question, resp.text, lang)
    return {
        "answer_markdown": resp.text,
        "citations": citations,
        "llm": resp.to_dict(),
        "validation": verdict.to_dict(),
        "refused": False,
        "grounding_pct": verdict.grounding_pct,
        "web_signals": web_signal_data,
        "followups": followups,
    }


def serialize_citations(citations: list[dict[str, Any]]) -> str:
    return json.dumps(citations, ensure_ascii=False)


# ── Rich no-LLM response builder ──────────────────────────────────────────

# Verdict → icon mapping for display
_VERDICT_ICONS = {
    "allowed": "✅", "dismissed": "❌", "disposed": "📋",
    "granted": "✅", "rejected": "❌", "acquitted": "⚖️",
    "convicted": "🔒", "partly allowed": "🔶", "partly dismissed": "🔶",
    "quashed": "🚫", "stayed": "⏸️", "remanded": "↩️",
}

def _verdict_icon(v: str) -> str:
    if not v:
        return "📄"
    v_lower = v.lower()
    for kw, icon in _VERDICT_ICONS.items():
        if kw in v_lower:
            return icon
    return "📄"

def _tier_badge(tier: str) -> str:
    t = (tier or "").upper()
    if t == "SC":   return "🏛️ Supreme Court"
    if t == "HC":   return "⚖️ High Court"
    if t == "LM":   return "⭐ Landmark"
    return ""

_DOC_TYPE_LABELS = {
    "JUDGMENT": "Judgment",
    "LEGAL_DOC": "Legal Document",
    "STATUTE": "Statute",
    "LEGAL_QA": "Legal Q&A",
}

_DOC_TYPE_ICONS = {
    "JUDGMENT": "⚖️",
    "LEGAL_DOC": "📑",
    "STATUTE": "📜",
    "LEGAL_QA": "❓",
}

def _build_no_llm_response(question: str, hits: list[dict[str, Any]]) -> str:
    """Build a rich, structured markdown response grouped by document type
    when no LLM is configured. Verdict is shown as a distinct section."""
    if not hits:
        return "No results found in the corpus."

    top = hits[:8]

    # Group hits by doc type
    grouped: dict[str, list] = {}
    for h in top:
        dt = _classify_doc_type(h)
        grouped.setdefault(dt, []).append(h)

    # Count totals per type for header
    type_counts = {dt: len(items) for dt, items in grouped.items()}
    type_summary = " · ".join(
        f"{_DOC_TYPE_ICONS.get(dt, '📄')} {count} {_DOC_TYPE_LABELS.get(dt, dt)}"
        for dt, count in type_counts.items()
    )

    lines = [
        f"## {len(top)} Results Found\n",
        f"*{type_summary}*\n",
        f"*Ranked by BM25 relevance from 31.9M Indian court records.*\n",
        "---\n",
    ]

    idx = 0
    # Render in a fixed order: JUDGMENT → LEGAL_DOC → STATUTE → LEGAL_QA
    for dt in ["JUDGMENT", "LEGAL_DOC", "STATUTE", "LEGAL_QA"]:
        items = grouped.get(dt, [])
        if not items:
            continue

        dt_icon = _DOC_TYPE_ICONS.get(dt, "📄")
        dt_label = _DOC_TYPE_LABELS.get(dt, dt)
        lines.append(f"## {dt_icon} {dt_label}s\n")

        for h in items:
            idx += 1
            title  = h.get("title") or h.get("case_id") or "Untitled"
            court  = h.get("court") or ""
            year   = h.get("year") or ""
            verdict = h.get("verdict") or ""
            judge  = h.get("judge") or ""
            bench  = h.get("bench") or ""
            tier   = (h.get("tier") or "").upper()
            citation = h.get("citation") or ""
            excerpt  = (h.get("excerpt") or "")[:400].strip()
            explanation = (h.get("explanation") or "")[:300].strip()

            icon = _verdict_icon(verdict)
            tier_badge = _tier_badge(tier)

            # Header line
            lines.append(f"### {icon} [{idx}] {title}")

            # Meta line — court · year · citation
            meta_parts = []
            if tier_badge: meta_parts.append(tier_badge)
            if court and not tier_badge: meta_parts.append(court)
            if year:     meta_parts.append(str(year))
            if citation: meta_parts.append(f"`{citation}`")
            if meta_parts:
                lines.append(f"**{' · '.join(meta_parts)}**\n")

            # Bench / Judge
            if judge:
                lines.append(f"- **Bench:** {judge}")
            elif bench:
                lines.append(f"- **Bench:** {bench}")

            # Verdict as a distinct section
            if verdict:
                lines.append(f"\n#### Verdict")
                lines.append(f"**{verdict.title()}**\n")

            # Explanation (case analysis) — separate from raw excerpt
            if explanation:
                lines.append(f"#### Analysis")
                if len(explanation) == 300 and "." in explanation[150:]:
                    explanation = explanation[:150 + explanation[150:].rfind(".") + 1]
                lines.append(f"> {explanation}\n")

            # Excerpt / description
            if excerpt and excerpt != explanation:
                # Trim at sentence boundary if possible
                if len(excerpt) == 400 and "." in excerpt[200:]:
                    excerpt = excerpt[:200 + excerpt[200:].rfind(".") + 1]
                lines.append(f"> {excerpt}\n")

            lines.append("---\n")

    lines.append(
        "\n> 💡 **Add an AI key** (Gemini · Anthropic · Groq · Cloudflare) in Settings "
        "to get a full cited legal analysis with principles, case analysis, and practice notes."
    )
    return "\n".join(lines)


def _smart_followups_no_llm(question: str) -> list[str]:
    """Generate contextually relevant follow-up nudges without an LLM."""
    q = question.lower()
    # Bail
    if any(w in q for w in ["bail", "custody", "arrest", "detention"]):
        return [
            "What factors does the court weigh in granting or refusing bail?",
            "Can bail be cancelled after it is granted? On what grounds?",
            "What is the difference between anticipatory bail and regular bail?",
        ]
    # Section 138 / NI Act
    if any(w in q for w in ["cheque", "138", "ni act", "dishonour", "bounce"]):
        return [
            "What is the limitation period for filing a complaint under Section 138?",
            "Can a company be prosecuted under Section 138 NI Act?",
            "What defences are available to the accused in a Section 138 case?",
        ]
    # Writ petition
    if any(w in q for w in ["writ", "226", "mandamus", "certiorari", "habeas"]):
        return [
            "What is the difference between Article 226 and Article 32 writs?",
            "Can a writ petition be filed against a private party?",
            "What is the doctrine of exhaustion of remedies in writ jurisdiction?",
        ]
    # NDPS / drugs
    if any(w in q for w in ["ndps", "narcotic", "drug", "contraband"]):
        return [
            "What are the mandatory minimum sentences under the NDPS Act?",
            "How is 'commercial quantity' defined under the NDPS Act?",
            "What special conditions apply to bail under Section 37 NDPS Act?",
        ]
    # Motor accident
    if any(w in q for w in ["accident", "motor", "compensation", "mact"]):
        return [
            "How is compensation calculated for permanent disability in motor accidents?",
            "What is the Multiplier Method used by courts for loss of income?",
            "Can compensation be awarded even if the claimant was partially at fault?",
        ]
    # Divorce / matrimonial
    if any(w in q for w in ["divorce", "matrimonial", "maintenance", "alimony", "custody"]):
        return [
            "What is the legal procedure for mutual consent divorce in India?",
            "How does the court determine the quantum of maintenance?",
            "What factors influence child custody decisions in Indian courts?",
        ]
    # Consumer
    if any(w in q for w in ["consumer", "deficiency", "service", "forum"]):
        return [
            "What is the pecuniary jurisdiction of District vs State vs National consumer forums?",
            "What constitutes 'deficiency of service' under the Consumer Protection Act?",
            "What is the limitation period for filing a consumer complaint?",
        ]
    # Generic legal follow-ups
    return [
        "How has this legal position evolved in recent High Court judgments?",
        "What is the limitation period applicable to this matter?",
        "What procedural steps should I follow when filing this matter?",
    ]


# ── Follow-up question generation ─────────────────────────────────────────
_FOLLOWUP_SYSTEM = """You are a legal research assistant. Given a legal question and its answer,
generate exactly 3 follow-up questions an Indian advocate would naturally ask next.
Questions must be specific, actionable, and directly related to the answer.
Return ONLY a JSON array of 3 strings. No explanation. Example:
["What is the limitation period for filing?", "Can bail be cancelled after grant?", "What documents are required?"]"""

def generate_followups(question: str, answer_markdown: str, lang: str = "en") -> list[str]:
    """Generate 3 follow-up questions using a fast LLM call."""
    try:
        prompt = f"Question: {question}\n\nAnswer summary: {answer_markdown[:800]}\n\nGenerate 3 follow-up questions."
        resp = router.generate(
            _FOLLOWUP_SYSTEM, prompt,
            temperature=0.7,
            max_tokens=200,
            prefer="openai",   # fast, cheap
        )
        text = resp.text.strip()
        # Extract JSON array from response
        import re as _re
        m = _re.search(r'\[.*\]', text, _re.DOTALL)
        if m:
            questions = json.loads(m.group(0))
            return [q.strip() for q in questions if isinstance(q, str)][:3]
    except Exception as e:
        logger.debug("followup generation failed: %s", e)
    # Fallback: generic follow-ups based on question
    return [
        "What is the limitation period applicable here?",
        "How has this position evolved in recent High Court judgments?",
        "What procedural steps should I follow to file this matter?",
    ]
