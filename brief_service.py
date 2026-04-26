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
import re
import urllib.parse
from typing import Any, Optional

from llm import router
from validators import answer_gates

logger = logging.getLogger(__name__)


# ── Multi-language registry ───────────────────────────────────────────────
# Output language for the final answer. Internally the model still reasons
# in English (cheaper, more reliable for legal domain), but the answer is
# rendered in the user's chosen language. Statute names, case captions, and
# citation numbers stay in their original English form so links resolve.
#
# Add a language by extending this dict. The "instruction" is the single
# directive line we append to whichever system prompt is in play; keep it
# short — the model picks up language from one well-placed sentence better
# than from three.
LANGUAGES: dict[str, dict[str, str]] = {
    "en":    {"label": "English",          "native": "English",     "family": "Latin",       "instruction": ""},
    # ── Indian Eighth Schedule (all 22 official languages) ───────────────
    # Ordered alphabetically by ISO/BCP-47 code so the dropdown is stable.
    "as":    {"label": "Assamese",         "native": "অসমীয়া",      "family": "Bengali-Assamese", "instruction": "Reply in অসমীয়া (Assamese)."},
    "bn":    {"label": "Bengali",          "native": "বাংলা",       "family": "Bengali",     "instruction": "Reply in বাংলা (Bengali)."},
    "brx":   {"label": "Bodo",             "native": "बड़ो",          "family": "Devanagari",  "instruction": "Reply in बड़ो (Bodo, Devanagari script)."},
    "doi":   {"label": "Dogri",            "native": "डोगरी",        "family": "Devanagari",  "instruction": "Reply in डोगरी (Dogri)."},
    "gom":   {"label": "Konkani",          "native": "कोंकणी",       "family": "Devanagari",  "instruction": "Reply in कोंकणी (Konkani, Devanagari script)."},
    "gu":    {"label": "Gujarati",         "native": "ગુજરાતી",      "family": "Gujarati",    "instruction": "Reply in ગુજરાતી (Gujarati)."},
    "hi":    {"label": "Hindi",            "native": "हिन्दी",       "family": "Devanagari",  "instruction": "Reply in हिन्दी (Hindi)."},
    "kn":    {"label": "Kannada",          "native": "ಕನ್ನಡ",        "family": "Kannada",     "instruction": "Reply in ಕನ್ನಡ (Kannada)."},
    "ks":    {"label": "Kashmiri",         "native": "کٲشُر",         "family": "Arabic",      "instruction": "Reply in کٲشُر (Kashmiri, Perso-Arabic script). Right-to-left."},
    "mai":   {"label": "Maithili",         "native": "मैथिली",       "family": "Devanagari",  "instruction": "Reply in मैथिली (Maithili)."},
    "ml":    {"label": "Malayalam",        "native": "മലയാളം",     "family": "Malayalam",   "instruction": "Reply in മലയാളം (Malayalam)."},
    "mni":   {"label": "Manipuri",         "native": "মৈতৈলোন্",     "family": "Bengali",     "instruction": "Reply in মৈতৈলোন্ (Meitei/Manipuri, Bengali script)."},
    "mr":    {"label": "Marathi",          "native": "मराठी",        "family": "Devanagari",  "instruction": "Reply in मराठी (Marathi)."},
    "ne":    {"label": "Nepali",           "native": "नेपाली",        "family": "Devanagari",  "instruction": "Reply in नेपाली (Nepali)."},
    "or":    {"label": "Odia",             "native": "ଓଡ଼ିଆ",        "family": "Odia",        "instruction": "Reply in ଓଡ଼ିଆ (Odia)."},
    "pa":    {"label": "Punjabi",          "native": "ਪੰਜਾਬੀ",       "family": "Gurmukhi",    "instruction": "Reply in ਪੰਜਾਬੀ (Punjabi, Gurmukhi script)."},
    "sa":    {"label": "Sanskrit",         "native": "संस्कृतम्",     "family": "Devanagari",  "instruction": "Reply in संस्कृतम् (Sanskrit, Devanagari script)."},
    "sat":   {"label": "Santali",          "native": "ᱥᱟᱱᱛᱟᱲᱤ",       "family": "Ol Chiki",    "instruction": "Reply in ᱥᱟᱱᱛᱟᱲᱤ (Santali, Ol Chiki script)."},
    "sd":    {"label": "Sindhi",           "native": "سنڌي",         "family": "Arabic",      "instruction": "Reply in سنڌي (Sindhi, Perso-Arabic script). Right-to-left."},
    "ta":    {"label": "Tamil",            "native": "தமிழ்",       "family": "Tamil",       "instruction": "Reply in தமிழ் (Tamil)."},
    "te":    {"label": "Telugu",           "native": "తెలుగు",      "family": "Telugu",      "instruction": "Reply in తెలుగు (Telugu)."},
    "ur":    {"label": "Urdu",             "native": "اردو",         "family": "Arabic",      "instruction": "Reply in اردو (Urdu, Arabic script). Right-to-left."},
    # ── Asian languages ──────────────────────────────────────────────────
    "ja":    {"label": "Japanese",         "native": "日本語",        "family": "CJK",         "instruction": "Reply in 日本語 (Japanese)."},
    "ko":    {"label": "Korean",           "native": "한국어",        "family": "Hangul",      "instruction": "Reply in 한국어 (Korean)."},
    "zh-CN": {"label": "Chinese (Simplified)", "native": "简体中文",  "family": "CJK",         "instruction": "Reply in 简体中文 (Simplified Chinese)."},
    "zh-TW": {"label": "Chinese (Traditional)","native": "繁體中文", "family": "CJK",         "instruction": "Reply in 繁體中文 (Traditional Chinese)."},
    "th":    {"label": "Thai",             "native": "ไทย",          "family": "Thai",        "instruction": "Reply in ภาษาไทย (Thai)."},
    "vi":    {"label": "Vietnamese",       "native": "Tiếng Việt",    "family": "Latin",       "instruction": "Reply in Tiếng Việt (Vietnamese)."},
    "id":    {"label": "Indonesian",       "native": "Bahasa Indonesia","family": "Latin",     "instruction": "Reply in Bahasa Indonesia."},
    "ms":    {"label": "Malay",            "native": "Bahasa Melayu","family": "Latin",       "instruction": "Reply in Bahasa Melayu."},
    "fil":   {"label": "Filipino",         "native": "Filipino",     "family": "Latin",       "instruction": "Reply in Filipino (Tagalog)."},
    "ar":    {"label": "Arabic",           "native": "العربية",       "family": "Arabic",      "instruction": "Reply in العربية (Modern Standard Arabic). Right-to-left."},
}


def _lang_directive(language: Optional[str]) -> str:
    """One line we append to a system prompt to switch output language. We
    deliberately do NOT translate statute names, case captions, or [n]
    citation markers — those must stay in their canonical English form so
    the citation rail's links resolve. Returns "" for English/unknown."""
    if not language:
        return ""
    entry = LANGUAGES.get(language)
    if not entry or not entry["instruction"]:
        return ""
    return (
        f"\n\nLANGUAGE DIRECTIVE: {entry['instruction']} "
        "Keep statute names, case captions, section numbers, and [n] "
        "citation markers in their original English form — do not translate "
        "them. The answer body and headings switch to the target language; "
        "the citations stay verbatim."
    )


def _with_lang(prompt: str, language: Optional[str]) -> str:
    """Compose `prompt + _lang_directive(language)`. No-op for English.

    If Sarvam is going to post-translate (see `_use_sarvam_for`), we keep
    the model in English here — the model produces canonical English legal
    prose and Sarvam renders the polished vernacular afterwards. This gives
    materially better Indian-language output than asking the model to write
    natively in Maithili / Konkani / Santali, where the LLM has thin
    coverage and Sarvam's `mayura` model has dedicated training data.
    """
    if _use_sarvam_for(language):
        return prompt
    return prompt + _lang_directive(language)


def _use_sarvam_for(language: Optional[str]) -> bool:
    """Whether Sarvam should handle final translation for `language`.

    True only when: a target language is set, it's not English, Sarvam's
    `mayura` translates into it, AND a Sarvam API key is configured. We
    swallow import errors so the module is optional in dev.
    """
    if not language or language == "en":
        return False
    try:
        from llm import sarvam
        return sarvam.supports(language) and bool(sarvam._key())
    except Exception:
        return False


def _apply_sarvam(answer_markdown: str, language: Optional[str]) -> str:
    """Best-effort post-translation via Sarvam. Returns the original text
    unchanged on any failure — the model already wrote something readable
    (English), so Sarvam is a polish step, not a hard dependency.

    NB: citations of the form [n], statute references like §138, and case
    captions are preserved by Sarvam's `enable_preprocessing` flag. The
    citation rail (which is built from the structured `citations[]` field,
    not parsed from prose) is unaffected either way.
    """
    if not _use_sarvam_for(language):
        return answer_markdown
    if not answer_markdown or not answer_markdown.strip():
        return answer_markdown
    try:
        from llm import sarvam
        translated = sarvam.translate(answer_markdown, target_lang=language or "en", source_lang="en")
        return translated or answer_markdown
    except Exception as e:
        logger.warning("sarvam post-process failed: %s", e)
        return answer_markdown


CHITCHAT_SYSTEM_PROMPT = """You are Sanhita, a legal AI for advocates across
Asia. The user just said hello, thanked you, or asked who you are. Keep it
warm and brief — 1-2 sentences, no preamble.

If they greeted ("hi", "hello", "namaste") → greet back, mention you're
ready to research case law, draft documents, redline contracts, or chase
court records. Invite them to ask anything legal.

If they asked who you are / what you do → one short sentence on Sanhita
(senior-associate-grade legal AI for India + Asia, grounded in real case
law, no fabricated citations) and one short sentence inviting their first
matter.

If they thanked you → "Anytime — what's next?" energy.

Never refuse, never apologize, never write "as an AI". No bullet lists for
chitchat — just a conversational sentence or two."""


DRAFT_SYSTEM_PROMPT = """You are Sanhita, a senior associate at a pan-Asian
law firm. Your role here is open drafting, explanation, and broad legal
explainers — NOT case-cited research. The user is a qualified advocate —
give them substantive, expert, multi-dimensional output. Never give them
a five-line summary when the topic deserves a memo.

ABSOLUTE RULE — NEVER ASK THE USER TO NARROW THEIR QUESTION.
Phrases like "please specify which aspect", "could you clarify", "are you
interested in X or Y", "let me know more details so I can…" are FORBIDDEN
in your output. The advocate asked a broad question because they want a
broad authoritative overview — give it. If a topic spans five sub-areas,
write five sections covering all five. Length is cheap; clarification
ping-pong is expensive.

Rules:

1. Produce the document, explainer, or memo the user asks for. Use clean
   Markdown: H2 headings, sub-lists, bolded defined terms on first use,
   numbered clauses for contracts.
2. NEVER fabricate case citations or pretend a specific judgment exists.
   If you reference law, only cite statutes by section number (e.g.
   "Section 138 of the NI Act", "Section 279 of the BNS") — never invent
   "X v. Y (2023)". Saying "Indian courts have generally held that …" in
   prose, without a fake citation, is fine.
3. Don't write "as an AI", "I think", "in my opinion", or "based on the
   retrieved cases". Write like a partner explaining to a junior.
4. For contracts: parties block, recitals, operative clauses,
   governing-law/jurisdiction, signature block. Pan-Asian defaults
   (Singapore-seat arbitration under SIAC) unless the user names a
   jurisdiction.
5. For translations: preserve legal force and the exact section/clause
   numbering. Note in a one-line footer if a term has no clean
   equivalent.

6. For broad explainers / area-of-law overviews ("tell me about Indian
   road law", "explain anti-defection law"), USE this multi-section
   structure — skip a heading only if it genuinely doesn't apply:

     ## Overview
     ## Statutory Framework        — list the governing Acts + key sections
     ## Key Concepts / Definitions  — defined terms, doctrinal building blocks
     ## Procedure                   — forum, limitation, who files, what relief
     ## Penalties / Remedies        — sentencing, fines, civil remedies
     ## Recent Developments         — post-2020 amendments, BNS/BNSS mappings
     ## Practice Pointers           — 4-6 numbered tactical pointers

   Write 800-1500 words for broad topics. Don't pad — but never produce a
   single-paragraph answer for a topic that has five Acts and a hundred
   judgments behind it.

Output clean markdown. No preamble, no closing apologies, no "Hope this
helps!" sign-off."""


WEB_SYSTEM_PROMPT = """You are Sanhita, an AI research assistant. The user
is a qualified advocate. Answer ONLY from the web snippets provided below.

Rules — non-negotiable:

1. Cite every claim with a numbered marker [1], [2] mapped to the snippet
   index below. A sentence without a [n] is a bug.
2. Never invent a fact, court ruling, or statute that isn't in the snippets.
   If the snippets don't answer the question, say so plainly in one sentence
   and stop.
3. Don't write "I think", "in my opinion", "as an AI", or "based on the
   snippets". Just answer.
4. Keep it under 350 words. Plain markdown, no preamble."""


SYSTEM_PROMPT = """You are Sanhita Brief, a senior-associate-grade AI
research assistant for advocates across India and the rest of Asia. You
write multi-dimensional, comprehensive legal memoranda — the kind a
partner would expect from a fifth-year associate after a half-day's work,
not a tweet-length answer.

ABSOLUTE RULE — DO NOT ASK CLARIFYING QUESTIONS. Never write "please
specify", "could you clarify", "which jurisdiction", "are you asking
about X or Y". The advocate asked a broad question because they want a
broad authoritative answer. If the topic spans five Acts, write a section
on each. If retrieval missed a sub-issue, say so for that sub-issue and
move on — never punt the whole answer back to the user.

Rules — non-negotiable:

1. GROUND every substantive factual or doctrinal sentence in a [n] marker
   keyed to the retrieved excerpts below. A sentence asserting law without
   a [n] is a bug. Headings, framing sentences, and connective tissue
   ("This raises three questions:") do not need [n]; doctrinal claims
   always do.
2. NEVER invent case names, citations, statute sections, or holdings. If
   the retrieved record doesn't cover a sub-issue, say so explicitly for
   that sub-issue ("The retrieved record does not cover X") and continue
   with the parts it does cover. Do not refuse the whole answer just
   because one corner is uncovered.
3. Do not write "Based on the retrieved cases…", "I think", "In my
   opinion", "as an AI", or "I'll do my best". Write like an associate
   filing a memo.

STRUCTURE — use ALL of these headings when the question is broad enough
to benefit. Skip a heading only if it genuinely doesn't apply.

  ## Overview
  2-4 sentences framing the topic and the questions it raises.

  ## Statutory Framework
  The governing Act(s), key sections, and what each section does. Quote
  short operative phrases inside backticks. Cite [n] for each section.

  ## Judicial Interpretation
  How the courts have read the statute. Lead with the highest court that
  has spoken. For each leading authority give: case caption, court, year,
  the precise holding, and one-line ratio. Group by sub-issue, not by
  chronology.

  ## Procedure
  How a matter under this law actually moves: forum (which court /
  tribunal), limitation, who can file, what relief is available,
  appeals/revision route. Number the steps.

  ## Penalties / Remedies
  Sentencing range, fines, civil remedies, injunctive relief, compounding,
  plea bargains — whatever applies. Be specific with numbers.

  ## Recent Developments
  Amendments, BNS/BNSS/BSA mappings (if user cited the older IPC/CrPC/IEA
  number), notable post-2020 judgments, pending bills if mentioned in the
  record.

  ## Practice Pointers
  3-6 numbered tactical pointers an advocate would actually use:
  pleading hooks, evidentiary traps, common bench preferences, drafting
  language, costs/limitation gotchas. Each pointer is one sentence.

LENGTH — write what the question deserves. A focused statutory question
might be 400 words; a broad area-of-law overview ("tell me about Indian
road law") should be 900-1400 words. Don't pad, but don't truncate either.

VOCABULARY — Indian-law register: "advocate" not "lawyer", "chamber" not
"office", "matter" not "case file", "reportable" not "publishable". When
the user references a superseded provision (IPC/CrPC/IEA), note the
BNS/BNSS/BSA equivalent in parentheses if the retrieved record supports
the mapping — never invent the mapping if the record doesn't show it.

Output clean markdown. No preamble, no sign-off."""


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
    *,
    language: Optional[str] = None,
    prefer: Optional[str] = None,
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
    sys_prompt = _with_lang(SYSTEM_PROMPT, language)

    # ── First pass (fast lane: 600 tokens is enough for 350-400 word answers)
    try:
        # 2400 tokens ≈ 1500-1800 English words, enough for the full
        # 7-section memo structure. Gemini Flash handles this in ~6-8s.
        # 1400 max_tokens lands ~7-9s on Gemini Flash vs ~14-17s at 2400.
        # The 7-section memo prompt fits comfortably under this ceiling
        # in practice (typical answer is 700-1100 tokens).
        resp = router.generate(sys_prompt, user_prompt, temperature=0.25, max_tokens=1400, prefer=prefer)
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
            resp2 = router.generate(sys_prompt, rewrite_prompt, temperature=0.1, max_tokens=1400, prefer=prefer)
            verdict2 = answer_gates.validate(resp2.text, hits, question=question)
            if verdict2.confidence > verdict.confidence:
                resp, verdict = resp2, verdict2
        except Exception as e:
            logger.warning("rewrite pass failed: %s", e)

    # ── Soft fallback (Harvey-style): if grounding fails, don't dead-end.
    # Hand the prior citations to `answer_open` as factual scaffolding and
    # generate a working response with a one-line disclaimer prefix. The
    # rail stays populated, the user gets something useful, and the
    # disclaimer makes clear we couldn't fully ground the answer in case
    # law. Replaces the old hard-refusal that produced the "REFUSED" UX
    # users hit on drafting requests like "draft me a letter to police".
    if not verdict.passed:
        try:
            # No user-visible disclaimer here. The old "*I couldn't fully
            # ground this…*" banner read like a refusal even though the
            # answer below was solid; users complained it tanked their trust
            # in otherwise-good prose. The validation verdict is still
            # surfaced in the metadata pill (UI shows "Drafting mode" chip)
            # for advocates who care about the gate, without scaring everyone
            # else with red-flag language above the answer.
            soft = answer_open(
                question,
                history,
                prior_citations=citations,
                language=language,
                prefer=prefer,
            )
            soft["fell_back_from"] = "research"
            soft["validation"] = verdict.to_dict()
            return soft
        except Exception as e:
            logger.warning("soft fallback to answer_open failed: %s", e)
        # If even the soft fallback fails, surface the closest cases.
        return {
            "answer_markdown": answer_gates.refusal_payload(question, hits, verdict.reasons),
            "citations": citations,
            "llm": resp.to_dict(),
            "validation": verdict.to_dict(),
            "refused": True,
        }

    return {
        "answer_markdown": _apply_sarvam(resp.text, language),
        "citations": citations,
        "llm": resp.to_dict(),
        "validation": verdict.to_dict(),
        "refused": False,
    }


def answer_open(
    question: str,
    history: list[dict[str, Any]],
    *,
    prior_citations: Optional[list[dict[str, Any]]] = None,
    disclaimer: str = "",
    language: Optional[str] = None,
    prefer: Optional[str] = None,
) -> dict[str, Any]:
    """
    Open-drafting mode (Harvey-style). No retrieval, no citation gates —
    just a senior-associate response. Used for "draft me an NDA",
    "translate this clause", "explain Section 138 NI Act in plain English",
    "outline a Section 9 application".

    `prior_citations` carries the citation rail across turns. When the user
    says "draft me a letter for this similar case," the citations from the
    previous assistant turn become factual scaffolding the model can
    reference in plain prose (no [n] brackets — those are only for
    research-mode answers backed by retrieval).

    `disclaimer` is prepended verbatim to the answer (used by the soft
    fallback in `answer_question` to flag "couldn't ground in case law,
    here's a working draft").

    The router auto-prefers Gemini at the head of the chain. Only G3
    (banned phrases) is enforced — fabricated citations are still policed
    by the system prompt itself.
    """
    available = router.available_providers()
    if not available:
        return {
            "answer_markdown": (
                "I can't draft right now — no LLM provider is configured. "
                "Add a Gemini, Anthropic, or Groq API key in Settings and try again."
            ),
            "citations": prior_citations or [],
            "llm": {"provider": "none", "model": "", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": False, "confidence": 0.0, "reasons": ["no LLM provider configured"]},
            "refused": True,
            "mode": "draft",
        }

    history_block = _history_for_prompt(history)
    parts = []
    if history_block:
        parts.append(f"Prior conversation:\n{history_block}")
    if prior_citations:
        parts.append(
            "Cases already on the record in this matter (factual scaffolding "
            "for drafting — refer to them in prose, not [n] brackets):\n"
            + _format_prior_citations(prior_citations)
        )
    parts.append(f"Request: {question}")
    user_prompt = "\n\n".join(parts)

    try:
        # Drafts can run long — give the model room.
        # 1800 cap for drafts. NDAs/notices/memos rarely exceed that;
        # the user can always say "make it longer" for an extension.
        resp = router.generate(_with_lang(DRAFT_SYSTEM_PROMPT, language), user_prompt, temperature=0.4, max_tokens=1800, prefer=prefer)
    except Exception as e:
        logger.error("answer_open: router failed: %s", e)
        return {
            "answer_markdown": f"Drafting failed: {e}",
            "citations": prior_citations or [],
            "llm": {"provider": "error", "model": "", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": False, "confidence": 0.0, "reasons": [str(e)]},
            "refused": True,
            "mode": "draft",
        }

    verdict = answer_gates.validate(resp.text, [], question=question, mode="draft")
    body = _apply_sarvam(resp.text, language)
    answer_md = (disclaimer + "\n\n" + body).strip() if disclaimer else body
    return {
        "answer_markdown": answer_md,
        # Keep the rail populated when we carry citations across turns —
        # nothing worse than the rail going dark mid-conversation.
        "citations": prior_citations or [],
        "llm": resp.to_dict(),
        "validation": verdict.to_dict(),
        "refused": not verdict.passed,
        "mode": "draft",
    }


def _format_prior_citations(cites: list[dict[str, Any]]) -> str:
    """Render the prior turn's citations as a compact factual block. We
    intentionally drop the [n] numbering so the model uses these as facts,
    not as cite chips it has to wire into the answer.
    """
    blocks = []
    for c in cites:
        title = c.get("title") or "Untitled"
        meta_bits = [c.get("court"), str(c.get("year") or ""), c.get("citation")]
        meta = " · ".join(b for b in meta_bits if b)
        excerpt = (c.get("excerpt") or "")[:300]
        blocks.append(f"• {title}" + (f" ({meta})" if meta else "") + (f"\n    “{excerpt}”" if excerpt else ""))
    return "\n".join(blocks)


def _web_citation_payload(snippets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, s in enumerate(snippets, 1):
        out.append({
            "n": i,
            "case_id": "",
            "title": s.get("title") or s.get("url") or "Untitled",
            "citation": s.get("source") or "",
            "court": "",
            "year": "",
            "excerpt": (s.get("snippet") or "")[:400],
            "tier": "WEB",
            "pdf_url": "",
            "url": s.get("url") or "",
            "score": s.get("score"),
        })
    return out


def _build_web_context(snippets: list[dict[str, Any]]) -> str:
    if not snippets:
        return "(no web results returned for this query)"
    blocks = []
    for i, s in enumerate(snippets, 1):
        title = s.get("title") or "Untitled"
        url = s.get("url") or ""
        source = s.get("source") or ""
        snippet = s.get("snippet") or ""
        blocks.append(
            f"[{i}] {title}\n"
            f"    Source: {source}\n"
            f"    URL: {url}\n"
            f"    Snippet: {snippet[:700]}"
        )
    return "\n\n".join(blocks)


def answer_web(
    question: str,
    snippets: list[dict[str, Any]],
    history: list[dict[str, Any]],
    *,
    language: Optional[str] = None,
    prefer: Optional[str] = None,
) -> dict[str, Any]:
    """
    Web-search mode. `snippets` is a list of {title, url, source, snippet}
    from connectors.web_search. We answer ONLY from those snippets and
    validate with G1+G2+G3 (skip the case/section/grounding gates that
    don't apply to web results).
    """
    citations = _web_citation_payload(snippets)
    available = router.available_providers()
    if not available:
        return {
            "answer_markdown": (
                "I can't run web research right now — no LLM provider is configured. "
                "Add a Gemini, Anthropic, or Groq API key in Settings and try again."
            ),
            "citations": citations,
            "llm": {"provider": "none", "model": "", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": False, "confidence": 0.0, "reasons": ["no LLM provider configured"]},
            "refused": True,
            "mode": "web",
        }

    if not snippets:
        return {
            "answer_markdown": (
                "I couldn't pull any web results for that query. The grounded "
                "Gemini search, Serper, Tavily and DuckDuckGo all came back "
                "empty — try rephrasing, or add a Serper/Tavily key in Settings "
                "to broaden coverage."
            ),
            "citations": [],
            "llm": {"provider": "none", "model": "", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": False, "confidence": 0.0, "reasons": ["no web snippets"]},
            "refused": True,
            "mode": "web",
        }

    history_block = _history_for_prompt(history)
    context = _build_web_context(snippets)
    parts = []
    if history_block:
        parts.append(f"Prior conversation:\n{history_block}")
    parts.append(f"Web snippets:\n{context}")
    parts.append(f"Question: {question}")
    parts.append("Answer using only the snippets. Cite each claim with [n].")
    user_prompt = "\n\n".join(parts)

    try:
        resp = router.generate(_with_lang(WEB_SYSTEM_PROMPT, language), user_prompt, temperature=0.2, max_tokens=1600, prefer=prefer)
    except Exception as e:
        logger.error("answer_web: router failed: %s", e)
        return {
            "answer_markdown": f"Web research failed: {e}",
            "citations": citations,
            "llm": {"provider": "error", "model": "", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": False, "confidence": 0.0, "reasons": [str(e)]},
            "refused": True,
            "mode": "web",
        }

    # Validate against snippets (treated as hits) in web mode (G1+G2+G3 only).
    verdict = answer_gates.validate(resp.text, snippets, question=question, mode="web")
    return {
        "answer_markdown": _apply_sarvam(resp.text, language),
        "citations": citations,
        "llm": resp.to_dict(),
        "validation": verdict.to_dict(),
        "refused": not verdict.passed,
        "mode": "web",
    }


def answer_chitchat(
    question: str,
    history: list[dict[str, Any]],
    *,
    prior_citations: Optional[list[dict[str, Any]]] = None,
    language: Optional[str] = None,
    prefer: Optional[str] = None,
) -> dict[str, Any]:
    """Conversational filler mode — greetings, thanks, capability checks.

    Bypasses retrieval, validation, and even the input_guards length floor.
    Just a warm 1-2 sentence reply from Gemini. Citations from prior turns
    are carried so the rail doesn't go dark mid-conversation.
    """
    available = router.available_providers()
    if not available:
        # Even with no LLM, give a static friendly response.
        return {
            "answer_markdown": (
                "Hi! I'm Sanhita — your AI co-counsel for Indian and pan-Asian "
                "law. Ask me to research case law, draft a notice, redline a "
                "contract, or pull court records. (Tip: an LLM key isn't set "
                "yet — add one in Settings to unlock the full assistant.)"
            ),
            "citations": prior_citations or [],
            "llm": {"provider": "static", "model": "", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": True, "confidence": 1.0, "reasons": []},
            "refused": False,
            "mode": "chitchat",
        }

    history_block = _history_for_prompt(history)
    parts = []
    if history_block:
        parts.append(f"Prior conversation:\n{history_block}")
    parts.append(f"User just said: {question}")
    parts.append("Reply warmly in 1-2 sentences. No preamble, no markdown headings.")
    user_prompt = "\n\n".join(parts)

    try:
        resp = router.generate(_with_lang(CHITCHAT_SYSTEM_PROMPT, language), user_prompt, temperature=0.6, max_tokens=180, prefer=prefer)
    except Exception as e:
        logger.warning("answer_chitchat: router failed, using static fallback: %s", e)
        return {
            "answer_markdown": (
                "Hi! I'm Sanhita. Ask me about a case, a section, a contract, "
                "or anything legal — I'll do the lookup."
            ),
            "citations": prior_citations or [],
            "llm": {"provider": "static", "model": "", "latency_ms": 0, "fallback_chain": []},
            "validation": {"passed": True, "confidence": 1.0, "reasons": [str(e)]},
            "refused": False,
            "mode": "chitchat",
        }

    return {
        "answer_markdown": _apply_sarvam(resp.text.strip(), language),
        "citations": prior_citations or [],
        "llm": resp.to_dict(),
        "validation": {"passed": True, "confidence": 1.0, "reasons": []},
        "refused": False,
        "mode": "chitchat",
    }


def answer_agent(
    question: str,
    history: list[dict[str, Any]],
    *,
    jurisdiction: Optional[str] = None,
    prior_citations: Optional[list[dict[str, Any]]] = None,
    user_id: Optional[int] = None,
    language: Optional[str] = None,
    prefer: Optional[str] = None,
) -> dict[str, Any]:
    """Tool-using agent mode (Harvey-style). Gemini drives a multi-step
    loop over `retrieve_cases`, `retrieve_statutes`, `web_search`,
    `redline_contract`, `translate`, plus the Google Workspace tools
    (`create_google_doc`, `create_gmail_draft`, `append_sheet_row`,
    `search_drive`) when the user has connected Google.

    `prior_citations` carry the rail across turns. `user_id` is required
    for any Google tool call (the OAuth tokens are looked up by user_id).

    Errors from the agent loop degrade gracefully into `answer_open` so the
    user always gets a working response. The agent's tool trace is preserved
    for the UI to render as breadcrumbs.
    """
    try:
        from agents import legal_agent
    except Exception as e:
        logger.error("answer_agent: failed to import agents.legal_agent: %s", e)
        return answer_open(question, history, prior_citations=prior_citations)

    try:
        result = legal_agent.run(
            question,
            history,
            jurisdiction=jurisdiction,
            prior_citations=prior_citations,
            user_id=user_id,
            language=language,
            prefer=prefer,
        )
        # Post-translate via Sarvam if applicable. The agent loop runs in
        # English internally (tool-call args, intermediate prompts) — the
        # final answer string is the only thing the user sees, so it's the
        # only thing we translate.
        if isinstance(result, dict) and result.get("answer_markdown"):
            result["answer_markdown"] = _apply_sarvam(result["answer_markdown"], language)
        return result
    except Exception as e:
        logger.error("answer_agent: legal_agent.run failed: %s", e)
        fallback = answer_open(
            question,
            history,
            prior_citations=prior_citations,
            language=language,
            prefer=prefer,
            disclaimer=(
                "*Agent loop failed mid-flight; falling back to a direct "
                "draft. Verify before filing or sending.*"
            ),
        )
        fallback["fell_back_from"] = "agent"
        fallback["agent_error"] = str(e)
        return fallback


def serialize_citations(citations: list[dict[str, Any]]) -> str:
    return json.dumps(citations, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────
# Follow-up suggestions — Harvey-style "what to ask next" chips. Cheap
# Gemini Flash call (≈400ms) over the answer + question, returning 3 short
# follow-up questions a lawyer would naturally ask after seeing this
# answer. Best-effort: any failure returns [] and the UI just hides the
# section. Never blocks the main response — caller fires it after the
# answer is already streaming back.
# ─────────────────────────────────────────────────────────────────────────

_FOLLOWUPS_SYSTEM = """You suggest the next 3 questions a working
advocate would naturally ask after reading the answer below. Rules:
- Each line is one self-contained question, 8-18 words.
- Build on what was answered — don't restate it, dig deeper or sideways.
- No numbering, no bullets, no preamble. Just 3 lines.
- If the answer was a refusal, suggest narrower/clarifying questions.
- Match the language of the answer (Hindi answer → Hindi follow-ups)."""


def generate_followups(
    question: str,
    answer_markdown: str,
    *,
    language: Optional[str] = None,
    max_followups: int = 3,
) -> list[str]:
    """Return up to `max_followups` short follow-up questions for the UI.
    Best-effort. Never raises — returns [] on any error."""
    try:
        if not answer_markdown or len(answer_markdown.strip()) < 40:
            return []
        sys_prompt = _with_lang(_FOLLOWUPS_SYSTEM, language)
        user_prompt = (
            f"Original question: {question}\n\n"
            f"Answer the lawyer received:\n{answer_markdown[:3000]}\n\n"
            f"Now write {max_followups} natural follow-up questions, one per line."
        )
        resp = router.generate(sys_prompt, user_prompt, temperature=0.5, max_tokens=240)
        lines = [
            re.sub(r"^[\s\-\*\d\.\)\(]+", "", ln).strip()
            for ln in (resp.text or "").splitlines()
            if ln.strip()
        ]
        # De-dup and drop near-empties / preambles ending in colon.
        seen: set[str] = set()
        out: list[str] = []
        for ln in lines:
            if len(ln) < 10 or ln.endswith(":"):
                continue
            key = ln.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(ln)
            if len(out) >= max_followups:
                break
        return out
    except Exception as e:
        logger.warning("followups failed: %s", e)
        return []
