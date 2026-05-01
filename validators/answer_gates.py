"""
Sanhita — Six-Gate Answer Validator.

Every Brief answer is run through this before it reaches the user. If it
fails any gate, we either rewrite (one retry with stricter prompt) or refuse
(return a "Closest 3 cases" payload — no prose).

Gates:
  G1  cite_present       — answer contains at least one [n] marker
  G2  cite_resolves      — every [n] marker resolves to a retrieved hit (1..k)
  G3  no_banned_phrases  — no hedge phrases that suggest hallucination
  G4  grounding_floor    — ≥60% of substantive sentences carry a [n] cite
  G5  scope_check        — no "X v. Y" case names appear that aren't in the
                            retrieved context (catches fabricated case names)
  G6  section_check      — every "Section NN of <Act>" reference appears in
                            either the question or the retrieved context

The result carries per-gate verdicts so the UI/eval can show *why*
something was refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


BANNED_PHRASES = [
    r"\bas an ai\b",
    r"\bi (?:think|believe|feel|guess)\b",
    r"\bin my opinion\b",
    r"\bi'?m not (?:sure|certain)\b",
    r"\bbased on the retrieved\b",       # preamble we explicitly forbade
    r"\bit is (?:widely|generally|commonly) (?:held|believed|known)\b",
]
BANNED_RE = re.compile("|".join(BANNED_PHRASES), re.IGNORECASE)

CITE_RE = re.compile(r"\[(\d+)\]")
CASE_NAME_RE = re.compile(r"\b([A-Z][A-Za-z.&'\-]+(?:\s+[A-Z][A-Za-z.&'\-]+){0,4})\s+v\.?s?\.?\s+([A-Z][A-Za-z.&'\-]+(?:\s+[A-Z][A-Za-z.&'\-]+){0,4})\b")
SECTION_RE = re.compile(r"\bSection\s+(\d+[A-Z]*)\s+(?:of\s+(?:the\s+)?)?([A-Z][A-Za-z &]{2,40})", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")

GATE_NAMES = ["cite_present", "cite_resolves", "no_banned_phrases", "grounding_floor", "scope_check", "section_check"]


@dataclass
class ValidationResult:
    passed: bool
    confidence: float                          # 0..1, fraction of gates passed
    gates: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    cite_indices_used: list[int] = field(default_factory=list)
    grounding_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "confidence": round(self.confidence, 2),
            "gates": self.gates,
            "reasons": self.reasons,
            "cite_indices_used": self.cite_indices_used,
            "grounding_pct": round(self.grounding_pct, 2),
        }


def _strip_markdown(text: str) -> str:
    text = re.sub(r"`[^`]*`", "", text)         # inline code
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return text


def _substantive_sentences(text: str) -> list[str]:
    body = _strip_markdown(text).strip()
    # Drop the "Practice note:" tail — it's a takeaway, not a claim
    body = re.sub(r"\n+Practice note:.*$", "", body, flags=re.IGNORECASE | re.DOTALL)
    sents = [s.strip() for s in SENTENCE_SPLIT_RE.split(body) if s.strip()]
    # Filter out trivially short or pure-list lines
    return [s for s in sents if len(s.split()) >= 6]


def validate(
    answer: str,
    hits: list[dict[str, Any]],
    question: str = "",
    *,
    grounding_floor: float = 0.6,
) -> ValidationResult:
    """Run all six gates. `hits` is the retrieved cases list (1-indexed by [n])."""
    gates: dict[str, bool] = {g: True for g in GATE_NAMES}
    reasons: list[str] = []
    k = len(hits)

    # ── G1 cite_present
    cite_idx = [int(m.group(1)) for m in CITE_RE.finditer(answer)]
    if not cite_idx:
        gates["cite_present"] = False
        reasons.append("no [n] citation markers found")

    # ── G2 cite_resolves
    bad_idx = [i for i in cite_idx if i < 1 or i > k]
    if bad_idx:
        gates["cite_resolves"] = False
        reasons.append(f"citation markers out of range (k={k}): {sorted(set(bad_idx))}")

    # ── G3 no_banned_phrases
    m = BANNED_RE.search(answer)
    if m:
        gates["no_banned_phrases"] = False
        reasons.append(f"banned phrase: '{m.group(0)}'")

    # ── G4 grounding_floor
    sents = _substantive_sentences(answer)
    if sents:
        cited = sum(1 for s in sents if CITE_RE.search(s))
        grounding_pct = cited / len(sents)
    else:
        grounding_pct = 0.0
    if grounding_pct < grounding_floor:
        gates["grounding_floor"] = False
        reasons.append(f"grounding {grounding_pct:.0%} < floor {grounding_floor:.0%}")

    # ── G5 scope_check (no fabricated case names)
    context_blob = " ".join(
        " ".join(str(h.get(k, "")) for k in ("title", "citation", "excerpt"))
        for h in hits
    ).lower()
    fabricated = []
    for cm in CASE_NAME_RE.finditer(_strip_markdown(answer)):
        a, b = cm.group(1).lower(), cm.group(2).lower()
        if a not in context_blob or b not in context_blob:
            fabricated.append(f"{cm.group(1)} v. {cm.group(2)}")
    if fabricated:
        gates["scope_check"] = False
        reasons.append(f"case names not in context: {fabricated[:3]}")

    # ── G6 section_check (statute references the corpus actually supports)
    q_blob = (question or "").lower()
    bad_sections = []
    for sm in SECTION_RE.finditer(answer):
        sec, act = sm.group(1).lower(), sm.group(2).lower().strip()
        token = f"section {sec}"
        if token not in context_blob and token not in q_blob:
            # one more chance — bare section number anywhere
            if not re.search(rf"\b{re.escape(sec)}\b", context_blob + " " + q_blob):
                bad_sections.append(f"Section {sm.group(1)} of {sm.group(2)}")
    if bad_sections:
        gates["section_check"] = False
        reasons.append(f"unsupported section refs: {bad_sections[:3]}")

    passed_count = sum(1 for v in gates.values() if v)
    confidence = passed_count / len(GATE_NAMES)
    # Hard-fail if cite_present, cite_resolves, or scope_check fail. Soft-fail
    # otherwise (the rewrite step gets a chance).
    hard_pass = gates["cite_present"] and gates["cite_resolves"] and gates["scope_check"]

    return ValidationResult(
        passed=hard_pass and passed_count >= 5,
        confidence=confidence,
        gates=gates,
        reasons=reasons,
        cite_indices_used=sorted(set(cite_idx)),
        grounding_pct=grounding_pct,
    )


def refusal_payload(question: str, hits: list[dict[str, Any]], reasons: list[str]) -> str:
    """Structured fallback when LLM answer can't be grounded — still useful."""
    if not hits:
        return (
            "No matching cases found in the corpus for this query. "
            "Try more specific terms — name a statute, section number, "
            "court, or describe the fact pattern in detail.\n\n"
            "**Tip:** Use *Court Search* for advanced filters across 31.9M records."
        )
    lines = [
        f"Here are **{min(len(hits), 6)} relevant cases** from the corpus "
        f"matching your query:\n",
    ]
    for i, h in enumerate(hits[:6], 1):
        t = h.get("title") or h.get("case_id") or "Untitled"
        court = h.get("court") or ""
        year = h.get("year") or ""
        cit = h.get("citation") or ""
        verdict = h.get("verdict") or ""
        excerpt = (h.get("excerpt") or "")[:250]

        meta_parts = []
        if court: meta_parts.append(court)
        if year: meta_parts.append(str(year))
        if verdict: meta_parts.append(f"*{verdict}*")
        meta = " · ".join(meta_parts)

        lines.append(f"**[{i}] {t}**")
        if cit and cit != t:
            lines.append(f"📋 {cit}")
        if meta:
            lines.append(f"🏛️ {meta}")
        if excerpt:
            lines.append(f"> {excerpt}")
        lines.append("")

    lines.append(
        "*For a detailed AI analysis, configure an LLM API key "
        "(Gemini, Groq, Anthropic, or Cloudflare) in your environment.*"
    )
    return "\n".join(lines)
