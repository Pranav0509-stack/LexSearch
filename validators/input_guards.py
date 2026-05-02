"""
Sanhita — Input Guardrails.

Runs before retrieval/LLM. Cheap, deterministic, no model calls. Catches:

  G_LEN     length cap                — DoS protection + cost cap
  G_INJECT  prompt-injection patterns — "ignore previous", "system:", role flips
  G_SCOPE   off-topic refusal         — only Indian-law questions answered
  G_PII     PII leak in question      — Aadhaar / PAN / phone / email auto-redact
  G_LANG    script gate               — accept Latin + Devanagari + English digits

Returns a verdict:
  GuardVerdict(allow=True,  redacted_question=..., notes=[...])
  GuardVerdict(allow=False, refusal_message="...", reason="...")

Refusal messages are user-facing and written for advocates — terse,
specific, no scolding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

MAX_QUESTION_CHARS = 4000
MIN_QUESTION_CHARS = 2

# ── Prompt-injection patterns (case-insensitive). Curated, not a kitchen sink.
INJECTION_PATTERNS = [
    r"ignore (?:all )?(?:previous|prior|above) (?:instructions?|prompts?|messages?)",
    r"disregard (?:the )?(?:system|above|previous)",
    r"forget (?:everything|all|previous)",
    r"you are now (?:a |an )?",
    r"new (?:instructions?|system prompt)\s*[:\-]",
    r"^\s*system\s*[:>]",
    r"\bact as (?:a |an )?(?!an? advocate|a lawyer|counsel)",  # role-flip, but allow "act as advocate"
    r"reveal (?:your )?(?:system )?prompt",
    r"print (?:your )?(?:system )?(?:prompt|instructions)",
    r"<\s*\|?\s*(?:im_start|system|assistant)\s*\|?\s*>",
    r"```\s*system",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

# ── Scope: must be plausibly an Indian-law question. Very permissive —
# lawyers ask in natural language ("road accident in goa", "my landlord is
# not returning deposit", "cheating case procedure").  We let almost
# everything through and let the retrieval + LLM handle off-topic gracefully.
#
# Only BLOCK queries that are *clearly* not law-related (recipes, code, etc.)
LEGAL_TOKENS = [
    # statutes / codes
    r"\bIPC\b", r"\bCrPC\b", r"\bCPC\b", r"\bIEA\b", r"\bBNS\b", r"\bBNSS\b", r"\bBSA\b",
    r"\bConstitution\b", r"\bArticle\s+\d", r"\bSection\s+\d", r"\bSec\.\s*\d",
    r"\bAct\b", r"\brules?\b", r"\bordinance\b", r"\bregulation\b",
    # courts / fora
    r"\bSupreme Court\b", r"\bHigh Court\b", r"\bSC\b", r"\bHC\b", r"\bDistrict Court\b",
    r"\bTribunal\b", r"\bNCLT\b", r"\bNCLAT\b", r"\bDRT\b", r"\bSAT\b", r"\bITAT\b",
    r"\beCourts?\b", r"\bMagistrate\b", r"\bSessions\b", r"\bCourt\b",
    # process / outcomes
    r"\bbail\b", r"\banticipatory\b", r"\bwrit\b", r"\bmandamus\b", r"\bcertiorari\b",
    r"\bhabeas corpus\b", r"\bquo warranto\b", r"\binjunction\b", r"\bstay\b",
    r"\barbitration\b", r"\bappeal\b", r"\brevision\b", r"\bremand\b",
    r"\bjudgment\b", r"\border\b", r"\bdecree\b", r"\bcitation\b", r"\bprecedent\b",
    r"\bFIR\b", r"\bcharge\s*sheet\b", r"\bwarrant\b", r"\bsummons\b",
    # subject areas
    r"\bcontract\b", r"\btort\b", r"\bcriminal\b", r"\bcivil\b", r"\bmatrimonial\b",
    r"\bproperty\b", r"\binheritance\b", r"\btenancy\b", r"\beviction\b",
    r"\bcompany\b", r"\binsolvency\b", r"\bIBC\b", r"\bcompetition\b", r"\bGST\b",
    r"\bincome[- ]?tax\b", r"\bcustoms\b", r"\bcyber\b", r"\bIT Act\b",
    r"\bPOCSO\b", r"\bdowry\b", r"\b498A\b", r"\b138\b",
    r"\bmurder\b", r"\bkilling\b", r"\btheft\b", r"\brobbery\b", r"\bfraud\b",
    r"\bcheating\b", r"\bforgery\b", r"\bkidnapping\b", r"\bextortion\b",
    r"\bassault\b", r"\bhurt\b", r"\bgrievous\b", r"\brape\b", r"\bmolestation\b",
    r"\baccident\b", r"\bcompensation\b", r"\binsurance\b", r"\bclaim\b",
    r"\bdivorce\b", r"\bcustody\b", r"\bmaintenance\b", r"\balimony\b",
    r"\blandlord\b", r"\btenant\b", r"\brent\b", r"\bevict\b", r"\bdeposit\b",
    r"\bbreach\b", r"\bdispute\b", r"\blitigat\b", r"\bproceed\b",
    r"\bdefamation\b", r"\bnuisance\b", r"\btrespass\b",
    # actors
    r"\badvocate\b", r"\bcounsel\b", r"\bclient\b", r"\bplaintiff\b", r"\bdefendant\b",
    r"\bpetitioner\b", r"\brespondent\b", r"\baccused\b", r"\bcomplainant\b",
    r"\blawyer\b", r"\bjudge\b", r"\bprosecutor\b", r"\bpolice\b", r"\bvictim\b",
    # misc
    r"\bvs?\.?\b", r"\bv\.\b", r"\bAIR\b", r"\bSCC\b", r"\bILR\b", r"\bCriLJ\b",
    # doctrines / common legal phrasing
    r"\bdoctrine\b", r"\bprinciple\b", r"\bestoppel\b", r"\blegitimate expectation\b",
    r"\badministrative law\b", r"\bjudicial review\b", r"\bpassing[- ]off\b",
    r"\btrademark\b", r"\bcopyright\b", r"\bpatent\b", r"\bIP(?:R)?\b",
    r"\bbona fide\b", r"\bex[- ]parte\b", r"\bmens rea\b", r"\bactus reus\b",
    r"\btax(?:es|ation)?\b", r"\brent(?:al)?\b", r"\btenan(?:t|cy)\b",
    r"\bcheque\b", r"\bdishonou?r\b", r"\bdefault\b", r"\bnegligence\b",
    r"\bdamages?\b", r"\bcompensation\b", r"\bsuit\b", r"\bcase\s+law\b",
    r"\blaw\s+on\b", r"\bunder\s+(?:the\s+)?[A-Z]",  # "under the X Act" pattern
    # ── Natural-language legal queries (how lawyers actually ask) ──
    r"\bcase\b", r"\bfind\b.*\bcase\b", r"\bsearch\b", r"\blegal\b",
    r"\blaw\b", r"\bright(?:s)?\b", r"\bpenalt(?:y|ies)\b", r"\bpunish\b",
    r"\boffence\b", r"\boffense\b", r"\bcrime\b", r"\billegal\b",
    r"\bconvict\b", r"\bacquit\b", r"\bsentence\b", r"\bimprison\b",
    r"\bfine\b", r"\bpenalty\b", r"\bliable\b", r"\bliability\b",
    r"\bnotice\b", r"\bcomplaint\b", r"\bpetition\b", r"\bapplication\b",
    r"\brelief\b", r"\bremedy\b", r"\bgrievance\b", r"\bapproach\b",
    r"\bprocedure\b", r"\bprocess\b", r"\bfiling\b", r"\bfile\b",
    r"\bhearing\b", r"\btrial\b", r"\bevidence\b", r"\bwitness\b",
    r"\baffidavit\b", r"\bsurety\b", r"\bbond\b",
    # place-specific (Indian states/cities — people ask "case in goa")
    r"\bIndia\b", r"\bIndian\b", r"\bgoa\b", r"\bdelhi\b", r"\bmumbai\b",
    r"\bbombay\b", r"\bchennai\b", r"\bmadras\b", r"\bkolkata\b", r"\bcalcutta\b",
    r"\bbangalore\b", r"\bhyderabad\b", r"\bahmedabad\b", r"\bjaipur\b",
    r"\blucknow\b", r"\bpatna\b", r"\bpunjab\b", r"\bharyana\b",
    r"\bkerala\b", r"\btamil\b", r"\bkarnataka\b", r"\bandhra\b", r"\btelangana\b",
    r"\bmaharashtra\b", r"\bgujarat\b", r"\brajasthan\b", r"\bMP\b", r"\bUP\b",
    r"\bbihar\b", r"\bwest bengal\b", r"\bassam\b", r"\bodisha\b",
    r"\bjharkhand\b", r"\bchhattisgarh\b", r"\buttarakhand\b", r"\bhimachal\b",
    r"\bjammu\b", r"\bkashmir\b", r"\bsikkim\b", r"\bmeghalaya\b",
    r"\bmanipuri?\b", r"\btripura\b", r"\bnagaland\b", r"\bmizoram\b", r"\barunachal\b",
    # ── Pan-Asia / cross-border ─────────────────────────────────────
    r"\bSingapor(?:e|ean)\b", r"\bHong Kong\b", r"\bUAE\b", r"\bDIFC\b",
    r"\bMalaysia(?:n)?\b", r"\bBangladesh(?:i)?\b", r"\bSri Lanka(?:n)?\b",
    r"\bNepal(?:i|ese)?\b", r"\bchoice[- ]of[- ]law\b",
    r"\bseat\s+of\s+arbitration\b", r"\bNew York Convention\b",
    r"\benforce(?:ment)?\b", r"\bjurisdiction(?:al)?\b",
    # ── General question words that suggest legal intent ────────────
    r"\bwhat\s+(?:is|are|should|can)\b",  # "what is the law on..."
    r"\bhow\s+(?:to|do|can|should)\b",    # "how to file..."
    r"\bcan\s+(?:I|we|a)\b",              # "can I sue..."
    r"\bis\s+it\s+(?:legal|illegal|lawful|unlawful)\b",
    r"\bwhat\s+(?:happens|if)\b",
    r"\bdatabase\b", r"\bfind\s+me\b",
]
LEGAL_RE = re.compile("|".join(LEGAL_TOKENS), re.IGNORECASE)

# ── PII patterns. We REDACT (don't refuse) — advocates routinely paste case facts.
AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PAN_RE     = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
PHONE_RE   = re.compile(r"\b(?:\+?91[\s\-]?)?[6-9]\d{9}\b")
EMAIL_RE   = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
CC_RE      = re.compile(r"\b(?:\d[ \-]?){13,16}\b")


@dataclass
class GuardVerdict:
    allow: bool
    redacted_question: str = ""
    refusal_message: str = ""
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "allow": self.allow,
            "refusal_message": self.refusal_message,
            "reason": self.reason,
            "notes": self.notes,
        }


def _redact_pii(text: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    out = text
    if AADHAAR_RE.search(out):
        out = AADHAAR_RE.sub("[REDACTED-AADHAAR]", out); notes.append("redacted Aadhaar")
    if PAN_RE.search(out):
        out = PAN_RE.sub("[REDACTED-PAN]", out); notes.append("redacted PAN")
    if PHONE_RE.search(out):
        out = PHONE_RE.sub("[REDACTED-PHONE]", out); notes.append("redacted phone")
    if EMAIL_RE.search(out):
        out = EMAIL_RE.sub("[REDACTED-EMAIL]", out); notes.append("redacted email")
    if CC_RE.search(out):
        out = CC_RE.sub("[REDACTED-CARD]", out); notes.append("redacted card-like number")
    return out, notes


def _is_followup(text: str, history_len: int) -> bool:
    # Any message in an existing thread is treated as a follow-up
    # (lawyers naturally pivot topics within a research session)
    return history_len > 0


def check(question: str, history_len: int = 0) -> GuardVerdict:
    """Run all input gates. Returns a verdict; caller honors `allow`."""
    q = (question or "").strip()

    # G_LEN
    if len(q) < MIN_QUESTION_CHARS:
        return GuardVerdict(
            allow=False,
            refusal_message="Your question is too short. Add a sentence of context — the section, the court, the fact pattern.",
            reason="length_min",
        )
    if len(q) > MAX_QUESTION_CHARS:
        return GuardVerdict(
            allow=False,
            refusal_message=f"Question exceeds the {MAX_QUESTION_CHARS:,}-character limit. Trim it or upload the brief as an attachment (coming in Phase B).",
            reason="length_max",
        )

    # G_INJECT
    m = INJECTION_RE.search(q)
    if m:
        return GuardVerdict(
            allow=False,
            refusal_message="That looks like a prompt-injection attempt. Sanhita answers legal questions grounded in retrieved judgments — it doesn't take instructions to override its own rules.",
            reason=f"injection:{m.group(0)[:40]}",
        )

    # G_SCOPE — very permissive: allow greetings, general questions, follow-ups,
    # and all non-Latin scripts (Hindi, Tamil, Telugu, etc. — if someone is
    # writing in an Indian language, it's almost certainly a legal query).
    is_conversational = bool(re.search(
        r"\b(?:hi|hello|hey|thanks|thank you|good|help|tell|explain|what|how|why|who|when|where|please|can you|could you|show|list|give|summarize|compare|analyze|draft|review|translate)\b",
        q, re.IGNORECASE
    ))
    # Detect non-Latin scripts (Devanagari, Tamil, Telugu, Kannada, Malayalam,
    # Bengali, Gujarati, Gurmukhi, Odia, Arabic/Urdu)
    has_indic_script = bool(re.search(
        r"[ऀ-ॿ"    # Devanagari (Hindi, Marathi)
        r"ঀ-৿"     # Bengali, Assamese
        r"਀-੿"     # Gurmukhi (Punjabi)
        r"઀-૿"     # Gujarati
        r"଀-୿"     # Odia
        r"஀-௿"     # Tamil
        r"ఀ-౿"     # Telugu
        r"ಀ-೿"     # Kannada
        r"ഀ-ൿ"     # Malayalam
        r"؀-ۿ]",   # Arabic/Urdu
        q
    ))
    if not _is_followup(q, history_len) and not LEGAL_RE.search(q) and not is_conversational and not has_indic_script:
        return GuardVerdict(
            allow=False,
            refusal_message="I'm Sanhita, your Indian legal research assistant. Ask me about any statute, case law, court process, or legal question under Indian law.",
            reason="off_scope",
        )

    # G_PII — redact, don't refuse
    redacted, notes = _redact_pii(q)

    return GuardVerdict(
        allow=True,
        redacted_question=redacted,
        notes=notes,
    )
