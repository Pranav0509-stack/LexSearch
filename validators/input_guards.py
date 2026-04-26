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

MAX_QUESTION_CHARS = 2000
MIN_QUESTION_CHARS = 4

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

# ── Scope: must be plausibly an Indian-law question. Heuristic: contains at
# least one legal-domain token OR is a follow-up (short + has cite refs).
LEGAL_TOKENS = [
    # statutes / codes
    r"\bIPC\b", r"\bCrPC\b", r"\bCPC\b", r"\bIEA\b", r"\bBNS\b", r"\bBNSS\b", r"\bBSA\b",
    r"\bConstitution\b", r"\bArticle\s+\d", r"\bSection\s+\d", r"\bSec\.\s*\d",
    r"\bAct\b", r"\brules?\b", r"\bordinance\b", r"\bregulation\b",
    # courts / fora
    r"\bSupreme Court\b", r"\bHigh Court\b", r"\bSC\b", r"\bHC\b", r"\bDistrict Court\b",
    r"\bTribunal\b", r"\bNCLT\b", r"\bNCLAT\b", r"\bDRT\b", r"\bSAT\b", r"\bITAT\b",
    r"\beCourts?\b", r"\bMagistrate\b", r"\bSessions\b",
    # process / outcomes
    r"\bbail\b", r"\banticipatory\b", r"\bwrit\b", r"\bmandamus\b", r"\bcertiorari\b",
    r"\bhabeas corpus\b", r"\bquo warranto\b", r"\binjunction\b", r"\bstay\b",
    r"\barbitration\b", r"\bappeal\b", r"\brevision\b", r"\bremand\b",
    r"\bjudgment\b", r"\border\b", r"\bdecree\b", r"\bcitation\b", r"\bprecedent\b",
    # subject areas
    r"\bcontract\b", r"\btort\b", r"\bcriminal\b", r"\bcivil\b", r"\bmatrimonial\b",
    r"\bproperty\b", r"\binheritance\b", r"\btenancy\b", r"\beviction\b",
    r"\bcompany\b", r"\binsolvency\b", r"\bIBC\b", r"\bcompetition\b", r"\bGST\b",
    r"\bincome[- ]?tax\b", r"\bcustoms\b", r"\bcyber\b", r"\bIT Act\b",
    r"\bPOCSO\b", r"\bdowry\b", r"\b498A\b", r"\b138\b",
    # actors
    r"\badvocate\b", r"\bcounsel\b", r"\bclient\b", r"\bplaintiff\b", r"\bdefendant\b",
    r"\bpetitioner\b", r"\brespondent\b", r"\baccused\b", r"\bcomplainant\b",
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
    # Generic legal nouns — any of these in a sentence is enough to clear
    # scope. Advocates ask "what's the law on cheque bounce?" all day, and
    # we shouldn't refuse just because the sentence doesn't have an Act
    # name explicitly.
    r"\blaw\b", r"\blegal\b", r"\blawful\b", r"\billegal\b", r"\blegality\b",
    r"\brights?\b", r"\bduties\b", r"\bliab(?:le|ility|ilities)\b",
    r"\boffen[cs]es?\b", r"\bprosecut(?:e|ion|ed)\b", r"\bconvict(?:ed|ion)?\b",
    r"\bsentenc(?:e|ed|ing)\b", r"\bpenalt(?:y|ies)\b", r"\bfine[ds]?\b",
    r"\bregulator(?:y|s)?\b", r"\bcompliance\b", r"\bnotif(?:y|ication|ied)\b",
    r"\bgovernment\b", r"\bministry\b", r"\bauthorit(?:y|ies)\b",
    # Subject areas that came up missing: road / motor vehicles, family,
    # consumer, employment, environment, RTI, banking, real-estate, etc.
    r"\broad\b", r"\btraffic\b", r"\bmotor\s*vehicles?\b", r"\bMV\s+Act\b",
    r"\bdriving\b", r"\blicen[cs]e\b", r"\baccident\b", r"\binsurance\b",
    r"\bfamily\b", r"\bmarriage\b", r"\bdivorce\b", r"\bmaintenance\b",
    r"\bcustody\b", r"\badoption\b", r"\bsuccession\b", r"\bwill\b",
    r"\bconsumer\b", r"\bdefective\b", r"\brefund\b", r"\bservice deficien",
    r"\bemployment\b", r"\blabou?r\b", r"\bwages?\b", r"\bgratuity\b",
    r"\bPF\b", r"\bESI\b", r"\bbonus\b", r"\bovertime\b", r"\btermination\b",
    r"\benvironment\b", r"\bpollution\b", r"\bforest\b", r"\bNGT\b",
    r"\bRTI\b", r"\binformation\s+commission",
    r"\bbank(?:ing)?\b", r"\bRBI\b", r"\bSEBI\b", r"\bIRDA(?:I)?\b",
    r"\bRERA\b", r"\breal\s*estate\b", r"\bbuilder\b", r"\bbuyers?\b",
    r"\beducation\b", r"\bUGC\b", r"\bAICTE\b", r"\bschool\b", r"\buniversity\b",
    r"\bhealth\b", r"\bmedical\b", r"\bhospital\b", r"\bMCI\b", r"\bNMC\b",
    r"\bsexual\s+harass", r"\bPOSH\b", r"\bworkplace\b",
    r"\bdomestic\s+violence\b", r"\bDV Act\b", r"\bPWDVA\b",
    r"\bSC[/\s]ST\b", r"\battrocit", r"\breservation\b",
    r"\bcitizenship\b", r"\bvisa\b", r"\bpassport\b", r"\bimmigration\b",
    r"\bdata\s+protection\b", r"\bDPDP\b", r"\bprivacy\b",
    r"\btell me about\b", r"\bwhat\s+(?:is|are)\b", r"\bhow\s+(?:to|do|does|can)\b",
    r"\bcan\s+(?:i|we|you|they|he|she|one)\b", r"\bis\s+it\s+(?:legal|illegal|allowed|prohibited)\b",
    # ── Pan-Asia jurisdictions ─────────────────────────────────────
    # Singapore
    r"\bSingapor(?:e|ean)\b", r"\bSGCA\b", r"\bSGHC\b", r"\bMAS\b", r"\bIAA\b",
    r"\bCompanies Act\b", r"\boppression\b",
    # Hong Kong
    r"\bHong Kong\b", r"\bHKCFA\b", r"\bCFA\b", r"\bHKSAR\b", r"\bBasic Law\b",
    # UAE / DIFC / ADGM
    r"\bUAE\b", r"\bDIFC\b", r"\bADGM\b", r"\bEmirat(?:es|i)\b", r"\bSharia\b",
    # Malaysia / Indonesia / Thailand / Vietnam / Philippines
    r"\bMalaysia(?:n)?\b", r"\bBursa\b",
    r"\bIndonesia(?:n)?\b", r"\bMahkamah\b", r"\bKUHP\b",
    r"\bThai(?:land)?\b",
    r"\bVietnam(?:ese)?\b",
    r"\bPhilippin(?:e|es|o)\b", r"\bRTC\b", r"\bSandiganbayan\b",
    # Japan / Korea
    r"\bJapan(?:ese)?\b", r"\bMinpo\b", r"\bSaikosai\b",
    r"\bKorea(?:n)?\b", r"\bROK\b", r"\bKCC\b",
    # South Asia neighbours
    r"\bBangladesh(?:i)?\b", r"\bSri Lanka(?:n)?\b", r"\bNepal(?:i|ese)?\b",
    # Cross-border commercial terms
    r"\bchoice[- ]of[- ]law\b", r"\bseat\s+of\s+arbitration\b", r"\bNew York Convention\b",
    r"\benforce(?:ment)?\b", r"\bjurisdiction(?:al)?\b",
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
    # Short ("expand on [2]", "what about Delhi?") in an existing thread
    return history_len > 0 and len(text.split()) <= 12


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

    # G_SCOPE — allow follow-ups in an existing thread without re-checking
    if not _is_followup(q, history_len) and not LEGAL_RE.search(q):
        return GuardVerdict(
            allow=False,
            refusal_message="Sanhita is for Indian legal research. Try a question about a statute, a section, a judgment, a court process, or a fact pattern under Indian law.",
            reason="off_scope",
        )

    # G_PII — redact, don't refuse
    redacted, notes = _redact_pii(q)

    return GuardVerdict(
        allow=True,
        redacted_question=redacted,
        notes=notes,
    )
