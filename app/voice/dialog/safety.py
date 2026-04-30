"""
Parallel safety classifier — plan §2.4 guardrails.

Short LLM prompt returns {category, severity, safe_reply}. Orchestrator can
short-circuit before spending a full LLM turn.
"""

import re
from dataclasses import dataclass


@dataclass
class SafetyVerdict:
    must_refuse: bool
    escalate: bool
    category: str        # 'imminent_harm' | 'asks_advice' | 'out_of_scope' | 'ok'
    safe_reply: str


# Hard-trigger patterns (multilingual). Real impl uses a tiny LLM classifier too.
_IMMINENT_HARM_PATTERNS = [
    r"\b(suicide|kill myself|end my life|mar jaana|atmahatya)\b",
    r"\b(beating me|beat me|maar raha|ghar mein mar)\b",
    r"\b(child abuse|rape|forced)\b",
]
_ADVICE_QUESTIONS = [
    r"\b(should i sue|will i win|is this legal|kya main sue karun)\b",
]


async def classify_safety(text: str, language: str) -> SafetyVerdict:
    low = text.lower()

    for pat in _IMMINENT_HARM_PATTERNS:
        if re.search(pat, low):
            return SafetyVerdict(
                must_refuse=True,
                escalate=True,
                category="imminent_harm",
                safe_reply=(
                    "Aapki suraksha sabse pehle hai. Main abhi aapko ek vakeel "
                    "aur helpline se jod raha hoon. Emergency ke liye 112 dial karein."
                ),
            )

    for pat in _ADVICE_QUESTIONS:
        if re.search(pat, low):
            return SafetyVerdict(
                must_refuse=False,
                escalate=True,
                category="asks_advice",
                safe_reply=(
                    "Main AI hoon — yeh decision ek vakeel hi kar sakte hain. "
                    "Main aapko ek panel-advocate se jod sakta hoon."
                ),
            )

    return SafetyVerdict(
        must_refuse=False, escalate=False, category="ok", safe_reply=""
    )
