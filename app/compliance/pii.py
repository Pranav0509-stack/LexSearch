"""PII redaction for logs + observability (plan §7.3)."""

import re

_PATTERNS = [
    (re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[AADHAAR]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    (re.compile(r"\b\d{9,18}\b"), "[ACCOUNT]"),
]


def redact(text: str) -> str:
    out = text
    for pat, sub in _PATTERNS:
        out = pat.sub(sub, out)
    return out


def redact_dict(data: dict) -> dict:
    out: dict = {}
    for k, v in data.items():
        if isinstance(v, str):
            out[k] = redact(v)
        elif isinstance(v, dict):
            out[k] = redact_dict(v)
        elif isinstance(v, list):
            out[k] = [redact(x) if isinstance(x, str) else x for x in v]
        else:
            out[k] = v
    return out
