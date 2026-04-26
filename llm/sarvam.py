"""
Sarvam AI — Indian-language translation post-processor.

Sarvam's `mayura` translation model is materially better at Indian-language
legal vernacular than direct Gemini output: it preserves transliteration for
proper nouns ("Sanhita", "BNS", "§138 NI Act") while rendering connectives
and explanations in idiomatic target-language prose. We use it as an
optional post-processor over the LLM router's English/Hindi answer when the
target is one of Sarvam's supported Indian languages.

Pipeline:
  router.generate(...) → English/Hindi markdown
                       → sarvam.translate(text, "en-IN", target)
                       → answer in target language, citations preserved

If Sarvam is unreachable or no key is set, we degrade silently to the
router's native multilingual output (which is still good — Gemini handles
all 22 languages, just with weaker legal vernacular).

API: https://api.sarvam.ai/translate
Auth: api-subscription-key header
Model: mayura:v1
Free tier: ~1000 requests/month at time of writing.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

SARVAM_ENDPOINT = "https://api.sarvam.ai/translate"
SARVAM_TIMEOUT = float(os.environ.get("SARVAM_TIMEOUT_S", "20"))
SARVAM_MODEL = os.environ.get("SARVAM_MODEL", "mayura:v1").strip() or "mayura:v1"

# Single chunk cap — Sarvam's API documents 1000 chars, so we conservatively
# split at ~900 to leave headroom for trailing punctuation. Longer answers
# are split on paragraph boundaries and re-joined.
SARVAM_MAX_CHARS = 900


# ── Eighth-Schedule code → Sarvam BCP-47-ish code ────────────────────────
# Sarvam's translate API expects codes like "en-IN", "hi-IN". Only the
# languages listed here are actually supported by mayura today; for the
# rest we return None and the caller falls back to direct Gemini output.
SARVAM_LANG_MAP: dict[str, str] = {
    "en":  "en-IN",
    "hi":  "hi-IN",
    "bn":  "bn-IN",
    "gu":  "gu-IN",
    "kn":  "kn-IN",
    "ml":  "ml-IN",
    "mr":  "mr-IN",
    "or":  "od-IN",   # Sarvam uses od-IN for Odia (Oriya)
    "pa":  "pa-IN",
    "ta":  "ta-IN",
    "te":  "te-IN",
}


def supports(lang_code: str) -> bool:
    """Whether Sarvam's mayura model can translate INTO `lang_code`."""
    return lang_code in SARVAM_LANG_MAP


def _key() -> str:
    """DB-first, env-fallback. Mirrors connectors._key shape."""
    try:
        import auth
        db_key = auth.get_connector_key("sarvam")
        if db_key:
            return db_key.strip()
    except Exception:
        pass
    return os.environ.get("SARVAM_API_KEY", "").strip()


def _split_for_translate(text: str) -> list[str]:
    """Split a long markdown blob into <=SARVAM_MAX_CHARS pieces along
    paragraph boundaries so each request stays under the API limit. We
    re-join with the same delimiter to preserve formatting."""
    if len(text) <= SARVAM_MAX_CHARS:
        return [text]
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in paragraphs:
        # Flush if adding this paragraph would overflow.
        if buf and (buf_len + len(p) + 2) > SARVAM_MAX_CHARS:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0
        # If a single paragraph is itself bigger than the limit, hard-split
        # on sentence boundaries as a last resort.
        if len(p) > SARVAM_MAX_CHARS:
            if buf:
                chunks.append("\n\n".join(buf))
                buf = []
                buf_len = 0
            for i in range(0, len(p), SARVAM_MAX_CHARS):
                chunks.append(p[i : i + SARVAM_MAX_CHARS])
            continue
        buf.append(p)
        buf_len += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _translate_chunk(text: str, source: str, target: str, key: str) -> str:
    body = json.dumps({
        "input": text,
        "source_language_code": source,
        "target_language_code": target,
        "model": SARVAM_MODEL,
        # `formal` register reads better for legal prose than the default
        # `modern-colloquial`. Sarvam silently ignores unknown values, so
        # this is safe across model versions.
        "mode": "formal",
        # Keep proper nouns (party names, statute citations) verbatim so
        # the [n] citation rail's links continue to resolve.
        "enable_preprocessing": True,
        "speaker_gender": "Male",
    }).encode("utf-8")
    req = urllib.request.Request(
        SARVAM_ENDPOINT,
        method="POST",
        data=body,
        headers={
            "Content-Type": "application/json",
            "api-subscription-key": key,
        },
    )
    with urllib.request.urlopen(req, timeout=SARVAM_TIMEOUT) as resp:
        raw = resp.read()
        data = json.loads(raw.decode("utf-8", errors="replace"))
    out = (data.get("translated_text") or "").strip()
    if not out:
        raise RuntimeError(f"sarvam empty response: {data}")
    return out


def translate(
    text: str,
    target_lang: str,
    *,
    source_lang: str = "en",
) -> Optional[str]:
    """Translate `text` from `source_lang` → `target_lang` via Sarvam.

    Both args are short ISO codes ("en", "hi", "bn", ...). Returns the
    translated string on success, or None if Sarvam isn't configured or
    the target language isn't supported. Never raises — callers degrade
    silently to whatever the LLM router produced natively.
    """
    if not text or not text.strip():
        return None
    src = SARVAM_LANG_MAP.get(source_lang)
    tgt = SARVAM_LANG_MAP.get(target_lang)
    if not src or not tgt or src == tgt:
        return None
    key = _key()
    if not key:
        return None
    try:
        chunks = _split_for_translate(text)
        translated = [_translate_chunk(c, src, tgt, key) for c in chunks]
        return "\n\n".join(translated)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.warning("sarvam translate failed (%s → %s): %s", src, tgt, e)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("sarvam translate unexpected error: %s", e)
        return None
