"""
Sanhita — LLM Router.

Provider chain (walked in order, circuit-breaker per provider):

  1. OpenAI GPT-4o       — PRIMARY  (best instruction-following, paid)
  2. Gemini 2.5 Flash    — SECONDARY (fast, free tier available)
  3. Groq Llama 3.3 70B  — TERTIARY  (sub-second, free tier)
  4. Anthropic Claude    — QUATERNARY (backup, paid)

Circuit breaker: 3 failures in 60s → provider skipped for 120s.

Environment:
  OPENAI_API_KEY    — required for OpenAI (primary)
  OPENAI_MODEL      — default "gpt-4o-mini"
  GEMINI_API_KEY    — required for Gemini
  GEMINI_MODEL      — default "gemini-2.5-flash"
  GROQ_API_KEY      — required for Groq
  GROQ_MODEL        — default "llama-3.3-70b-versatile"
  ANTHROPIC_API_KEY — required for Claude
  ANTHROPIC_MODEL   — default "claude-sonnet-4-5"
  LLM_TIMEOUT_S     — per-call timeout (default 45)
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl.create_default_context()

logger = logging.getLogger(__name__)

# ── Credentials ────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL     = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()

GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL     = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()

GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL       = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL   = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5").strip()

LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "45"))

_UA = "Sanhita-Brief/2.0 (+https://sanhita.law)"


# ── Circuit breaker ────────────────────────────────────────────────────────
@dataclass
class _Breaker:
    name: str
    fail_window_s: float = 60.0
    fail_threshold: int = 3
    open_for_s: float = 120.0
    failures: list[float] = field(default_factory=list)
    opened_at: float = 0.0

    def is_open(self) -> bool:
        if self.opened_at and (time.monotonic() - self.opened_at) < self.open_for_s:
            return True
        if self.opened_at:
            self.opened_at = 0.0
            self.failures.clear()
        return False

    def record_success(self) -> None:
        self.failures.clear()
        self.opened_at = 0.0

    def record_failure(self) -> None:
        now = time.monotonic()
        self.failures = [t for t in self.failures if now - t < self.fail_window_s]
        self.failures.append(now)
        if len(self.failures) >= self.fail_threshold:
            self.opened_at = now
            logger.warning("[router] breaker OPEN for %s (cooldown %.0fs)", self.name, self.open_for_s)


_BREAKERS: dict[str, _Breaker] = {
    "openai": _Breaker("openai"),
    "gemini": _Breaker("gemini"),
    "groq": _Breaker("groq"),
    "anthropic": _Breaker("anthropic"),
}


# ── HTTP helper ───────────────────────────────────────────────────────────
def _http_post_json(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "User-Agent": _UA, **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:600]
        except Exception:
            err_body = ""
        raise RuntimeError(f"HTTP {e.code}: {err_body or e.reason}") from e


# ── Provider implementations ──────────────────────────────────────────────
def _call_openai(system: str, user: str, *, temperature: float, max_tokens: int) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    data = _http_post_json(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {OPENAI_API_KEY}"},
        {
            "model": OPENAI_MODEL,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        },
        timeout=LLM_TIMEOUT_S,
    )
    return data["choices"][0]["message"]["content"].strip()


def _call_gemini(system: str, user: str, *, temperature: float, max_tokens: int) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:
        raise RuntimeError(f"google-generativeai not installed: {e}")
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system)
    resp = model.generate_content(
        user,
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
    )
    return (getattr(resp, "text", "") or "").strip()


def _call_groq(system: str, user: str, *, temperature: float, max_tokens: int) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    data = _http_post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {"Authorization": f"Bearer {GROQ_API_KEY}"},
        {
            "model": GROQ_MODEL,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        },
        timeout=LLM_TIMEOUT_S,
    )
    return data["choices"][0]["message"]["content"].strip()


def _call_anthropic(system: str, user: str, *, temperature: float, max_tokens: int) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    data = _http_post_json(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
        {
            "model": ANTHROPIC_MODEL,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=LLM_TIMEOUT_S,
    )
    blocks = data.get("content") or []
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()


# ── Router ────────────────────────────────────────────────────────────────
@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: int
    fallback_chain: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "fallback_chain": self.fallback_chain,
        }


_CHAIN: list[tuple[str, Any, str]] = [
    ("openai",     _call_openai,     OPENAI_MODEL),
    ("gemini",     _call_gemini,     GEMINI_MODEL),
    ("groq",       _call_groq,       GROQ_MODEL),
    ("anthropic",  _call_anthropic,  ANTHROPIC_MODEL),
]

_HAS_CREDS = {
    "openai":    lambda: bool(OPENAI_API_KEY),
    "gemini":    lambda: bool(GEMINI_API_KEY),
    "groq":      lambda: bool(GROQ_API_KEY),
    "anthropic": lambda: bool(ANTHROPIC_API_KEY),
}


def available_providers() -> list[str]:
    return [name for name, check in _HAS_CREDS.items() if check()]


def generate(
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    prefer: Optional[str] = None,
) -> LLMResponse:
    """Walk the provider chain, return first success."""
    chain = _CHAIN
    if prefer:
        chain = sorted(_CHAIN, key=lambda x: 0 if x[0] == prefer else 1)

    tried: list[str] = []
    last_err: Optional[Exception] = None

    for name, fn, model_name in chain:
        if not _HAS_CREDS[name]():
            continue
        breaker = _BREAKERS[name]
        if breaker.is_open():
            tried.append(f"{name}:open")
            continue
        t0 = time.monotonic()
        tried.append(name)
        try:
            text = fn(system, user, temperature=temperature, max_tokens=max_tokens)
            if not text:
                raise RuntimeError("empty response")
            breaker.record_success()
            return LLMResponse(
                text=text,
                provider=name,
                model=model_name,
                latency_ms=int((time.monotonic() - t0) * 1000),
                fallback_chain=tried,
            )
        except Exception as e:
            last_err = e
            breaker.record_failure()
            logger.warning("[router] %s failed: %s", name, e)

    raise RuntimeError(f"all providers failed (tried={tried}): {last_err}")
