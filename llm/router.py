"""
Sanhita — LLM Router (paid-tier).

Four-provider stack walked in order, with per-provider circuit breakers:

  1. Gemini 2.5 Flash             — PRIMARY  (fast, cheap, long-context)
  2. Anthropic Claude Sonnet 4.5  — fallback (best reasoning, paid)
  3. Groq Llama 3.3 70B           — helper   (sub-second TTFT, fast lane)
  4. Cloudflare Workers AI 8B     — emergency helper (cheap classify/rewrite)

Why this order: Gemini 2.5 Flash is the sweet spot for grounded legal Q&A —
sub-second TTFT, 1M-token context for multi-PDF Vault, and roughly 10× cheaper
per token than Sonnet. Anthropic Sonnet 4.5 sits second as the heavyweight
reasoning fallback for any Gemini outage. Groq Llama-70B is third for further
resilience. Cloudflare is for cheap helper tasks like query rewrite or scope
classification. Every provider speaks plain HTTP (no SDK dependencies).

Each provider is wrapped in a circuit breaker: 3 failures inside 60s opens
the breaker for 120s (no calls land on it during that window). The router
walks the chain in order, returning the first success and the provider
metadata so the UI can show which model answered.

Environment:
  ANTHROPIC_API_KEY       — required for Claude (primary)
  ANTHROPIC_MODEL         — default "claude-sonnet-4-5-20250929"
  GROQ_API_KEY            — required for Groq
  GROQ_MODEL              — default "llama-3.3-70b-versatile"
  GEMINI_API_KEY          — required for Gemini
  GEMINI_MODEL            — default "gemini-2.5-pro"
  CF_ACCOUNT_ID           — required for Cloudflare
  CF_API_TOKEN            — required for Cloudflare
  CF_MODEL                — default "@cf/meta/llama-3.1-8b-instruct"
  LLM_TIMEOUT_S           — per-call timeout (default 30)
"""

from __future__ import annotations

import logging
import os
import ssl
import time
import urllib.error
import urllib.request
import json
from dataclasses import dataclass, field
from typing import Any, Optional

# Use certifi's CA bundle if available (fixes macOS python.org SSL trust issues).
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl.create_default_context()

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929").strip()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()

CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "").strip()
CF_API_TOKEN = os.environ.get("CF_API_TOKEN", "").strip()
CF_MODEL = os.environ.get("CF_MODEL", "@cf/meta/llama-3.1-8b-instruct").strip()

LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "30"))


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
        if self.opened_at:  # cooldown elapsed → close
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
    "anthropic": _Breaker("anthropic"),
    "groq": _Breaker("groq"),
    "gemini": _Breaker("gemini"),
    "cloudflare": _Breaker("cloudflare"),
}


_UA = "Sanhita-Brief/1.0 (+https://sanhita.law)"


def _http_post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    merged = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": _UA,
        **headers,
    }
    req = urllib.request.Request(url, data=body, headers=merged, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # surface the response body so the caller logs the real reason
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            err_body = ""
        raise RuntimeError(f"HTTP {e.code}: {err_body or e.reason}") from e


# ── Providers ──────────────────────────────────────────────────────────────
def _call_anthropic(system: str, user: str, *, temperature: float, max_tokens: int) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    data = _http_post_json(
        "https://api.anthropic.com/v1/messages",
        {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        {
            "model": ANTHROPIC_MODEL,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=LLM_TIMEOUT_S,
    )
    # Response shape: { content: [{type: "text", text: "..."}], ... }
    blocks = data.get("content") or []
    text_parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
    return "".join(text_parts).strip()


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
                {"role": "user", "content": user},
            ],
        },
        timeout=LLM_TIMEOUT_S,
    )
    return data["choices"][0]["message"]["content"].strip()


def _call_cloudflare(system: str, user: str, *, temperature: float, max_tokens: int) -> str:
    if not (CF_ACCOUNT_ID and CF_API_TOKEN):
        raise RuntimeError("CF_ACCOUNT_ID / CF_API_TOKEN not set")
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_MODEL}"
    data = _http_post_json(
        url,
        {"Authorization": f"Bearer {CF_API_TOKEN}"},
        {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=LLM_TIMEOUT_S,
    )
    # CF response shape: { result: { response: "..." }, success: true }
    if not data.get("success"):
        raise RuntimeError(f"cloudflare error: {data.get('errors')}")
    return (data.get("result", {}).get("response") or "").strip()


def _call_gemini(system: str, user: str, *, temperature: float, max_tokens: int) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    # Use plain REST (no SDK dep). v1beta supports system_instruction + generationConfig.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    data = _http_post_json(
        url,
        {},
        {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        },
        timeout=LLM_TIMEOUT_S,
    )
    candidates = data.get("candidates") or []
    if not candidates:
        # Some safety blocks return promptFeedback only.
        feedback = data.get("promptFeedback") or {}
        raise RuntimeError(f"gemini empty response: {feedback or data}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────
# Gemini function-calling — single round-trip primitive used by the agent
# loop. Caller passes the system prompt, the running conversation
# (`contents`) and the tool catalog (`tools`); we return a structured dict
# with either {"text": "..."} (model finished) or {"function_calls": [...]}
# (model wants to invoke a tool). Caller is responsible for executing tools
# and appending the responses back into `contents`, then calling again.
#
# This is gemini-only by design — the agent loop is not provider-portable.
# Anthropic / Groq fallback is for non-agentic use only.
# ─────────────────────────────────────────────────────────────────────────

def gemini_tool_call(
    system: str,
    contents: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> dict[str, Any]:
    """Single Gemini tool-calling round-trip.

    Args:
      system:    System instruction (the agent's persona / scope).
      contents:  The running conversation. Each entry is
                 {"role": "user"|"model", "parts": [...]}. Tool responses
                 are sent as a "user"-role message containing a
                 functionResponse part.
      tools:     [{"functionDeclarations": [{name, description, parameters}, ...]}]
                 — Gemini's standard tool spec.

    Returns a dict with these keys:
      - "text":            assistant text part (may be empty)
      - "function_calls":  list of {name, args} pending dispatch
      - "finish_reason":   STOP | MAX_TOKENS | SAFETY | TOOL_CALL | ...
      - "raw":             the candidate content dict (so caller can append
                           it verbatim as the next "model"-role message in
                           contents — Gemini wants the round-trip to include
                           its own functionCall part).

    Raises RuntimeError if the API rejects the call or returns nothing.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload: dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if tools:
        payload["tools"] = tools
    data = _http_post_json(url, {}, payload, timeout=LLM_TIMEOUT_S)
    candidates = data.get("candidates") or []
    if not candidates:
        feedback = data.get("promptFeedback") or {}
        raise RuntimeError(f"gemini empty response: {feedback or data}")
    cand = candidates[0]
    content = cand.get("content") or {}
    parts = content.get("parts") or []

    text_chunks: list[str] = []
    function_calls: list[dict[str, Any]] = []
    for p in parts:
        if "text" in p:
            text_chunks.append(p["text"])
        if "functionCall" in p:
            fc = p["functionCall"]
            function_calls.append({
                "name": fc.get("name", ""),
                "args": fc.get("args", {}) or {},
            })
    return {
        "text": "".join(text_chunks).strip(),
        "function_calls": function_calls,
        "finish_reason": cand.get("finishReason", ""),
        "raw": content,
    }


# ── Router ─────────────────────────────────────────────────────────────────
@dataclass
class LLMResponse:
    text: str
    provider: str       # "groq" | "cloudflare" | "gemini"
    model: str
    latency_ms: int
    fallback_chain: list[str]   # which providers were tried

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "fallback_chain": self.fallback_chain,
        }


_CHAIN: list[tuple[str, callable, str]] = [
    ("gemini", _call_gemini, GEMINI_MODEL),
    ("anthropic", _call_anthropic, ANTHROPIC_MODEL),
    ("groq", _call_groq, GROQ_MODEL),
    ("cloudflare", _call_cloudflare, CF_MODEL),
]


def available_providers() -> list[str]:
    out = []
    if ANTHROPIC_API_KEY: out.append("anthropic")
    if GROQ_API_KEY: out.append("groq")
    if GEMINI_API_KEY: out.append("gemini")
    if CF_ACCOUNT_ID and CF_API_TOKEN: out.append("cloudflare")
    return out


def generate(
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 900,
    prefer: Optional[str] = None,
) -> LLMResponse:
    """
    Walk the provider chain. Returns the first successful response.

    `prefer="cloudflare"` short-circuits to the small/fast helper for
    classification/rewrite tasks (where we don't need 70B).

    Raises RuntimeError if every available provider fails / breaker is open.
    """
    chain = _CHAIN
    if prefer:
        chain = sorted(_CHAIN, key=lambda x: 0 if x[0] == prefer else 1)

    tried: list[str] = []
    last_err: Optional[Exception] = None

    for name, fn, model_name in chain:
        breaker = _BREAKERS[name]
        if breaker.is_open():
            tried.append(f"{name}:open")
            continue
        # skip providers without creds
        if name == "anthropic" and not ANTHROPIC_API_KEY: continue
        if name == "groq" and not GROQ_API_KEY: continue
        if name == "cloudflare" and not (CF_ACCOUNT_ID and CF_API_TOKEN): continue
        if name == "gemini" and not GEMINI_API_KEY: continue

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

    raise RuntimeError(f"all LLM providers failed (tried={tried}): {last_err}")


# ─────────────────────────────────────────────────────────────────────────
# Gemini text embeddings — `text-embedding-004`. 768-dim vectors. Used for
# semantic vault search (cosine similarity) which we then RRF-merge with
# BM25 to get the "Harvey-quality" recall the user asked for.
#
# Free tier: 1500 RPM, 1M tokens/day. Comfortably enough for vault upload
# (typical 10-page contract = ~30 chunks = 30 calls one-time on upload).
# ─────────────────────────────────────────────────────────────────────────

EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "text-embedding-004").strip()
EMBED_DIM = 768  # text-embedding-004 fixed output


def embed_texts(
    texts: list[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    timeout: int = 30,
) -> list[list[float]]:
    """Embed a batch of texts via Gemini's embedContent endpoint.

    `task_type` matters: for indexing chunks use RETRIEVAL_DOCUMENT, for
    embedding the user query use RETRIEVAL_QUERY. Same model, different
    projection — relevance is materially better when matched.

    Returns one 768-float vector per input. On failure raises so the
    caller can fall back to BM25-only retrieval (which is already good).
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set — cannot embed")
    if not texts:
        return []
    # batchEmbedContents handles up to 100 inputs per request. For tiny
    # batches we still use it (single round-trip, simpler error path).
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{EMBED_MODEL}:batchEmbedContents?key={GEMINI_API_KEY}"
    )
    out: list[list[float]] = []
    for i in range(0, len(texts), 100):
        chunk = texts[i : i + 100]
        payload = {
            "requests": [
                {
                    "model": f"models/{EMBED_MODEL}",
                    "content": {"parts": [{"text": t[:8000]}]},
                    "taskType": task_type,
                }
                for t in chunk
            ]
        }
        data = _http_post_json(url, {}, payload, timeout=timeout)
        embeddings = data.get("embeddings") or []
        if len(embeddings) != len(chunk):
            raise RuntimeError(
                f"gemini embed: requested {len(chunk)} got {len(embeddings)}"
            )
        for e in embeddings:
            vals = (e.get("values") or [])
            if not vals:
                raise RuntimeError("gemini embed: empty vector")
            out.append([float(v) for v in vals])
    return out


def embed_one(text: str, *, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    """Convenience: embed a single string. Used for query-time embedding."""
    return embed_texts([text], task_type=task_type)[0]
