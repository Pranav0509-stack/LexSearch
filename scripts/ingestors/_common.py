"""Shared helpers for the GitHub-data ingestors.

Each per-source ingestor exposes a single `ingest(*, limit=None)` that
yields `Document` instances. Driver script (`ingest_github_data.py`)
glues them together and persists to `bm25.pkl`.

We keep network access minimal: prefer raw github content URLs (no API
quota) over `api.github.com` calls. When the API is needed (listing repo
contents in an org), use `gh_api()` with optional GITHUB_TOKEN for the
60 → 5000 rph bump.
"""

from __future__ import annotations

import json
import logging
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

# Local Python on dev machines often lacks system root CAs (we saw
# CERTIFICATE_VERIFY_FAILED in the FastAPI logs). Try to use certifi if
# installed (it's a transitive dep of requests), else fall back to the
# unverified context — the ingestors only fetch public github content,
# so MITM risk is low and predictable.
try:
    import certifi  # type: ignore
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL_CTX = ssl._create_unverified_context()  # noqa: SLF001


HTTP_TIMEOUT = 30.0
USER_AGENT = "Sanhita-Ingestor/1.0 (+https://sanhita.ai)"


def _key(name: str) -> str:
    """Read API keys from the auth.connector_keys table first, env var
    second. Mirrors the resolver in `connectors._key`."""
    try:
        import auth  # type: ignore
        v = auth.get_connector_key(name)
        if v:
            return v
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get(f"{name.upper()}_API_KEY", "").strip() or os.environ.get(name.upper(), "").strip()


def http_get(url: str, *, accept: str = "*/*", timeout: float = HTTP_TIMEOUT) -> bytes:
    """Fetch raw bytes from a URL. Raises on non-2xx."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": accept},
    )
    token = _key("github")
    if token and "api.github.com" in url:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
        return r.read()


def gh_raw(user: str, repo: str, path: str, ref: str = "HEAD") -> str:
    """Build a raw.githubusercontent.com URL."""
    return f"https://raw.githubusercontent.com/{user}/{repo}/{ref}/{path.lstrip('/')}"


def gh_api(path: str) -> Any:
    """Hit the GitHub REST API and decode JSON."""
    url = f"https://api.github.com{path}"
    body = http_get(url, accept="application/vnd.github+json")
    return json.loads(body.decode("utf-8"))


# ─────────────────────────────────────────────────────────────────────────
# Field normalisers — every ingestor benefits from cheap, lenient parsing
# of years, citations, and tier hints out of free-text columns.
# ─────────────────────────────────────────────────────────────────────────

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_HK_CITATION_RE = re.compile(r"\[(\d{4})\]\s*HK([A-Z]{2,4})", re.IGNORECASE)
_IN_CITATION_RE = re.compile(r"\b(AIR|SCC|SCR|SCJ|SCALE)\b", re.IGNORECASE)
_SG_CITATION_RE = re.compile(r"\[(\d{4})\]\s*SG([A-Z]{2,4})", re.IGNORECASE)


def extract_year(*candidates: str) -> int | None:
    """Find the first 4-digit year in any of the given strings."""
    for c in candidates:
        if not c:
            continue
        m = _YEAR_RE.search(str(c))
        if m:
            try:
                y = int(m.group(1))
                if 1800 < y < 2100:
                    return y
            except ValueError:
                pass
    return None


def hk_tier_from_citation(citation: str) -> str:
    """HK case tier: CFA = Court of Final Appeal, CA = Court of Appeal,
    CFI = Court of First Instance, DC = District Court, etc.

    Reads the suffix after `[YYYY] HK<XXX>` — see
    https://www.hklii.hk/eng/hk/cases/."""
    m = _HK_CITATION_RE.search(citation or "")
    if not m:
        return ""
    return m.group(2).upper()


def sg_tier_from_citation(citation: str) -> str:
    m = _SG_CITATION_RE.search(citation or "")
    if not m:
        return ""
    return m.group(2).upper()


def stable_case_id(*parts: Any) -> str:
    """Deterministic id from any combination of parts. Lower-case,
    alnum-only, slug-safe so we can reuse it across re-ingests."""
    raw = "-".join(str(p) for p in parts if p)
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", raw).strip("-").lower()[:80] or "anon"


def clean_text(s: str | None, *, max_chars: int = 8000) -> str:
    """Collapse whitespace, strip BOMs, cap length."""
    if not s:
        return ""
    out = re.sub(r"\s+", " ", str(s)).strip()
    if len(out) > max_chars:
        out = out[:max_chars]
    return out
