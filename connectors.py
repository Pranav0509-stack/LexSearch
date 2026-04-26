"""
Sanhita — live data connectors.

Implements the pluggable-source architecture from the production plan:

  SourceConnector (abstract surface) → {
      indian_kanoon, ecourts, seed_corpus, web_search, egov_japan,
      lawnet_sg, hklii, dubai_pulse, klri, clj, jdih
  }

Each connector returns hits in the unified shape consumed by brief_service:
    {case_id, title, citation, court, year, tier, excerpt, score, jurisdiction,
     source, url, s3_key, pdf_name}

Keys are loaded DB-first (auth.connector_keys, settable via /api/settings/keys),
falling back to env vars (INDIAN_KANOON_API_KEY, ECOURTS_API_KEY, SERPER_API_KEY,
TAVILY_API_KEY, LAWNET_SG_API_KEY, DUBAI_PULSE_API_KEY, CLJ_API_KEY).

When unset, the connector is silently skipped and the caller falls back to the
next source in the chain.

No external deps beyond `urllib.request` — ships in stdlib.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import urllib.request
import urllib.error
from typing import Any

logger = logging.getLogger(__name__)

HTTP_TIMEOUT = float(os.environ.get("CONNECTOR_TIMEOUT_S", "6"))
USER_AGENT = "Sanhita/1.0 (+https://sanhita.ai)"


# ── key loader (DB-first, env-fallback) ───────────────────────────────
def _key(name: str) -> str:
    """Return the API key for `name`, prefering the DB-backed keystore.

    Lookup order:
      1. auth.get_connector_key(name)         — set via /api/settings/keys
      2. os.environ[f"{NAME}_API_KEY"]        — bootstrap from launch.json

    Defensive against import order: if auth isn't ready yet, fall straight
    through to env. Empty string means "no key configured".
    """
    name = (name or "").strip().lower()
    if not name:
        return ""
    try:
        import auth  # local import — avoid circular at module load
        db_key = auth.get_connector_key(name)
        if db_key:
            return db_key.strip()
    except Exception:
        pass
    return os.environ.get(f"{name.upper()}_API_KEY", "").strip()


# ── shared http helper ────────────────────────────────────────────────
def _http_json(url: str, *, method: str = "GET", headers: dict | None = None,
               body: bytes | None = None, timeout: float = HTTP_TIMEOUT) -> dict | list | None:
    """GET/POST a URL, return parsed JSON, None on error. Cheap + defensive."""
    try:
        req = urllib.request.Request(url, method=method, data=body)
        req.add_header("User-Agent", USER_AGENT)
        req.add_header("Accept", "application/json")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8", errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as e:
        logger.warning("connector fetch failed (%s): %s", url[:80], e)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("connector unexpected error (%s): %s", url[:80], e)
        return None


def _http_text(url: str, *, timeout: float = HTTP_TIMEOUT) -> str | None:
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", USER_AGENT)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        logger.warning("connector text fetch failed (%s): %s", url[:80], e)
        return None


# ── Indian Kanoon API ─────────────────────────────────────────────────
def indian_kanoon_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """
    Indian Kanoon public API. Key required (~₹500 free on signup).
    Endpoint: POST https://api.indiankanoon.org/search/?formInput=<q>&pagenum=0
    Auth: Authorization: Token <key>
    """
    key = _key("indian_kanoon")
    if not key:
        return []
    encoded_q = urllib.parse.quote_plus(query)
    url = f"https://api.indiankanoon.org/search/?formInput={encoded_q}&pagenum=0"
    data = _http_json(url, method="POST", headers={"Authorization": f"Token {key}"})
    if not isinstance(data, dict):
        return []
    docs = data.get("docs", [])[:k]
    out: list[dict[str, Any]] = []
    for i, d in enumerate(docs):
        tid = str(d.get("tid") or d.get("docid") or "")
        out.append({
            "case_id": f"IK-{tid}",
            "title": d.get("title") or "Untitled",
            "citation": d.get("publishdate") or "",
            "court": d.get("docsource") or "Indian Kanoon",
            "year": _extract_year(d.get("publishdate", "")),
            "tier": "SC" if "Supreme Court" in (d.get("docsource") or "") else "HC",
            "excerpt": _strip_html(d.get("headline") or d.get("fragments") or "")[:600],
            "score": 1.0 - (i * 0.05),
            "jurisdiction": "IN",
            "source": "indian_kanoon",
            "url": f"https://indiankanoon.org/doc/{tid}/",
            "s3_key": None,
            "pdf_name": None,
        })
    return out


# ── eCourts India ─────────────────────────────────────────────────────
def ecourts_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """
    eCourts India partner API. Key required (₹200 free credits on signup).
    Endpoint: GET https://webapi.ecourtsindia.com/api/partner/search?q=<q>
    """
    key = _key("ecourts")
    if not key:
        return []
    url = f"https://webapi.ecourtsindia.com/api/partner/search?q={urllib.parse.quote_plus(query)}&limit={k}"
    data = _http_json(url, headers={"Authorization": f"Bearer {key}"})
    if not isinstance(data, dict):
        return []
    hits = data.get("results") or data.get("hits") or []
    out: list[dict[str, Any]] = []
    for i, h in enumerate(hits[:k]):
        cnr = h.get("cnr") or h.get("case_number") or ""
        out.append({
            "case_id": f"ECT-{cnr}",
            "title": h.get("case_title") or h.get("title") or "Untitled",
            "citation": h.get("citation") or cnr,
            "court": h.get("court") or "eCourts India",
            "year": _extract_year(h.get("filing_date", "")),
            "tier": h.get("court_tier") or "HC",
            "excerpt": (h.get("summary") or "")[:600],
            "score": 1.0 - (i * 0.05),
            "jurisdiction": "IN",
            "source": "ecourts",
            "url": h.get("url") or "",
            "s3_key": None,
            "pdf_name": None,
        })
    return out


# ── Japan e-Gov Laws API (FREE, no auth) ──────────────────────────────
def egov_japan_search(query: str, k: int = 3) -> list[dict[str, Any]]:
    """
    e-Gov Laws API v2 — free, no key required.
    https://laws.e-gov.go.jp/api/2/keyword?keyword=<q>
    """
    try:
        enc = urllib.parse.quote(query)
        url = f"https://laws.e-gov.go.jp/api/2/keyword?keyword={enc}&limit={k}"
        data = _http_json(url)
        if not isinstance(data, dict):
            return []
        hits = data.get("laws") or data.get("results") or []
        out: list[dict[str, Any]] = []
        for i, h in enumerate(hits[:k]):
            lid = h.get("law_id") or h.get("lawId") or ""
            out.append({
                "case_id": f"JP-{lid}",
                "title": h.get("law_name") or h.get("lawName") or "Japanese Statute",
                "citation": lid,
                "court": "Government of Japan (e-Gov)",
                "year": h.get("promulgation_date", "")[:4],
                "tier": "Statute",
                "excerpt": (h.get("preamble") or h.get("excerpt") or "Statute reference from Japan e-Gov Laws API")[:600],
                "score": 0.9 - (i * 0.05),
                "jurisdiction": "JP",
                "source": "egov_japan",
                "url": f"https://laws.e-gov.go.jp/law/{lid}",
                "s3_key": None,
                "pdf_name": None,
            })
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("egov japan failed: %s", e)
        return []


# ── Singapore — LawNet (Singapore Academy of Law) ─────────────────────
def lawnet_sg_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """
    LawNet API (Singapore Academy of Law). Key required.
    Public docs: https://www.lawnet.sg/lawnet/web/lawnet/api
    """
    key = _key("lawnet_sg")
    if not key:
        return []
    url = "https://api.lawnet.sg/v1/search"
    body = json.dumps({"query": query, "limit": k}).encode("utf-8")
    data = _http_json(
        url, method="POST", body=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    if not isinstance(data, dict):
        return []
    hits = data.get("results") or data.get("hits") or []
    out: list[dict[str, Any]] = []
    for i, h in enumerate(hits[:k]):
        doc_id = h.get("doc_id") or h.get("id") or ""
        out.append({
            "case_id": f"SG-LN-{doc_id}",
            "title": h.get("title") or h.get("case_title") or "Untitled",
            "citation": h.get("citation") or "",
            "court": h.get("court") or "Singapore Courts",
            "year": _extract_year(h.get("decision_date", "") or h.get("year", "")),
            "tier": h.get("court_tier") or "SC",
            "excerpt": (h.get("summary") or h.get("snippet") or "")[:600],
            "score": 1.0 - (i * 0.05),
            "jurisdiction": "SG",
            "source": "lawnet_sg",
            "url": h.get("url") or f"https://www.lawnet.sg/lawnet/web/lawnet/document/{doc_id}",
            "s3_key": None,
            "pdf_name": None,
        })
    return out


# ── Hong Kong — HKLII (free) ──────────────────────────────────────────
def hklii_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """
    HKLII free-text JSON endpoint. No auth required.
    Endpoint shape derived from HKLII's CanLII-style search.
    """
    enc = urllib.parse.quote_plus(query)
    url = f"https://www.hklii.hk/api/search?q={enc}&limit={k}"
    data = _http_json(url)
    if not isinstance(data, dict):
        return []
    hits = data.get("results") or data.get("documents") or []
    out: list[dict[str, Any]] = []
    for i, h in enumerate(hits[:k]):
        doc_id = h.get("doc_id") or h.get("id") or ""
        out.append({
            "case_id": f"HK-{doc_id}",
            "title": h.get("title") or "Untitled",
            "citation": h.get("citation") or "",
            "court": h.get("court") or "HK Courts",
            "year": _extract_year(h.get("date", "")),
            "tier": h.get("tier") or "SC",
            "excerpt": (h.get("snippet") or h.get("summary") or "")[:600],
            "score": 0.95 - (i * 0.05),
            "jurisdiction": "HK",
            "source": "hklii",
            "url": h.get("url") or f"https://www.hklii.hk/eng/hk/cases/{doc_id}.html",
            "s3_key": None,
            "pdf_name": None,
        })
    return out


# ── UAE — Dubai Pulse legislation API ─────────────────────────────────
def dubai_pulse_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Dubai Pulse legislation API. Key required."""
    key = _key("dubai_pulse")
    if not key:
        return []
    enc = urllib.parse.quote_plus(query)
    url = f"https://www.dubaipulse.gov.ae/api/legislation/search?q={enc}&limit={k}"
    data = _http_json(url, headers={"X-API-Key": key})
    if not isinstance(data, dict):
        return []
    hits = data.get("results") or data.get("legislations") or []
    out: list[dict[str, Any]] = []
    for i, h in enumerate(hits[:k]):
        lid = h.get("id") or h.get("legislation_id") or ""
        out.append({
            "case_id": f"AE-DP-{lid}",
            "title": h.get("title_en") or h.get("title") or "Untitled UAE legislation",
            "citation": h.get("citation") or h.get("number") or "",
            "court": "Government of Dubai",
            "year": _extract_year(h.get("issued_date", "")),
            "tier": "Statute",
            "excerpt": (h.get("summary_en") or h.get("summary") or "")[:600],
            "score": 0.9 - (i * 0.05),
            "jurisdiction": "AE",
            "source": "dubai_pulse",
            "url": h.get("url") or f"https://www.dubaipulse.gov.ae/legislation/{lid}",
            "s3_key": None,
            "pdf_name": None,
        })
    return out


# ── Korea — KLRI English statutes (free, rate-limited) ────────────────
def klri_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """
    Korea Legislation Research Institute English statute search.
    No auth — rate-limited (~1 rps).
    """
    enc = urllib.parse.quote_plus(query)
    url = f"https://elaw.klri.re.kr/api/v1/statutes/search?q={enc}&limit={k}"
    data = _http_json(url)
    if not isinstance(data, dict):
        return []
    hits = data.get("statutes") or data.get("results") or []
    out: list[dict[str, Any]] = []
    for i, h in enumerate(hits[:k]):
        sid = h.get("statute_id") or h.get("id") or ""
        out.append({
            "case_id": f"KR-{sid}",
            "title": h.get("title_en") or h.get("title") or "Korean statute",
            "citation": h.get("citation") or sid,
            "court": "Government of Korea (KLRI)",
            "year": _extract_year(h.get("promulgation_date", "")),
            "tier": "Statute",
            "excerpt": (h.get("summary_en") or h.get("summary") or "")[:600],
            "score": 0.85 - (i * 0.05),
            "jurisdiction": "KR",
            "source": "klri",
            "url": h.get("url") or f"https://elaw.klri.re.kr/eng_service/lawView.do?hseq={sid}",
            "s3_key": None,
            "pdf_name": None,
        })
    return out


# ── Malaysia — CLJ Law (paid) ─────────────────────────────────────────
def clj_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """CLJ Law (Malaysia) commercial API. Key required."""
    key = _key("clj")
    if not key:
        return []
    enc = urllib.parse.quote_plus(query)
    url = f"https://www.cljlaw.com/api/search?q={enc}&limit={k}"
    data = _http_json(url, headers={"Authorization": f"Bearer {key}"})
    if not isinstance(data, dict):
        return []
    hits = data.get("results") or data.get("cases") or []
    out: list[dict[str, Any]] = []
    for i, h in enumerate(hits[:k]):
        cid = h.get("case_id") or h.get("id") or ""
        out.append({
            "case_id": f"MY-CLJ-{cid}",
            "title": h.get("title") or "Untitled MY case",
            "citation": h.get("citation") or "",
            "court": h.get("court") or "Malaysia Courts",
            "year": _extract_year(h.get("decision_date", "")),
            "tier": h.get("tier") or "SC",
            "excerpt": (h.get("summary") or "")[:600],
            "score": 0.9 - (i * 0.05),
            "jurisdiction": "MY",
            "source": "clj",
            "url": h.get("url") or f"https://www.cljlaw.com/case/{cid}",
            "s3_key": None,
            "pdf_name": None,
        })
    return out


# ── Indonesia — JDIH BPHN (free scrape) ───────────────────────────────
def jdih_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """JDIH BPHN (Indonesia) public legal documentation. No auth."""
    enc = urllib.parse.quote_plus(query)
    url = f"https://jdihn.go.id/search/api?keyword={enc}&limit={k}"
    data = _http_json(url)
    if not isinstance(data, dict):
        return []
    hits = data.get("data") or data.get("results") or []
    out: list[dict[str, Any]] = []
    for i, h in enumerate(hits[:k]):
        did = h.get("id") or h.get("doc_id") or ""
        out.append({
            "case_id": f"ID-{did}",
            "title": h.get("judul") or h.get("title") or "Indonesian legal document",
            "citation": h.get("nomor") or "",
            "court": "Government of Indonesia (BPHN)",
            "year": _extract_year(h.get("tahun", "") or h.get("year", "")),
            "tier": "Statute",
            "excerpt": (h.get("abstrak") or h.get("summary") or "")[:600],
            "score": 0.85 - (i * 0.05),
            "jurisdiction": "ID",
            "source": "jdih",
            "url": h.get("url") or f"https://jdihn.go.id/legislation/view/{did}",
            "s3_key": None,
            "pdf_name": None,
        })
    return out


# ── Web search (Serper → Tavily → DuckDuckGo) ─────────────────────────
def web_search(query: str, k: int = 5, *, restrict_domain: str = "") -> list[dict[str, Any]]:
    """Tiered web search: paid → free-tier → grounded-LLM → free-scrape.

    Order:
      1. Serper.dev (Google) — best results, paid (2.5K free)
      2. Tavily — good free tier (1K/mo)
      3. Gemini grounded search — uses google_search built-in tool. Always
         works when GEMINI_API_KEY is set (which it is in our deploys),
         which is exactly the case where Serper/Tavily/DDG all fail today.
      4. DuckDuckGo HTML scrape — last resort. SSL fails on some hosts,
         which is why we keep Gemini-grounded ahead of it.
    """
    q = f"{query} site:{restrict_domain}".strip() if restrict_domain else query
    serper_key = _key("serper")
    if serper_key:
        hits = _serper_search(q, k, serper_key)
        if hits:
            return hits
    tavily_key = _key("tavily")
    if tavily_key:
        hits = _tavily_search(q, k, tavily_key)
        if hits:
            return hits
    gemini_hits = _gemini_grounded_search(q, k)
    if gemini_hits:
        return gemini_hits
    return _duckduckgo_search(q, k)


def web_search_snippets(query: str, k: int = 6) -> list[dict[str, Any]]:
    """
    Distinct snippet-shaped output for /api/brief/web. Each item is
    {title, url, source, snippet} — exactly what brief_service.answer_web
    expects.
    """
    raw = web_search(query, k=k)
    out: list[dict[str, Any]] = []
    for r in raw:
        out.append({
            "title": r.get("title") or "Untitled",
            "url": r.get("url") or r.get("citation") or "",
            "source": r.get("source") or "web",
            "snippet": r.get("excerpt") or "",
            "score": r.get("score") or 0.0,
        })
    return out


def _serper_search(query: str, k: int, key: str) -> list[dict[str, Any]]:
    body = json.dumps({"q": query, "num": k}).encode("utf-8")
    data = _http_json(
        "https://google.serper.dev/search",
        method="POST",
        body=body,
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
    )
    if not isinstance(data, dict):
        return []
    results = data.get("organic", [])[:k]
    out = []
    for i, r in enumerate(results):
        out.append({
            "case_id": f"WEB-{i+1}",
            "title": r.get("title", "")[:200],
            "citation": r.get("link", ""),
            "court": "Web result",
            "year": "",
            "tier": "Web",
            "excerpt": (r.get("snippet") or "")[:600],
            "score": 0.7 - (i * 0.05),
            "jurisdiction": "",
            "source": "serper",
            "url": r.get("link", ""),
            "s3_key": None,
            "pdf_name": None,
        })
    return out


def _tavily_search(query: str, k: int, key: str) -> list[dict[str, Any]]:
    """Tavily API. Free tier: 1000 searches/mo. Auth in body, not header."""
    body = json.dumps({
        "api_key": key,
        "query": query,
        "max_results": k,
        "search_depth": "basic",
        "include_answer": False,
    }).encode("utf-8")
    data = _http_json(
        "https://api.tavily.com/search",
        method="POST",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    if not isinstance(data, dict):
        return []
    results = data.get("results") or []
    out = []
    for i, r in enumerate(results[:k]):
        out.append({
            "case_id": f"WEB-{i+1}",
            "title": (r.get("title") or "")[:200],
            "citation": r.get("url") or "",
            "court": "Web result",
            "year": "",
            "tier": "Web",
            "excerpt": (r.get("content") or r.get("snippet") or "")[:600],
            "score": float(r.get("score") or (0.65 - i * 0.05)),
            "jurisdiction": "",
            "source": "tavily",
            "url": r.get("url") or "",
            "s3_key": None,
            "pdf_name": None,
        })
    return out


def _gemini_grounded_search(query: str, k: int) -> list[dict[str, Any]]:
    """Gemini grounded search via the built-in `google_search` tool.

    Uses the existing GEMINI_API_KEY (no extra paid keys required) — this
    is the always-on fallback that turns "no Serper/Tavily key, broken DDG
    SSL" from a refusal into a working answer.

    The Gemini API runs Google search server-side, then returns citations
    via `groundingMetadata.groundingChunks[].web.{uri,title}` plus
    `groundingSupports[]` linking text spans to chunk indices. We unpack
    that into our standard hit shape so brief_service.answer_web sees
    exactly what it expects from Serper/Tavily.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return []
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    body = json.dumps({
        "contents": [{
            "role": "user",
            "parts": [{
                "text": (
                    f"Search the web for: {query}\n\n"
                    f"Return a concise factual summary in 4-6 sentences. "
                    f"Cite the most relevant {k} sources."
                ),
            }],
        }],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 700},
    }).encode("utf-8")
    data = _http_json(
        url,
        method="POST",
        body=body,
        headers={"Content-Type": "application/json"},
        timeout=15.0,
    )
    if not isinstance(data, dict):
        return []
    candidates = data.get("candidates") or []
    if not candidates:
        return []
    cand0 = candidates[0]
    parts = (cand0.get("content") or {}).get("parts") or []
    summary = "".join(p.get("text", "") for p in parts).strip()
    grounding = cand0.get("groundingMetadata") or {}
    chunks = grounding.get("groundingChunks") or []
    supports = grounding.get("groundingSupports") or []

    # Collect text segments per chunk index so each citation gets a
    # specific snippet (not just the global summary).
    snippet_for: dict[int, list[str]] = {}
    for sup in supports:
        seg = (sup.get("segment") or {}).get("text") or ""
        if not seg:
            continue
        for idx in sup.get("groundingChunkIndices") or []:
            try:
                snippet_for.setdefault(int(idx), []).append(seg)
            except (TypeError, ValueError):
                continue

    out: list[dict[str, Any]] = []
    for i, ch in enumerate(chunks[:k]):
        web = ch.get("web") or {}
        uri = (web.get("uri") or "").strip()
        title = (web.get("title") or "").strip() or "Web result"
        if not uri:
            continue
        # Per-chunk snippet (joined text spans). Falls back to the model
        # summary if the API didn't link any segments to this chunk.
        per = " ".join(snippet_for.get(i, [])).strip()
        excerpt = (per or summary)[:600]
        out.append({
            "case_id": f"WEB-{i+1}",
            "title": title[:200],
            "citation": uri,
            "court": "Web result",
            "year": "",
            "tier": "Web",
            "excerpt": excerpt,
            "score": 0.7 - (i * 0.05),
            "jurisdiction": "",
            "source": "gemini_search",
            "url": uri,
            "s3_key": None,
            "pdf_name": None,
        })
    return out


_DDG_LINK_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.+?)</a>.*?'
    r'<a[^>]+class="result__snippet"[^>]*>(.+?)</a>',
    re.DOTALL,
)


def _duckduckgo_search(query: str, k: int) -> list[dict[str, Any]]:
    """Scrape DDG's HTML no-JS endpoint. Free forever, rate-limit yourself."""
    enc = urllib.parse.quote_plus(query)
    html = _http_text(f"https://html.duckduckgo.com/html/?q={enc}")
    if not html:
        return []
    out: list[dict[str, Any]] = []
    for i, m in enumerate(_DDG_LINK_RE.finditer(html)):
        if i >= k:
            break
        raw_url = urllib.parse.unquote(m.group(1))
        # DDG redirect: /l/?uddg=<real-url>
        if "uddg=" in raw_url:
            try:
                raw_url = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query).get("uddg", [raw_url])[0]
            except Exception:
                pass
        out.append({
            "case_id": f"WEB-{i+1}",
            "title": _strip_html(m.group(2))[:200],
            "citation": raw_url,
            "court": "Web result",
            "year": "",
            "tier": "Web",
            "excerpt": _strip_html(m.group(3))[:600],
            "score": 0.6 - (i * 0.05),
            "jurisdiction": "",
            "source": "duckduckgo",
            "url": raw_url,
            "s3_key": None,
            "pdf_name": None,
        })
    return out


# ── helpers ───────────────────────────────────────────────────────────
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    if not s:
        return ""
    return _HTML_TAG_RE.sub(" ", s).replace("&nbsp;", " ").replace("&amp;", "&").strip()


def _extract_year(s: str) -> str:
    m = re.search(r"\b(19|20)\d{2}\b", s or "")
    return m.group(0) if m else ""


# ── Orchestrator ──────────────────────────────────────────────────────
# Per-juris domain hints for restricted web search. Sanhita's three
# supported jurisdictions all have first-party APIs or BM25 corpora —
# the web hint is a defensive fallback for queries the index can't
# answer (very recent rulings not yet ingested, news commentary, etc.).
_DOMAIN_HINTS: dict[str, str] = {
    "IN": "indiankanoon.org",
    "SG": "elitigation.sg",
    "HK": "hklii.hk",
}


def bm25_search(query: str, k: int = 6, *, jurisdiction: str | None = None) -> list[dict[str, Any]]:
    """Query the live `BM25Index` populated by `scripts/ingest_github_data.py`.

    Returns hits in the standard shape so the caller treats them like
    any other connector. Silently returns [] if the index isn't loaded
    (during cold start, or in deployments without rank_bm25). The
    `_ensure_bm25` helper in `server.py` is the canonical loader; we
    import it lazily to avoid a circular import."""
    try:
        import server  # type: ignore
        idx = server._ensure_bm25()  # noqa: SLF001
    except Exception as e:  # noqa: BLE001
        logger.debug("bm25_search: server._ensure_bm25 unavailable: %s", e)
        return []
    if idx is None:
        return []
    try:
        from retrieval_pkg import doc_to_retrieve_hit
    except Exception as e:  # noqa: BLE001
        logger.debug("bm25_search: retrieval_pkg unavailable: %s", e)
        return []
    try:
        results = idx.query(query, k=k, jurisdiction=jurisdiction)
    except Exception as e:  # noqa: BLE001
        logger.warning("bm25_search: idx.query failed: %s", e)
        return []
    return [doc_to_retrieve_hit(d, s, query) for d, s in results]


def retrieve_hybrid(
    query: str,
    *,
    jurisdiction: str | None = None,
    sources: list[str] | None = None,
    k: int = 6,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval — fans out to enabled connectors, merges, dedupes,
    returns top-k by score.

    `sources` — allowlist subset of:
       indian_kanoon, ecourts, egov_japan, lawnet_sg, hklii, dubai_pulse,
       klri, clj, jdih, web, seed.
       None → default stack for the jurisdiction.
    """
    if sources is None:
        sources = _default_sources_for(jurisdiction)

    merged: list[dict[str, Any]] = []

    # bm25 first — the live GitHub-ingested case-law corpus. When the
    # index is warm and contains relevant matches this is by far the
    # cheapest hop (no network).
    if "bm25" in sources:
        merged.extend(bm25_search(query, k=k, jurisdiction=jurisdiction))

    if "indian_kanoon" in sources:
        merged.extend(indian_kanoon_search(query, k=k))
    if "ecourts" in sources:
        merged.extend(ecourts_search(query, k=k))
    if "egov_japan" in sources:
        merged.extend(egov_japan_search(query, k=3))
    if "lawnet_sg" in sources:
        merged.extend(lawnet_sg_search(query, k=k))
    if "hklii" in sources:
        merged.extend(hklii_search(query, k=k))
    if "dubai_pulse" in sources:
        merged.extend(dubai_pulse_search(query, k=k))
    if "klri" in sources:
        merged.extend(klri_search(query, k=k))
    if "clj" in sources:
        merged.extend(clj_search(query, k=k))
    if "jdih" in sources:
        merged.extend(jdih_search(query, k=k))
    if "web" in sources:
        domain = _DOMAIN_HINTS.get(jurisdiction or "", "")
        merged.extend(web_search(query, k=4, restrict_domain=domain))
    if "seed" in sources or not merged:  # always fall back to seed if nothing else
        try:
            import seed_corpus
            merged.extend(seed_corpus.query(query, k=k, jurisdiction=jurisdiction))
        except Exception as e:  # noqa: BLE001
            logger.warning("seed_corpus fallback failed: %s", e)

    # dedupe by case_id; keep the highest-scored copy
    by_id: dict[str, dict[str, Any]] = {}
    for h in merged:
        cid = h.get("case_id") or h.get("title", "")[:40]
        if cid not in by_id or h.get("score", 0) > by_id[cid].get("score", 0):
            by_id[cid] = h

    out = sorted(by_id.values(), key=lambda h: h.get("score", 0), reverse=True)
    return out[:k]


def _default_sources_for(jurisdiction: str | None) -> list[str]:
    """Per-jurisdiction connector stack. Order matters — first hit wins.

    Sanhita serves three jurisdictions: India, Singapore, Hong Kong. Each
    one starts with `bm25` (the live GitHub-ingested case-law index),
    falls through to remote APIs / web search, and finally to the seed
    corpus as a deterministic safety net.
    """
    j = (jurisdiction or "").upper()
    table: dict[str, list[str]] = {
        "IN": ["bm25", "indian_kanoon", "web", "seed"],
        "SG": ["bm25", "web", "seed"],
        "HK": ["bm25", "web", "seed"],
    }
    if j in table:
        return table[j]
    # No / unknown jurisdiction: try the index across all juris, then web.
    return ["bm25", "web", "seed"]


def available_connectors() -> dict[str, bool]:
    """Connector availability for IN / SG / HK. The dropped Asian-market
    connectors (egov_japan, lawnet_sg, dubai_pulse, klri, clj, jdih)
    remain in this file as dormant code — they're not surfaced anywhere
    after the 3-jurisdiction trim."""
    return {
        "bm25": True,                                  # live GitHub corpus
        "indian_kanoon": bool(_key("indian_kanoon")),
        "ecourts": bool(_key("ecourts")),
        "web_serper": bool(_key("serper")),
        "web_tavily": bool(_key("tavily")),
        "web_duckduckgo": True,                        # always available
        "seed_corpus": True,
    }
