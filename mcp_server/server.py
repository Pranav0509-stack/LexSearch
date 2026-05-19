"""Sanhita MCP server.

Exposes Sanhita's Indian-law capabilities to any MCP client (Claude
Desktop, Claude Code, ChatGPT, etc.) as 6 tools:

    search_caselaw        Hybrid BM25 + semantic over 70M Indian court records
    get_document          Full text + metadata for any case_id, statute, or QA
    list_templates        Browse Sanhita's 26 templates (5 verbatim Govt forms)
    draft_document        Generate from template + slots (deterministic)
    lookup_statute        Pull verbatim statute + top interpreting cases
    compliance_check      Run 8 plug-ins (DPDP / RBI / SEBI / IBC / GST / POSH / Stamp / IT Act)

Runs as a STDIO server (the standard MCP transport that Claude Desktop and
Claude Code spawn locally) — no network ports to expose, no auth surface.
The HTTP backend at localhost:8080 is consulted for actual work.

Install:
    pip install mcp httpx

Wire to Claude Desktop:
    Add to ~/Library/Application Support/Claude/claude_desktop_config.json:
    {
      "mcpServers": {
        "sanhita": {
          "command": "python3",
          "args": ["/Users/pranav/Desktop/LexSearch-main 2/mcp_server/server.py"],
          "env": {"SANHITA_BACKEND": "http://localhost:8080"}
        }
      }
    }

Wire to Claude Code:
    claude mcp add sanhita "python3 /Users/pranav/Desktop/LexSearch-main 2/mcp_server/server.py"
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import httpx

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("FATAL: `pip install mcp` to run the Sanhita MCP server.", file=sys.stderr)
    sys.exit(2)


# ── Config ──────────────────────────────────────────────────────────────

BACKEND = os.environ.get("SANHITA_BACKEND", "http://localhost:8080").rstrip("/")
TIMEOUT = 30.0


# ── Helpers ─────────────────────────────────────────────────────────────

async def _post(path: str, body: dict) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as cli:
        r = await cli.post(f"{BACKEND}{path}", json=body)
        r.raise_for_status()
        return r.json()


async def _get(path: str, params: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as cli:
        r = await cli.get(f"{BACKEND}{path}", params=params or {})
        r.raise_for_status()
        return r.json()


def _fmt(obj: Any, ttl: str | None = None) -> str:
    """Pretty-print a result for the MCP text response."""
    out = []
    if ttl:
        out.append(f"# {ttl}\n")
    out.append(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    return "\n".join(out)


# ── Tool implementations ───────────────────────────────────────────────

async def search_caselaw(args: dict) -> str:
    q = args["query"]
    mode = args.get("mode", "hybrid")
    limit = max(1, min(int(args.get("limit", 10)), 50))
    body = {"q": q, "mode": mode, "limit": limit}
    for k in ("court", "state", "year_from", "year_to", "verdict",
              "judge", "act", "doc_type", "source_table", "sort"):
        if args.get(k) is not None:
            body[k] = args[k]
    try:
        res = await _post("/api/cases/smart-search", body)
    except Exception as e:
        return f"ERROR: search failed ({e}). Is Sanhita backend running at {BACKEND}?"
    hits = res.get("hits", [])
    if not hits:
        return f"No results for: {q!r} (mode={mode}). Try broader terms."
    lines = [f"# {len(hits)} hits for: {q!r} (mode={mode}, engine={res.get('engine','-')})"]
    for i, h in enumerate(hits, 1):
        court = h.get("court") or "—"
        year = h.get("year") or ""
        title = (h.get("title") or "").strip()
        snip = (h.get("snippet") or h.get("summary") or "")[:200].strip()
        sid = h.get("doc_id") or h.get("case_id") or ""
        st = h.get("source_table") or "?"
        lines.append(f"\n## {i}. {title[:160]}")
        lines.append(f"   • id: `{sid}`   source: {st}   court: {court}   year: {year}")
        if snip:
            lines.append(f"   • snippet: {snip}…")
    lines.append(f"\nUse `get_document` with the `id` to read the full text.")
    return "\n".join(lines)


async def get_document(args: dict) -> str:
    case_id = args["id"]
    try:
        d = await _get(f"/api/cases/document/{case_id}")
    except Exception as e:
        return f"ERROR: {e}. Try `search_caselaw` first to find a valid id."
    out: list[str] = []
    out.append(f"# {d.get('title','(untitled)')}")
    meta = []
    if d.get("court"):         meta.append(f"**Court:** {d['court']}")
    if d.get("year"):          meta.append(f"**Year:** {d['year']}")
    if d.get("judgment_date"): meta.append(f"**Date:** {d['judgment_date']}")
    if d.get("citation"):      meta.append(f"**Citation:** `{d['citation']}`")
    if d.get("outcome"):       meta.append(f"**Outcome:** {d['outcome']}")
    if meta: out.append(" · ".join(meta))
    parties = d.get("parties") or {}
    if parties.get("petitioner") or parties.get("respondent"):
        out.append(f"**Parties:** {parties.get('petitioner','?')} **vs** {parties.get('respondent','?')}")
    judges = d.get("judges") or []
    if judges: out.append(f"**Bench:** {', '.join(filter(None, judges))}")
    acts = d.get("acts_cited") or []
    if acts:   out.append(f"**Acts cited:** {', '.join(map(str, acts[:8]))}")
    if d.get("has_pdf") and d.get("pdf_url"):
        out.append(f"**PDF:** {d['pdf_url']}")
    body = d.get("full_text") or d.get("summary") or ""
    if body:
        out.append("\n## Body\n")
        out.append(body[:12000])
        if len(body) > 12000:
            out.append(f"\n… ({len(body)-12000:,} more characters available; the lawyer should open in Sanhita)")
    return "\n".join(out)


async def list_templates(args: dict) -> str:
    domain = args.get("domain")
    params = {"domain": domain} if domain else None
    try:
        ts = await _get("/api/contract/templates", params)
    except Exception as e:
        return f"ERROR: {e}"
    if isinstance(ts, dict):
        ts = ts.get("templates", [])
    lines = [f"# Sanhita templates ({len(ts)})  "
             f"— 4 tiers: Verbatim Govt Form / Statutory / Court-aligned / Practice-standard"]
    from collections import defaultdict
    by_dom: dict[str, list[dict]] = defaultdict(list)
    for t in ts:
        by_dom[t.get("domain","?")].append(t)
    for dom in sorted(by_dom):
        lines.append(f"\n## {dom}")
        for t in by_dom[dom]:
            lines.append(f"   • `{t['id']}` — {t.get('title','?')}")
    lines.append("\nUse `draft_document` with `template_id` to generate; required slots are listed in the template detail.")
    return "\n".join(lines)


async def draft_document(args: dict) -> str:
    body = {
        "template_id": args["template_id"],
        "slots": args.get("slots", {}),
        "mode": args.get("mode", "deterministic_only"),
    }
    try:
        d = await _post("/api/contract/draft", body)
    except Exception as e:
        return f"ERROR: {e}"
    out = [
        f"# {d.get('template_id','?')} — draft `{d.get('draft_id')}`",
        f"**Words:** {d.get('word_count', 0):,}  **Risk:** {d.get('risk_score', '-')}",
        "",
        d.get("body_md", "(empty)")[:15000],
    ]
    body_md = d.get("body_md", "")
    if len(body_md) > 15000:
        out.append(f"\n… (truncated; full {len(body_md):,} chars available via `get_draft`)")
    return "\n".join(out)


async def lookup_statute(args: dict) -> str:
    anchors = [{"act": args["act"], "section": args.get("section", "")}]
    body = {"anchors": anchors, "topn": int(args.get("topn", 5))}
    try:
        d = await _post("/api/contract/citations", body)
    except Exception as e:
        return f"ERROR: {e}"
    enriched = (d.get("anchors") or [])
    if not enriched:
        return f"No cases found for {args['act']} {args.get('section','')}"
    a = enriched[0]
    out = [
        f"# {a['act']} — {a.get('section') or '(no section)'}",
        f"Found **{a['count']}** interpreting cases:\n",
    ]
    for i, c in enumerate(a.get("cases", []), 1):
        out.append(f"### {i}. {(c.get('title') or '').strip()[:160]}")
        out.append(f"   • {c.get('court')}  {c.get('year') or ''}  {c.get('citation') or ''}")
        if c.get("snippet"):
            out.append(f"   • {c['snippet'][:200]}…")
        out.append("")
    return "\n".join(out)


async def compliance_check(args: dict) -> str:
    body = {
        "body_md": args["text"],
        "doc_type": args.get("doc_type", ""),
    }
    try:
        d = await _post("/api/contract/compliance", body)
    except Exception as e:
        return f"ERROR: {e}"
    findings = d.get("findings", [])
    if not findings:
        return "✅ Compliance: no plug-in fired any warning."
    out = [f"# Compliance check — {d.get('count', 0)} findings"]
    by_sev = {"block": [], "warn": [], "info": []}
    for f in findings:
        by_sev.get(f.get("severity","info"), by_sev["info"]).append(f)
    for sev in ("block", "warn", "info"):
        if by_sev[sev]:
            out.append(f"\n## {sev.upper()}  ({len(by_sev[sev])})")
            for f in by_sev[sev]:
                out.append(f"- **{f.get('plugin','?')}** · `{f.get('rule_id','-')}`")
                out.append(f"  {f.get('finding','')}")
                if f.get("remediation"):
                    out.append(f"  → *{f['remediation']}*")
    return "\n".join(out)


# ── MCP wiring ─────────────────────────────────────────────────────────

server = Server("sanhita")

TOOLS: list[Tool] = [
    Tool(
        name="search_caselaw",
        description=(
            "Search Sanhita's corpus of 70M+ Indian court records (16.9M HC + 53M district + 86K SC + tribunals + 11.6M legal docs + 2K statutes + 1.3M Q&A). "
            "Returns a ranked list of cases / documents with court, year, snippet, and an `id` you can pass to get_document. "
            "Default mode is 'hybrid' (BM25 + semantic via RRF) — use 'keyword' for exact-phrase, 'semantic' for intent-aware queries."
        ),
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query":   {"type": "string", "description": "Free-text search. Examples: 'section 138 cheque dishonour', 'anticipatory bail elderly accused', 'Bhajan Lal quashing'"},
                "mode":    {"type": "string", "enum": ["keyword", "semantic", "hybrid"], "default": "hybrid"},
                "year_from": {"type": "integer"},
                "year_to":   {"type": "integer"},
                "verdict":   {"type": "string", "description": "e.g. 'Allowed', 'Dismissed'"},
                "court":     {"type": "string", "description": "e.g. 'Supreme Court', 'Bombay High Court'"},
                "act":       {"type": "string", "description": "Act name to filter, e.g. 'Indian Contract Act'"},
                "source_table": {"type": "string", "enum": ["judgments","legal_docs","pipeline_docs","statutes","legal_qa","documents"]},
                "sort":      {"type": "string", "enum": ["relevance","latest","oldest","most_cited"], "default": "relevance"},
                "limit":     {"type": "integer", "minimum": 1, "maximum": 50, "default": 10}
            }
        },
    ),
    Tool(
        name="get_document",
        description=(
            "Fetch full text + structured metadata (court, parties, judges, date, outcome, acts cited, PDF URL) for any case_id, statute, or legal-doc id returned by search_caselaw. "
            "Supports prefixed ids (doc_*, statute_*, qa_*) and bare CNRs / pipeline doc_ids."
        ),
        inputSchema={
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "The `id` from a search_caselaw hit"}
            }
        },
    ),
    Tool(
        name="list_templates",
        description=(
            "Browse Sanhita's 26 Indian-law document templates across 9 practice areas. Includes 5 VERBATIM Government-prescribed forms (CPC Schedule I App.A Form 1, Form 4 Order 37 Summary Suit, RTI Form A, NCLT IBC §7 Form 1, ISA §276 Probate) plus statutory + court-aligned + practice-standard tiers. "
            "Each template id can be passed to draft_document."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Filter to one domain: litigation_filings, statutory_forms, corporate, disputes, real_estate, banking, corporate_tx, employment, family_trust"}
            }
        },
    ),
    Tool(
        name="draft_document",
        description=(
            "Generate a complete Indian legal document from a Sanhita template. Returns the full Markdown body, ready for filing or signature. "
            "Slots are template-specific (call list_templates and inspect the slot schema). Mode 'deterministic_only' fills slots verbatim with no AI; 'one_pass' / 'multi_pass' add LLM polish on top — preserves all statute citations exactly."
        ),
        inputSchema={
            "type": "object",
            "required": ["template_id", "slots"],
            "properties": {
                "template_id": {"type": "string"},
                "slots":       {"type": "object", "description": "Field values keyed by slot name"},
                "mode":        {"type": "string", "enum": ["deterministic_only","one_pass","multi_pass"], "default": "deterministic_only"}
            }
        },
    ),
    Tool(
        name="lookup_statute",
        description=(
            "Return the top N Supreme Court / High Court cases interpreting a given Indian Act + Section. Uses Sanhita's hybrid FTS5 retrieval over 11.6M legal_docs + judgments. Each case returns a holding-window snippet anchored to the section reference."
        ),
        inputSchema={
            "type": "object",
            "required": ["act"],
            "properties": {
                "act":     {"type": "string", "description": "e.g. 'Indian Contract Act, 1872'"},
                "section": {"type": "string", "description": "e.g. 'Section 27' or '§27'"},
                "topn":    {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}
            }
        },
    ),
    Tool(
        name="compliance_check",
        description=(
            "Run 8 Indian-law compliance plug-ins against a draft body: DPDP (Digital Personal Data Protection Act 2023), RBI/FEMA (cross-border financial), SEBI (insider trading / PIT Regs), IBC (§14 moratorium), GST, IT Act §43A (sensitive personal data), POSH (Sexual Harassment Act 2013), Stamp Duty + Registration Act §17. Returns block/warn/info severity findings with rule_id and remediation."
        ),
        inputSchema={
            "type": "object",
            "required": ["text"],
            "properties": {
                "text":     {"type": "string", "description": "The contract / pleading body to audit"},
                "doc_type": {"type": "string", "description": "Optional — e.g. 'employment_agreement' enables POSH-specific checks"}
            }
        },
    ),
]

HANDLERS = {
    "search_caselaw":   search_caselaw,
    "get_document":     get_document,
    "list_templates":   list_templates,
    "draft_document":   draft_document,
    "lookup_statute":   lookup_statute,
    "compliance_check": compliance_check,
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, args: dict) -> list[TextContent]:
    fn = HANDLERS.get(name)
    if not fn:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    try:
        out = await fn(args)
    except Exception as e:
        out = f"Tool error: {e}"
    return [TextContent(type="text", text=out)]


async def main() -> None:
    async with stdio_server() as (read, write):
        init_opts = server.create_initialization_options()
        await server.run(read, write, init_opts)


if __name__ == "__main__":
    asyncio.run(main())
