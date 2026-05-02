"""
Sanhita Legal Agent — Gemini-powered tool-calling orchestrator.

This is the Harvey-style "do tasks for the lawyer" layer. One user request
("Find leading SC cases on cheque dishonour, then draft a Section 138
demand notice using the holdings as legal basis") becomes a multi-step
agent run: Gemini decides which tools to call, in what order, until it has
enough context to compose the final answer.

Tool catalog (kept tight on purpose — fewer tools = better tool-pick
behaviour):

  retrieve_cases(query, jurisdiction?, k?)        — case-law search (BM25 + connectors)
  retrieve_statutes(query, jurisdiction?, kind?)  — Library DB lookup
  web_search(query, k?, restrict_domain?)         — Serper → Tavily → DDG
  redline_contract(text)                          — runs structured redline workflow
  translate(text, direction)                      — en↔hi translation

The loop is hard-capped at 6 tool turns — anything that needs more steps
should ask the user a clarifying question instead of churning.

Usage:
    out = legal_agent.run(question, history, jurisdiction="IN")
    out["answer_markdown"]   # final answer
    out["citations"]         # accumulated cites from tool calls
    out["trace"]             # [{tool, args, result_summary, ms}] for the UI
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from llm import router

logger = logging.getLogger(__name__)

MAX_TURNS = 6
PER_TOOL_MAX_RESULTS = 6


# ─────────────────────────────────────────────────────────────────────────
# AGENT SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are Sanhita, a senior Asian-jurisdiction legal
associate. You have these tools:

Legal research / drafting:
  • retrieve_cases     — search case law (Indian Kanoon / BM25 / connectors)
  • retrieve_statutes  — pull from the curated Library (BNS, BNSS, NI Act, SG/HK/JP/UAE statutes, contract templates)
  • web_search         — open-web search via Serper / Tavily
  • redline_contract   — produces a JSON list of remove/replace/add suggestions
  • translate          — formal en↔hi translation

Google Workspace (only if user has connected — tool will say so if not):
  • create_google_doc  — save a draft contract / notice / memo to user's Drive
  • create_gmail_draft — compose a Gmail DRAFT (not sent — user clicks Send)
  • append_sheet_row   — log a matter into the user's Sanhita Matter Tracker
  • search_drive       — find prior Sanhita-created Docs in user's Drive

How to act:

1. Decompose the user's request into the smallest set of tool calls that
   will give you enough grounding. Don't over-call — one good retrieve is
   better than three vague ones.
2. After each tool result, decide whether you have enough context. If yes,
   compose the final answer. If no, call another tool.
3. NEVER fabricate citations. Cite only sources returned by tools, using
   [n] brackets that map to the citation list you accumulate.
4. For drafting tasks (letters, notices, contracts, petitions) you may
   compose without retrieval — but if the user references "this case" or
   "the holdings above," ground the draft in the cases your tools returned.
5. When you have everything, return a clean markdown answer with [n]
   citations only where they support specific factual or legal claims.
   Drafted documents (letters, contracts, etc.) don't need [n] inside the
   document body — those are for legal-research answers.
6. If the question is out of scope (criminal advice for a specific person,
   medical, etc.), say so plainly. Don't invent.

Google tool etiquette:
  • Use create_google_doc when the user says "save to Docs", "put it in my
    Drive", "make a Doc", or asks for a long drafted artifact they'll edit.
    Always FIRST compose the doc content in the answer, THEN call the tool
    with the same content. After the tool returns, mention the URL in the
    answer.
  • Use create_gmail_draft when the user says "email it to X", "send to X",
    "draft an email to X". You only create a DRAFT — the user must click
    Send themselves. Make this clear in the answer.
  • Use append_sheet_row at the end of a meaningful workflow (drafted a
    notice, finished research) so the user has a log. Skip for chitchat.
  • Use search_drive when the user references a prior matter ("the NDA I
    drafted last week", "find the Acme contract"). Drive scope is per-file
    so only Sanhita-created Docs are visible — say so if results are empty.
  • If a Google tool returns {"error": "Google not connected..."}, finish
    the legal work without it and tell the user to connect Google in
    Settings to enable the integration. Don't retry.

Voice: senior Asian commercial / litigation associate. Plain English,
specific section numbers, concrete next steps. No filler, no disclaimers
unless materially relevant.
"""


# ─────────────────────────────────────────────────────────────────────────
# TOOL DECLARATIONS — Gemini's functionDeclarations spec
# ─────────────────────────────────────────────────────────────────────────

TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "functionDeclarations": [
            {
                "name": "retrieve_cases",
                "description": (
                    "Search Asian case law for judgments matching a legal query. "
                    "Returns up to k cases with title, court, year, citation, and excerpt. "
                    "Use this for any factual or legal claim that should be supported by "
                    "decided authority."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language legal query (e.g. 'cheque dishonour Section 138 NI Act standard of proof').",
                        },
                        "jurisdiction": {
                            "type": "string",
                            "description": "ISO-style country code: IN, JP, SG, HK, MY, ID, KR, AE, * (pan-Asia). Default IN.",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results, 1-6. Default 4.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "retrieve_statutes",
                "description": (
                    "Look up curated statutes, contract templates, and pleading skeletons "
                    "from the Library DB. Use this when the user mentions a specific section "
                    "(BNS §103, NI Act §138, Companies Act §157A) or wants a standard contract template."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Section number or template name to search for.",
                        },
                        "jurisdiction": {
                            "type": "string",
                            "description": "IN, JP, SG, HK, AE, * (pan-Asia).",
                        },
                        "kind": {
                            "type": "string",
                            "description": "statute | contract | pleading. Optional.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "web_search",
                "description": (
                    "Open-web search via Serper → Tavily → DuckDuckGo. Use ONLY when the "
                    "question is about current events, recent rulings, or anything past the "
                    "case-law index cutoff. Never use this for established law."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "description": "Max results, 1-8. Default 5."},
                        "restrict_domain": {
                            "type": "string",
                            "description": "Optional domain to restrict to (e.g. 'sci.gov.in', 'judiciary.gov.sg').",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "redline_contract",
                "description": (
                    "Run the structured redline workflow. Returns a JSON list of "
                    "remove/replace/add suggestions tailored for Asian-jurisdiction commercial law."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The full contract text (50-40000 chars).",
                        },
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "translate",
                "description": (
                    "Formal English↔Hindi translation. Use this only when the user "
                    "explicitly asks for a translation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "direction": {
                            "type": "string",
                            "description": "'en->hi' or 'hi->en'.",
                        },
                    },
                    "required": ["text", "direction"],
                },
            },
            # ── Google Workspace tools (require user to have connected) ──
            {
                "name": "create_google_doc",
                "description": (
                    "Save a draft into the user's Google Drive as a Google Doc. "
                    "Use this AFTER you've composed a contract, letter, notice, or memo "
                    "when the user asks for it in Docs/Drive/'save as a doc'/'put it in "
                    "google docs'. Returns a docs.google.com URL the user can open and "
                    "edit. Requires Google OAuth — fails clearly if not connected."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Doc title — short, e.g. 'NDA — Acme × Beta — 2026-04-25'.",
                        },
                        "content_md": {
                            "type": "string",
                            "description": "Full body of the document (markdown / plain text).",
                        },
                    },
                    "required": ["title", "content_md"],
                },
            },
            {
                "name": "create_gmail_draft",
                "description": (
                    "Create a Gmail DRAFT (NOT sent) in the user's mailbox. The user "
                    "must click Send themselves — this is intentional safety behavior "
                    "for legal correspondence. Use when the user asks to 'email this', "
                    "'send to opposing counsel', etc. Requires Google OAuth."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient email address."},
                        "subject": {"type": "string"},
                        "body_md": {"type": "string", "description": "Email body."},
                    },
                    "required": ["to", "subject", "body_md"],
                },
            },
            {
                "name": "append_sheet_row",
                "description": (
                    "Log a matter into the user's Sanhita Matter Tracker (auto-created "
                    "Google Sheet on first use). Columns: Date, Matter, Jurisdiction, "
                    "Court, Stage, Action, Sanhita Doc URL, Notes. Use this whenever "
                    "the user finishes a workflow they want to track."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "matter": {"type": "string", "description": "Matter name / case caption."},
                        "jurisdiction": {"type": "string"},
                        "court": {"type": "string"},
                        "stage": {"type": "string", "description": "e.g. 'Drafted notice', 'Hearing 25-Apr-2026'."},
                        "action": {"type": "string", "description": "What you just did, one short phrase."},
                        "doc_url": {"type": "string", "description": "Optional Sanhita-created Doc URL."},
                        "notes": {"type": "string"},
                    },
                    "required": ["matter", "action"],
                },
            },
            {
                "name": "search_drive",
                "description": (
                    "Search the user's Google Drive for prior Sanhita-created documents "
                    "(NDAs, notices, prior matters). Only files Sanhita has touched are "
                    "visible (drive.file scope). Use when the user says 'find the NDA "
                    "I drafted last week' or similar."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "description": "Max results, 1-20. Default 8."},
                    },
                    "required": ["query"],
                },
            },
        ]
    }
]


# ─────────────────────────────────────────────────────────────────────────
# TOOL DISPATCHERS — the Python implementation each declared tool maps to
# ─────────────────────────────────────────────────────────────────────────

def _tool_retrieve_cases(args: dict[str, Any]) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    jurisdiction = (args.get("jurisdiction") or "IN").strip().upper()
    k = max(1, min(int(args.get("k") or 4), PER_TOOL_MAX_RESULTS))
    if not query:
        return {"error": "query is required", "hits": []}

    # Try BM25 first — it's the LexSearch-built 16M-doc index. Falls through
    # to connectors (Indian Kanoon / eCourts / e-Gov / web) and finally
    # seed_corpus so the agent never gets nothing back.
    hits: list[dict[str, Any]] = []
    try:
        import server  # _ensure_bm25 lives there
        idx = server._ensure_bm25()  # noqa: SLF001
        if idx is not None:
            results = idx.query(query, k=k, tier=None)
            from server import doc_to_retrieve_hit
            hits = [doc_to_retrieve_hit(d, s, query) for d, s in results]
    except Exception as e:  # noqa: BLE001
        logger.debug("BM25 path failed in agent retrieve_cases: %s", e)

    if not hits:
        try:
            import connectors
            hits = connectors.retrieve_hybrid(
                query, jurisdiction=jurisdiction or None, k=k
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("connectors.retrieve_hybrid failed in agent: %s", e)

    if not hits:
        try:
            import seed_corpus
            hits = seed_corpus.query(query, k=k, jurisdiction=jurisdiction or None)
        except Exception as e:  # noqa: BLE001
            logger.debug("seed_corpus failed in agent: %s", e)

    # Normalize to a small payload Gemini can read cheaply.
    out = []
    for h in hits[:k]:
        out.append({
            "title": h.get("title") or h.get("case_id") or "Untitled",
            "court": h.get("court") or "",
            "year": h.get("year") or "",
            "citation": h.get("citation") or "",
            "excerpt": (h.get("excerpt") or "")[:400],
            "tier": (h.get("tier") or "").upper(),
            "score": h.get("score"),
            # Tool-side identifiers we'll rebuild into the final citation list
            "_case_id": h.get("case_id") or "",
            "_pdf_name": h.get("pdf_name") or "",
            "_s3_key": h.get("s3_key") or "",
        })
    return {"hits": out, "count": len(out)}


def _tool_retrieve_statutes(args: dict[str, Any]) -> dict[str, Any]:
    query = (args.get("query") or "").strip().lower()
    jurisdiction = (args.get("jurisdiction") or "").strip().upper() or None
    kind = (args.get("kind") or "").strip().lower() or None
    if not query:
        return {"error": "query is required", "items": []}
    try:
        import auth
        rows = auth.library_list(jurisdiction, kind)
    except Exception as e:  # noqa: BLE001
        return {"error": f"library lookup failed: {e}", "items": []}

    matched = []
    for r in rows:
        title = (r.get("title") or "").lower()
        if query in title:
            matched.append(r)
    if not matched:
        # Fall back to any title token match.
        tokens = [t for t in query.split() if len(t) > 2]
        for r in rows:
            title = (r.get("title") or "").lower()
            if any(t in title for t in tokens):
                matched.append(r)

    out = []
    for r in matched[:PER_TOOL_MAX_RESULTS]:
        # We need the full body for the agent to actually use the statute.
        try:
            import auth
            full = auth.library_get(int(r["id"]))
        except Exception:  # noqa: BLE001
            full = r
        out.append({
            "title": full.get("title") or "",
            "jurisdiction": full.get("jurisdiction") or "",
            "kind": full.get("kind") or "",
            "body_md": (full.get("body_md") or "")[:4000],
            "source_url": full.get("source_url") or "",
            "_library_id": full.get("id"),
        })
    return {"items": out, "count": len(out)}


def _tool_web_search(args: dict[str, Any]) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    k = max(1, min(int(args.get("k") or 5), 8))
    restrict_domain = (args.get("restrict_domain") or "").strip() or None
    if not query:
        return {"error": "query is required", "snippets": []}
    try:
        import connectors
        snippets = connectors.web_search_snippets(
            query, k=k, restrict_domain=restrict_domain
        )
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "snippets": []}
    return {"snippets": snippets, "count": len(snippets)}


def _tool_redline_contract(args: dict[str, Any]) -> dict[str, Any]:
    text = (args.get("text") or "").strip()
    if len(text) < 50:
        return {"error": "text must be at least 50 chars"}
    if len(text) > 40000:
        return {"error": "text capped at 40000 chars"}
    try:
        import workflows
        return workflows.redline_contract(text)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _tool_translate(args: dict[str, Any]) -> dict[str, Any]:
    text = (args.get("text") or "").strip()
    direction = (args.get("direction") or "en->hi").strip()
    if not text:
        return {"error": "text is required"}
    try:
        import workflows
        return workflows.translate(text, direction=direction)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _tool_create_google_doc(args: dict[str, Any], *, user_id: int) -> dict[str, Any]:
    title = (args.get("title") or "").strip()
    content_md = args.get("content_md") or ""
    if not title or not content_md:
        return {"error": "title and content_md are required"}
    try:
        import google_service
        return google_service.create_doc(user_id, title, content_md)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _tool_create_gmail_draft(args: dict[str, Any], *, user_id: int) -> dict[str, Any]:
    to = (args.get("to") or "").strip()
    subject = (args.get("subject") or "").strip()
    body_md = args.get("body_md") or ""
    if not to or not subject:
        return {"error": "to and subject are required"}
    try:
        import google_service
        return google_service.create_gmail_draft(user_id, to, subject, body_md)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _tool_append_sheet_row(args: dict[str, Any], *, user_id: int) -> dict[str, Any]:
    matter = (args.get("matter") or "").strip()
    action = (args.get("action") or "").strip()
    if not matter or not action:
        return {"error": "matter and action are required"}
    import datetime as _dt
    row = [
        _dt.date.today().isoformat(),
        matter,
        (args.get("jurisdiction") or "").strip(),
        (args.get("court") or "").strip(),
        (args.get("stage") or "").strip(),
        action,
        (args.get("doc_url") or "").strip(),
        (args.get("notes") or "").strip(),
    ]
    try:
        import google_service
        return google_service.append_matter_row(user_id, row)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _tool_search_drive(args: dict[str, Any], *, user_id: int) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    k = max(1, min(int(args.get("k") or 8), 20))
    if not query:
        return {"error": "query is required"}
    try:
        import google_service
        files = google_service.search_drive(user_id, query, k=k)
        return {"files": files, "count": len(files)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# Tools that require the user_id (look up Google OAuth tokens).
_GOOGLE_TOOL_NAMES = frozenset({
    "create_google_doc",
    "create_gmail_draft",
    "append_sheet_row",
    "search_drive",
})


_DISPATCH = {
    "retrieve_cases":     _tool_retrieve_cases,
    "retrieve_statutes":  _tool_retrieve_statutes,
    "web_search":         _tool_web_search,
    "redline_contract":   _tool_redline_contract,
    "translate":          _tool_translate,
    "create_google_doc":  _tool_create_google_doc,
    "create_gmail_draft": _tool_create_gmail_draft,
    "append_sheet_row":   _tool_append_sheet_row,
    "search_drive":       _tool_search_drive,
}


# ─────────────────────────────────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────────────────────────────────

def _format_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert thread history rows into Gemini `contents` format. We keep
    only the last 6 turns so the agent stays focused — full history goes
    into the system prompt instead if it ever matters."""
    out = []
    for m in history[-6:]:
        role = "user" if m.get("role") == "user" else "model"
        text = (m.get("content") or "").strip()
        if not text:
            continue
        out.append({"role": role, "parts": [{"text": text}]})
    return out


def run(
    question: str,
    history: list[dict[str, Any]],
    *,
    jurisdiction: Optional[str] = None,
    prior_citations: Optional[list[dict[str, Any]]] = None,
    user_id: Optional[int] = None,
    language: Optional[str] = None,
    max_turns: int = MAX_TURNS,
    prefer: Optional[str] = None,  # noqa: ARG001  agent loop is Gemini-only by design (tool calling)
) -> dict[str, Any]:
    """Run the agent loop. Returns:

      {
        answer_markdown: str,
        citations: [...],           # accumulated from retrieve_cases hits
        llm: {provider, model, latency_ms, fallback_chain},
        trace: [{tool, args, result_preview, ms}, ...],
        validation: {passed: True, confidence: 0.85, reasons: []},
        refused: bool,
        mode: "agent",
      }
    """
    if not router.GEMINI_API_KEY:
        return {
            "answer_markdown": (
                "I can't run the agent right now — `GEMINI_API_KEY` isn't "
                "configured. Add it in Settings or `.claude/launch.json`."
            ),
            "citations": [],
            "llm": {"provider": "none", "model": "", "latency_ms": 0, "fallback_chain": []},
            "trace": [],
            "validation": {"passed": False, "confidence": 0.0, "reasons": ["GEMINI_API_KEY not set"]},
            "refused": True,
            "mode": "agent",
        }

    # Compose the agent's system prompt with an optional language directive.
    # We import lazily to avoid a circular import (brief_service imports this
    # module via answer_agent's lazy `from agents import legal_agent`).
    try:
        from brief_service import _lang_directive  # type: ignore
        sys_prompt = AGENT_SYSTEM_PROMPT + _lang_directive(language)
    except Exception:
        sys_prompt = AGENT_SYSTEM_PROMPT

    # Seed contents with the prior conversation + current request.
    contents: list[dict[str, Any]] = _format_history(history)
    user_text_parts: list[str] = []
    if jurisdiction:
        user_text_parts.append(f"[Jurisdiction: {jurisdiction}]")
    if prior_citations:
        # Surface the rail across turns so "this case" / "the same matter"
        # references resolve. The model treats these as facts in evidence,
        # not as cite-chips it must re-emit with [n] brackets.
        cite_lines = []
        for c in prior_citations[:8]:
            title = c.get("title") or "Untitled"
            meta_bits = [c.get("court"), str(c.get("year") or ""), c.get("citation")]
            meta = " · ".join(b for b in meta_bits if b)
            cite_lines.append(f"  • {title}" + (f" ({meta})" if meta else ""))
        if cite_lines:
            user_text_parts.append(
                "Cases already on the record in this matter (treat as facts; "
                "do NOT re-fetch unless the user explicitly asks):\n"
                + "\n".join(cite_lines)
            )
    user_text_parts.append(question)
    contents.append({"role": "user", "parts": [{"text": "\n\n".join(user_text_parts)}]})

    # Seed accumulated_hits with prior citations so the final rail stays
    # populated even if the agent doesn't call retrieve_cases this turn.
    accumulated_hits_seed = list(prior_citations or [])

    accumulated_hits: list[dict[str, Any]] = []
    # Carry prior rail through to the final citations payload.
    _prior_rail: list[dict[str, Any]] = list(accumulated_hits_seed)
    trace: list[dict[str, Any]] = []
    final_text = ""
    last_provider = "gemini"
    last_model = router.GEMINI_MODEL
    t_start = time.monotonic()

    for turn in range(max_turns):
        try:
            resp = router.gemini_tool_call(
                sys_prompt,
                contents,
                TOOL_DECLARATIONS,
                temperature=0.2,
                max_tokens=2400,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("agent gemini_tool_call failed (turn %d): %s", turn, e)
            return {
                "answer_markdown": f"Agent failed: {e}",
                "citations": _build_citations(_prior_rail + accumulated_hits),
                "llm": {"provider": last_provider, "model": last_model, "latency_ms": int((time.monotonic() - t_start) * 1000), "fallback_chain": [last_provider]},
                "trace": trace,
                "validation": {"passed": False, "confidence": 0.0, "reasons": [str(e)]},
                "refused": True,
                "mode": "agent",
            }

        function_calls = resp["function_calls"]
        text = resp["text"]

        # If the model returned a final answer (no tool calls), we're done.
        if not function_calls:
            final_text = text or "(empty response)"
            break

        # Otherwise, append the model's tool-call message verbatim, then
        # dispatch each call and append responses.
        contents.append({"role": "model", "parts": resp["raw"].get("parts", [])})

        for call in function_calls:
            name = call["name"]
            args = call["args"]
            t_tool = time.monotonic()
            if name not in _DISPATCH:
                result = {"error": f"unknown tool: {name}"}
            else:
                try:
                    # Google tools need user_id for OAuth lookup; legal tools
                    # ignore it. Both signatures accepted via kwargs.
                    fn = _DISPATCH[name]
                    if name in _GOOGLE_TOOL_NAMES:
                        if user_id is None:
                            result = {"error": "Google tools require a logged-in user."}
                        else:
                            result = fn(args, user_id=user_id)
                    else:
                        result = fn(args)
                except Exception as e:  # noqa: BLE001
                    logger.warning("tool %s raised: %s", name, e)
                    result = {"error": str(e)}
            ms = int((time.monotonic() - t_tool) * 1000)

            # Accumulate citations from retrieve_cases for the final UI rail.
            if name == "retrieve_cases" and isinstance(result, dict) and result.get("hits"):
                accumulated_hits.extend(result["hits"])

            trace.append({
                "tool": name,
                "args": args,
                "result_preview": _preview(result),
                "ms": ms,
            })

            contents.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": name,
                        "response": _truncate_for_gemini(result),
                    }
                }],
            })

        # If we're at the last turn and the model still wants tools, force
        # a final answer on the next iteration by clearing the tool list.
        # (Gemini's loop will obey because finishReason will go STOP.)
        if turn == max_turns - 2:
            # one more turn allowed — let it finalise
            continue

    if not final_text:
        # Hit max_turns without a STOP — ask Gemini one more time WITHOUT
        # tools so it must compose an answer.
        try:
            resp_final = router.gemini_tool_call(
                sys_prompt
                + "\n\nYou have used all tool budget. Compose the final answer "
                + "now using only the tool results you already have.",
                contents,
                tools=[],
                temperature=0.2,
                max_tokens=2400,
            )
            final_text = resp_final["text"] or "(empty response)"
        except Exception as e:  # noqa: BLE001
            final_text = f"Agent ran out of turns. Error during finalisation: {e}"

    # Merge accumulated_hits with prior rail (dedupe by title+citation in
    # _build_citations). Prior rail comes first so its [n] indexing is stable
    # across turns.
    citations = _build_citations(_prior_rail + accumulated_hits)
    elapsed = int((time.monotonic() - t_start) * 1000)
    return {
        "answer_markdown": final_text,
        "citations": citations,
        "llm": {
            "provider": last_provider,
            "model": last_model,
            "latency_ms": elapsed,
            "fallback_chain": [last_provider],
        },
        "trace": trace,
        "validation": {
            "passed": True,
            "confidence": 0.85 if citations else 0.6,
            "reasons": [f"{len(trace)} tool call(s)"] if trace else ["no tools called"],
        },
        "refused": False,
        "mode": "agent",
    }


# ─────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────

def _preview(result: Any) -> str:
    """Compact one-line preview of a tool result for the UI trace."""
    if not isinstance(result, dict):
        return str(result)[:160]
    if "error" in result:
        return f"error: {result['error'][:120]}"
    if "hits" in result:
        return f"{result.get('count', len(result['hits']))} cases"
    if "items" in result:
        return f"{result.get('count', len(result['items']))} statutes"
    if "snippets" in result:
        return f"{result.get('count', len(result['snippets']))} web snippets"
    if "translation" in result:
        return f"translated {len(result['translation'])} chars"
    if "suggestions" in result:
        return f"{len(result['suggestions'])} redline suggestions"
    # Fallback — first key + length
    return json.dumps({k: ("…" if isinstance(v, (list, dict)) else v) for k, v in list(result.items())[:3]}, default=str)[:160]


def _truncate_for_gemini(result: Any) -> Any:
    """Drop large fields (full PDFs, long bodies) before sending tool
    results back into Gemini's context."""
    if not isinstance(result, dict):
        return {"value": str(result)[:2000]}
    out: dict[str, Any] = {}
    for k, v in result.items():
        if isinstance(v, str):
            out[k] = v[:6000]
        elif isinstance(v, list):
            out[k] = v[:8] if len(v) > 8 else v
        else:
            out[k] = v
    return out


def _build_citations(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert accumulated retrieve_cases hits into the same citation shape
    `brief_service._citation_payload` produces, so the UI's source rail
    renders identically. Dedupe by (title, citation)."""
    seen: set[tuple[str, str]] = set()
    out = []
    for h in hits:
        key = (h.get("title", ""), h.get("citation", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "n": len(out) + 1,
            "case_id": h.get("_case_id") or "",
            "title": h.get("title") or "Untitled",
            "citation": h.get("citation") or "",
            "court": h.get("court") or "",
            "year": h.get("year") or "",
            "excerpt": (h.get("excerpt") or "")[:400],
            "tier": (h.get("tier") or "").upper(),
            "pdf_url": "",
            "score": h.get("score"),
        })
    return out
