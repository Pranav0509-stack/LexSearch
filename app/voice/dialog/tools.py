"""
LLM tool-call router — dispatches tool calls emitted during a dialog turn.

Tools exposed:
  - judgment_search(query, top_k, filters)   → BM25 RAG over judgment corpus
  - lawyer_search(domain, city, language)    → partner panel lookup
"""

from typing import Any

from app.voice.session import CallSession

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "judgment_search",
            "description": "Search Indian HC+SC judgments to cite relevant case law.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "court": {"type": "string"},
                            "year_min": {"type": "integer"},
                        },
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lawyer_search",
            "description": "Find a panel lawyer by domain, city, language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "city": {"type": "string"},
                    "language": {"type": "string"},
                },
                "required": ["domain"],
            },
        },
    },
]


async def execute_tool_call(call: dict, session: CallSession) -> dict[str, Any]:
    name = call.get("name")
    args = call.get("arguments", {}) or {}

    if name == "judgment_search":
        from app.rag.judgment_tool import judgment_search
        return await judgment_search(**args)

    if name == "lawyer_search":
        from app.lawyers.matcher import propose_lawyers
        return await propose_lawyers(**args)

    return {"error": f"unknown tool: {name}"}
