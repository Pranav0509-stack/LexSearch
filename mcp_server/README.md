# Sanhita MCP Server

Exposes Sanhita's Indian-law stack to any MCP-aware client (Claude
Desktop, Claude Code, ChatGPT, Cursor, etc.) as 6 tools.

## Tools

| Tool | What it does |
|---|---|
| `search_caselaw` | Hybrid BM25 + semantic search over **70M+ Indian court records** (16.9M HC + 53M district + 86K SC + tribunals + 11.6M legal_docs + 2K statutes + 1.3M Q&A) |
| `get_document` | Full text + structured metadata (court, parties, judges, outcome, acts cited, PDF URL) for any case |
| `list_templates` | Browse **26 templates** across 9 practice areas; **5 verbatim Government-prescribed forms** (CPC Schedule I Appendix A Form 1, Form 4 Order 37, RTI Form A, NCLT IBC §7 Form 1, Probate §276 ISA) |
| `draft_document` | Generate a complete Indian legal document; preserves all statute citations exactly |
| `lookup_statute` | Top N SC/HC cases interpreting a given Act + Section, with holding-window snippets |
| `compliance_check` | Run 8 Indian-law plug-ins: DPDP, RBI/FEMA, SEBI, IBC §14, GST, IT Act §43A, POSH, Stamp Duty |

## Install

```bash
pip install mcp httpx
```

The Sanhita backend must be running at `localhost:8080` (or set `SANHITA_BACKEND` env).

## Wire to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sanhita": {
      "command": "python3",
      "args": ["/Users/pranav/Desktop/LexSearch-main 2/mcp_server/server.py"],
      "env": { "SANHITA_BACKEND": "http://localhost:8080" }
    }
  }
}
```

Restart Claude Desktop. In any chat, you can now ask:

> Use the sanhita tool to find Supreme Court cases on anticipatory bail
> for elderly accused with medical grounds.

> Draft a CPC Form 1 Plaint for recovery of INR 12 lakh based on a
> dishonoured promissory note dated 14 Jan 2025.

> Run a DPDP compliance check on this MSA: [paste body]

## Wire to Claude Code

```bash
claude mcp add sanhita "python3 /Users/pranav/Desktop/LexSearch-main\ 2/mcp_server/server.py"
```

## Smoke test (standalone)

```bash
python3 - <<'PY'
import asyncio, sys, json
sys.path.insert(0, "/Users/pranav/Desktop/LexSearch-main 2/mcp_server")
from server import search_caselaw, draft_document, lookup_statute

async def main():
    print(await search_caselaw({"query": "section 138 cheque dishonour", "mode": "hybrid", "limit": 3}))
    print("---")
    print(await lookup_statute({"act": "Indian Contract Act, 1872", "section": "Section 27", "topn": 2}))

asyncio.run(main())
PY
```
