# Sanhita — Next Session Roadmap (May 3, 2026)

## What's Done (committed)

### Session 1 — Core Features
- 30M+ record FTS5 search working (<6s)
- 9 panes: Assistant, Court Search, Vault, Workflows, Editor, Dashboard, Library, Clients, Settings
- India-only (stripped SG/HK)
- Auth flow with demo code
- LLM router (Gemini > Claude > Groq > Cloudflare)

### Session 2 — Roadmap Items
- Legal QA dedicated endpoint (`/api/qa/search`) — 144ms vs 8.8s
- 3 new draft templates: consumer_complaint, affidavit, vakalatnama
- Workflow output -> Draft Editor integration (sessionStorage bridge)
- Judge Analytics panel with search, verdict breakdown, court distribution
- Citation Graph (SVG) with cited-by/cites relationships
- Compare mode: select 2-4 cases to compare in Assistant
- TipTap editor (Google Docs-style) for legal documents
- Design restored to original warm parchment theme

### Session 3 (today) — Structured Documents
- `_classify_doc_type()` — JUDGMENT / LEGAL_DOC / STATUTE / LEGAL_QA
- `_build_context()` — full_text (4K chars), separate VERDICT section, doc type in LLM context
- `_citation_payload()` — includes doc_type, full_text, explanation, bench, date_decided
- `_build_no_llm_response()` — grouped by doc type, verdict as distinct section
- Case cards: colored doc type badges (amber/blue/green/purple)
- Detail drawer: structured sections (Type -> Metadata -> Verdict -> Analysis -> Content)
- "Use in Assistant" fetches full case detail (full_text via /api/cases/{id})
- Compare mode sends 4K chars per case with doc type + verdict
- Library pane no longer truncates body_md

## Known Bugs (MUST FIX)

### 1. Judge Profile SQL Error
**Files:** `server.py` (line ~879), `sanhita_adapter.py` (judge_profile method)
**Bug:** References `j.petitioner` and `j.respondent` which don't exist in the `judgments` table.
**Fix:** Use `j.title` and parse petitioner/respondent from it (the `clean_title()` function already does this).

### 2. Auth Cookie Flow
**Bug:** Chat returns `{"detail":"not signed in"}` for valid sessions intermittently.
**Fix:** Verify cookie name/path/domain settings end-to-end. Check `SameSite` attribute.

### 3. Dead Connector Code
**Bug:** `connectors.retrieve_hybrid` referenced in some code paths — connectors.py was for multi-jurisdiction.
**Fix:** Remove all references. FTS5 is the only retrieval path.

### 4. Frontend API Path Inconsistency
**Bug:** Mix of `/search`, `/api/cases/search`, `/retrieve` in some edge cases.
**Fix:** Audit all `fetch()` calls in frontend, standardize to `/api/*`.

## What To Do Next (Priority Order)

### Priority 1: Fix All Bugs (30 min)
1. Fix judge_profile SQL — replace `j.petitioner`/`j.respondent` with parsed title
2. Fix auth cookie flow — test login -> search -> chat -> analytics E2E
3. Remove dead connector references from server.py
4. Audit frontend fetch paths

### Priority 2: Make Search Results Richer (45 min)
The search hits currently show excerpt text but it's unstructured. Need:
- **Highlight matched terms** in excerpts (BM25 doesn't return snippets, need to do client-side)
- **Show acts/sections cited** in each case card (data exists in legal_docs table: `acts_cited` column)
- **Petitioner vs Respondent** display in case cards (parse from title)
- **PDF preview** — when a case has `pdf_available=true`, show a "Preview" button that loads the PDF inline

### Priority 3: Assistant Response Quality (1 hour)
The LLM responses need to be more structured:
- **Citation cards in responses** — when the assistant cites [1], render it as a clickable chip that expands to show the case card inline
- **Follow-up suggestions** — already built (`generate_followups()`), wire into the UI below each response
- **Thinking indicator** — show what the assistant is doing: "Searching 31.9M records..." -> "Found 8 relevant cases" -> "Analyzing..."
- **Source panel** — right sidebar showing all cited cases with expandable details while reading the answer

### Priority 4: Deployment (1 hour)
- Create `Dockerfile` for backend
- Create `railway.toml` or equivalent
- Clean `requirements.txt`
- Document env vars (INDIA_COURTS_DB, API keys, SECRET_KEY, DEMO_CODE)
- Static export of Next.js frontend for Vercel
- Upload 84.7 GB SQLite to persistent volume

### Priority 5: DB Optimization (30 min)
- Create covering indexes: `idx_judgments_court_year`, `idx_judgments_cnr`, `idx_legal_docs_court`
- Run `ANALYZE` for query planner stats
- Precompute analytics tables (court_case_counts, yearly_case_counts already done)
- Cap FTS5 inner query at LIMIT 1000 with court/year filters as WHERE clauses

### Priority 6: Mobile Responsive Polish (30 min)
- Sidebar collapse on mobile (hamburger menu)
- Search bar full-width on small screens
- Case cards stack vertically
- Detail drawer goes full-screen on mobile
- Touch-friendly tap targets (min 44px)

### Priority 7: Advanced Features (2+ hours)
- **Semantic search** — add vector embeddings alongside BM25 (needs embedding model)
- **Case comparison view** — side-by-side structured comparison (already have compare mode, need dedicated view)
- **Court filter bug** — some court_code values don't match between tables
- **Export** — export search results as CSV, export assistant answers as PDF

## File Map (for orientation)

| File | What it does |
|---|---|
| `server.py` | FastAPI backend, 53+ API routes, ~1400 lines |
| `brief_service.py` | Assistant brain: context building, LLM routing, citation formatting |
| `auth.py` | Session auth, demo codes, user management |
| `workflows.py` | 7 legal draft templates with multi-step AI workflows |
| `doc_editor.py` | TipTap editor backend (AI complete, improve, write section) |
| `llm/router.py` | 4-provider LLM chain with circuit breakers |
| `sanhita-react/web/src/app/app/page.tsx` | Main app shell, sidebar, chat UI |
| `sanhita-react/web/src/app/app/court-search-pane.tsx` | Search UI, case cards, detail drawer, judge analytics |
| `sanhita-react/web/src/app/app/editor-pane.tsx` | TipTap document editor |
| `sanhita-react/web/src/app/app/workflows-pane.tsx` | Legal draft workflow UI |
| `sanhita-react/web/src/app/globals.css` | Global styles, warm parchment theme |
| `india-judgments-corpus/scripts/db.py` | SQLite FTS5 queries, get_case(), search() |
| `india-judgments-corpus/scripts/sanhita_adapter.py` | FTS5Index class wrapping db.py for server.py |

## Repo Locations

- **LexSearch (backend + submodule):** `/Users/pranav/Desktop/LexSearch-main 2/`
- **Frontend (submodule):** `/Users/pranav/Desktop/LexSearch-main 2/sanhita-react/web/`
- **Data scripts + adapter:** `/Users/pranav/Desktop/india-judgments-corpus/`
- **Database:** `india_courts.db` (84.7 GB, 30.5M records)

## Key Design Decisions
- **SQLite over Postgres** — single-file deployment, FTS5 built-in, no connection pool complexity
- **BM25 over vector search** — faster, deterministic, no embedding cost; semantic search planned for later
- **Warm parchment theme** — user explicitly rejected cold blue/navy; keep `--bg: #faf7f2`, `--accent: #6b4f1d`
- **India-only** — SG/HK/JP had zero data, stripped entirely
- **No LLM fallback** — when no API key configured, show structured case results directly (no hallucination risk)
