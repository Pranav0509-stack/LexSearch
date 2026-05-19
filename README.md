# Sanhita — India's AI Legal Research Platform

**83M+ indexed legal records. 16.9M HC judgments. 53.3M HC + district registry rows. 11.6M curated docs. 1.31M legal Q&A. 2,333 statutes. 2K tribunal / regulator orders.**

The largest structured Indian legal corpus, searchable in under 50 ms warm, sub-3 s cold across all 6 corpora.

Built by [NyayaSathi AI](https://github.com/Nyayasathi-AI) — because Indian lawyers deserve better than hallucinated citations.

---

## Why Sanhita Exists

| | GPT-4 / Claude | Harvey AI | **Sanhita** |
|---|---|---|---|
| Indian case law | ~0 (hallucinates) | US/UK only | **83M+ real records, 6 corpora** |
| Structured metadata | None | Limited | **CNR, judge, bench, court, petitioner, respondent, verdict, PDF** |
| Verified citations | No | Yes (US) | **Every citation links to a real judgment** |
| NJDG pending cases | No | No | **44.1M records across 30 states** |
| Indian statutes | Training data | US/UK | **2,333 indexed bare acts** |
| Tribunal coverage | No | No | **NCLAT, RBI, SEBI, PRS, IndiaCode** |
| Languages | English | English | **13 Indian languages** |
| Price | $20-200/mo | $2,000/user/mo | **Free tier + Pro** |

## Architecture

```
Sanhita Platform
├── FastAPI Backend
│   ├── /api/cases/smart-search — HybridSearchEngine: BM25 ⨁ FAISS over 6 corpora
│   │     · parallel fan-out across tables (6 worker threads, own SQLite handles)
│   │     · FTS5 `rank` short-circuit replaces per-row bm25() calls
│   │     · cold-cache "bail" + ALL sources: 2.6 s  (was >60 s sequential)
│   ├── /api/cases/latest — cross-corpus newest, round-robined 4-per-source
│   ├── /api/brief/chat     — legacy RAG: retrieval → LLM → 6-gate validator
│   ├── /api/brief/chat-v2  — planner + multi-corpus retrieve + synth + validator
│   │     · sub-question decomposition · practice-area auto-classify
│   │     · returns {answer_markdown, citations, sub_questions, validation}
│   ├── /api/contract/* — draft, review, redline, compliance, quick-edit
│   ├── /api/vault/* — document upload + multi-doc Q&A
│   ├── /api/legal-aid/* — intake form + admin queue
│   └── /api/analytics/* — court efficiency, bail intelligence, corpus stats
├── React Frontend (Next.js 16, Turbopack)
│   ├── Assistant — chat hitting /api/brief/chat-v2 (planner+validator)
│   ├── Court Search — filter-driven auto-rerun, 6 source tabs incl. tribunals
│   ├── Drafter Studio — 26 templates (4 verbatim Govt forms first, then statute-anchored)
│   ├── Editor — TipTap with full toolbar + Export menu (HTML/MD/TXT/Print/Clipboard)
│   ├── Workflows — n8n-style drag-drop canvas + 27 prebuilt recipes
│   │     · 12 Indian-litigation procedural (cheque-bounce §138, bail pipeline,
│   │       civil-suit bundle, §80 CPC, §34 setaside, quashing §482, morning
│   │       cause-list, limitation tracker, order-copy fetcher, client
│   │       onboarding, cross-exam prep, bill of costs)
│   │     · 15 transactional / compliance — covers EVERY Lexity Clickflow
│   │       (Agreement Audit · Chronology of Events · Compliance Gap ·
│   │        Due Diligence · Focused Summarizer · Closing Checklist ·
│   │        Privilege Review · Playbook Audit · Matter Triage ·
│   │        Claim Challenger · New Case Assessment · Quick Agreement
│   │        Analyzer · Loan Compliance Tracker · Execution-vs-Final Diff ·
│   │        Chain of Ownership) — all anchored to Indian statutes
│   ├── Document Vault — upload & query your case files
│   └── Clients / Legal Aid / History panes
├── LLM Router (4-provider chain with circuit breakers)
│   ├── Gemini Flash → Anthropic Claude → Groq Llama 70B → Cloudflare
│   └── No-LLM fallback: returns structured case results (never refuses)
├── Reasoner pipeline (scripts/assistant/legal_reasoner.py)
│   ├── Planner LLM → sub-questions + practice area
│   ├── Retrieve evidence per sub-question via HybridSearchEngine
│   ├── Synthesise answer with inline [E1..En] markers
│   └── Validate via 6 answer gates (cite_present / resolves / banned / grounding / scope / section)
├── 6-Gate Answer Validator (validators/answer_gates.py)
│   ├── G1: cite_present — answer has [n] markers
│   ├── G2: cite_resolves — markers map to real cases
│   ├── G3: no_banned_phrases — no hallucination hedges
│   ├── G4: grounding_floor — 60%+ sentences cited
│   ├── G5: scope_check — no fabricated case names
│   └── G6: section_check — statute refs verified
└── SQLite + FTS5 + FAISS (152 GB on disk, WAL + mmap 8 GiB)
    ├── judgments      (16.9M) — all 25 High Courts, 1950-2025
    ├── pipeline_docs  (53.3M) — HC + district + tribunal registry
    ├── legal_docs     (11.6M) — KanoonGPT HC text, SC, statutes-as-cases
    ├── legal_qa       ( 1.31M) — question-answer pairs
    ├── statutes       ( 2,333) — IPC, CrPC, CPC, BNS, BNSS, BSA, India Code
    ├── documents      ( 2,038) — NCLAT, RBI, SEBI, PRS, IndiaCode acts
    └── njdg_*         (44.1M) — pending case records
```

## What landed in the May 18 release

| Change | Why it matters |
|---|---|
| **Parallel BM25 fan-out** across 6 corpora (`scripts/search/engine.py`) | Cold-cache "bail" ALL-sources: **>60 s → 2.6 s** (UI no longer times out) |
| **FTS5 `rank` short-circuit** replaces per-row `bm25()` | Individual table cold time **30 s → 3-7 s** |
| **`documents` corpus exposed** (NCLAT/RBI/SEBI/PRS/IndiaCode) | Previously invisible; now searchable + appears in Latest tab |
| **`/api/brief/chat-v2`** wraps the legal_reasoner pipeline | Chat now gets sub-question planning + 6-gate validation + 83M-row grounding |
| **`api_cases_latest` cross-corpus union** | Latest tab was judgments-only — now shows 4 per corpus, date-sorted |
| **Auto-rerun on filter change** in Court Search | Year / source / engine changes now refire the query (250 ms debounce) |
| **Tabs surface all 6 corpora** | All Sources · HC Judgments · Case Records · Legal Docs · Statutes · Legal Q&A · Tribunals & Regulators |
| **Drafter quick-edit messaging** | Refusal-with-reason now renders as a friendly tip, not an error |
| **Editor Export menu** | HTML / Markdown / TXT / Print → PDF / Copy-all (replaces broken empty-onClick button) |
| **Template sort by authority** | 4 verbatim Govt forms (RTI, CPC Form 1, CPC Form 4, NCLT IBC §7) now sort first |
| **Acronym aliasing in search** | RBI / SEBI / NCLAT / IBC / CrPC / BNS / 19 more expand to full-name OR-groups so the documents corpus is reachable by acronym |
| **+5 transactional workflows** | Agreement Audit (14-dim, FEMA-aware) · Chronology of Events · Compliance Gap (8 plugins) · Due Diligence M&A · Focused Summarizer — the Indian-flavoured answer to Lexity's Clickflows |
| **Hardened criminal recipes** | Cheque-bounce §138 + Bail pipeline now include citator gate (no overruled cites surface), WhatsApp client tracker, and Vault snapshot for reuse on appeal |
| **+10 Lexity-parity workflows** | Closing Checklist · Privilege Review (§126 IEA) · Playbook Audit · Matter Triage · Claim Challenger · New Case Assessment · Quick Agreement Analyzer · Loan Compliance Tracker · Execution-vs-Final Diff · Chain of Ownership — every Lexity Clickflow now has an India-anchored counterpart |
| **Workflow runner hardening** | `executeNode` never breaks the chain on a single backend miss; soft-fail rows show what would have run; "pick best body" finds the right input field automatically (no more "missing brief_facts") |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Nyayasathi-AI/Nyayasathi_main.git
cd Nyayasathi_main

# 2. Install
pip install -r requirements.txt

# 3. Symlink the database (from data_main repo)
ln -s ../india-judgments-corpus/india_courts.db india_courts.db

# 4. Run
uvicorn server:app --reload --port 8080
```

Open [http://localhost:8080](http://localhost:8080). Login with demo code: `SNHT-DEMO-2026`

### Optional: Configure LLM for AI-composed answers

```bash
# Any ONE of these enables the Brief assistant's AI mode:
export GROQ_API_KEY=gsk_...        # Free — https://console.groq.com/keys
export GEMINI_API_KEY=AI...        # Free — https://aistudio.google.com/apikey
export ANTHROPIC_API_KEY=sk-ant-...# Paid — best quality
```

Without an LLM key, the assistant still returns structured case results from the 31.9M corpus — no refusals, 100% grounded.

## Key Features

### Sanhita Brief (AI Assistant)
- Ask any Indian law question in natural language
- Gets 10 relevant cases from 31.9M corpus via FTS5 BM25
- LLM composes answer with mandatory [n] citations
- 6-gate validator ensures every claim is grounded
- Live web signals from Bar & Bench, LiveLaw, SC Observer, PRS India
- 13 Indian languages: Hindi, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi, Odia, Assamese, Urdu

### Court Search
- Search across 25 High Courts + Supreme Court
- Filter by court, year, judge, case type, verdict, CNR
- PDF viewer with in-browser reading
- Related cases via citation graph

### Analytics Dashboard
- Court efficiency metrics (disposal rates, avg time)
- Bail intelligence (grant/rejection rates by court)
- Corpus statistics (cases by year, court, verdict)
- Document type distribution

### Document Vault
- Upload case files (PDF, DOCX)
- Multi-document Q&A with grounded citations

### Draft Generator
- 10 legal document templates (bail applications, writ petitions, etc.)
- Auto-fills with case citations from corpus

## API Endpoints (53 routes)

| Category | Endpoint | Description |
|---|---|---|
| Search | `GET /api/cases/search?q=...&court=...&year_from=...` | FTS5 search with filters |
| Search | `GET /api/cases/latest` | Newest judgments |
| Search | `GET /api/cases/{case_id}` | Single case detail |
| Search | `GET /api/cases/related/{case_id}` | Citation graph — related cases |
| Chat | `POST /api/brief/chat` | RAG: retrieve + compose + validate |
| Chat | `GET /api/brief/threads` | List chat threads |
| News | `GET /api/news` | Latest legal news (5 sources) |
| News | `GET /api/news/search?q=...` | Search legal news |
| Analytics | `GET /api/analytics/corpus-stats` | Corpus-wide statistics |
| Analytics | `GET /api/analytics/court-efficiency` | Court disposal metrics |
| Analytics | `GET /api/analytics/bail-intelligence` | Bail grant/rejection rates |
| Vault | `POST /api/vault/upload` | Upload documents |
| Vault | `POST /api/vault/ask` | Q&A over uploaded docs |
| Draft | `POST /api/draft` | Generate legal documents |
| Templates | `GET /api/templates` | List draft templates |
| Auth | `POST /api/login` | Session login |
| Health | `GET /health` | Server health + index stats |

## Environment Variables

See [`.env.example`](.env.example) for full list. Key variables:

| Variable | Required | Description |
|---|---|---|
| `INDIA_COURTS_DB` | No | Path to SQLite DB (auto-detected) |
| `GROQ_API_KEY` | No | Groq API key (free, primary LLM) |
| `GEMINI_API_KEY` | No | Gemini API key (fallback LLM) |
| `ANTHROPIC_API_KEY` | No | Claude API key (best quality) |
| `LEXSEARCH_SECRET_KEY` | Prod | Session cookie signing key |
| `LEXSEARCH_ADMIN_TOKEN` | Prod | Admin reload endpoint token |

## Data Sources

- [indian-high-court-judgments](https://github.com/vanga/indian-high-court-judgments) (CC-BY-4.0) — 16.9M HC judgments
- [KanoonGPT](https://huggingface.co/datasets/Exploration-Lab/KanoonGPT) — 13.6M legal documents
- [Indian Legal QA](https://huggingface.co/) — 1.36M question-answer pairs
- [NJDG](https://njdg.ecourts.gov.in/) — 44.1M pending case records
- Live feeds: Bar & Bench, LiveLaw, Supreme Court Observer, PRS India

## Repos

| Repo | Purpose |
|---|---|
| [Nyayasathi_main](https://github.com/Nyayasathi-AI/Nyayasathi_main) | Backend + Frontend (this repo) |
| [data_main](https://github.com/Nyayasathi-AI/data_main) | Data pipeline, ingestors, SQLite DB scripts |
| [plan_main](https://github.com/Nyayasathi-AI/plan_main) | Product roadmap, architecture plans |

## License

Proprietary. Data sources are individually licensed (see above).

---

Built by [Pranav](https://github.com/Pranav0509-stack) at [NyayaSathi AI](https://github.com/Nyayasathi-AI)
