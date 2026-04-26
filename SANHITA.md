# Sanhita — Pan-Asia Legal Research Counsel

> **What it is:** an AI legal research assistant for advocates in India,
> Singapore, and Hong Kong. Ask a question → get a grounded answer with
> case citations, draft contracts, search live court-case databases,
> redline documents, draft notices, translate into 33 languages, and
> ship the result by email or Google Doc.

---

## Table of contents

1. [What we built (capability inventory)](#1-what-we-built-capability-inventory)
2. [Architecture — how the pieces fit](#2-architecture--how-the-pieces-fit)
3. [Tech stack](#3-tech-stack)
4. [What it can do today (user-visible surface)](#4-what-it-can-do-today-user-visible-surface)
5. [What we can build next (12-week roadmap)](#5-what-we-can-build-next-12-week-roadmap)
6. [What's needed (infra, accounts, data)](#6-whats-needed-infra-accounts-data)
7. [Cost model](#7-cost-model-pilot--scale-tiers)
8. [Local development](#8-local-development)
9. [Deployment notes](#9-deployment-notes)
10. [Security & data handling](#10-security--data-handling)

---

## 1. What we built (capability inventory)

### 1.1 The chat itself (Assistant pane)
- **Multi-mode chat** — research (BM25 + connectors), draft (open-canvas, no retrieval), web (Tavily/Serper/Gemini-grounded snippets), agent (tool-using loop).
- **Live "thinking" panel** — ChatGPT-style phase strip ("Searching the case-law index…" → "Drafting the memo…" → "Checking citations resolve…") so the user never sees a silent spinner.
- **Citation chips** — every `[n]` in an answer is a clickable superscript that highlights the matching source card in the rail.
- **Multi-section memos** — system prompt enforces a 7-section structure (Overview, Statutory Framework, Key Concepts, Procedure, Penalties, Recent Developments, Practice Pointers) so answers read like a real research memo, not a Wikipedia stub.
- **Action row on every answer** — Copy (plain text), Email (opens mail client pre-filled), Save as Doc (Google Docs API if connected, else Markdown download).
- **Suggested next steps** — after each answer, three Harvey-style follow-up cards in a 2-column grid.
- **Model picker** — Auto · Gemini 2.5 Flash (default), Claude Sonnet 4.5, Llama 3.3 70B (Groq), Workers AI. Threaded as `prefer` through `llm.router.generate`.
- **Language picker** — 33 languages, including all 22 Eighth Schedule Indian languages. Routes through Sarvam AI (`mayura:v1`) for Indian languages where Gemini quality drops; native Gemini for everything else.

### 1.2 Court Search (NEW)
- Dedicated sidebar pane between Workflows and Clients.
- **Search tab** — full-text BM25 over the live GitHub-ingested corpus, filter by jurisdiction (IN/SG/HK) and tier (SC, HC, CFA, CA, CFI…), 20 results per page.
- **Latest tab** — newest cases ingested, per jurisdiction, sorted by `(added_at, year)` desc.
- **Result cards** — title in Fraunces serif, court · year · citation chip in accent gold, 3-line excerpt, "Open" → side drawer with full text + source link, "Use in Assistant" → seeds a chat with the case as context.
- **Header chip** — live `"X cases indexed — N IN · N SG · N HK"` so you always know corpus depth.

### 1.3 Storage / Vault
- Multipart upload, list, delete (`/api/vault/*`).
- Per-document Q&A using BM25 over the upload + Gemini synthesis.

### 1.4 Workflows
- **Draft** — template + facts JSON → drafted document.
- **Review** — clauses → flag unusual terms.
- **Translate** — EN ↔ 33 languages.
- **Citator** — case title + holdings → cite-checked.
- **Redline** — large-text diff → JSON suggestions {remove, replace, add}.
- 9 generic flows from `workflows.py` (summarize, expand, simplify, …) wired into one card grid.

### 1.5 Clients
- Inbox of leads pulled from the NyayaSathi WhatsApp/voice surface.
- Click a row → opens the matter as a fresh chat thread, jurisdiction pre-set.

### 1.6 History
- SQLite-backed thread search (LIKE over `messages.content`) at `/api/brief/threads/search`.
- Date-range filter chips: today / this week / this month / all.

### 1.7 Library
- Curated statutes + contract templates, filterable by jurisdiction (IN/SG/HK) and kind (statute / contract / pleading).
- "Use this in chat" injects the doc as context into the Assistant pane.

### 1.8 Settings
- Per-connector API key plug-in surface (Indian Kanoon, eCourts, Serper, Tavily, Sarvam, …).
- DB-backed keystore (`auth.connector_keys`) — keys read DB-first, env fallback. Set / revoke from the UI.
- Google Workspace OAuth panel (Gmail draft + Docs export).

### 1.9 Retrieval engine (Part B of the plan)
- `retrieval_pkg/index.py` — `BM25Index` class on `rank_bm25.BM25Okapi`. Pure Python, ~5ms/query at 100K docs.
- `Document` dataclass — fields: `case_id, title, text, court, year, citation, jurisdiction, tier, url, source, added_at, extra`.
- `bm25.pkl` — pickled corpus + tokens (BM25 re-fits on first query post-load, ~3s at 100K).
- `add()` is idempotent — re-running the same ingestor does not duplicate.

### 1.10 Ingestion engine (Part C of the plan)
- `scripts/ingest_github_data.py` — driver CLI.
- `scripts/ingestors/_common.py` — shared GitHub helpers (`gh_raw`, `gh_api`, `extract_year`, `hk_tier_from_citation`, `clean_text`, `stable_case_id`).
- `scripts/ingestors/hk_cuthchow_csv.py` — pulls 6,091 HK judgments from `cuthchow/Hong-Kong-Courts/case_df.csv`, emits 882 unique Documents (after dedup) with court, year, citation, tier extracted from the case_name string.
- Driver supports `--source NAME`, `--all`, `--limit N`, `--top-up`. Atomic save (`bm25.pkl.tmp` → rename).

### 1.11 LLM router
- Provider chain: Gemini 2.5 Flash → Anthropic Claude Sonnet 4.5 → Groq Llama 3.3 70B → Cloudflare Workers AI.
- Circuit-breaker per provider (open after consecutive failures, half-open after cooldown).
- `prefer="X"` reorders the chain so the named provider is tried first.

### 1.12 Validation gates
- 6-gate `validators/answer_gates.py`: G1 citation present, G2 citation resolves, G3 banned phrases (no "as a language model"), G4 grounding ratio ≥ 60%, G5 no fabricated case names, G6 quote spans match retrieved text.
- Per-mode relaxation: research = all 6, draft = G3 only, web = G1+G2+G3.

### 1.13 Other infra
- SQLite (auth, sessions, threads, messages, library_docs, connector_keys, clients).
- Three FastAPI rate-limit buckets (chat, draft, web, agent).
- Structured input guards (`validators/input_guards.py`) for length / scope / PII redaction.

---

## 2. Architecture — how the pieces fit

```
                              Browser
                                 │
       ┌─────────────────────────┴──────────────────────────┐
       │           Sanhita Frontend (Next.js 16)            │
       │  /app  ─── 8 panes: Assistant, Storage, Workflows,│
       │              Court Search, Clients, History,      │
       │              Library, Settings                    │
       └─────────────────────────┬──────────────────────────┘
                                 │  fetch() w/ ls_session cookie
                                 ▼
       ┌────────────────────────────────────────────────────┐
       │            Sanhita Backend (FastAPI)               │
       │                                                    │
       │  /api/brief/{chat,draft,web,agent}                 │
       │  /api/cases/{search,latest,:id}        ← NEW       │
       │  /api/vault/{upload,docs,chat}                     │
       │  /api/workflows + /api/{draft,review,translate,…}  │
       │  /api/library + /api/clients + /api/settings/keys  │
       │  /api/google/{oauth/start,oauth/callback,docs/…}   │
       └─────────────┬───────────────────────┬──────────────┘
                     │                       │
                     ▼                       ▼
           ┌──────────────────┐    ┌────────────────────┐
           │  retrieval_pkg   │    │   connectors.py    │
           │   BM25Index      │◄───│  retrieve_hybrid   │
           │   (882 HK docs)  │    │  bm25 → kanoon →   │
           │                  │    │  web → seed        │
           └────────┬─────────┘    └─────────┬──────────┘
                    │                        │
        ┌───────────┴───────────┐  ┌─────────┴─────────┐
        │  scripts/ingestors/   │  │   llm/router.py   │
        │  hk_cuthchow_csv.py   │  │   Gemini → Claude │
        │  + 5 more (planned)   │  │   → Groq → CF     │
        └───────────┬───────────┘  └───────────────────┘
                    │
                    ▼
            GitHub raw content
            (parquet / CSV / JSON dumps)
```

---

## 3. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Frontend | Next.js 16 (App Router, Turbopack) | Fast HMR, native font optimisation, edge-friendly |
| Styling | Tailwind 4 + custom CSS variables | Beige/serif legal aesthetic; Fraunces (display) + Inter (body) |
| Backend | FastAPI + Pydantic | Async, easy to deploy, sane request validation |
| Auth | SQLite + signed cookies | Single-tenant pilot; trivially upgradable to Postgres + Auth.js |
| LLM | Gemini 2.5 Flash (primary) | Best quality/$ for grounded Q&A in 2026; multilingual native |
| LLM fallback | Anthropic, Groq, Cloudflare | Circuit-breaker degrades gracefully |
| Translation | Sarvam AI `mayura:v1` | 11 Indian languages, better than Gemini for low-resource scripts |
| Retrieval | `rank_bm25` + pickle | Pure-Python, no infra; great recall for legal-keyword queries |
| Web search | Tavily → Serper → Gemini grounded → DuckDuckGo | Tier-falls-through; Tavily key live |
| Persistence | SQLite (`lexsearch.db`) | Zero ops; migrate to Postgres at >100 concurrent users |

---

## 4. What it can do today (user-visible surface)

| User says | What happens |
|---|---|
| "Limitation period for §138 NI Act?" | BM25 → 6 hits → Gemini drafts a 7-section memo with `[n]` citations |
| "Draft a 2-clause NDA between Acme and Kyoto Holdings" | Skips retrieval, drafts a 4,400-char NDA in ~6s |
| "Latest 2025 SC ruling on PMLA bail" | Web mode — Tavily / Serper / Gemini grounded search → snippet-cited answer |
| "Find me HK Court of First Instance defamation cases" | Court Search → BM25 over the 882-doc HK corpus → ranked cards |
| "Use this case in my chat" | Seeds a fresh thread with the case as context |
| "Translate the answer into Tamil" | Same answer, reply language flipped → Sarvam translates the markdown post-hoc |
| "Save this as a Google Doc" | If Google connected → opens in Docs; else → downloads as `.md` |
| "Email this to my client" | Opens mail client with answer pre-filled |
| "Redline this NDA" | Workflows pane → returns JSON of {remove, replace, add} suggestions |
| "Add my Indian Kanoon API key" | Settings pane → DB-backed keystore, used immediately on next request |

---

## 5. What we can build next (12-week roadmap)

### Phase 1 — finish the corpus (week 1-3)
- **Ingest the rest of the 6 GitHub sources** (~110K docs total):
  - `india_vanga_hc` — 80K Indian HC judgments (parquet)
  - `india_openjustice` — 15K openjustice-in/* dumps
  - `sg_lacuna` — 8K SG cases (lacuna-db)
  - `sg_codelah` — 1K SG mixed (codelah/singapore)
  - `hk_ylchan_list` — 1K active HK case list
- **Nightly cron** — `scripts/ingest_github_data.py --all --top-up` via Render Scheduler.

### Phase 2 — semantic recall (week 3-5)
- Pipe each Document through Gemini's `text-embedding-004` (768-dim) on ingest.
- Store vectors in **Qdrant Cloud** (free tier: 1GB, ~250K vectors).
- **Hybrid retrieval** — BM25 + cosine similarity → Reciprocal Rank Fusion (k=60). Lifts recall on synonym-heavy queries.

### Phase 3 — productisation (week 5-8)
- **Multi-tenant** — strip the single-org assumption; per-org library, vault, settings; row-level isolation in SQLite/Postgres.
- **Stripe billing** — usage-based pricing (per-query-tier or seat). Free tier capped at 50 queries/day.
- **Postgres migration** — drop SQLite. `auth.py` is already DB-agnostic; only schema migration needed.
- **Org admin pane** — invite users, set role (admin/member), see usage dashboard.

### Phase 4 — agent muscle (week 8-10)
- **Agent clarifies before drafting** — "Drafting a road-construction contract — which jurisdiction? Indian buyer or Indian seller? Is this for a public PWD tender or private?" Then runs retrieval + drafts.
- **Multi-document agent** — given 5 contracts in Vault, "Find every clause that conflicts with our standard NDA template" → returns line-anchored diff list.
- **Streaming** — switch `/api/brief/chat` to Server-Sent Events so token-by-token UI works.

### Phase 5 — distribution (week 10-12)
- **NyayaSathi → Sanhita redirect** — once a NyayaSathi WhatsApp/voice user clicks "Open in Sanhita", land them in the right thread with their conversation pre-filled.
- **Mobile responsive polish** — current pane works on tablet but the rail drawer is awkward on phone.
- **Custom domain** + email outbound from `noreply@sanhita.ai` (postmark / resend).

---

## 6. What's needed (infra, accounts, data)

### Required to ship the v1
| Item | Source | Cost (per month) |
|---|---|---|
| **Gemini API key** (you already have one) | https://aistudio.google.com | $0 free tier ≤ 1500 RPM |
| **Hosting — backend** | Render / Railway / Fly | $7-25 |
| **Hosting — frontend** | Vercel hobby / Netlify | $0 |
| **Domain** | sanhita.ai | $50/year ($4/mo) |
| **TLS / DNS** | Cloudflare free | $0 |
| **Persistent disk for `bm25.pkl`** | Render disk 5GB | $1.50 |
| **SQLite → start with Render disk** | Same as above | included |

### Required for serious traction (week 4+)
| Item | Source | Cost (per month) |
|---|---|---|
| **Postgres (managed)** | Neon free tier → paid | $0-19 |
| **Qdrant Cloud** | qdrant.cloud free tier → paid | $0-25 |
| **Tavily web search** (you have a dev key) | tavily.com | $0 (1k/mo) → $30 (10k) |
| **Serper backup** | serper.dev | $0 (2.5k/mo) → $50 (50k) |
| **Sarvam AI** for Indian-language translate | sarvam.ai | usage-priced (see §7) |
| **Indian Kanoon API key** | indiankanoon.org/api | ~₹2000/mo (~$24) for 100k requests |
| **Stripe** for billing | stripe.com | 2.9% + 30¢ per charge |
| **Postmark / Resend** outbound email | resend.com | $0 (100/day) → $20 (50k/mo) |
| **Google Workspace OAuth client** | console.cloud.google.com | $0 |
| **Sentry** for error tracking | sentry.io | $0 (5k events) → $26 |
| **Linear / GitHub Issues** | included | $0 |

### Nice-to-have (optional)
| Item | Why | Cost (per month) |
|---|---|---|
| Anthropic Claude API | Smarter fallback when Gemini struggles | usage-priced |
| Groq API | Sub-second answers; great for chitchat | $0 free tier |
| Render Scheduler | Nightly ingest cron | $0 included |
| Mixpanel / PostHog | Funnel analytics | $0 → $20 |

---

## 7. Cost model — pilot vs scale tiers

### Pilot (0-50 active lawyers, ≤5K queries/month)
| Bucket | Spend |
|---|---|
| Hosting (Render starter + disk + Vercel hobby) | **$15** |
| Domain (annualised) | **$4** |
| Gemini Flash @ 5K queries × ~2K tok in / ~1K tok out × $0.15/1M in / $0.60/1M out | **$3-5** |
| Sarvam translate (occasional) | **$2** |
| Web search (Tavily free tier covers 1K/mo) | **$0** |
| **Total** | **≈$25/month** |

A pilot can run on **$25-30/month** end-to-end. The free tiers of Gemini + Tavily + Vercel cover most of it.

### Scale (500 active lawyers, 50K queries/month)
| Bucket | Spend |
|---|---|
| Hosting (Render Pro + 20GB disk + Vercel Pro) | **$70** |
| Postgres (Neon scale) | **$19** |
| Qdrant Cloud (1GB → paid) | **$25** |
| Domain + SSL | **$4** |
| Gemini @ 50K queries × ~2K tok in / ~1K tok out | **$30-50** |
| Anthropic fallback (≈5% of traffic) | **$15-25** |
| Tavily 10K/mo plan | **$30** |
| Serper backup 50K/mo | **$50** |
| Sarvam translate (5K calls) | **$10-15** |
| Indian Kanoon API | **$24** |
| Resend outbound email | **$20** |
| Sentry | **$26** |
| Stripe fees (assumed 100 paying @ $19/mo = $1900 revenue × 3% + $0.30) | **$87** |
| **Total** | **≈$420-500/month** |

At **$19/seat × 500 lawyers = $9,500/mo revenue**, infra is ≤ 6% — healthy SaaS economics.

### Per-query cost (Gemini Flash 2.5)
- Input tokens: ~2,000 (system prompt + history + retrieval hits) → $0.0003
- Output tokens: ~1,200 → $0.00072
- **Per answer ≈ $0.001** (a tenth of a cent). 1,000 answers = $1.

### Per-query cost (Claude Sonnet 4.5 fallback)
- ~$0.012 per answer (12× more than Gemini). Use only for fallback or when user explicitly picks it from the model dropdown.

---

## 8. Local development

```bash
# 1. Clone
git clone https://github.com/<owner>/sanhita.git
cd sanhita

# 2. Backend deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Frontend deps
cd web && npm install && cd ..

# 4. Configure local env
cp .env.example .env
# Edit .env — at minimum set GEMINI_API_KEY

# 5. Bootstrap the case-law index (one-time, ~3s for HK only)
python3 scripts/ingest_github_data.py --source hk_cuthchow_csv

# 6. Run backend (port 8080)
LEXSEARCH_BM25_ENABLED=true uvicorn server:app --port 8080

# 7. Run frontend (port 3001) — separate terminal
cd web && BACKEND_ORIGIN=http://localhost:8080 npm run dev -- --port 3001

# 8. Open http://localhost:3001/login
#    Demo creds (dev only): demo@sanhita.ai / changeme
```

---

## 9. Deployment notes

### Recommended target: **Render**

- Backend: **Web Service** running `uvicorn server:app --port $PORT --host 0.0.0.0`
- Disk: 5GB attached at `/data` for `bm25.pkl` and `lexsearch.db`
- Build command: `pip install -r requirements.txt`
- Env vars: copy from `.env.example`, fill in real keys via the Render dashboard
- Scheduler: nightly `python scripts/ingest_github_data.py --all --top-up`

Frontend: **Vercel** project pointing to `web/` with `BACKEND_ORIGIN=https://sanhita-backend.onrender.com`.

`render.yaml` and `Procfile` already in repo — both targets need only an account + the keys pasted in.

### Alternatives
- **Railway** — same shape, slightly cheaper at scale
- **Fly.io** — needed if we ever do per-region BM25 sharding for latency
- **AWS Fargate + RDS** — if the user demands SOC 2; ~3× the cost for the same load

---

## 10. Security & data handling

- **No live keys in the repo.** `.claude/launch.json` is gitignored. `.env.example` ships placeholders. Production keys live in Render's encrypted env-var store.
- **Per-user threads** — every `/api/brief/threads/:id` query checks `messages.user_id = current_session.user_id`.
- **PII redaction** — `validators/input_guards.py` strips Aadhaar, PAN, phone numbers before storing the question to SQLite.
- **No raw client data sent to GitHub-ingested datasets.** The corpus is read-only public case law.
- **Google Workspace OAuth** uses the standard offline flow; tokens stored encrypted in `auth.connector_keys`.
- **Rate limits** — chat 30/min, draft 20/min, web 15/min, agent 10/min per IP.
- **Validation gates** — answers fail closed (refusal) if grounding ratio < 60% in research mode.

---

## Quick numbers

| Metric | Value |
|---|---|
| Lines of code (Python + TS) | ~25,000 |
| Components (React panes) | 8 |
| API endpoints | 40+ |
| LLM providers wired | 4 |
| Translation languages | 33 |
| Court cases indexed today | 882 (HK) |
| Court cases indexed at end of Phase 1 | ~110,000 (IN + SG + HK) |
| Time per answer (Gemini Flash, research) | 6-10s |
| Time per answer (Gemini Flash, draft) | 8-15s |
| Cost per answer | ~$0.001 |

---

*Last updated: 2026-04-26.*
