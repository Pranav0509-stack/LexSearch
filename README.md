# Sanhita — Pan-Asia Legal Research Counsel

> AI legal research assistant for advocates in **India, Singapore, and Hong Kong**.
> Grounded answers with case citations · live court-case database (1,135 cases ingested from open GitHub datasets) · drafting · 33 languages · realtime in-house dashboard.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Pranav0509-stack/sanhita)
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https%3A%2F%2Fgithub.com%2FPranav0509-stack%2Fsanhita)

> **Quickest demo path:** click the **Deploy to Render** button → sign in with GitHub → Render reads `render.yaml`, spins up both backend (FastAPI + Socket.io) and frontend (Next.js) services with a 5GB persistent disk for the BM25 index → Render asks you for `GEMINI_API_KEY` → wait ~3 minutes → public HTTPS URL ready. The `BACKEND_ORIGIN` for the frontend is filled in automatically by Render from the backend service.

---

## What Sanhita is

A working chat product for legal research, built on top of a real BM25 case-law engine. Sign in with an access code, ask a legal question, and get back a 7-section memo with `[n]` citations to actual judgments — not a hallucinated summary.

The same brain powers four Harvey-style modes:

| Mode | Toggle | What it does |
|---|---|---|
| **Research** | default | BM25 over the 1,135-case corpus + Gemini synthesises a memo with citations |
| **Draft** | Canvas | Skips retrieval. "Draft an NDA between an Indian SaaS company and a Japanese client" → 4,400-char document in ~6s |
| **Web** | Search | Live web (Tavily → Serper → Gemini grounded → DDG) → snippet-cited answer for current events |
| **Agent** | wand | Tool-using loop: retrieve cases, redline contracts, translate, send Gmail drafts |

Plus 8 dedicated panes in the sidebar: **Assistant · Storage · Workflows · Court Search · Clients · History · Library · Settings · Dashboard**.

## Highlights

- **Court Search pane** — full-text BM25 over 1,135 ingested cases (HK 882 · SG 239 · IN 14), filterable by jurisdiction & tier (SC, HC, CFA, CA, CFI…), with a **Compare cases** flow that hands up to 4 cases to the Assistant for side-by-side analysis.
- **In-house Dashboard** — At-a-glance stats, system health, user list with revoke, and a live **Activity feed** that streams audit-log events over Socket.io. Two admins on the same screen stay in sync without F5.
- **GitHub-ingested case law** — 3 live ingestors (`hk_cuthchow_csv`, `sg_lacuna`, `india_seed_promote`) + 4 documented stubs. Run `python scripts/ingest_github_data.py --all` to bootstrap.
- **33 languages** — All 22 Eighth Schedule Indian languages + Asian markets, routed through Sarvam AI for low-resource scripts and native Gemini for everything else.
- **Action row on every answer** — Copy / Email (mail-client pre-filled) / Save as Doc (Google Docs API → `.md` fallback).
- **DB-adapter** — Auto-routes to **NeonDB Postgres** when `DATABASE_URL` is set, else SQLite. Same code, both modes.
- **Validation gates** — 6-gate validator (citation present, citation resolves, banned phrases, ≥60% grounding, no fabricated names, quote-span match).

## Quick start (local)

```bash
# 1. Clone + install
git clone https://github.com/Pranav0509-stack/sanhita.git
cd sanhita
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
(cd web && npm install)

# 2. Configure
cp .env.example .env
# edit .env — at minimum set GEMINI_API_KEY (free tier OK)

# 3. Bootstrap the case-law index (1-2 minutes)
python3 scripts/ingest_github_data.py --all

# 4. Run backend (port 8080) — the socketio_app target enables realtime
LEXSEARCH_BM25_ENABLED=true uvicorn server:socketio_app --port 8080

# 5. Run frontend (port 3001) — separate terminal
cd web && BACKEND_ORIGIN=http://localhost:8080 npm run dev -- --port 3001

# 6. Open http://localhost:3001/login
#    Demo access code: SNHT-DEMO-2026
```

## Deploy

### Railway (recommended — `railway.toml` is in the repo)

1. **Backend service**: New Project → Deploy from GitHub → `Pranav0509-stack/sanhita`. Railway auto-detects the `Dockerfile`. Add a 5GB disk mounted at `/data` for `bm25.pkl`.
2. **Frontend service**: New Service → same repo → root dir `web/`. Railway picks up `web/railway.toml` and runs Next.js.
3. **Env vars** (Backend service):
   - `GEMINI_API_KEY` — required
   - `LEXSEARCH_BM25_ENABLED=true`
   - `LEXSEARCH_BM25_PATH=/data/bm25.pkl`
   - `DATABASE_URL` — optional; set to a Neon connection string to flip from SQLite → Postgres
   - `TAVILY_API_KEY`, `SERPER_API_KEY` — optional, for web search
4. **Env vars** (Frontend service):
   - `BACKEND_ORIGIN` — set to the backend service's public URL
5. Hit Deploy. Wait ~3 minutes. Visit the frontend URL.
6. (Optional) Add a Railway Cron job: `python scripts/ingest_github_data.py --all --top-up` nightly.

### Render

`render.yaml` + `Procfile` are in the repo. Connect the repo and Render auto-creates the Web Service. Same env-var list as above.

### Vercel (frontend only)

The "Deploy with Vercel" button at the top of this README clones into Vercel pointing at `web/`. Set `BACKEND_ORIGIN` to your Railway/Render backend URL.

## Architecture

```
                              Browser
                                 |
       +-------------------------+--------------------------+
       |           Sanhita Frontend (Next.js 16)            |
       |  /app  --- 9 panes (Assistant, Court Search,       |
       |   Dashboard, Storage, Workflows, Clients,          |
       |   History, Library, Settings)                      |
       +-------------------------+--------------------------+
                                 |  fetch + WebSocket
                                 v
       +----------------------------------------------------+
       |        Sanhita Backend (FastAPI + uvicorn)         |
       |                                                    |
       |  /api/brief/{chat,draft,web,agent}                 |
       |  /api/cases/{search,latest,:id}                    |
       |  /api/dashboard/{stats,users,activity,system}      |
       |  /socket.io/* (realtime fanout)                    |
       |  /api/{vault,workflows,library,settings,…}         |
       +---------+-----------------------+------------------+
                 |                       |
                 v                       v
       +------------------+    +--------------------+
       |  retrieval_pkg   |    |   db_adapter       |
       |   BM25Index      |    |   Postgres OR      |
       |   1,135 cases    |    |   SQLite (auto)    |
       |   IN·SG·HK       |    |                    |
       +--------+---------+    +--------------------+
                |
       +--------+------------+
       |  scripts/ingestors  |
       |  hk_cuthchow_csv    |
       |  sg_lacuna          |
       |  india_seed_*       |
       +---------------------+
```

## Tech stack

| Layer | Choice |
|---|---|
| Frontend | Next.js 16 + Tailwind 4 + Fraunces / Inter |
| Backend | FastAPI + uvicorn |
| Realtime | python-socketio + socket.io-client |
| DB | NeonDB Postgres in prod, SQLite in dev (auto via `db_adapter.py`) |
| LLM | Gemini 2.5 Flash → Anthropic → Groq → Cloudflare (router with circuit breakers) |
| Translation | Sarvam AI `mayura:v1` for Indian languages |
| Retrieval | `rank_bm25` + pickle |
| Web search | Tavily → Serper → Gemini grounded → DuckDuckGo |

## Roadmap

The detailed plan lives in [SANHITA.md](SANHITA.md). Headline phases:

1. **Finish the corpus** — wire up `india_vanga_hc` (80K judgments via AWS S3) and `india_openjustice` to grow from 1,135 → ~110K cases.
2. **Semantic recall** — Gemini text-embeddings → Qdrant Cloud → hybrid BM25 + cosine RRF.
3. **Call surface** — voice / WhatsApp / SMS interface. Self-contained brief at [CALL_SURFACE_PLAN.md](CALL_SURFACE_PLAN.md).
4. **Productisation** — multi-tenant, Stripe billing, org admin pane, OAuth.

## Cost

- **Per answer**: ~$0.001 (Gemini Flash 2.5).
- **Pilot (≤50 lawyers, 5K queries/mo)**: ~$25/month all-in.
- **Scale (500 lawyers, 50K queries/mo)**: ~$420-500/month all-in. At $19/seat, infra is ≤6% of revenue.

Detailed breakdown in [SANHITA.md §8](SANHITA.md#8-cost-model--pilot--scale-tiers).

## Data sources

| Source | Country | License |
|---|---|---|
| [`cuthchow/Hong-Kong-Courts`](https://github.com/cuthchow/Hong-Kong-Courts) | HK | check upstream |
| [`hueyy/lacuna-db`](https://github.com/hueyy/lacuna-db) | SG | check upstream |
| [`vanga/indian-high-court-judgments`](https://github.com/vanga/indian-high-court-judgments) | IN | CC-BY-4.0 |
| [`openjustice-in/ecourts`](https://github.com/openjustice-in/ecourts) | IN | check upstream |

All ingested data is public legal-record material.

## Documents

- [SANHITA.md](SANHITA.md) — master doc: architecture, capabilities, costs, roadmap.
- [CALL_SURFACE_PLAN.md](CALL_SURFACE_PLAN.md) — voice/WhatsApp surface plan, self-contained for a fresh session.
- [Sanhita.pdf](Sanhita.pdf) — combined PDF, exec-ready.

## License

Source: TBD (currently private). Ingested case-law data: per upstream licenses (CC-BY-4.0 for India HC, others vary).

## Demo

Demo access code: `SNHT-DEMO-2026` (works on local dev; wipe / change before production).

Live demo URL: _add after Railway deploy completes_.
