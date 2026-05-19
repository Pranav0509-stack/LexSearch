# Sanhita — Deployment Runbook

Last revision: 2026-05-17

## Architecture (one diagram)

```
                    ┌─────────────────────────────────────────┐
                    │           Internet / Lawyers            │
                    └────────────────────┬────────────────────┘
                                         │ HTTPS
                    ┌────────────────────▼────────────────────┐
                    │   sanhita.ai  (Cloudflare / Vercel)     │
                    │   Static — Next.js export (`npm run     │
                    │   build`) — 24 pages, 12 plugin SSGs    │
                    └────────────┬────────────────────────────┘
                                 │ /api/* rewrite (next.config.ts)
                    ┌────────────▼────────────────────────────┐
                    │   FastAPI on :8080  (server.py)         │
                    │   120 routes · 7 routers                │
                    │     /api/contract/*    (Drafter)        │
                    │     /api/cases/*       (Court Search)   │
                    │     /api/legal-aid/*   (Intake)         │
                    │     /api/brief/*       (Assistant)      │
                    │     /api/vault/*       (Files)          │
                    │     /api/editor/*      (Doc editor)     │
                    │     /pdf/* /doc-pdf/*  (PDF proxy)      │
                    └────────────┬────────────────────────────┘
                                 │ SQLite (file)
                    ┌────────────▼────────────────────────────┐
                    │  india_courts.db  (~360 GB after FTS)   │
                    │  16.9M judgments · 53.3M pipeline_docs  │
                    │  11.6M legal_docs · 1.3M Q&A · 2.3K     │
                    │  statutes · 2K documents · 26 templates │
                    │  FAISS index_meta (40K vectors)         │
                    └─────────────────────────────────────────┘
```

## Prerequisites

| Item                       | Version / detail                                  |
|----------------------------|---------------------------------------------------|
| Python                     | 3.13+                                             |
| Node                       | 18+ (for the Next.js frontend)                    |
| SQLite                     | 3.40+ (must support FTS5 and `pragma cache_size`) |
| Disk                       | 400 GB free (DB + FAISS + S3 PDF cache)          |
| Memory                     | 16 GB minimum (FAISS load + FTS5 buffers)        |
| AWS S3                     | Read-only on `indian-high-court-judgments` bucket |
| GEMINI_API_KEY             | required for AI scope (Polish / Cite / Summary)   |

## Environment variables

| Env var                 | Purpose                                | Default                                                          |
|------------------------|----------------------------------------|------------------------------------------------------------------|
| `SANHITA_BACKEND`       | Backend URL for MCP / Word add-in     | `http://localhost:8080`                                          |
| `GEMINI_API_KEY`        | Gemini Flash for Drafter polish / cite | none — drafter falls back to deterministic-only without it       |
| `ANTHROPIC_API_KEY`     | Optional fallback LLM                  | none                                                             |
| `BACKEND_ORIGIN`        | Next.js rewrite target                 | `http://localhost:8080`                                          |
| `_HTTPX_VERIFY`         | TLS verify for S3 fetch                | `True`                                                           |
| `SANHITA_INTAKE_SECRET` | HMAC for NyayaSathi handoff           | none                                                             |

## First-boot sequence (clean machine)

```bash
# 1. Clone + Python deps
git clone <repo> sanhita && cd sanhita
pip install -r requirements.txt

# 2. Place the corpus DB at the expected path
#    (~360 GB; ship via tarball + checksum to a mounted volume)
ls -la india-judgments-corpus/india_courts.db

# 3. Verify FTS5 + FAISS are intact
sqlite3 india-judgments-corpus/india_courts.db \
    "SELECT count(*) FROM pipeline_docs_fts; SELECT count(*) FROM legal_docs_fts;"

# 4. Frontend deps + build
cd "sanhita-react/web"
npm install
npm run build          # produces .next/ — 24 routes, 12 plugin SSGs

# 5. Boot both services (see start.sh below)
./deploy/start.sh
```

## Production start (systemd / pm2 / kubectl — pick one)

### Local-machine demo (single-host)

```bash
# Backend (FastAPI on :8080)
nohup python3 -m uvicorn server:app --host 0.0.0.0 --port 8080 \
    --workers 1 \
    > /var/log/sanhita-backend.log 2>&1 &

# Frontend (Next.js standalone on :3000)
cd sanhita-react/web
nohup npm run start > /var/log/sanhita-frontend.log 2>&1 &

# MCP stdio server is spawned by Claude Desktop / Claude Code on demand
# (see mcp_server/README.md for `claude_desktop_config.json` snippet)
```

### Kill-restart script

```bash
#!/usr/bin/env bash
# deploy/kill-restart.sh
pgrep -f "uvicorn server"      | xargs -r kill -9 2>/dev/null
pgrep -f "next-server"          | xargs -r kill -9 2>/dev/null
pgrep -f "node .*next/dist"     | xargs -r kill -9 2>/dev/null
sleep 2
cd "/Users/pranav/Desktop/LexSearch-main 2"
nohup python3 -m uvicorn server:app --host 0.0.0.0 --port 8080 > /tmp/uvicorn.log 2>&1 &
cd sanhita-react/web
nohup npm run start > /tmp/next.log 2>&1 &
echo "restarted; logs in /tmp/{uvicorn,next}.log"
```

## Healthcheck endpoints

| Path                                              | Expected                                            |
|---------------------------------------------------|-----------------------------------------------------|
| `GET  /api/contract/health`                       | `{"status":"ok","templates":26,"clauses":143,…}`    |
| `GET  /api/cases/engine-status`                   | `{"engine_available":true,"semantic_available":…}`  |
| `GET  /api/cases/doc-types`                       | array of 5 doc-type entries (200)                   |
| `POST /api/cases/smart-search`                    | `{"hits":[…]}` ≤ 300 ms warm                        |
| `POST /api/contract/draft`                        | `{"draft_id":"d_…","word_count":…}` ≤ 100 ms        |
| `GET  /pdf/<s3 key>`                              | `Content-Type: application/pdf · inline`            |

A passing health probe:

```bash
curl -sf http://localhost:8080/api/contract/health > /dev/null && \
curl -sf http://localhost:8080/api/cases/engine-status > /dev/null && \
curl -sf -X POST http://localhost:8080/api/cases/smart-search \
    -H 'Content-Type: application/json' \
    -d '{"q":"section 138","mode":"keyword","limit":1}' > /dev/null && \
echo "Sanhita: green"
```

## Smoke test — full

```bash
./deploy/smoke.sh
```

See `deploy/smoke.sh` (added in this commit) — fires:
1. contract/health
2. cases/engine-status
3. smart-search (keyword / semantic / hybrid)
4. document fetch (legal_docs + judgments + pipeline_docs)
5. draft generation (NDA)
6. compliance check
7. legal-aid application
8. frontend `/`, `/app`, `/plugins`, `/plugins/litigation`, `/legal-aid`

Exits 0 on green, non-zero on any failure.

## What the Drafter / Court Search depend on

- **`india_courts.db`** — single SQLite file. WAL mode. Must be on a fast-IO mount.
- **FTS5 virtual tables** — `judgments_fts`, `legal_docs_fts`, `pipeline_docs_fts`, `statutes_fts`, `legal_qa_fts`, `documents_fts`. Built once via the parquet bulk ingest; do not drop.
- **FAISS index** — `index_meta` table (40K vectors over legal_docs subset). Loaded lazily on first semantic query.
- **26 templates** seeded from `scripts/contract/templates/*.md` via `scripts/contract/seed_templates.py`.
- **S3 read access** to `indian-high-court-judgments` (HC PDFs) + `indian-supreme-court-judgments` (SC PDFs) — public, no auth needed.

## Common pitfalls

1. **WAL file balloons during batch UPDATEs** — if a long-running script blocks the writer, `*.db-wal` can grow to 30+ GB. Mitigation: `PRAGMA wal_checkpoint(TRUNCATE)` post-batch. Server restart triggers automatic checkpoint.
2. **First semantic query is slow** — FAISS loads lazily (≈ 4 s). Hit `/api/cases/engine-status` once at boot to warm.
3. **PDF iframe shows black in headless browsers** — Chrome's PDF plugin isn't initialised in headless mode; this is a known limitation of automated screenshots. Real users in Chrome / Firefox / Safari see the PDF page rendered. The full-text fallback below the PDF ensures content is always visible.
4. **Legacy `/api/cases/search` first-hit slow** — BM25 index lazy-load. Subsequent hits < 300 ms.

## Roll-back procedure

```bash
# Stop services
pgrep -f "uvicorn server"  | xargs -r kill
pgrep -f "next-server"      | xargs -r kill

# Restore DB from latest pre-deploy snapshot
cp /backup/india_courts.db.YYYYMMDD-HHMMSS /Users/pranav/Desktop/india-judgments-corpus/india_courts.db

# Boot prior commit
git checkout <prior-sha>
./deploy/kill-restart.sh
```

## Monitoring / observability (to wire in production)

- Uptime: ping `/api/contract/health` every 30 s; page on 3 consecutive failures
- Latency P95: smart-search > 1 s sustained = degraded
- Disk: alarm if `india_courts.db` mount > 90 % full
- S3 cost: track GET requests on the HC bucket; ITAT cache should be < 1 % of total

## Launch checklist

- [x] Backend `/api/contract/health` green
- [x] `/api/cases/engine-status` reports `engine_available + semantic_available`
- [x] All three search modes (keyword / semantic / hybrid) return hits
- [x] Side panel opens with real SC 2025 judgment, full text auto-loads, PDF button present
- [x] Drafter NDA generates 1,903 words deterministically (< 100 ms)
- [x] Plugin pages prebuilt: 12 routes × `/plugins/[slug]`
- [x] Legal-Aid intake form submits and creates `laa_*` reference
- [x] Production `next build` succeeds — 24 static pages
- [x] `pipeline_docs_fts` (53.3M rows) indexed and queryable
- [x] DEPLOY.md + deploy/smoke.sh in repo
- [ ] DNS record `sanhita.ai` → frontend CDN
- [ ] DNS record `api.sanhita.ai` → backend behind a TLS reverse-proxy
- [ ] Database snapshot every 6 hours to S3 (off-site backup)
- [ ] Sentry / OpenTelemetry wiring (post-pilot)
