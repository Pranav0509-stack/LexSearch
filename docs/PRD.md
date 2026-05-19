# Sanhita AI — Product Requirements Document (PRD)

**Status:** Living draft v0.1
**Owner:** [Founder]
**Last updated:** 2026-05-08
**Audience:** Founders, engineers, design, GTM, advisors, prospective investors

> 📌 Import to Google Docs: File → Open → Upload → drop this `.md`. Or copy-paste — formatting carries.

---

## 1. Vision

> **Make the last hour of an Indian lawyer's or CA's research the first 60 seconds.**

Sanhita is the AI-native research, drafting, and citator platform for Indian legal and tax practice. One database, two specialised lenses (Lawyer / CA), priced for solo + small firms — same coverage as Manupatra and Taxmann combined, at one-tenth the cost, with capabilities (citator, headnotes, PDF chat, real-time circular feed) that incumbents structurally cannot ship because they're built on 1990s tech.

---

## 2. Mission (12-month version)

By **May 2027**, Sanhita is:

- The **default research tool for ≥1,000 Indian solo / small-firm CAs and lawyers**
- The **largest indexed corpus** of Indian judgments + tribunal orders + circulars on the public internet (≥30M docs)
- **Cited by users** in scrutiny replies, appeal grounds, writ petitions, and opinions — with our paragraph-level citations
- A **real revenue business**: ≥₹2 Cr ARR at gross margin ≥80%

If we hit those, we're a defensible legal-tech platform. If we miss, we re-strategise.

---

## 3. Problem

### 3.1 Who has the problem

Indian legal and tax professionals — primarily:

| Segment | Approx. count in India | Why they hurt |
|---|---|---|
| **Practising CAs (solo + small firm)** | ~150K of 400K total | Pay ₹50K–₹2L/yr for Taxmann + Manupatra + CCH; can't afford the full stack alone |
| **Tax professionals (non-CA)** | ~50K | Same tools, same cost burden |
| **Litigating advocates (solo + small firm)** | ~1.5M total advocates; ~500K active research users | Manupatra/SCC are out-of-reach; rely on Indian Kanoon (free, no enrichment) |
| **In-house counsel (corporate)** | ~50K | Use enterprise contracts; pain is fragmentation across tools |
| **Articleship trainees / juniors** | ~200K | Do most actual research; have no power to choose tools |

**Initial wedge:** solo and 2–10 person firms across CA + lawyer practices. We expand to enterprise after we've earned the right.

### 3.2 What the problem actually is

Indian legal/tax research today suffers from five structural failures:

1. **Coverage fragmentation.** No single tool gives you HC + SC + ITAT + NCLT + CBDT + CBIC + India Code. Each portal is its own walled garden. Even within one tool, tribunal coverage is patchy.
2. **No citator on free tools.** Indian Kanoon (the free incumbent) has no "is this still good law?" signal. Manupatra has it but costs ₹50K+/year per seat.
3. **No semantic / AI search.** All tools are 1990s keyword search. Users have to know the exact phrase to find anything.
4. **No real-time notification firehose.** CBDT issues a clarification at 6 PM Friday — solo CAs find out from a WhatsApp group on Monday morning.
5. **Hostile drafting workflow.** Scrutiny replies, appeals, briefs all take 4–8 hours of copy-paste. Templates exist as Word files emailed between juniors. Nothing connects research to drafting.

### 3.3 Evidence

- **Manupatra** charges ~₹50K–₹1L/yr per seat, has ~3M judgments. Mid-tier firms (10+ partners) pay it.
- **Taxmann** charges ~₹40K–₹60K/yr per seat for tax-specific research; market leader for CAs.
- **SCC Online** charges ~₹50K/yr for SC + HC + tribunal coverage with editorial enrichment.
- **Indian Kanoon** is free, has 5M+ judgments, but no headnote, no citator, no enrichment. ~70% of practising lawyers use it as a daily tool.
- **CaseMine** — AI-search startup, ~₹15K/yr, ~5M judgments, no tax-specific tribunal depth.
- **Total addressable annual spend** on Indian legal/tax research tools (combined firms ≥10 seats): ~₹1,500 Cr.
- **Solo + small-firm spend** today: most spend ₹0 (Indian Kanoon) or ₹40K–₹50K (one tool only). The ₹999–₹2,500/seat/mo SaaS sweet spot is unserved.

---

## 4. Users + Personas

### 4.1 Persona A — "Priya, the solo CA"

- **Role:** Solo or 2-person firm CA, 5–15 years post-qualification
- **Practice:** Mostly tax compliance, scrutiny reply drafting, occasional ITAT appeal
- **Tools today:** ICAI material + Indian Kanoon + WhatsApp groups + occasional Taxmann access borrowed from a senior
- **Annual tool spend:** ₹0 to ₹40K
- **Pain:** Loses 6 hours per scrutiny reply hunting for the right ITAT precedent
- **What she wants:** Fast tribunal search, citator that flags overruled cases, draft generator
- **What she doesn't want:** A ₹1L bill or a tool that asks her to learn boolean operators
- **Conversion trigger:** First time she finds an ITAT order in 30 seconds that took her 3 hours on Indian Kanoon

### 4.2 Persona B — "Rahul, the litigating advocate"

- **Role:** Solo or junior in a 5–10 person litigation chamber
- **Practice:** Civil writ + criminal bail + commercial disputes in HC
- **Tools today:** Indian Kanoon (daily) + ChatGPT (sometimes for boilerplate) + Manupatra in chambers (limited access)
- **Annual tool spend:** ₹0–₹25K personal
- **Pain:** Cites a case, opposing counsel says it's been overruled, judge raps him
- **What he wants:** Citator with overruled flag, judge analytics ("how does Justice X decide bail under Section 37 NDPS?")
- **What he doesn't want:** A tool that takes 30 minutes to learn before a hearing tomorrow
- **Conversion trigger:** A live demo where Sanhita finds a case Indian Kanoon couldn't, with a citator showing it's still good law

### 4.3 Persona C — "Mehta & Co., 8-partner firm"

- **Role:** Mixed CA + tax-litigation firm, 30 staff, articles, juniors, partners
- **Tools today:** Taxmann (firm seats) + Manupatra (2 seats) + CCH (1 seat); ~₹2L/yr total
- **Pain:** Articles take 4 hours on research the partner could finish in 15 minutes; firm wants to scale juniors faster
- **What they want:** Firm-wide seats, audit trail, training-time reduction, lower bill
- **What they don't want:** A startup that disappears in 18 months
- **Conversion trigger:** A 30-day pilot with 3 articles where output quality matches the partner's work

### 4.4 Persona D — "the article / junior"

- The actual user. Persona A/B/C is the buyer. The article does the work.
- They will adopt anything that makes them look smart in front of the partner without effort.
- We must be lovable to them within 5 minutes of first use.

---

## 5. Goals + Success Metrics

### 5.1 90-day goals (post-PRD-finalisation)

| Goal | Metric | Target |
|---|---|---|
| **Coverage** | Total searchable docs | ≥25M |
| **Tribunals** | Tribunal sources live | ≥4 (NCLAT done; ITAT, NCLT, CESTAT next) |
| **Citator** | Citation edges | ≥10M |
| **Headnotes** | Cases with auto-headnote | ≥100K (top-cited first) |
| **Discovery** | Customer interviews completed | ≥30 |
| **Trial** | Free-trial signups | ≥200 |
| **Pilot** | Active paying pilots (any size) | ≥10 |
| **Revenue** | MRR | ≥₹50K |
| **NPS** | NPS from active trial users | ≥35 |

### 5.2 12-month north-star metrics

| Metric | 12-month target |
|---|---|
| Paying users | ≥1,000 |
| ARR | ≥₹2 Cr |
| Net revenue retention | ≥110% |
| Weekly active users / total | ≥60% |
| Average searches per WAU per week | ≥40 |
| % of users using citator monthly | ≥80% |
| % of users using PDF chat monthly | ≥50% |
| Mean response latency (search) | <100 ms p50 |
| Mean response latency (PDF chat) | <3 s p50 |

### 5.3 Anti-goals (things we are explicitly NOT optimising for)

- Maximising daily active users via gamification or notifications
- Adding social / community features (we are a tool, not a forum)
- Replacing the tax-filing software stack (Tally/Winman) — we complement, not compete
- Selling case data to anyone (privacy is a moat, not a revenue line)
- Free-forever consumer model (we are a B2B SaaS)

---

## 6. Scope

### 6.1 In scope (what Sanhita IS)

- Search engine over judgments + statutes + circulars + tribunals
- Citator (cited-by, cites, overruled, distinguished, PageRank)
- Auto-generated headnotes via LLM
- PDF reader + paragraph anchors
- PDF-to-chat (vault) — drop a document, ask questions, get cited answers
- Drafting assistant (scrutiny replies, appeals, briefs) — Phase 3
- Multi-language interface (English + Hindi at launch; 11 Indian languages later)
- Voice channel (NyayaSathi) for citizens, distinct product but shared corpus
- Two specialised views: Lawyer mode + CA mode (same engine, different defaults / templates / alerts)

### 6.2 Out of scope (for v1)

- Tax filing / GST returns / TDS reconciliation (existing market is well-served)
- Court fee / e-filing automation
- Practice management (matter tracking, time billing, client invoicing) — Phase 4 candidate
- Case docket scraping for live court status (separate product, future)
- International law beyond what's cited by Indian courts

---

## 7. Product surface (current state + roadmap)

### 7.1 What's live today (v0)

| Capability | State |
|---|---|
| **Search** — FTS5 across 4 tables (judgments, legal_docs, statutes, legal_qa) + new documents | ✅ |
| **Intent router** — classifies query as FIND_CASE / STATUTE / QA / DOCUMENT | ✅ |
| **Live suggestions** — Google-style autocomplete | ✅ |
| **Court Search Pane** — judgment-mode / documents-mode toggle, filters | ✅ |
| **PDF viewer + full-text reader** — auto-detect availability, both modes | ✅ |
| **PDF-to-chat (vault)** — drop a file in chat, ask grounded questions | ✅ |
| **HC judgment PDFs** — 16.8M, all reachable via S3 proxy | ✅ |
| **India Code** — central + state acts being ingested | ✅ |
| **Documents corpus** — SEBI, RBI, PRS, NCLAT live | ✅ |
| **Citation extractor** — regex bank for SCC/AIR/SCR/SCC OnLine + sections + articles | ✅ |
| **Citator** — cited-by, cites, edge types (followed/distinguished/overruled), PageRank | ✅ |
| **Headnotes** — Gemini-driven JSON-mode generator + FTS index | ✅ |
| **Citations API** — `/api/cases/{id}/citations` | ✅ |
| **Citator UI panel** — Cited-by / Cites tabs on case detail | ✅ |
| **NyayaSathi voice** — separate citizen-facing product, sibling repo | ✅ |

### 7.2 Phase 1 — Coverage parity (next 4–6 weeks)

| Capability | State |
|---|---|
| **SC judgments** (~120K) | 🔜 |
| **ITAT orders** (~400K) | 🔜 — needs AJAX RE |
| **NCLT orders** (~150K, 16 benches) | 🔜 |
| **CESTAT orders** (~80K) | 🔜 |
| **NGT orders** (~15K) | 🔜 |
| **SAT, AFT, CCI, TDSAT** (small but specialised) | 🔜 |
| **NCDRC consumer** (~250K) | 🔜 |
| **CBDT, CBIC notifications** (full backfill) | 🔜 |
| **eGazette** (5M+, requires Playwright session work) | 🔜 |
| **Citation extraction over judgments full_text** (when populated) | 🔜 |
| **OCR pipeline** for scanned tribunal PDFs | 🔜 |

### 7.3 Phase 2 — Editorial moat (concurrent with Phase 1)

| Capability | State |
|---|---|
| **Headnote backfill** — top-cited 100K cases | 🔜 |
| **Editorial tags** — multi-label classifier on point-of-law | 🔜 |
| **Conflicting-precedent detector** — surface HCs that diverge | 🔜 |
| **Paragraph-level anchor links** in PDFs | 🔜 |
| **Treatment classifier** — auto-tag followed / distinguished / overruled with higher recall | 🔜 |

### 7.4 Phase 3 — Drafting assistant

| Capability | State |
|---|---|
| **Scrutiny-reply drafter** (CA killer feature) | 🔜 |
| **Appeal-grounds generator** (post-scrutiny outcome) | 🔜 |
| **Writ-petition skeleton** (lawyer side) | 🔜 |
| **Template library** (~30 starting templates per practice area) | 🔜 |
| **Brief writer** — fact pattern → argument structure with cites | 🔜 |
| **Side-by-side judgment compare** | 🔜 |

### 7.5 Phase 4 — Practice infrastructure

| Capability | State |
|---|---|
| **Saved searches + alerts** (email + WhatsApp) | 🔜 |
| **Notification firehose with smart filters** | 🔜 |
| **Matter / client folders + collaboration** | 🔜 |
| **Audit trail** — who searched what, when (firm tier) | 🔜 |
| **API access** (enterprise tier) | 🔜 |
| **Mobile app** (iOS + Android) | 🔜 |
| **Voice search** (English + Hindi + 9 Indian languages) | 🔜 |

### 7.6 Phase 5 — Data moat

| Capability | State |
|---|---|
| **Old reports digitization** (AIR, SCR, SCC, ITR, GLR pre-2000) | 🔜 long-term |
| **Daily orders** (HC + district court) | 🔜 |
| **Cause lists** (next-day hearing schedules) | 🔜 |
| **Bench / judge analytics** (full coverage) | 🔜 |
| **Embeddings (BGE) for semantic search** | 🔜 |

---

## 8. Architecture (one-page view)

```
┌──────────────────────── USER LAYER ────────────────────────┐
│  Web (Next.js)   Mobile (later)   Voice (NyayaSathi)       │
└────────────────────────┬───────────────────────────────────┘
                         │
┌────────────────────── API LAYER ──────────────────────────┐
│  FastAPI server (server.py)                               │
│  /api/cases/search, /api/cases/{id}, /api/cases/{id}/text │
│  /api/cases/{id}/citations, /pdf/*, /doc-pdf/*            │
│  /api/vault/{upload, chat}, /api/auth/*                   │
└────────────────────────┬───────────────────────────────────┘
                         │
┌──────────────── INTELLIGENCE LAYER ───────────────────────┐
│  query_router (intent classifier)                         │
│  sanhita_adapter (FTS5 + dedup + citator-aware ranking)   │
│  citation_extractor / citator_stats / pagerank            │
│  headnote_generator (Gemini)                              │
│  llm/router (Gemini → Claude → Groq → Cloudflare cascade) │
└────────────────────────┬───────────────────────────────────┘
                         │
┌──────────────────── DATA LAYER ────────────────────────────┐
│  india_courts.db (SQLite + WAL)  — ~80GB                   │
│  Tables:                                                   │
│   judgments (16.8M) • legal_docs (11.5M) • legal_qa (1.3M)│
│   statutes (2.3K) • documents (~2K, growing) • district  │
│   citation_edges • citator_stats • enrichments • runs    │
│  FTS5 indexes per table + auto-sync triggers              │
│  S3: indian-high-court-judgments bucket (PDFs)            │
└─────────────────────────────────────────────────────────────┘

┌─────────────── INGESTION + ENRICHMENT ────────────────────┐
│  scripts/ingest/* — sebi, rbi, prs, indiacode, nclat,     │
│                    egazette (deferred), …                 │
│  scripts/enrich/* — citation_extractor, citator_stats,    │
│                    headnote, tagger                       │
│  Orchestrator: run_all.py with checkpoints + retry        │
└─────────────────────────────────────────────────────────────┘
```

### 8.1 Why SQLite (not Postgres)

- 80GB single-file DB is portable and trivially backed up
- FTS5 on SQLite is faster than Postgres + pgvector for our access pattern (<50ms p50)
- WAL + cache_size tuning gives us multi-process read concurrency
- We migrate to Postgres only when (a) write concurrency exceeds 10 ingest workers, or (b) we need pgvector for embeddings at scale

### 8.2 Why Gemini-2.0-flash for headnotes

- JSON-mode reliable output (validated)
- ₹0.20 per case at 30K input chars + 8K output tokens
- Anthropic Claude is fallback (better quality, 3× cost)
- Local Llama-3-70B is the cost-floor option for hot cases

### 8.3 Privacy + security posture

- Vault uploads encrypted at rest
- No client data ever leaves our infra to LLM providers without explicit user opt-in
- Per-user audit log of every query (firm tier)
- SOC2 Type II target by month 9

---

## 9. Pricing (working hypothesis — validate via discovery)

| Tier | Price | What's included |
|---|---|---|
| **Free** | ₹0 | Search across HC + statutes; 10 citator lookups/day; no PDF chat |
| **Solo** | ₹999/seat/mo | All tribunals + circulars + unlimited citator + PDF chat (10 docs/mo) + headnotes |
| **Firm (5–25 seats)** | ₹2,499/seat/mo | + PDF chat unlimited + saved searches + alerts + audit log |
| **Firm Plus (25+ seats)** | ₹4,999/seat/mo | + API + custom templates + dedicated support + SOC2 reports |
| **Enterprise** | quote | + SSO + on-prem option + custom SLA |

Annual: 2 months free.

**Comparison anchor:** Manupatra ₹50K–₹1L/seat/yr → ours at ₹12K–₹30K/seat/yr.

---

## 10. Go-to-market (at a glance)

### 10.1 Channels (in order)

1. **Founder-led discovery + sales** (next 90 days). 30 conversations → 10 pilots → 3 paying.
2. **CA & lawyer associations** — speak at ICAI study circles, Bar Councils, Tax Bar Associations.
3. **Content / SEO** — every search query that fails on Indian Kanoon is a future SEO landing page.
4. **Word of mouth** — every active user gets a "share with peer for 1 free month" link.
5. **LinkedIn organic** — founder posting, building "Indian legal tech" presence (already 5K likes / 220K impressions on LinkedIn launch — leverage).

### 10.2 Anti-channels (won't waste budget here yet)

- Paid Google Ads — CPC for "Manupatra alternative" is ₹40+; ROI poor at our stage
- Conferences (₹2L for a booth that yields 5 leads)
- Direct mail / cold email blasts (Indian legal market is relationship-driven)

### 10.3 Pilot motion

- 30-day free trial → 30-day paid pilot at 50% off → annual contract
- Every trial gets a personal onboarding call (founder-led)
- Every pilot gets a weekly check-in for the first month

---

## 11. Risks + mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Manupatra/Taxmann drop prices** | Med | High | Move fast on coverage and AI features they can't ship; build network effects (citator depth) |
| **AI hallucination → user cites wrong case** | High | Critical | Every AI claim links to source paragraph; refuse to answer if no source; user-visible "no answer found" never auto-generated |
| **Site changes break scrapers** | High | Med | Resume + retry + checkpoint design; 1 person-week/month maintenance budget |
| **Cost of LLM per headnote** | Med | Med | Batch enrich top 100K only; long tail uses cheaper models or extractive summarization |
| **Storage cost at 200GB+** | Low | Low | SQLite single-file; S3 cold tier for PDFs (already there) |
| **Compliance / data privacy challenge** | Med | High | DPDP Act compliance from day 1; no client-data egress; SOC2 in flight |
| **Bar Council / ICAI restrictions on tools** | Low | Med | Stay clearly research-only; never claim to replace professional judgment |
| **Founder fatigue / scope creep** | High | High | 90-day plans with hard boundaries; this PRD is the source of truth |

---

## 12. Open questions (unresolved — to validate via discovery)

> Each open question MUST be resolved by interview data, not assumption. Tag closures with the meeting # that resolved them.

| # | Question | Owner | Status |
|---|---|---|---|
| Q1 | Do CAs prefer ITAT-first or HC-first search defaults? | Founder | Open |
| Q2 | What's the killer feature: scrutiny drafter or notification firehose? | Founder | Open |
| Q3 | Will firms pay ₹2,500/seat/mo or top out at ₹1,500? | Founder | Open |
| Q4 | Do lawyers actually use citator daily or only before hearings? | Founder | Open |
| Q5 | Is voice / WhatsApp interface a "must" or a "nice"? | Founder | Open |
| Q6 | What % of Indian Kanoon users would move for a citator + ₹0 entry tier? | Founder | Open |
| Q7 | Do firms want a per-firm contract or per-seat self-serve? | Founder | Open |
| Q8 | Is OCR'd-PDF search a top-3 pain or top-10? | Founder | Open |
| Q9 | Would CAs trust AI-drafted scrutiny replies as final, first draft, or never? | Founder | Open |
| Q10 | How many states' AAR rulings does a typical GST practitioner need? | Founder | Open |

---

## 13. Glossary (keep current)

| Term | Meaning |
|---|---|
| **AAR** | Authority for Advance Ruling (state + central) |
| **AAAR** | Appellate AAR |
| **AY** | Assessment Year |
| **BNS / BNSS** | Bharatiya Nyaya Sanhita / Bharatiya Nagarik Suraksha Sanhita (post-2024 successors to IPC / CrPC) |
| **CA** | Chartered Accountant (ICAI) |
| **CASS** | Computer-Aided Scrutiny Selection (income tax) |
| **CBDT** | Central Board of Direct Taxes |
| **CBIC** | Central Board of Indirect Taxes & Customs |
| **CIT(A)** | Commissioner of Income Tax (Appeals) |
| **CESTAT** | Customs Excise & Service Tax Appellate Tribunal |
| **Citator** | A tool that shows whether a judgment has been followed / distinguished / overruled |
| **DPDP Act** | Digital Personal Data Protection Act (2023) |
| **FTS5** | SQLite Full-Text Search v5 — our primary search engine |
| **GST** | Goods & Services Tax |
| **Headnote** | Short editorial summary of a judgment (200 words typical) |
| **IBC** | Insolvency & Bankruptcy Code |
| **ICAI** | Institute of Chartered Accountants of India |
| **ITAT** | Income Tax Appellate Tribunal (63 benches) |
| **NCLT / NCLAT** | National Company Law Tribunal / Appellate Tribunal |
| **NCDRC** | National Consumer Disputes Redressal Commission |
| **PageRank** | Random-walk graph algorithm; here, applied to citation graph to surface landmark cases |
| **PRD** | This document |
| **SEBI / RBI / MCA** | Securities and Exchange Board of India / Reserve Bank / Ministry of Corporate Affairs |
| **TP** | Transfer Pricing |
| **Vault** | The user's private document store (uploaded PDFs); also the chat surface over those docs |

---

## 14. Changelog

| Version | Date | Author | Changes |
|---|---|---|---|
| v0.1 | 2026-05-08 | Founder | Initial PRD — vision, mission, problem, users, scope, architecture, pricing, GTM, risks, open questions |

---

## 15. Appendix — definitions of "done"

A capability is **launched** when:

1. It works end-to-end for at least 80% of inputs in a representative sample
2. It has API + UI + tests
3. It does not regress p50 latency for any other endpoint
4. Failure modes are visible to users (never silent)
5. A founder demo of it is recorded and stored
6. At least one non-founder user has used it without help
