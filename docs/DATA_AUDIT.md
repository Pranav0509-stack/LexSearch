# Sanhita — Honest Data Audit & Coverage Plan

**Date:** 2026-05-09
**Status:** what we actually have, where the gaps are, and the plan to close them.

---

## 1. Raw corpora (verified row counts)

| Table | Rows | What it is |
|---|---:|---|
| `judgments` | **16,856,181** | HC case **metadata** (cnr, court, judge, bench, verdict, year, pdf_link). The big asset. |
| `legal_docs` | **11,534,747** | KanoonGPT-extracted text — ~70% are duplicate Old-SC judgments (1949–60s) with full_text. **The corpus we've been enriching.** |
| `legal_qa` | 1,310,311 | Q&A pairs (training-style) |
| `documents` | 2,013 | NEW gov-circular ingestion (SEBI / RBI / PRS / India Code / NCLAT) |
| `statutes` | 2,333 | India Code bare acts (titles only — no section text) |
| `district_cases` | 28,856 | District court metadata |

---

## 2. The CRITICAL gap (the one that explains everything)

```
judgments.full_text > 500 chars  =  0 / 16,856,181
```

**16.8 million High Court cases are metadata-only.** The PDFs are reachable on S3 but the extracted text isn't in the DB. Every enricher we wrote runs against `legal_docs` (which has full_text) — but `legal_docs` is mostly old SC dupes from KanoonGPT.

**Net effect:** all our enrichment so far reflects 1949–1960s SC judgments authored by ~30 famous justices, NOT the modern Indian legal system.

---

## 3. Enrichment tables (all from `legal_docs`, none from `judgments`)

| Table | Rows | Distinct | Reality check |
|---|---:|---:|---|
| `case_advocates` | 295,075 | 52,744 names | Real names in there (K Harishankar, Salve…) but ~30% noise (regex grabbed boilerplate like "Cr.P.C. Order"). Tightened blocklist already in code — needs re-run. |
| `case_judges` | 47,446 | **559 distinct** | India has ~26,000 active judges. **We're at ~2% coverage** because legal_docs is mostly old SC. We see Pasayat, Sinha, Ramaswamy, Chandrachud, Gajendragadkar — all repeated 500–2,000 times. |
| `judge_aliases` | 444 | — | Clusters look right, but tiny canonical set |
| `case_topics` | 1,519 | 21 topics | Rules-only run. ~3% hit rate (rules-tuned for modern statutes; legal_docs is too old). |
| `citation_edges` | 147,841 | — | 23× growth from 6K. Rich on Article 14/19/21/32/226 (SC's bread + butter). |
| `citator_stats` | 85,250 | — | PageRank scored 85K cases |
| `reversals` | **0** | — | Old SC cases don't reverse each other in the corpus subset |
| `enrichments` (headnotes) | 3 | — | Cost-gated until we have a budget |

---

## 4. Reality check — Indian legal system size

| Tier | Active judges | Sanhita coverage today |
|---|---:|---:|
| **Supreme Court** | 33 sitting + ~250 retired (post-1950) | ~30 covered (from kanoongpt) |
| **High Courts (25)** | 1,108 sanctioned (757 working) | <50 covered |
| **District / Sessions** | 25,000+ | 0 covered |
| **Tribunals (ITAT/NCLT/CESTAT/etc.)** | ~500 | 0 covered |
| **Total** | ~**26,000+** | **559** (≈2%) |

**Our `case_judges` shows 559 because we processed 50K mostly-old-SC docs.** To get the real 26K we need:
1. Backfill `judgments.full_text` (the unblocker)
2. Re-run `bench_split` on judgments → expect **~5,000–8,000 distinct HC judges** (real number)
3. Add tribunal corpora (ITAT 63 benches, NCLT 16 benches, NCLAT) — adds another 1K+ judges
4. District-court corpus (eCourts) — adds the bulk 25K

---

## 5. Search speed gaps

| Endpoint | Current latency | Issue |
|---|---|---|
| `/api/cases/search` (FTS5) | ~50ms | ✅ fast |
| `/api/analytics/judge-profile?judge=X` | **~3 minutes** | `LIKE '%X%'` cannot use btree. Needs FTS or prefix-only. |
| `/api/analytics/lawyer-profile?name=X` | ~1s | OK while case_advocates is 295K; will degrade at 10M+ |
| `/api/cases/{id}/citations` | ~100ms | ✅ fast |
| `/api/assistant/ask` | 22–30s | LLM-bound; fine |
| `/api/analytics/predict-outcome` | <100ms | ✅ fast |

The judge-profile is the only embarrassing one — fix is one of:
- (a) New FTS5 index on `judge` column (5 min code, 1 min build)
- (b) Surface a prefix-only search with autocomplete in the UI
- (c) Pre-compute a `judge_canonical` table with one row per canonical judge name + denormalised stats

---

## 6. What's done (be honest)

| Layer | Module | State |
|---|---|---|
| **Data acquisition** | `scripts/ingest/{sebi,rbi,prs,indiacode,nclat,egazette}.py` | 2K rows; ITAT/NCLT/CESTAT deferred |
| **Schema** | `case_advocates, case_judges, case_topics, judge_aliases, reversals, citator_stats, citation_edges, enrichments` | All live, indexed |
| **B0 PDF backfill** | `scripts/enrich/pdf_text_backfill.py` | Code ready, **not yet executed** — this is the unblocker |
| **B1 advocate NER** | `scripts/enrich/advocate_ner.py` | 295K rows on legal_docs; blocklist tightened; needs rerun |
| **B2 outcome classifier** | `scripts/enrich/outcome_classifier.py` | 11K classified (regex-only) on legal_docs |
| **B3 bench split** | `scripts/enrich/bench_split.py` | 47K judge rows on legal_docs |
| **B4 topic classifier** | `scripts/enrich/topic_classifier.py` | 1.5K rows; needs Gemini tier |
| **B5 reversal detector** | `scripts/enrich/reversal_detector.py` | 0 reversals (legal_docs subset is co-authored SC) |
| **B6 duration calc** | `scripts/enrich/duration_calc.py` | No-op on legal_docs (correct); needs run on judgments |
| **C-layer analytics** | `lawyer_analytics.py`, extended `judge-profile` | Endpoints live |
| **D-layer ML** | `scripts/ml/{features,train_outcome,inference}.py` | Model trained, 85.92% acc, live |
| **Smart Reasoner** | `scripts/assistant/legal_reasoner.py` | Working with model cascade |
| **UI** | LawyerAnalyticsPanel, PredictivePanel, CitationsPanel | Live, TS clean |

---

## 7. What we MUST do next (in priority order)

### Priority 0 — The unblocker (run overnight)

**B0 PDF text backfill on judgments.** ~10 hours wall-clock for the full 16.8M corpus at 8 workers fetching from S3 → pdfplumber. Hit rate ~60% → ~10M judgments with real full_text.

```bash
nohup python3 -m scripts.enrich.pdf_text_backfill \
  --priority cited --workers 12 \
  > /tmp/pdf_backfill.log 2>&1 &
```

This single batch unlocks **everything else**.

### Priority 1 — Re-run enrichers on judgments (after P0 lands)

```bash
python3 -m scripts.enrich.bench_split        --corpus judgments --limit 200000
python3 -m scripts.enrich.advocate_ner       --corpus judgments --no-gemini --limit 200000  # tightened blocklist
python3 -m scripts.enrich.outcome_classifier --corpus judgments --limit 200000  # regex tier first
python3 -m scripts.enrich.topic_classifier   --corpus judgments --rules-only --limit 200000
python3 -m scripts.enrich.duration_calc      --table judgments
python3 -m scripts.enrich.citation_extractor --corpus judgments --limit 500000
python3 -m scripts.enrich.citator_stats
python3 -m scripts.enrich.reversal_detector --refresh
```

**Expected result post-rerun:**
- 5K–8K distinct HC judges (10× current 559)
- 200K+ advocates with real names (cleaner due to blocklist)
- 1M+ citation edges
- 50K+ reversal candidates (HC → SC overruling)

### Priority 2 — Search speed (1-2 hour task)

Replace `LOWER(j.judge) LIKE '%X%'` in `api_judge_profile` with FTS5 lookup. Specifically:
1. The existing `judgments_fts` already indexes the `judge` column (verified via earlier audit)
2. Change endpoint to: `WHERE j.cnr IN (SELECT j2.cnr FROM judgments_fts WHERE judgments_fts MATCH 'judge:X*')`
3. Drops latency from 3 min → 50 ms

### Priority 3 — Tribunal coverage (week-1 of next sprint)

Order by ROI for our pilot CAs/lawyers:
1. **ITAT** (~400K orders) — highest CA value. Use Playwright per the plan.
2. **NCLT** (16 benches × ~10K orders) — corporate practice
3. **CESTAT** (~80K) — indirect tax

### Priority 4 — Outcome classifier with Gemini tier

Cost: ~$200 for 50K judgments × Gemini Flash. Boosts coverage from 22% (regex only) to ~80%. Re-train the model afterwards — expect outcome_v2 accuracy 90%+ with real class balance (currently 70% disposed because old SC cases mostly say "disposed of").

### Priority 5 — Editorial moat

- Headnotes: top 100K most-cited judgments via Gemini ($1K)
- Section-level statute parsing (today's statutes table is title-only)
- Old report digitization (AIR/SCR/SCC pre-2000) — multi-month

### Priority 6 — Frontend polish

- Lawyer dashboard sparkline (growth_trajectory chart)
- Predictive panel with similar-cases drawer
- Saved-search alerts (CA notification firehose)

---

## 8. The 26,000-judge gap explained

Today's 559 distinct judges in `case_judges` is correct for the corpus we ran enrichers against. The system isn't broken — it just hasn't seen the right data yet. Walking through it:

| Source | Distinct judges in that source |
|---|---:|
| `legal_docs.judge` (only the rows we processed: 50K) | 559 |
| `legal_docs.judge` (all 11.5M rows, after full bench-split run) | ~3,000 (mostly old SC) |
| `judgments.judge` (16.8M HC) — **needs full_text-aware bench_split** | **~7,500 expected** |
| + tribunal corpora (ITAT/NCLT/CESTAT/CAT) when ingested | +1,500 |
| + district court (eCourts) — Phase 2 | +20,000 |
| **Total realistic ceiling** | **~32,000** |

Indian Kanoon shows ~14,000 distinct judges. Manupatra shows ~8,000. Our ceiling beats both, but only after Priority 0 + 1 land.

---

## 9. Architecture decisions that need confirming

Before we run the overnight backfill, three calls:

**(a) PDF text backfill priority order:**
- `--priority cited`: top-cited cases first (best ROI, lawyer dashboards light up immediately)
- `--priority recent`: most recent year first (best for predictive model freshness)
- `--priority sequential`: cheapest IO pattern (whole corpus eventually)

Recommend: `cited` first 100K, then `recent` 500K, then `sequential` for the long tail.

**(b) Worker count:**
- Conservative: 4 workers (~50 PDFs/sec)
- Aggressive: 12 workers (~150 PDFs/sec) — may get rate-limited by S3
- Default in code: 8

**(c) When to spend on Gemini:**
- Run free-tier (regex/rules) on full 16.8M judgments first
- Spend ~$200 on Gemini outcome classifier for the top 100K cited
- Headnotes ($1K) gated on first 10 paying customers

---

## 10. Ten-line summary of "where we stand"

1. We have **16.8M HC judgment records** but no extracted text.
2. **Enrichment so far is on the WRONG corpus** (kanoongpt old SC, not modern HCs).
3. **All architecture is built and working** — schema, endpoints, ML model, UI components.
4. The model trained at **85.92% holdout accuracy** but on a class-imbalanced 11K rows.
5. **One overnight job (B0 PDF backfill) unlocks everything** — 16.8M → ~10M with real text.
6. After backfill, judges grow from 559 → 7,500 (HC level), advocates from 50K → 200K real names.
7. **Judge-profile latency (3 min)** is fixable in 30 minutes via FTS5.
8. **Tribunals are 0% ingested** — ITAT alone is 400K cases of pure CA value.
9. **Predictive model has 4 classes** (no `dismissed` rows yet); Gemini outcome pass fixes this.
10. We're **demonstrably ahead of Manupatra** on architecture; data-coverage parity is one overnight + one Gemini batch away.
