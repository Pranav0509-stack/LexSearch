# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

NyayaSathi — the citizen-facing voice helpline half of a two-product platform. The lawyer/compliance half is **Sanhita**, a sibling repo at `../LexSearch-main 2/sanhita-react/`. They integrate via `POST /api/nyaya/intake` (HMAC-signed) on the Sanhita side. Read [`docs/MASTER-PLAN.pdf`](docs/MASTER-PLAN.pdf) for the canonical 20-page plan covering both products.

## Common commands

```bash
# Dev server (port 3000)
node server.js

# Eval — gates every commit
node eval/harness.js                              # ad-hoc run
node eval/harness.js --tag <name>                 # named report at eval/reports/<name>.json
node eval/harness.js --filter safety              # only category=safety questions
node eval/harness.js --filter DSTR-               # only id-prefix matches
node eval/harness.js --threshold 0.85             # exit non-zero below threshold
node eval/harness.js --max 10                     # limit to N questions

# Re-generate the master plan PDF after editing the script
python3 docs/generate_master_plan.py

# Server smoke test (one query)
curl -s -X POST localhost:3000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"sessionId":"smoke","message":"FIR कैसे करें","lang":"hi-IN"}' | jq .
```

The eval harness exits 1 on **any** safety failure or below the threshold, 2 on setup error (server unreachable, malformed input), 0 on pass. CI uses these exit codes to gate merges.

## Architecture (the parts that need multi-file reading)

### Two products, one platform

NyayaSathi (this repo) and Sanhita (sibling repo, ~7,839 LOC Python + Next.js 16) are designed to integrate. **We do not rebuild what Sanhita already has.** Specifically, Sanhita owns:

- `validators/answer_gates.py` — six-gate validator (G1 cite_present, G2 cite_resolves, G3 banned_phrases, G4 grounding_floor, G5 scope_check, G6 section_check). NyayaSathi adopts a JS port at `eval/checks/banned_phrases.js` and will get a runtime port at `validators/answer-gates.js` Sprint 1.
- `llm/router.py` — four-provider chain (Gemini → Claude → Groq → Cloudflare) with circuit breakers. NyayaSathi will adopt this pattern Sprint 1, replacing the homegrown `Promise.any` cascade in `server.js`.
- `retrieval_pkg/` — BM25 index over the case-law corpus (1,135 docs today, 110K target). NyayaSathi will share this corpus Sprint 1+ via a `tier="landmark"` filter.
- `auth.py` — already has the `nyaya_clients` table with `intake_summary`, `intake_transcript`, `language`, `jurisdiction`, `status`, `assigned_user_id`, `thread_id`. The handoff endpoint `/api/nyaya/intake` already exists at `server.py:1857`. Sprint 0 Day 2 wires NyayaSathi to it via HMAC.

### The architectural promise (R1–R4)

AI does NOT render judgment, interpret unmarked, take accountability, or substitute for humans in regulated decisions. Enforced by validator gates:

- **G3** (existing in Sanhita, ported here) — bans `as an ai`, `i think`, `in my opinion`, etc.
- **G7** (Sprint 1) — bans `you should`, `i recommend`, `your best move is`.
- **G8** (Sprint 1) — wraps unmarked interpretations in `{interpretation_alert: true}`.
- **G9** (Sprint 1) — bans `i'll handle it`, `leave it to me`.
- **G10** (Sprint 2) — predictive output must include base rate + sample size.

`eval/checks/banned_phrases.js` already enforces G3, G7, and G9 on every eval question. New violations found in production must be appended to the `HEDGE`, `JUDGMENT`, or `ACCOUNTABILITY` arrays in that file.

### Request flow (NyayaSathi voice)

`server.js` is currently 2,400 lines and houses all of this in one file (Sprint 0 Day 5 will split it):

```
/api/ask | /api/ask-stream
  → sanitize() + getOrCreateSession()
  → distress-detector.js (keyword scan, sub-3ms)
      ├─ critical → safety bypass; helpline reply with TTFB <200ms
      └─ continue
  → L1 guardrail (isLegalQuery)
  → matchFAQ() → response cache → buildContext (RAG)
  → buildSystemPrompt (case-state injection from Phase C)
  → callGemini || callSarvam (Promise.any race, 8s wall-clock)
  → applyRules (rules-engine.js)
  → if !reply: getSmartFallback() — case-aware FAQ lookup, never bare error
  → appendToSession (persists to data/cases/<id>.json + fires entity-extractor.js async)
  → segmentForTTS → generateTTS (Sarvam Bulbul)
```

The smart-fallback machinery (case-store.js, entity-extractor.js, distress-detector.js, intent-classifier.js, slot-templates.js) was built across phases A–F documented in earlier conversation. Read those files' header comments before changing them — each explains what it owns and what it deliberately does NOT own.

### Eval is a peer of src, not an afterthought

`eval/v1/questions.jsonl` is **append-only**. Every regression found in production becomes a new line. The check modules under `eval/checks/` enforce one concern each:

- `citations.js` — required section refs present; aliases handle BNS/IPC, BNSS/CrPC, Hindi variants
- `banned_phrases.js` — hedge/judgment/accountability bans + per-question `forbid_phrases`
- `helplines.js` — required numbers, with critical-distress requirement that helpline appear in first sentence
- `grounding.js` — `must_mention` terms + `max_words` budget
- `language.js` — script-ratio check (Devanagari ≥40% for hi-IN; Latin ≥60% for en-IN)

Reports under `eval/reports/` are **kept forever**. They form a time-series of model behaviour we diff with `jq`.

### Data persistence

Currently file-backed under `data/` (gitignored). Will migrate to Sanhita's `nyaya_clients` Postgres table Sprint 0 Day 2. The case schema in `case-store.js` matches Sanhita's `nyaya_clients` schema deliberately — minimal change at migration time.

## Commit discipline

Conventional Commits with two extra trailers — see `CONTRIBUTING.md`. Every commit must have:

- `Refs: section <X> sprint <Y> day <Z>` linking back to `docs/MASTER-PLAN.pdf`
- `Eval: <before>% → <after>% (<report-file>.json)` showing measured impact

Allowed types: `feat fix refactor chore docs test safety data`. Allowed scopes: `server case safety matching validators llm voice rag audit eval docs routes vault workflows sanhita-bridge contract-intel`.

## Sprint structure

Three weeks, eval-gated:

- **Sprint 0** (Days 1–7): bridge to Sanhita + safety floor. Tag `v0.5.0`. Threshold ≥85%.
- **Sprint 1** (Days 8–14): real lawyer matching, female-voice auto-switch, G7–G9 ports. Tag `v0.7.0`. Threshold ≥90%.
- **Sprint 2** (Days 15–21): multi-modal vault, audit trail, Exotel call-masking, launch. Tag `v1.0.0`. Threshold ≥92%.

Day-by-day breakdown is in `docs/CHANGELOG.md` and `docs/MASTER-PLAN.pdf` section 11.

## Files that have one specific reason to exist

| File | Reason | Don't touch unless... |
|---|---|---|
| `distress-detector.js` | Sub-3ms safety triage; LLM-bypass on critical signals | You're adding a language or new critical signal; **never lower the bar** |
| `case-store.js` | File-backed case persistence + phone-keyed callback lookup | You're migrating to Postgres (Sprint 0 Day 2) |
| `entity-extractor.js` | Background Gemini JSON-mode extractor; 8s timeout | Schema additions for new case types |
| `intent-classifier.js` | Keyword-based case-type classifier; sticky | Adding case types |
| `slot-templates.js` | Required slots per case type; powers `renderCaseContext` | Adding case types or refining priorities |
| `rules-engine.js` | Post-LLM cleanup (banned phrases, citation enforcement) | Adding rules; extends Sanhita's G3 |
| `lawyers.json` + `lawyer-match.js` | **DEPRECATED** — Sprint 0 Day 2 deletes both | Don't add to these; use `sanhita-client.js` instead |

## When something breaks in production

1. Read the latest `eval/reports/<latest>.json` — find the question, replay locally.
2. Reproduce with `node eval/harness.js --filter <id>`.
3. Add the failing case as a new line in `eval/v1/questions.jsonl` (it becomes a permanent regression test).
4. Fix in code.
5. Re-run eval; commit with `fix(<scope>):` and the eval delta.

## What to avoid

- Lowering eval thresholds to ship. The whole point is the discipline.
- Deleting failing questions instead of fixing the code.
- Adding to `lawyers.json` — it's deprecated.
- Building from scratch what Sanhita already has — read the master plan section 1 first.
- Letting `server.js` grow further before the Sprint 0 Day 5 split.
