# Architecture — One-Page Entry Point

> **Read this first. Then `docs/MASTER-PLAN.pdf` for the full reasoning.**
> This file is the 60-second orientation for any new contributor or future Claude session.

---

## What we are building

**Two products, one platform:**

1. **NyayaSathi** — free citizen-facing voice helpline (Indian languages, voice-first).
   Repo: this one. Lives on `localhost:3000` in dev.

2. **Sanhita** — paid lawyer/compliance workbench. Lives in the sibling repo
   `LexSearch-main 2/sanhita-react/` (port 8080 in dev).

   - **Sanhita Research** ($19/seat/mo) — solo lawyer research
   - **Sanhita Practice** ($49/seat/mo) — research + clients inbox + multi-modal vault
   - **Sanhita Compliance** ($199–499/seat/mo) — Practice + audit trail + RBAC + contract intelligence

The **integration** is `POST /api/nyaya/intake` (HMAC-signed) on the Sanhita side. The endpoint and the `nyaya_clients` table already exist on Sanhita; we wire NyayaSathi to call it.

---

## The architectural promise (non-negotiable)

**AI does NOT do these. Humans do.**

- **R1** — render judgment ("you should sue")
- **R2** — interpret ambiguous law as if settled
- **R3** — take accountability ("I'll handle it")
- **R4** — substitute for a human in regulated decisions

These are enforced by validator gates **G7–G10** in code (banned-phrase regex + interpretation alert wrapping + accountability refusal + human-signoff routing).

---

## How a request flows (NyayaSathi voice)

```
Citizen dials 1800
  → Exotel webhook
  → DPDP + recording disclosure (10s, mandatory)
  → STT (Sarvam Saaras)
  → Distress detector (keyword AND LLM tier-1.5 paraphrase classifier)
      ├─ critical → bypass LLM, return helpline in <200ms TTFB
      └─ none / high → continue
  → L1 guardrail (legal-scope check)
  → FAQ template match → response cache → RAG retrieval
  → LLM router (Gemini → Claude → Groq → Cloudflare, with circuit breaker)
  → Validator gates G1–G10
      ├─ pass → reply
      └─ fail → smart fallback (case-aware FAQ or soft prompt)
  → TTS (Sarvam Bulbul, emotion-mapped)
  → background entity extractor (updates case state for next turn)
  → if user accepts lawyer offer → POST /api/nyaya/intake to Sanhita (HMAC)
```

---

## The four customer journeys

| | Audience | Outcome |
|---|---|---|
| **A** | Citizen, problem solved without lawyer | ~70% of calls; ₹0 revenue; the moat-builder |
| **B** | Citizen → matched to Sanhita lawyer | ~20% of calls; lawyer pays seat fee + commission |
| **C** | Solo lawyer using Sanhita Practice | $49/mo; case mgmt + multi-modal vault + leads |
| **D** | Compliance team (e.g., bank GC) | $199–499/mo; audit trail + policies + contract intelligence |

---

## Code map (current state, Sprint 0 Day 1)

```
nyayasathi/                            # this repo
├── docs/
│   ├── MASTER-PLAN.pdf                # the canonical plan
│   ├── ARCHITECTURE.md                # this file
│   ├── CONTRIBUTING.md                # commit + comment conventions
│   ├── CHANGELOG.md                   # per-sprint changes
│   └── generate_master_plan.py        # re-run to refresh PDF
├── server.js                          # 2.4K lines; will be split Sprint 0 Day 5
├── case-store.js                      # case persistence (file-backed for now)
├── intent-classifier.js               # case-type from keywords
├── slot-templates.js                  # required slots per case type
├── entity-extractor.js                # background Gemini JSON-mode
├── distress-detector.js               # keyword scan; Sprint 0 adds LLM paraphrase
├── rag.js / rag-tiers.js              # retrieval (will share Sanhita's BM25 Sprint 1)
├── rules-engine.js                    # post-processing; will become validators/
├── security.js                        # PII guards
├── voice-engine.js                    # STT/TTS adapters
├── lawyers.json                       # DEPRECATED — delete Sprint 1 Day 9
├── lawyer-match.js                    # DEPRECATED — replaced by sanhita-client
├── public/index.html                  # web UI
├── eval/                              # NEW Sprint 0 Day 1
│   ├── v1/questions.jsonl             # versioned eval prompts
│   ├── harness.js                     # CLI runner
│   ├── checks/                        # 5 gate modules
│   ├── reports/                       # versioned JSON reports (kept forever)
│   └── fixtures/                      # mock data
└── data/                              # gitignored
    ├── cases/
    ├── sessions/
    └── safety-events/
```

---

## Sprint 0 — what we're shipping in week 1

| Day | What ships | Eval target |
|---|---|---|
| 1 | Eval harness + 30 baseline questions + baseline run | establish baseline |
| 2 | HMAC bridge to Sanhita /api/nyaya/intake | +20 handoff questions |
| 3 | DPDP + recording + AI disclosure first 10s | safety subset = 100% |
| 4 | Distress copy in 11 languages + LLM paraphrase classifier | distress subset = 100% |
| 5 | Sentry + safety-event writer + duty-officer alert | — |
| 6 | Fix worst 5 fail patterns from Day 5 eval | — |
| 7 | Final Sprint 0 eval against 100 questions; tag v0.5.0 | ≥85% |

Sprint 1 → real lawyer matching with Sanhita backend (Days 8–14, target ≥90%).
Sprint 2 → multi-modal vault + audit trail + Exotel call-masking (Days 15–21, target ≥92%, **launch v1.0.0**).

---

## Where to look when X happens

| Symptom | First file to read |
|---|---|
| User reports wrong answer | `eval/reports/<latest>.json` — find the question, replay |
| Distress not bypassing | `distress-detector.js` then `eval/checks/safety.js` |
| Citation hallucinated | `rules-engine.js` + Sanhita's `validators/answer_gates.py` G5 |
| Handoff to Sanhita failing | `sanhita-client.js` (Sprint 0 Day 2) + Sanhita logs |
| LLM cascade timing out | `server.js` LLM router section (will be `src/llm/router.js`) |
| Cache poisoned | response-cache section + `eval/checks/banned_phrases.js` |

---

## Why this design — read the PDF

Every design choice has a reason; the reasons live in `MASTER-PLAN.pdf`. If you find yourself disagreeing with a decision, the PDF either justifies it or shows it should change. Update the PDF first, then the code.
