# Contributing to NyayaSathi

NyayaSathi is the citizen-facing voice helpline; **Sanhita** (sibling repo) is the lawyer/compliance workbench. Together they form one platform. Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) first, then [`docs/MASTER-PLAN.pdf`](docs/MASTER-PLAN.pdf) for the full reasoning behind every design decision.

---

## Getting started

```bash
git clone <repo-url> nyayasathi && cd nyayasathi
npm install
cp .env.example .env       # add GEMINI_API_KEY, SARVAM_API_KEY
node server.js             # → http://localhost:3000
```

For the eval harness:

```bash
node eval/harness.js                    # runs eval/v1/questions.jsonl
node eval/harness.js --tag baseline     # saves to eval/reports/baseline.json
node eval/harness.js --filter safety    # only safety-tagged questions
```

---

## The architectural promise (non-negotiable)

**AI does NOT do these. Humans do.**

- **R1** — render judgment ("you should sue")
- **R2** — interpret ambiguous law as if settled
- **R3** — take accountability ("I'll handle it")
- **R4** — substitute for a human in regulated decisions

Every PR that touches user-facing answers must keep these enforced. Validator gates G7–G10 catch most violations automatically; some require taste.

---

## Commit message format (Conventional Commits, lightly extended)

```
<type>(<scope>): <imperative subject under 60 chars>

<body — wrap at 72 chars; explain *why* not *what*>

Refs: section <X> sprint <Y> day <Z>
Eval: <before>% → <after>% (<report-file>.json)
Co-Authored-By: <name> <email>
```

### Allowed types

| Type | Use for |
|---|---|
| `feat` | User-visible new capability |
| `fix` | Bug fix |
| `refactor` | Internal restructuring without behaviour change |
| `chore` | Tooling, deps, config |
| `docs` | Documentation only |
| `test` | Eval question additions, unit tests |
| `safety` | Distress, helpline, DPDP, audit-trail changes (always reviewed) |
| `data` | Corpus or eval-data changes (additive only) |

### Allowed scopes

```
server · case · safety · matching · validators · llm · voice · rag · audit
eval · docs · routes · vault · workflows · sanhita-bridge · contract-intel
```

### Two examples

```
safety(distress): translate critical helpline copy to 11 languages

Adds Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi,
Odia plus existing Hindi/English. Bengali DV victim was getting Hindi
response before — empirically zero conversion to dialing 181.

Refs: section 5 sprint 0 day 4
Eval: 78% → 84% on distress subset (sprint0_d4.json)
```

```
feat(matching): replace lawyers.json with sanhita-client HMAC bridge

Deletes mock lawyers.json + lawyer-match.js. NyayaSathi now POSTs to
Sanhita /api/nyaya/intake with HMAC-SHA256 signature. Sanhita's match
engine returns ranked candidates from the lawyers_profile table.

Refs: section 9.1 sprint 1 day 9
Eval: 87% → 89% (sprint1_d9.json)
```

---

## Pre-merge checklist

- [ ] Eval ran with pass rate ≥ current sprint target (S0=85%, S1=90%, S2=92%)
- [ ] **No regression on safety subset** — must stay at 100%
- [ ] New failures added as regression tests in `eval/v1/questions.jsonl`
- [ ] Conventional Commit message with `Refs:` and `Eval:` lines
- [ ] `docs/CHANGELOG.md` updated for the current sprint
- [ ] No `lawyers.json`-style dead code being added (we delete, not duplicate)

---

## Comment style for code

**Block comments at module head** explain *why* the module exists, what it owns, what it does NOT own:

```javascript
/**
 * distress-detector.js — sub-millisecond safety triage.
 *
 * Why keyword scan, not LLM? Latency. A user saying "मार रहा है" cannot
 * wait 5–8 seconds for Gemini. They need 181 / 112 / 1098 NOW.
 *
 * Does NOT own: emotional tone classification (that's the LLM tier-1.5
 * paraphrase classifier in distress-llm.js). This module is keyword-only.
 */
```

**Inline comments** mark non-obvious decisions and trade-offs:

```javascript
const TIMEOUT_MS = 8000;
// 8s budget — generous because this runs after the user reply is sent.
// JSON-mode generation tends to be slower than free-form on Flash.
```

**Never** document *what* the code does at line level — the code says that already.

---

## What we need help with

### High impact (Sprint 0 priority)
- **Eval question authoring** — add real-world scenarios to `eval/v1/questions.jsonl`
- **11-language distress copy** — translate critical helpline messages (`distress-detector.js`)
- **Regional FAQ templates** — pre-built answers in your regional language

### Good first issues
- Add helpline numbers to FAQ answers (verify they work in your state)
- Improve TTS pronunciation for specific languages
- Add suggestion chips for more languages in `public/index.html`
- Web UI accessibility / mobile fixes

### Advanced
- Hybrid retrieval (BM25 + dense vectors) — Sprint 1+
- Multi-modal vault extractors (audio/video/image) — Sprint 2
- Compliance edition primitives (workspaces, policies, audit chain) — Sprint 2+

---

## Questions?

Open an issue with the `question` label — we triage daily.
