# CHANGELOG

All notable changes to NyayaSathi. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with sprint markers.

## [Unreleased] — Sprint 0 (week 1)

### Day 1 — Eval foundation
- **Added** `docs/MASTER-PLAN.pdf` — 20-page canonical plan covering integration with Sanhita, three product editions, four customer journeys, four refusals (R1–R4), multi-modal data pipeline, three stages of casework, predictive/compliance/contract surfaces, five data moats, 21-day sprint plan, eval policy, codebase organisation, commit conventions.
- **Added** `docs/ARCHITECTURE.md` — one-page entry point for new contributors and future Claude sessions.
- **Added** `docs/CHANGELOG.md` — this file.
- **Added** `docs/generate_master_plan.py` — re-runnable PDF generator (reportlab).
- **Updated** `CONTRIBUTING.md` — Conventional Commit format with `Refs:` + `Eval:` lines, comment style guide, pre-merge checklist.
- **Added** `eval/` scaffold — `v1/questions.jsonl`, `harness.js`, `checks/`, `reports/`, `fixtures/`.
- **Added** 30 baseline eval questions covering FIR (10), family/DV (8), property (6), cheque-bounce/fraud (4), distress safety (2).
- **Added** 5 check modules: `citations.js`, `banned_phrases.js`, `helplines.js`, `grounding.js`, `language.js`.
- **Captured** baseline eval report at `eval/reports/baseline.json`. **Result: 0/30 = 0% pass rate.** Honest starting point. The harness correctly hard-failed on 2 safety questions (DSTR-001, DSTR-002 — distress bypass fires but helpline isn't in first sentence). Top failure modes across the other 28: (a) missing canonical citations because answers cite without using the exact alias the check expects (fixable by extending `eval/checks/citations.js` ALIASES), (b) missing must-mention terms because answers in Hindi don't include English term names like "Protection Officer" (fixable by adding `|`-alternatives in question terms), (c) missing helplines because answers route to FAQ paths that already include NALSA but use spaced-digits not present in our variant generator (fixable in `helplines.js`).
- **Added** `CLAUDE.md` — orientation file for future Claude Code sessions covering commands, architecture, sprint structure, and the don't-touch-unless rules.

### Day 2 — Sanhita HMAC bridge (planned)
- **Add** `sanhita-client.js` — HMAC-SHA256 POST to Sanhita `/api/nyaya/intake`.
- **Delete** `lawyers.json` and `lawyer-match.js` (deprecated mock).
- **Add** 20 eval questions covering handoff scenarios.

### Day 3 — DPDP + recording disclosure (planned)
- First 10 seconds of every call: AI-disclosure + recording-consent + DPDP-consent in user's language.

### Day 4 — Distress 11 languages + LLM paraphrase (planned)
- Translate critical helpline copy to 11 languages.
- Add LLM tier-1.5 paraphrase classifier in parallel with keyword scan.

### Day 5 — Observability (planned)
- Sentry integration.
- Safety-event JSON writer to `data/safety-events/<ts>.json`.
- Slack/email duty-officer alert pipeline.

### Day 6 — Fix worst patterns (planned)
- Address top 5 failure patterns from Day 5 eval.
- Author 30 more eval questions covering safety paraphrases.

### Day 7 — Sprint 0 freeze (planned)
- Final eval against 100 questions; block if <85%.
- Tag `v0.5.0`.

---

## Sprint 1 — Polish + real lawyer matching (week 2)

(Days 8–14, target ≥90% pass rate, tag `v0.7.0`.)

---

## Sprint 2 — Multi-modal + audit + connection (week 3)

(Days 15–21, target ≥92% pass rate, tag `v1.0.0` — **launch**.)

---

## How to update this file

Each commit appends to the **current sprint's day** section. At sprint freeze, the day-by-day list collapses into a single tagged release entry. Format:

```markdown
## [v0.5.0] — Sprint 0 (YYYY-MM-DD → YYYY-MM-DD)

### Added
- Item with link to commit / issue / PR

### Changed
- Item

### Removed
- Item

### Fixed
- Item

### Eval
- Sprint 0 final: 87% pass rate (eval/reports/sprint0_final.json)
- Safety subset: 100%
- Distress paraphrase: 100% on 12-question subset
```
