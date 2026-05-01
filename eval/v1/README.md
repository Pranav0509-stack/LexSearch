# Eval v1 — questions.jsonl

The canonical question bank. **Append-only.** Every regression found in production becomes a new line here. We never delete questions; failing questions get fixed in code, not removed from the suite.

## Format (one JSON per line)

```json
{
  "id": "FIR-001",
  "category": "fir",
  "lang": "hi-IN",
  "text": "FIR कैसे दर्ज करवाएं?",
  "expect": {
    "must_mention": ["FIR", "थाना"],
    "must_cite_section": ["BNSS 173"],
    "must_include_helpline": ["NALSA 15100"],
    "forbid_phrases": ["i think", "you should"],
    "max_words": 120,
    "is_critical": false
  }
}
```

## Field reference

- `id` — unique, immutable. Format `<CATEGORY>-<NNN>`.
- `category` — one of: `fir`, `dv`, `divorce`, `maintenance`, `custody`, `property`, `land`, `salary`, `bail`, `fraud`, `cheque_bounce`, `caste`, `false_case`, `distress`.
- `lang` — `hi-IN` / `en-IN` / `bn-IN` / `te-IN` / `ta-IN` / `mr-IN` / `gu-IN` / `kn-IN` / `ml-IN` / `pa-IN` / `od-IN`.
- `text` — the user's question, in their language.
- `expect.must_mention` — terms that must appear in the answer. Use `|` for alternatives: `"FIR|एफआईआर"` accepts either form.
- `expect.must_cite_section` — canonical section refs. See `eval/checks/citations.js` for aliases.
- `expect.must_include_helpline` — phone numbers as plain digits. The check accepts digit-spaced and Hindi-words variants.
- `expect.forbid_phrases` — question-specific bans on top of the global hedge/judgment/accountability bans.
- `expect.max_words` — voice-mode budget (90–160 typical). Long answers are useless on phone.
- `expect.is_critical` — distress questions; helpline must appear in first sentence; safety-bypass first-byte under 300 ms.

## Distribution (matches NALSA 2022 helpline call distribution)

| Category | Sprint 0 count | Sprint 2 final |
|---|---|---|
| fir | 10 | 25 |
| dv / divorce / maintenance / custody | 8 | 15 |
| property / land | 6 | 12 |
| cheque_bounce / fraud | 4 | 10 |
| bail | 2 | 8 |
| salary | 2 | 8 |
| consumer | 0 | 7 |
| caste | 0 | 5 |
| false_case | 1 | 5 |
| rti / scheme | 0 | 3 |
| distress | 2 | 12 |
| **total** | **35** | **110+** |

## How to add a question

1. Pick the next free `id` for the category.
2. Append a new JSON line at the bottom of `questions.jsonl`.
3. Run `node eval/harness.js --filter <id>` to verify it parses and returns expected fields.
4. Commit with a `test(eval): add <category> regression for <issue>` message.
5. Update `eval/v1/README.md` count table.
