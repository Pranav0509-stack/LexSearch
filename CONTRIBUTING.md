# Contributing to NyayaSathi

Thanks for wanting to help make legal awareness accessible to every Indian. Here's how you can contribute.

## Getting Started

1. Fork this repo
2. Clone your fork: `git clone https://github.com/<your-username>/Nyayasathi_.git`
3. Install dependencies: `npm install`
4. Copy `.env.example` to `.env` and add your free API keys
5. Run: `node server.js` → open `http://localhost:3000`

## What We Need Help With

### High Impact
- **New language FAQ templates** — Add pre-built legal answers in your regional language
- **Supreme Court judgments** — Add landmark cases to the RAG corpus in `rag.js`
- **Village/rural scenarios** — More FAQ templates for common rural legal problems
- **STT accuracy** — Improve speech recognition for noisy phone environments

### Good First Issues
- Add suggestion chips for more languages in `public/index.html`
- Add new legal helpline numbers to FAQ answers
- Fix TTS pronunciation issues for specific languages
- Improve the web UI (accessibility, mobile responsiveness)

### Advanced
- Better RAG retrieval accuracy in `rag-tiers.js`
- New post-processing rules in `rules-engine.js`
- Exotel IVR flow improvements in `server.js`
- Eval test cases in `eval-engine.js`

## How to Add a New FAQ Template

In `server.js`, find `FAQ_TEMPLATES` and add your entry:

```javascript
{
  patterns: ["keyword1", "keyword2", "कीवर्ड"],  // Mix of romanized + native script
  answer: "Your pre-built legal answer with specific Act, Section, and helpline numbers."
}
```

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Test your changes locally before submitting
- If adding legal content, cite the specific Act/Section
- Run `node eval-engine.js` to check nothing breaks

## Code Structure

| File | What it does |
|---|---|
| `server.js` | Main server — API routes, AI logic, FAQ templates, Exotel webhooks |
| `voice-engine.js` | TTS normalization, speaker config, transcript scoring |
| `rules-engine.js` | 10-rule post-processing pipeline |
| `rag.js` / `rag-tiers.js` | Legal knowledge retrieval (RAG) |
| `citation-guard.js` | Citation verification |
| `public/index.html` | Frontend (vanilla JS) |
| `eval-engine.js` | Test suite (68 tests) |

## Questions?

Open an issue — we're happy to help you get started.
