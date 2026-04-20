# NyayaSathi — Voice AI Legal Advocate for India

Dial a number, talk to an AI in any of 10 Indian languages, walk away with an eSigned legal letter in your inbox and a booked paid consultation with a real lawyer.

> **Not legal advice.** NyayaSathi is a legal-information + document-drafting platform. For advice, consult a licensed advocate. BCI-safe positioning, DPDP-compliant.

## What it does

- **Voice-first.** PSTN call → streaming ASR → dialog orchestrator → streaming TTS. p95 round-trip target <1.8s.
- **10 Indian languages.** Hindi, English, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi.
- **Legal document drafting.** 8 Jinja templates (S.138 cheque-bounce, consumer complaint, landlord/tenant, employment, RTI, general notice, FIR, defamation). SMS-OTP eSign (IT Act §3A) → WeasyPrint PDF → SES email in <5 min.
- **Judgment RAG.** 16.7M+ Indian High Court judgments (CC-BY-4.0 corpus, preserved from LexSearch) exposed as a tool the LLM cites inline in speech and in the PDF footnotes.
- **Lawyer handoff.** Partner adapters (LegalKart / Vakilsearch / LawRato) with Razorpay Route auto-split (25% platform commission, 75% partner). Callback booking via SMS, not live transfer.

## Tech stack

| Layer | Stack |
|---|---|
| Telephony | Plivo Media Streams (bidirectional WS, 8kHz μ-law) |
| ASR / TTS / LLM | Sarvam AI (Saaras, Bulbul, Sarvam-M) + provider-abstraction fallbacks |
| API | FastAPI + Uvicorn |
| State | Postgres 15 (SQLAlchemy 2.0 async, Alembic) + Redis 7 |
| Async | Celery (post-call pipeline, doc generation, lawyer match) |
| PDF | Jinja2 + WeasyPrint (Noto Indic fonts) + pypdf (signature stamping) |
| Payments | Razorpay Route |
| Email / Storage | AWS SES + S3 (ap-south-1, SSE-KMS) |
| Observability | structlog + Prometheus + Sentry + OpenTelemetry |

## Repo layout

```
app/
  main.py, config.py, deps.py
  api/            lexsearch.py (legacy judgment search preserved), sign_page.py
  voice/          plivo_webhook.py, plivo_media_ws.py, sarvam_client.py, session.py
  voice/dialog/   orchestrator.py, safety.py, intents.py, tools.py
  rag/            judgment_tool.py
  docs/           generator.py, esign.py, delivery.py, templates/*.jinja
  lawyers/        adapter.py, matcher.py, handoff.py, {legalkart,vakilsearch,lawrato,mock}_*.py
  payments/       razorpay_client.py, route.py, webhooks.py
  compliance/     disclaimer.py, consent.py, dpdp.py, pii.py
  db/             models.py, session.py, migrations/
  tasks/          celery_app.py, post_call.py, doc_generation.py, lawyer_match.py
  observability/  logging.py, metrics.py
server.py         thin entry — from app.main import create_app
```

## Local dev

```bash
cp .env.example .env   # fill in Plivo, Sarvam, AWS, Razorpay keys

docker compose up -d postgres redis
alembic upgrade head
uvicorn server:app --reload --port 8080
```

Or the whole stack (app + celery worker + postgres + redis):

```bash
docker compose up --build
```

Health check: `curl localhost:8080/healthz` · Legacy judgment search at `/` · Voice webhook at `/voice/plivo/answer`.

## Compliance

- **BCI-safe:** disclaimer played in user's language at start of every call; recorded as consent evidence. Every PDF footer says "not legal advice."
- **DPDP Act 2023:** `/api/user/data-export` and `/api/user/delete` endpoints. Retention: audio 90d, transcripts 2y, docs 7y.
- **PII redaction:** phone, Aadhaar, PAN, account numbers stripped from logs/Sentry via structlog processor.
- **Rate limiting (learned from the 500/hr LinkedIn incident):** per-phone (5/day, 20/month), per-Sarvam-key semaphore, Celery queue-length alarm, Plivo edge limits.

## Data source

Judgment corpus: [vanga/indian-high-court-judgments](https://github.com/vanga/indian-high-court-judgments) (CC-BY-4.0).

---

*NyayaSathi is the production successor to LexSearch. The judgment-search UI is preserved at `/` and reused as the RAG backbone.*
