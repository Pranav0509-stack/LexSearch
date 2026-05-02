# Sanhita — Call surface plan

> **Goal:** let an Indian advocate query Sanhita by **phone call or
> WhatsApp voice note**, in their own language, and get back a
> grounded legal memo by SMS / WhatsApp / call-back. Same answers, same
> citations, same Gemini brain — just a non-browser entry point.

This doc is self-contained for a fresh Claude session. Read it, then
read `SANHITA.md` for the existing app surface.

---

## What "the call part" means

Three distinct surfaces to build, in priority order:

1. **Inbound voice call** — advocate dials a Twilio Indian DID, speaks
   their question (Hindi / Tamil / English / etc.), gets the answer
   read back by TTS in the same language, with the full memo + cites
   sent via SMS / WhatsApp afterwards.

2. **Inbound WhatsApp voice note** — same model but async: lawyer
   sends a WhatsApp voice note, Sanhita transcribes (Sarvam Saaras),
   answers, replies with the audio answer + a text memo.

3. **Outbound call-back from the web app** — from the Assistant pane,
   click "Call me with the answer" → Twilio places an outbound call
   reading the latest answer aloud in the chosen language. Useful for
   advocates driving to court who already have a thread open.

All three reuse the **existing `brief_service.answer_question` /
`answer_open` pipeline** — no new LLM, no new validator. Just adapters.

---

## Architecture

```
Phone call --> Twilio Voice  ----+
                                 |
WhatsApp voice --> Twilio WA ----+--> /api/call/{voice,whatsapp}
                                 |       |
                                 |       v
                                 |   Sarvam Saaras (STT, IN langs)
                                 |       |
                                 |       v
                                 |   brief_service.answer_question()  <-- same brain
                                 |       |
                                 |       v
                                 |   Sarvam Bulbul (TTS, IN langs)
                                 |       |
                                 v       v
                              Twilio TwiML <Say>/<Play>
                                 |
                                 +--> SMS/WhatsApp: full memo + [n] cites
```

---

## What to build

### Phase A — Twilio inbound voice (~3 days)

**New file: `call_service.py`**
- `handle_incoming_call(from_number, language_hint) -> TwiML`
  - Greet caller in Hindi by default ("Sanhita mein swagat hai. Apna
    sawal poochiye.")
  - `<Gather input="speech" speechTimeout="auto" language="hi-IN"
    action="/api/call/voice/answer">`
- `handle_voice_answer(speech_result, from_number) -> TwiML`
  - Light-weight intent detection (chit-chat vs research) using the
    existing `_is_chitchat` / `_detect_intent` from `server.py`.
  - Call `brief_service.answer_question(speech_result, hits=[],
    language=detected_lang)` — `hits=[]` lets the BM25 path fire if
    enabled, otherwise the model just talks.
  - Truncate the answer to the first 2 sentences for `<Say>`. Send
    the full memo over SMS / WhatsApp follow-up.
  - Persist as a row in a new `call_transcripts` SQLite table so the
    Dashboard's Activity feed shows "+91-9876543210 asked about
    anticipatory bail".

**New endpoints (`server.py`):**
- `POST /api/call/voice` — Twilio Voice webhook
- `POST /api/call/voice/answer` — speech-result handler
- `POST /api/call/voice/sms-followup` — sends the full memo as SMS

**TwiML helpers** — `twilio` SDK is already on PyPI. Add
`twilio>=9.3.0` to `requirements.txt`.

**Twilio config:**
- Buy an Indian DID (~$1/mo) at console.twilio.com.
- Set the Voice URL to `https://api.sanhita.ai/api/call/voice`.
- Enable speech recognition for Hindi, Tamil, Telugu, Bengali, English (multi-lang Gather is supported in Twilio).

**Smoke test:**
- Buy a $1 Indian Twilio DID.
- Call it from your phone.
- Speak: "Section 138 ki limitation kya hai?"
- Expect: Hindi voice answer "138 ke liye limitation 30 din hain
  notice ke baad…" + SMS with full memo + cites.

---

### Phase B — WhatsApp voice note (~2 days)

**Twilio WhatsApp Sandbox** is free; production needs a BSP.

- New endpoint: `POST /api/call/whatsapp` — Twilio WA webhook.
- If message contains a `MediaUrl0`, fetch the audio, run through
  Sarvam Saaras STT, then `answer_question`, then send back two
  WhatsApp messages: (1) Sarvam Bulbul TTS audio file, (2) text memo
  with cites.
- If text-only message, skip STT, go straight to answer.

**Sarvam wiring:**
- `llm/sarvam.py` already has `translate()`. Add `stt(audio_bytes,
  language) -> str` and `tts(text, language) -> bytes` helpers using
  the same `mayura:v1` API surface.

---

### Phase C — Outbound call-back from web (~1 day)

**Frontend:**
- In `ChatBubble` action row (next to Copy / Email / Save as Doc),
  add a "Call me" button that opens a small modal asking for a phone
  number + language.

**Backend:**
- `POST /api/call/outbound` — `{phone, message_id, language}` →
  `twilio.client.calls.create(to=phone, from_=DID, url=…)` where the
  URL hosts a TwiML that `<Say>`s the answer in the chosen language.

**Cost guard:** rate-limit to 3 outbound calls per user per hour.

---

## Files to create / modify

| File | What |
|---|---|
| **NEW** `call_service.py` | TwiML builders + STT/TTS bridges |
| **NEW** `llm/sarvam.py` (extend) | Add `stt()` and `tts()` helpers |
| `server.py` | Add `/api/call/{voice,voice/answer,whatsapp,outbound}` endpoints |
| `auth.py` | Add `call_transcripts` table |
| `requirements.txt` | Add `twilio>=9.3.0` |
| `.env.example` | Add `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_VOICE_DID`, `TWILIO_WA_FROM` |
| `web/src/app/app/page.tsx` | "Call me" button in `ChatBubble` action row |

---

## Existing code to reuse

- `brief_service.answer_question()` — drop in directly.
- `_is_chitchat()` and `_detect_intent()` from `server.py`.
- `validators/input_guards.py` — same length / scope / PII guards.
- `llm/sarvam.py` `translate()` — pattern to follow for `stt()`/`tts()`.
- `realtime.broadcast()` — fan out a `call:answered` event so the
  Dashboard shows live call activity alongside web admin actions.

---

## Costs

| Item | Cost |
|---|---|
| Twilio India voice (inbound) | $0.014/min |
| Twilio India SMS | $0.0075/SMS |
| Twilio India DID | ~$1/mo |
| WhatsApp Business API | $0.0065/conversation |
| Sarvam Saaras (STT) | usage-priced (~$0.001/sec) |
| Sarvam Bulbul (TTS) | usage-priced |
| Gemini Flash answer | ~$0.001/answer |

Per call (90 sec, ~30 sec speech, 60 sec TTS playback, 1 SMS):
- Voice: $0.014 × 1.5 = $0.021
- STT: $0.001 × 30 = $0.030
- TTS: ~$0.005
- Gemini: $0.001
- SMS: $0.0075
- **~$0.065 per call**

At ₹5 (~$0.06) revenue per call (typical Indian micro-payment), the
unit economics work if 90%+ of calls convert to a satisfactory answer.

---

## Smoke checklist

- [ ] Twilio Indian DID purchased and pointed at `/api/call/voice`.
- [ ] `TWILIO_*` env vars set in Railway / Render.
- [ ] `requirements.txt` has `twilio>=9.3.0`; `pip install` clean.
- [ ] `server.py` route `POST /api/call/voice` returns valid TwiML
      (use `curl -X POST` with form data shaped like Twilio's webhook).
- [ ] End-to-end: call the DID, ask a §138 NI Act question, hear an
      answer in Hindi, receive an SMS with the full memo.
- [ ] Dashboard's Activity feed shows the call event live (Socket.io
      `call:answered` broadcast).

---

## Out of scope for this work-stream

- Native iOS/Android app (Phase 7+).
- IVR menus / multi-step voice flows (just a single Q→A loop).
- Voice-to-voice conversation memory across calls (each call is its
  own thread; tie to a user via `from_number` lookup).
- Compliance: TRAI DLT for SMS in India is required for outbound SMS
  to non-Twilio-customers. Inbound is fine; outbound needs a header
  registered with one of the operator panels (Airtel/Jio/Vi).
