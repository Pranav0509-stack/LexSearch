# NyayaSathi v15 — System Architecture

## Voice Flow (End-to-End)

```
User Speaks (mic)
    │
    ▼
┌──────────────────┐
│  Frontend (HTML)  │
│  MediaRecorder    │
│  webm/opus 16kHz  │
└────────┬─────────┘
         │ POST /api/stt (FormData: audio blob + lang)
         ▼
┌──────────────────┐
│  Sarvam STT      │   ← 3500ms timeout
│  saarika:v2.5    │
│  11 languages    │
└────────┬─────────┘
         │ transcript
         ▼
┌──────────────────┐
│  Frontend send() │
│  POST /api/ask-stream (NDJSON)
│  body: { message, lang, speaker, history }
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  SERVER /api/ask-stream                       │
│                                               │
│  1. L1 Guardrail (isLegalQuery)              │
│     └─ blocks non-legal queries              │
│                                               │
│  2. RAG Retrieval (BM25 3-tier)              │
│     ├─ SC_CORPUS (Supreme Court judgments)    │
│     ├─ HC_CORPUS (High Court cases)          │
│     └─ DC_CORPUS (District Court + rural)    │
│     Cross-lingual: 65+ Hindi→English maps    │
│                                               │
│  3. Gemini STREAMING (new architecture)      │
│     URL: streamGenerateContent?alt=sse       │
│     ├─ Tokens stream in via SSE              │
│     ├─ Buffer until sentence boundary (.!?।) │
│     ├─ Each sentence → TTS immediately       │
│     └─ Audio chunk sent via NDJSON instantly  │
│                                               │
│  4. Fallback (if streaming fails):           │
│     ├─ callGemini() non-streaming (3000ms)   │
│     ├─ callSarvam() 200ms delayed (4000ms)   │
│     └─ Promise.race, first wins              │
│                                               │
│  5. Post-processing (10-rule pipeline)       │
│     stripMarkdown → enforceLanguage →        │
│     verifyCitations → detectEmergency →      │
│     ensureNALSA → enforceWordLimit →         │
│     filterNonLegal → convertNumbers →        │
│     filterProfanity → guardHallucination     │
│                                               │
│  6. NDJSON Response Chunks:                  │
│     { type: "reply", reply, model, aiMs }    │
│     { type: "audio", audio: base64, index }  │
│     { type: "done", aiMs, ttsMs, segments }  │
└──────────────────────────────────────────────┘
         │
         │ NDJSON stream
         ▼
┌──────────────────────────────────────────────┐
│  Frontend Audio Queue                         │
│                                               │
│  audioQ[0] ──play──► audioQ[1] ──play──► ... │
│                                               │
│  Gap handling: skip undefined entries         │
│  200ms pause between chunks                  │
│  Safety: 20s busy timeout auto-resets        │
│                                               │
│  When all chunks played + streamDone:        │
│  └─ finishSend() → busy=false → startRec()  │
└──────────────────────────────────────────────┘
```

## Speaker System (Voice Consistency)

```
Frontend: speakerPref = 'male' | 'female'
    │
    │ Sent in every API call: speaker: speakerPref
    ▼
Server: generateTTS(text, langCode, customSpeaker)
    │
    ├─ customSpeaker === "female"?
    │   ├─ YES → getVoiceParams(lang, "female") → LANG_VOICE_FEMALE
    │   └─ NO  → getVoiceParams(lang, undefined) → LANG_VOICE (male)
    │
    ▼
TTS Cache Key: `${lang}:${speaker}:${text}` ← INCLUDES SPEAKER
    │
    ├─ Cache hit → return cached audio (correct speaker)
    └─ Cache miss → Sarvam Bulbul v3 API → cache + return

MALE SPEAKERS:
  hi-IN: rahul    en-IN: kabir    bn-IN: kabir
  te-IN: anand    ta-IN: anand    mr-IN: kabir
  gu-IN: anand    kn-IN: kabir    ml-IN: anand
  pa-IN: anand    od-IN: kabir

FEMALE SPEAKERS:
  hi-IN: ritu     en-IN: ishita   bn-IN: simran
  te-IN: kavitha  ta-IN: kavitha  mr-IN: shruti
  gu-IN: rupali   kn-IN: kavya   ml-IN: kavitha
  pa-IN: simran   od-IN: pooja

FILLER AUDIO: Also keyed by speaker
  Cache: fillerCache[`${lang}:female`] or fillerCache[lang]
  Frontend prefetch: /api/filler?lang=hi-IN&speaker=female
```

## Speed Architecture (Target: <2s perceived)

```
OLD FLOW (4-6s):
  STT(1.5s) → full LLM(2s) → all TTS chunks(1.5s) → play
  Total: ~5s before first audio

NEW FLOW (1.5-2.5s perceived):
  STT(1s) → Gemini STREAMS tokens → first sentence(0.5s) → TTS(0.5s) → PLAY
  │                                    │
  │                                    └─ User hears first sentence at ~2s
  │
  └─ Remaining sentences TTS in parallel while user listens

TIMEOUTS:
  STT:           3500ms (was 4500)
  Gemini Flash:  3000ms (was 3500) / 5000ms streaming
  Gemini Gemma:  6000ms (was 8000)
  Sarvam LLM:   4000ms (was 6000 race timeout)
  TTS per chunk: 3000ms (was 4000)
  apiFetch:      8000ms default
  Frontend:      15s stream timeout, 20s busy safety
```

## Anti-Hallucination Pipeline

```
1. ACT_SECTION_RANGES (26 Acts validated)
   BNS(395), IPC(511), BNSS(531), CrPC(484), CPC(158),
   BSA(170), IEA(167), POCSO(46), POSH(30), RERA(92),
   NI(147), IT Act(90), RTI(31), Consumer(107),
   HMA(30), MVA(217), Companies(484), DV Act(37),
   NDPS(83), SC/ST(22), Land Acquisition(114),
   TPA(137), Contract(238), Arbitration(87),
   Arms(41), HSA(40), Maternity(28)

2. Citation Grounding Score (threshold: 0.7)
   - Extract citations from response
   - Verify against RAG chunks retrieved
   - Score < 0.7 → strip unverified sections
   - Salvage grounded sentences, add NALSA helpline

3. guardHallucination()
   - Regex matches Act+Section patterns
   - Validates section number within Act's range
   - Strips sentences with out-of-range sections
```

## Bug Fixes Applied (v15.1)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Voice switches mid-chat | TTS cache key = `lang:text` (no speaker) | Cache key = `lang:speaker:text` |
| Flow dies after 1st response | audioQ gap when TTS fails → playNext stalls | Skip undefined gaps, simpler completion check |
| App freezes permanently | busy=true never reset if audio stalls | 20s safety timeout auto-resets busy |
| Filler plays wrong voice | Pre-warmed with male only, no speaker param | Filler endpoint accepts speaker, cache keyed by speaker |
| 4-6s latency | Wait for full LLM → then start TTS | Gemini SSE streaming → TTS per sentence as it arrives |
| Languages sound foreign | FILLERS in Roman transliteration | All 11 languages in native scripts |

## File Map

| File | Purpose | Lines |
|------|---------|-------|
| `server.js` | All endpoints, LLM race, TTS/STT, phone | ~2100 |
| `voice-engine.js` | TTS normalization, speaker maps, segmentation | ~380 |
| `rules-engine.js` | 10-rule post-LLM pipeline, hallucination guard | ~430 |
| `rag-tiers.js` | 3-tier BM25 RAG, Hindi mapping, corpus | ~700 |
| `citation-guard.js` | Citation extraction, grounding score | ~320 |
| `public/index.html` | Frontend SPA, voice UI, audio queue | ~1850 |
| `eval-engine.js` | Automated test suite | ~200 |
