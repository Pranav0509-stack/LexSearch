# न्यायसाथी (NyayaSathi) v12

India's free AI legal helpline — voice-driven, phone-callable, 11 languages.

## Features

| Feature | Details |
|---|---|
| **11 Languages** | Hindi, English, Bengali, Telugu, Tamil, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia |
| **Voice Engine** | Sarvam Bulbul v3 TTS — optimized speakers per language, low temperature for clarity |
| **Pure Native Script** | LLM generates pure Devanagari/native script — no English words in Hindi/regional responses |
| **Phone Integration** | Exotel webhook — real phone number, multi-language IVR (each option in its own language) |
| **AI Brain** | Gemini 2.5 Flash + Sarvam parallel race — fastest response wins |
| **RAG** | BM25 retrieval over Indian legal corpus — SC judgments, BNS, IPC, RERA, DV Act |
| **FAQ Templates** | Pre-built Hindi/English responses for common queries (FIR, cheque bounce, DV, etc.) |
| **Self-Correction** | Transcript confidence scoring — detects poor audio, asks to repeat |
| **3-Layer Guardrails** | Pre-AI keyword + LLM system prompt + Post-AI validator |
| **Eval Engine** | 68-test self-eval suite — `node eval-engine.js` |

## Quick Start

```bash
npm install
cp .env.example .env
# Fill in your API keys in .env
npm start
# Open http://localhost:3000
```

## Environment Variables

```env
SARVAM_API_KEY=       # From dashboard.sarvam.ai
GEMINI_API_KEY=       # From aistudio.google.com
EXOTEL_API_KEY=       # From Exotel dashboard → Settings → API
EXOTEL_API_TOKEN=     # From Exotel dashboard → Settings → API
EXOTEL_SID=           # Your Exotel Account SID
EXOTEL_CALLER_ID=     # Your ExoPhone number (e.g. 09513886363)
PUBLIC_URL=           # Your public URL (run: cloudflared tunnel --url http://localhost:3000)
```

## Phone Integration (Exotel)

1. Get Exotel account at [exotel.com](https://exotel.com)
2. Run: `cloudflared tunnel --url http://localhost:3000` — copy the HTTPS URL
3. Set `PUBLIC_URL=https://xxx.trycloudflare.com` in `.env`
4. In Exotel dashboard → ExoPhone → set incoming webhook to:
   `https://your-public-url/api/phone/exotel/call-start`
5. Call your ExoPhone — hear the multi-language IVR, press a digit, speak!

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/ask` | POST | RAG + AI + TTS |
| `/api/ask-stream` | POST | Streaming response |
| `/api/stt` | POST | Speech to text |
| `/api/tts` | POST | Text to speech |
| `/api/filler` | GET | Filler audio (thinking sound) |
| `/api/health` | GET | Health + cache stats |
| `/api/phone/exotel/call-start` | POST | Exotel incoming call webhook |
| `/api/phone/exotel/gather` | POST | DTMF language selection |
| `/api/phone/exotel/audio` | POST | Recording → STT → AI → TTS |
| `/api/phone/exotel/call` | POST | Trigger outbound test call |
| `/api/phone/exotel/info` | GET | Exotel setup status |
| `/api/ivr-prompt` | GET | Multi-language IVR audio |

## Helplines Referenced

| Helpline | Number |
|---|---|
| NALSA (Free Lawyer) | 15100 |
| Tele-Law | 1516 |
| Consumer | 1915 |
| Women | 181 |
| Cyber Crime | 1930 |
| Police | 112 |
| Child | 1098 |

## Voice Configuration

Each language uses the clearest Bulbul v3 speaker at low temperature (0.3–0.45) for rural users:

| Language | Speaker | Notes |
|---|---|---|
| Hindi | rahul | Clear male, natural diction |
| English | shubh | Indian accent, v3 default |
| Bengali | simran | Clear female |
| Telugu | priya | Fast & clear |
| Tamil | kavitha | Slightly slower for clarity |
| Marathi | ritu | Clear female |
| Gujarati | anand | Clear male |
| Kannada | kavya | Clear female |
| Malayalam | kavitha | Slightly slower |
| Punjabi | anand | Clear male |
| Odia | pooja | Clear female |

---

*NyayaSathi provides legal information, not legal advice. For formal legal representation, contact NALSA 15100.*
