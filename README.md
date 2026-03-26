# न्यायसाथी (NyayaSathi) v13

**India's Free AI Legal Helpline** — Just call and talk. 11 languages. No app. No fees.

> *A rickshaw driver in Lucknow got cheated of Rs 40,000. He didn't know Section 138 of the Negotiable Instruments Act could get his money back — with interest. He never went to court. Because he didn't know he could. This is the story of 92% of India. So I built something about it.*

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What It Does

NyayaSathi is a voice-first AI legal helpline that picks up your call, listens in your language, and tells you exactly which law protects you — with the specific Act, Section, Court, and step-by-step action plan.

**No app download. No typing. No lawyer fees. Just call and talk.**

```
You Speak (11 languages)
    → Sarvam STT (Speech-to-Text)
        → Gemini AI + Sarvam 105B (Legal Brain + RAG)
            → Sarvam TTS Bulbul v3 (Text-to-Speech)
                → You hear the answer
```

## Features

| Feature | Details |
|---|---|
| **11 Languages** | Hindi, English, Bengali, Telugu, Tamil, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia |
| **Voice Engine** | Sarvam Bulbul v3 TTS — `shubh` (male) / `kavya` (female) voices, temperature 0.5 |
| **AI Brain** | Gemini 2.5 Flash + Sarvam 105B raced concurrently — fastest response wins |
| **RAG** | BM25 retrieval over 50+ Supreme Court judgments, BNS/IPC/BNSS mapping, RERA, DV Act |
| **FAQ Templates** | 16+ pre-built answers (FIR, cheque bounce, land grab, MGNREGA, caste violence, ration card, panchayat, false cases) |
| **Phone Integration** | Exotel 1800 toll-free webhook — real phone number, multi-language IVR |
| **TTS Normalization** | Numbers → words, Sections → spoken form, URLs → readable, abbreviations letter-spaced |
| **Self-Correction** | Transcript confidence scoring — detects poor audio, asks to repeat |
| **10-Rule Pipeline** | stripMarkdown → enforceLanguage → verifyCitations → detectEmergency → ensureNALSA → enforceWordLimit → filterNonLegal → convertNumbers → filterProfanity → guardHallucination |
| **Pure Native Script** | Hindi responses in pure Devanagari — zero English words leak through |
| **Eval Engine** | 68-test self-eval suite — `node eval-engine.js` |

## Quick Start

```bash
git clone https://github.com/pranavpandey2511/NyayaSathi.git
cd NyayaSathi
npm install
cp .env.example .env
# Add your API keys (both are FREE):
#   SARVAM_API_KEY from https://dashboard.sarvam.ai
#   GEMINI_API_KEY from https://aistudio.google.com
node server.js
# Open http://localhost:3000
```

### One-Command Start
```bash
npm start
```

## How to Use

### Web Browser (Easiest)
1. Open `http://localhost:3000`
2. Click **Start Free Call**
3. Pick your language (Hindi, English, or 9 more)
4. **Speak** your legal question — or type it
5. Get instant legal guidance with voice response

### Phone Call (via Exotel)
1. Dial your Exotel toll-free number
2. Listen to the multi-language IVR menu
3. Press a digit to select language (1=Hindi, 2=English...)
4. Speak your legal problem after the beep
5. Hear the AI lawyer respond in your language

### Quick Suggestion Chips
The UI shows common queries per language:
- Hindi: किराया समस्या, ऑनलाइन धोखाधड़ी, तनख्वाह नहीं मिली, एफआईआर कैसे करें
- English: Rent problem, Online fraud, Salary not paid, How to file FIR

### Village / Rural Scenarios
Built specifically for rural India — handles queries like:
- "ज़मीन पर कब्ज़ा हो गया" → Land grab legal steps + BNS Section 330
- "ऊँची जाति वालों ने मारा" → SC/ST Atrocities Act + FIR + compensation
- "मनरेगा का पैसा नहीं आया" → MGNREGA rights + 15-day deadline + helpline
- "राशन कार्ड नहीं मिल रहा" → National Food Security Act + complaint process
- "सरपंच पैसे खा रहा है" → RTI + Prevention of Corruption Act
- "झूठे केस में फँसाया" → Anticipatory bail + FIR quashing + DK Basu rights

## Environment Variables

```env
# Required (both FREE)
SARVAM_API_KEY=       # https://dashboard.sarvam.ai (Free tier)
GEMINI_API_KEY=       # https://aistudio.google.com (Free tier)

# Optional — Phone Integration (Exotel)
EXOTEL_API_KEY=       # Exotel dashboard → Settings → API
EXOTEL_API_TOKEN=     # Exotel dashboard → Settings → API
EXOTEL_SID=           # Your Exotel Account SID
EXOTEL_CALLER_ID=     # Your ExoPhone number
EXOTEL_WEBHOOK_TOKEN= # Custom webhook auth token
PUBLIC_URL=           # Your public HTTPS URL (see below)
```

## Exotel Phone Integration (Detailed Setup)

### Prerequisites
- [Exotel](https://exotel.com) account with an ExoPhone number
- [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) (free) or [ngrok](https://ngrok.com)

### Step-by-Step Setup

**1. Get your public URL:**
```bash
# Option A: Cloudflare Tunnel (free, no account needed)
cloudflared tunnel --url http://localhost:3000

# Option B: ngrok
ngrok http 3000
```
Copy the HTTPS URL (e.g., `https://xxx.trycloudflare.com`)

**2. Set environment variables:**
```env
EXOTEL_API_KEY=your_api_key
EXOTEL_API_TOKEN=your_api_token
EXOTEL_SID=your_account_sid
EXOTEL_CALLER_ID=your_exophone_number
PUBLIC_URL=https://xxx.trycloudflare.com
```

**3. Configure Exotel webhook:**
- Go to Exotel Dashboard → ExoPhone → your number
- Set incoming call webhook to: `POST https://your-url/api/phone/exotel/call-start`

**4. Start the server:**
```bash
node server.js
```
You'll see:
```
Exotel Webhook: https://xxx.trycloudflare.com/api/phone/exotel/call-start
Test Call:      POST /api/phone/exotel/call {"to":"+919XXXXXXXXX"}
```

**5. Test with an outbound call:**
```bash
curl -X POST http://localhost:3000/api/phone/exotel/call \
  -H "Content-Type: application/json" \
  -d '{"to": "+919876543210"}'
```

### Call Flow
```
Incoming Call
  → IVR: "Hindi ke liye 1 dabayein, English ke liye 2..."
    → User presses digit
      → Greeting in selected language
        → Record user's voice (up to 15 seconds)
          → Sarvam STT → AI (Gemini/Sarvam 105B race) → Rules Engine → Sarvam TTS
            → Play response audio
              → Record next question (auto-continues)
```

### Exotel API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/phone/exotel/call-start` | POST | Incoming call webhook — starts IVR |
| `/api/phone/exotel/gather` | POST | DTMF digit received — sets language |
| `/api/phone/exotel/audio` | POST | Voice recording → STT → AI → TTS → play |
| `/api/phone/exotel/call` | POST | Trigger outbound test call |
| `/api/phone/exotel/info` | GET | Exotel configuration status |

## All API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/ask` | POST | Question → RAG + AI + TTS (JSON response) |
| `/api/ask-stream` | POST | Streaming NDJSON — text + audio chunks |
| `/api/stt` | POST | Speech-to-Text (audio file → text) |
| `/api/tts` | POST | Text-to-Speech (text → WAV audio) |
| `/api/filler` | GET | Filler/thinking audio per language |
| `/api/health` | GET | Health check + cache stats |
| `/api/ivr-prompt` | GET | Multi-language IVR prompt audio |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    server.js                         │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Sarvam   │  │ Gemini   │  │ Sarvam 105B      │  │
│  │ STT v2.5 │  │ 2.5 Flash│  │ (MoE, 105B, FREE)│  │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│       │              └────────┬────────┘             │
│       ▼                       ▼                      │
│  ┌─────────┐  ┌──────────────────────────────────┐  │
│  │  FAQ    │  │  Promise.race(gemini, sarvam105b) │  │
│  │ Templates│  │  First response wins              │  │
│  └────┬────┘  └──────────────┬───────────────────┘  │
│       │                       │                      │
│       └───────────┬───────────┘                      │
│                   ▼                                  │
│  ┌────────────────────────────────────────────────┐  │
│  │         rules-engine.js (10-rule pipeline)     │  │
│  │  stripMarkdown → enforceLanguage → citations   │  │
│  │  → emergency → NALSA → wordLimit → nonLegal    │  │
│  │  → numbers → profanity → hallucination         │  │
│  └──────────────────────┬─────────────────────────┘  │
│                         ▼                            │
│  ┌────────────────────────────────────────────────┐  │
│  │     voice-engine.js (TTS normalization)        │  │
│  │  Numbers→words, Sections→spoken, URLs removed  │  │
│  │  Abbreviations letter-spaced, phones digit-by  │  │
│  └──────────────────────┬─────────────────────────┘  │
│                         ▼                            │
│  ┌──────────┐  ┌──────────────────────────────────┐  │
│  │ Sarvam   │  │ Exotel Phone Integration         │  │
│  │ TTS v3   │  │ IVR → DTMF → Record → Loop      │  │
│  │ Bulbul   │  │ ExoML XML responses              │  │
│  └──────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Voice Configuration

All languages use Sarvam Bulbul v3 with optimized pace per language:

| Language | Male Speaker | Female Speaker | Pace |
|---|---|---|---|
| Hindi | shubh | kavya | 1.0 |
| English | shubh | kavya | 1.05 |
| Bengali | shubh | kavya | 0.95 |
| Telugu | shubh | kavya | 0.95 |
| Tamil | shubh | kavya | 0.9 |
| Marathi | shubh | kavya | 0.95 |
| Gujarati | shubh | kavya | 0.95 |
| Kannada | shubh | kavya | 0.9 |
| Malayalam | shubh | kavya | 0.85 |
| Punjabi | shubh | kavya | 0.95 |
| Odia | shubh | kavya | 0.95 |

## Legal Domains Covered

- FIR Filing (BNSS Section 173)
- Cheque Bounce (NI Act Section 138)
- Online Fraud / Cyber Crime (IT Act + BNS)
- Domestic Violence (DV Act 2005 + BNS 85)
- Consumer Complaints (Consumer Protection Act 2019)
- Property / Land Disputes (RERA, Revenue Courts)
- Divorce / Family Law (Hindu Marriage Act, Muslim Personal Law)
- Bail & Arrest (BNSS Section 482, DK Basu guidelines)
- RTI (Right to Information Act 2005)
- Salary / Labour Issues (Code on Wages 2019, ID Act)
- **Village/Rural**: Land Grabbing, Caste Violence (SC/ST Act), MGNREGA Wages, Ration Card, Panchayat Corruption, False Cases

## Helplines Referenced

| Helpline | Number | For |
|---|---|---|
| NALSA | 15100 | Free lawyer (mentioned in every response) |
| Tele-Law | 1516 | Free legal consultation |
| Consumer | 1915 | Consumer complaints |
| Women | 181 | Domestic violence, harassment |
| Cyber Crime | 1930 | Online fraud (money freeze) |
| Police | 112 | Emergency |
| Child | 1098 | Child abuse |
| Labour | 14434 | Salary / job issues |
| MGNREGA | 1800-111-0100 | Rural employment |

## Running the Eval Suite

```bash
node eval-engine.js
```
Runs 68 test scenarios covering:
- Language purity (no English in Hindi responses)
- Legal accuracy (correct Act + Section citations)
- Guardrail effectiveness (blocks non-legal queries)
- Response time benchmarks

## Tech Stack

| Component | Technology | Cost |
|---|---|---|
| AI Brain | Google Gemini 2.5 Flash + Sarvam 105B | Free |
| Speech-to-Text | Sarvam Saarika v2.5 | Free tier |
| Text-to-Speech | Sarvam Bulbul v3 (39 voices) | Free tier |
| Phone | Exotel (1800 toll-free) | Paid |
| Server | Node.js + Express | Free |
| Tunnel | Cloudflare Tunnel | Free |
| Legal Corpus | 50+ SC Judgments + BNS/IPC Mapping | Open |

**Total API cost: Rs 0** (Sarvam + Gemini both offer free tiers)

## File Structure

```
├── server.js          # Main server — all API routes, AI race, Exotel webhooks
├── voice-engine.js    # TTS normalization, speaker config, transcript scoring
├── rules-engine.js    # 10-rule post-LLM response pipeline
├── eval-engine.js     # 68-test evaluation suite
├── public/
│   └── index.html     # Single-page frontend (vanilla JS, no framework)
├── legal-data/        # RAG corpus — SC judgments, BNS/IPC mapping
├── .env.example       # Environment variable template
├── package.json
└── README.md
```

## Contributing

PRs welcome. Key areas:
- More regional language FAQ templates
- Additional Supreme Court judgments in RAG corpus
- Better STT accuracy for noisy phone environments
- Accessibility improvements for the web UI

## License

MIT

---

**NyayaSathi provides legal information, not legal advice. For formal legal representation, contact NALSA 15100.**

*Knowledge of law shouldn't be a privilege of the rich.*
