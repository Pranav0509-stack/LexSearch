/**
 * NyayaSathi v14 — India's Advanced AI Legal Agent
 * ════════════════════════════════════════════════
 * Features:
 *  - RAG: Three-tier BM25 (Supreme Court / High Court / District Court)
 *  - Security: SSRF protection, webhook auth, crypto sessions, prompt injection defense
 *  - Rules Engine: 10-rule post-LLM pipeline (citation verify, hallucination guard, etc.)
 *  - Latency: Promise.race LLM pattern, timing instrumentation
 *  - Self-correction: transcript confidence + response validation
 *  - Phone dial-in: Exotel + Sarvam telephony webhooks
 *  - Advanced guardrails: 3-layer (pre-AI, LLM system, post-AI)
 *  - All 11 Sarvam languages with optimal voice + pace per language
 *  - Gender detection + female speaker switch
 *  - Silence detection + 5-minute call duration limit
 *  - Number/section/currency expansion for natural TTS
 *  - Eval engine integration (/api/eval/run)
 *  - Streaming-ready architecture
 */

require("dotenv").config({ path: require("path").join(__dirname, ".env") });
const express = require("express");
const multer = require("multer");
const path = require("path");

// Internal modules
const { buildContext } = require("./rag.js");
const { computeGroundingScore, sanitizeResponse } = require("./citation-guard.js");
const {
    normalizeTTS, segmentForTTS, getVoiceParams,
    scoreTranscript, getClarificationPrompt, validateResponse,
} = require("./voice-engine.js");
const {
    isAllowedRecordingUrl, verifyWebhookToken, appendWebhookToken,
    generateSessionId, sanitizeForLLM, timer,
} = require("./security.js");
const { applyRules, stripMarkdown } = require("./rules-engine.js");
const caseStore = require("./case-store.js");
const intent = require("./intent-classifier.js");
const slots = require("./slot-templates.js");
const { extractAndUpdate } = require("./entity-extractor.js");
const distress = require("./distress-detector.js");
const lawyerMatch = require("./lawyer-match.js");

const app = express();

// Security headers
try {
    const helmet = require("helmet");
    app.use(helmet({ contentSecurityPolicy: false, crossOriginEmbedderPolicy: false }));
} catch { console.log("[SECURITY] helmet not installed — run: npm install helmet"); }

// Rate limiting (replaces manual implementation)
try {
    const rateLimit = require("express-rate-limit");
    app.use("/api", rateLimit({ windowMs: 60000, max: 100, standardHeaders: true, legacyHeaders: false }));
} catch { console.log("[SECURITY] express-rate-limit not installed — using manual rate limiter"); }

app.use(express.json({ limit: "4mb" }));
app.use(express.urlencoded({ extended: true, limit: "1mb" })); // Exotel sends form-encoded data
app.use(express.static(path.join(__dirname, "public")));
// Fallback: serve index.html from root if it exists there too
app.get("/", (req, res, next) => {
    const pubIndex = path.join(__dirname, "public", "index.html");
    const rootIndex = path.join(__dirname, "index.html");
    const fs = require("fs");
    if (fs.existsSync(pubIndex)) return res.sendFile(pubIndex);
    if (fs.existsSync(rootIndex)) return res.sendFile(rootIndex);
    next();
});
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

const SK = process.env.SARVAM_API_KEY;
const GK = process.env.GEMINI_API_KEY;
const HEADERS = { "api-subscription-key": SK, "Content-Type": "application/json" };

// ═══════════════════════════════════════════════════════════
//  MODEL ROTATION — Smart quota tracking + cascade
// ═══════════════════════════════════════════════════════════
const MODEL_PRIORITY = ["gemini-2.5-flash", "gemini-2.0-flash", "gemma-3-27b-it"];
const MODEL_LIMITS = {
    "gemini-2.5-flash": 18,   // keep 2 buffer from 20/day free-tier limit
    "gemini-2.0-flash": 18,   // same free-tier structure
    "gemma-3-27b-it": Infinity,
};
const modelUsage = {}; // { model: { count, resetDate } }

function getTodayStr() { return new Date().toISOString().slice(0, 10); }

function recordUsage(model) {
    const today = getTodayStr();
    if (!modelUsage[model] || modelUsage[model].resetDate !== today) {
        modelUsage[model] = { count: 0, resetDate: today };
    }
    modelUsage[model].count++;
    console.log(`[QUOTA] ${model}: ${modelUsage[model].count}/${MODEL_LIMITS[model]} used today`);
}

function getAvailableModel() {
    const today = getTodayStr();
    for (const model of MODEL_PRIORITY) {
        const limit = MODEL_LIMITS[model];
        if (limit === Infinity) return model;
        const usage = modelUsage[model];
        if (!usage || usage.resetDate !== today) return model; // fresh day
        if (usage.count < limit) return model;
        console.log(`[QUOTA] ${model} exhausted (${usage.count}/${limit}), trying next...`);
    }
    return "gemma-3-27b-it"; // ultimate fallback
}

function getModelStatus() {
    const today = getTodayStr();
    return MODEL_PRIORITY.map(m => {
        const usage = modelUsage[m];
        const used = (usage && usage.resetDate === today) ? usage.count : 0;
        return `${m}: ${used}/${MODEL_LIMITS[m]}`;
    }).join(", ");
}

// ═══════════════════════════════════════════════════════════
//  LAYER 1 GUARDRAILS — Pre-AI keyword classification
// ═══════════════════════════════════════════════════════════

const LEGAL_WORDS = new Set([
    // Core legal terms
    "kanoon", "kanooni", "legal", "law", "act", "section", "court", "police", "fir", "complaint",
    "lawyer", "vakeel", "advocate", "judge", "bail", "arrest", "theft", "fraud", "scam",
    "cheating", "murder", "assault", "harassment", "stalking", "cyber", "cybercrime",
    // Property
    "property", "rent", "landlord", "tenant", "deposit", "builder", "flat", "house", "plot",
    "rera", "registry", "mutation", "encroachment", "possession", "eviction", "lease",
    "agreement", "stamp", "mortgage", "conveyance", "title", "deed", "khata", "patta",
    // Family
    "divorce", "custody", "maintenance", "alimony", "dowry", "domestic", "violence",
    "marriage", "shaadi", "talaq", "wife", "husband", "child", "baccha", "guardian",
    "adoption", "dv", "498a", "protection", "order", "streedhan",
    // Labour
    "salary", "wages", "employer", "boss", "fired", "termination", "labour", "pf", "esi", "gratuity",
    "overtime", "bonus", "retrenchment", "workmen", "industrial", "dispute", "epf", "esic",
    "notice", "period", "appointment", "relieving", "certificate", "naukri", "job", "kaam",
    // Consumer
    "consumer", "refund", "defective", "product", "warranty", "forum", "commission",
    "service", "deficiency", "unfair", "trade", "practice", "misleading",
    // Accident & Insurance
    "accident", "insurance", "claim", "challan", "motor", "vehicle", "mact", "hit", "run",
    "compensation", "injury", "negligence", "rash", "driving",
    // Government & Rights
    "rti", "information", "government", "corruption", "bribe", "lokpal", "lokayukta",
    // Financial crimes
    "cheque", "bounce", "dishonour", "negotiable", "forgery", "embezzlement",
    "money", "laundering", "benami", "hawala", "ponzi", "chit",
    // Succession
    "will", "succession", "inheritance", "nominee", "death", "probate", "intestate",
    "heir", "ancestral", "partition", "coparcener",
    // Social justice
    "caste", "reservation", "atrocity", "scheduled", "tribe", "obc", "ews", "discrimination",
    "pocso", "minor", "abuse", "rape", "juvenile", "trafficking",
    // Civil
    "contract", "breach", "damages", "relief", "injunction", "specific", "performance",
    "suit", "decree", "execution", "attachment", "garnishee",
    // Procedure
    "petition", "appeal", "revision", "writ", "pil", "stay", "hearing",
    "summons", "warrant", "cognizable", "bailable", "anticipatory",
    // Legal aid
    "nalsa", "dlsa", "helpline", "aid", "free", "tele-law",
    // Education legal
    "ragging", "antiragging", "rte", "fee", "fees", "admission", "scholarship",
    "college", "school", "university", "vidyalaya",
    // Revenue / land mutation
    "mutation", "tehsil", "revenue", "zameen", "jameen", "khasra", "khatauni",
    "patwari", "registry", "daakhil", "kharij", "intkal",
    // Hindi financial
    "paisa", "rupay", "karz", "loan", "emi", "bank", "udhaar", "fareb",
    // Police/admin transliterated
    "thana", "daroga", "inspector", "dsp", "sp", "ssp",
    "adalat", "nyayalay", "panchayat", "tehsil", "collector",
    // Property Hindi
    "kiraya", "makaan", "ghar", "zameen", "jamin", "khet", "dukaan",
    // Help signals
    "problem", "samasya", "pareshani", "dikkat", "issue", "takleef", "mushkil",
    "madad", "help", "haq", "adhikar", "right",
    // Revenue
    "naksha", "tehsildar", "patwari", "lekhpal", "sarpanch", "pradhan",
    // Misc legal
    "pension", "retirement", "defamation", "slander", "libel",
    "trespass", "nuisance", "easement", "arbitration", "mediation", "conciliation",
    "notary", "affidavit", "oath", "declaration", "surety", "guarantor",
    "limitation", "caveat", "restitution", "passport", "visa", "immigration",
    "vyapar", "business", "partner", "company", "firm", "gst", "tax", "manager", "hr",
    "gaali", "maar", "peet", "chori", "loot", "dhamki", "threat", "dhoka",
    "posh", "icc", "lcc", "ngt", "environment", "pollution",
    // BNS/BNSS references
    "bns", "bnss", "bsa", "crpc", "ipc", "iea", "nia", "nia2",
]);

const LEGAL_PHRASES = [
    "police station", "legal aid", "15100", "1516", "1915", "181", "112", "1930", "1098",
    "fir darj", "fir nahi", "paise wapas", "salary nahi", "rent wapas",
    "cheque bounce", "domestic violence", "online fraud", "cyber crime", "digital arrest",
    "consumer court", "district court", "high court", "supreme court",
    "lok adalat", "family court", "labour court", "legal notice", "zero fir",
    "anticipatory bail", "motor accident", "hit and run",
    "property dispute", "land dispute", "sexual harassment",
    "child custody", "wrongful termination", "identity theft",
    "tenant rights", "police complaint", "kanooni madad", "vakeel chahiye", "scam ho gaya",
    "atm fraud", "upi fraud", "credit card fraud", "bank fraud",
    "emi bounce", "loan recovery", "sarfaesi", "drt", "nclt",
    "noise pollution", "ngt complaint", "tribal rights", "forest rights",
    "rti application", "information commission", "whistleblower",
    "posh complaint", "icc committee", "maternity benefit",
];

const BLOCK_WORDS = new Set([
    // Sports
    "cricket", "ipl", "score", "game", "khel", "match", "team", "player", "captain",
    "batsman", "bowler", "wicket", "goal", "football", "tennis", "badminton",
    "olympic", "medal", "champion", "worldcup", "cup", "jeeta", "jeeti", "haar", "haari",
    // Entertainment
    "movie", "film", "song", "recipe", "cooking", "gana", "gaana", "singer", "actor",
    "actress", "heroine", "hero", "bollywood", "hollywood", "netflix", "webseries", "serial", "drama",
    // Food
    "khana", "pakana", "biryani", "pizza", "burger", "chai", "coffee", "restaurant",
    // Weather
    "weather", "mausam", "barish", "garmi", "sardi", "thand", "dhoop", "temperature",
    // Humor
    "joke", "mazak", "poem", "story", "kahani", "shayari", "hasna", "funny", "comedy", "standup",
    // Medical (not injury-related)
    "dawa", "dawai", "tablet", "syrup", "injection", "fever", "bukhar",
    "dard", "headache", "stomach", "treatment", "upchar", "bimari", "rog", "clinic", "pathology", "xray",
    // Academics (only block generic study queries, not legal ones like ragging/RTE/fees)
    "padhai", "exam", "homework", "assignment",
    // Finance (speculative)
    "stock", "bitcoin", "crypto", "trading", "mutual", "sip",
    // Romance
    "girlfriend", "boyfriend", "breakup", "dating", "love", "pyar", "romance",
    // Travel/shopping
    "travel", "tourism", "hotel", "flight", "booking", "shopping", "amazon", "flipkart", "discount", "sale", "offer",
    // Astrology
    "astrology", "horoscope", "kundli", "rashifal", "vastu", "numerology",
    // Beauty
    "makeup", "beauty", "skincare", "fashion", "style",
    // Tech (not legal-related)
    "programming", "python", "java", "javascript", "code", "coding", "software", "hardware",
    "ayurveda", "homeopathy", "yoga", "diet", "exercise", "gym",
]);

// Hindi/Devanagari block list
const BLOCK_DEVANAGARI = [
    "वर्ल्ड कप", "क्रिकेट", "आईपीएल", "मैच", "स्कोर", "खेल", "गोल",
    "फिल्म", "मूवी", "गाना", "गाने", "बॉलीवुड", "एक्टर", "हीरो", "हीरोइन",
    "खाना", "रेसिपी", "पकाना", "बिरयानी",
    "मौसम", "बारिश", "गर्मी", "सर्दी", "तापमान",
    "चुटकुला", "जोक", "कहानी", "शायरी", "कविता",
    "डॉक्टर", "दवाई", "हॉस्पिटल", "इलाज", "बीमारी",
    "पढ़ाई", "एग्जाम", "स्कूल", "कॉलेज",
    "शेयर मार्केट", "बिटकॉइन", "क्रिप्टो", "ट्रेडिंग",
    "प्यार", "ब्रेकअप", "रिश्ता", "गर्लफ्रेंड", "बॉयफ्रेंड",
    "ट्रैवल", "होटल", "फ्लाइट", "बुकिंग",
    "शॉपिंग", "डिस्काउंट", "ऑफर",
    "कुंडली", "राशिफल", "ज्योतिष", "वास्तु",
];

function isLegalQuery(text) {
    const origLower = text.toLowerCase();

    // Devanagari block check
    for (const phrase of BLOCK_DEVANAGARI) {
        if (text.includes(phrase) || origLower.includes(phrase.toLowerCase())) {
            const words = origLower.replace(/[^\w\s]/g, " ").split(/\s+/);
            const hasLegal = words.some(w => LEGAL_WORDS.has(w));
            if (!hasLegal) return { allow: false, reason: "devanagari_block" };
        }
    }

    const t = origLower.replace(/[^\w\s]/g, " ");
    const words = t.split(/\s+/).filter(Boolean);

    let hasLegal = false, hasBlock = false;
    for (const w of words) {
        if (LEGAL_WORDS.has(w)) hasLegal = true;
        if (BLOCK_WORDS.has(w)) hasBlock = true;
    }

    if (!hasLegal) {
        for (const p of LEGAL_PHRASES) {
            if (t.includes(p)) { hasLegal = true; break; }
        }
    }

    // Short messages (greetings, help, etc.) — allow through to AI
    if (words.length <= 4 && !hasBlock) return { allow: true, reason: "short_message" };

    if (hasBlock && !hasLegal) return { allow: false, reason: "block_word_no_legal" };
    return { allow: true, reason: hasLegal ? "legal_confirmed" : "no_signals_allow" };
}

// ═══════════════════════════════════════════════════════════
//  LAYER 2 — LLM SYSTEM PROMPTS (per language)
// ═══════════════════════════════════════════════════════════

const LANG_PROMPTS = {
    "en-IN": (rag) => `You are NyayaSathi, India's free AI legal helpline agent on a LIVE phone call.

DO NOT think out loud. DO NOT start with "Okay", "Let me", "The user", or any reasoning. Start DIRECTLY with your empathetic response.

CRITICAL: Your output goes DIRECTLY to a Text-to-Speech engine. Write ONLY plain spoken English sentences. NO Devanagari. NO markdown. NO bullets. NO asterisks. NO numbered lists. NO digits — write all numbers as words (say "one hundred thirty-eight" not "138").

STYLE: Think of yourself as a calm, experienced friend who happens to be a great lawyer. The person calling you is scared or confused. Talk to them like a human — warm, patient, reassuring. Use simple words. Short sentences. No legal jargon unless you immediately explain it. Avoid cold phrases like "procedural violation" or "limitation period" — say "it's past the deadline" or "the police didn't follow the rule." You name exact Acts, Sections, Courts, Forms, Helplines so they know what to do next, but you wrap them in plain language, not a lecture.

LEGAL REFERENCES:
${rag || "No specific legal reference found. Direct caller to NALSA helpline fifteen-one-hundred for a free lawyer. Do not cite any section or act number."}

HOW TO RESPOND:
Start with one warm line — "I understand, this must be stressful" or "Don't worry, you have rights here." Make them feel heard.
Then tell them clearly what the law says on their side — the Act, Section, or case — in a sentence a non-lawyer can understand.
Then give them the next two or three concrete steps — where to go, what to file, what to ask for.
End with a reassuring line. Only ask a follow-up question if you genuinely cannot answer without more info — and when you do, make it a SPECIFIC, USEFUL question like "Did the police refuse to write it down?" — not a generic "tell me more." Do NOT ask a question every turn. Most of the time, just answer.

HARD RULES:
- MAX 90 words. This is a phone call — keep it tight.
- Tone: warm, comforting, human. NEVER robotic or lecturing.
- ONLY Indian law questions. Non-legal: "I can only help with legal matters. Please tell me your legal problem."
- Be SPECIFIC with actions: "file Form A at District Consumer Commission within two years."
- ALWAYS mention NALSA fifteen-one-hundred for free lawyer.
- Ask a follow-up question ONLY when truly needed — and only a relevant one. Default: no question.
- NEVER use Devanagari script or Hindi words.
- Write numbers as words: "Section one hundred thirty-eight" not "Section 138".
- ONLY cite Acts and Sections from the LEGAL REFERENCES above. If the reference doesn't cover their issue, say so honestly and direct them to NALSA.`,

    "hi-IN": (rag) => `आप न्यायसाथी हैं — भारत की मुफ़्त कानूनी हेल्पलाइन। आप फ़ोन पर बात कर रहे हैं।

ज़रूरी: अपनी सोच प्रक्रिया मत लिखें। "Okay", "Let me", "The user" जैसे अंग्रेज़ी शब्दों से शुरू मत करें। सीधे हिंदी में जवाब दें।

सबसे ज़रूरी बात: आपका जवाब सीधे बोलने वाली मशीन में जाएगा। इसलिए:
- पूरा जवाब शुद्ध हिंदी देवनागरी में लिखें। कोई भी अंग्रेज़ी शब्द मत लिखें।
- कानूनी शब्द हिंदी में लिखें: "धारा" (Section), "अधिनियम" (Act), "अदालत" (Court), "उच्चतम न्यायालय" (Supreme Court), "ज़िला न्यायालय" (District Court), "उपभोक्ता आयोग" (Consumer Commission), "प्रथम सूचना रिपोर्ट" (FIR), "ज़मानत" (Bail), "याचिका" (Petition)
- सभी अंक हिंदी शब्दों में लिखें: "एक सौ अड़तीस" (138), "तीस दिन" (30 days), "पंद्रह सौ" (1500)
- फ़ोन नंबर भी हिंदी में: "एक पाँच एक शून्य शून्य" (15100)
- कोई मार्कडाउन, बुलेट, तारा चिह्न, या सूची नहीं। सीधे बोलचाल वाले वाक्य लिखें।

अंदाज़: आप एक समझदार बड़े भाई या दीदी हैं जो वकील भी हैं। सामने वाला परेशान है, डरा हुआ है। उससे इंसान की तरह बात कीजिए — प्यार से, धीरे से, हिम्मत देते हुए। आसान शब्द। छोटे वाक्य। कानूनी भाषा तब ही बोलिए जब साथ में आसान शब्दों में समझा दें। "प्रक्रियात्मक उल्लंघन" जैसे भारी शब्द नहीं — बस बोलिए "पुलिस ने नियम तोड़ा है" या "समय निकल गया है।" धारा और अधिनियम बताना ज़रूरी है ताकि अगला कदम साफ़ हो, लेकिन भाषा सरल रखें — भाषण मत दीजिए।

कानूनी संदर्भ:
${rag || "कोई विशेष कानूनी संदर्भ नहीं मिला। कॉलर को नालसा हेल्पलाइन एक पाँच एक शून्य शून्य पर फ़ोन करने को बोलें। कोई धारा या अधिनियम न बताएं।"}

जवाब कैसे दें:
पहले एक गर्मजोशी वाली लाइन — "मैं आपकी परेशानी समझ रहा हूँ, घबराइए मत" या "चिंता मत कीजिए, आपके पास अधिकार हैं।" उन्हें लगे कि कोई सुन रहा है।
फिर साफ़ बताइए क़ानून क्या कहता है — कौन सा अधिनियम, कौन सी धारा — पर आसान शब्दों में।
फिर दो-तीन ठोस कदम — कहाँ जाना है, क्या लिखवाना है, क्या माँगना है।
आख़िर में हिम्मत देने वाली एक लाइन। नालसा "एक पाँच एक शून्य शून्य" ज़रूर बताएं।
सवाल सिर्फ़ तब पूछें जब सच में ज़रूरी हो — और पूछें तो ठीक काम का सवाल, जैसे "क्या पुलिस ने लिखने से मना किया था?" — सामान्य "और बताइए" नहीं। हर बार सवाल मत पूछिए। ज़्यादातर सीधे जवाब दीजिए।

सख्त नियम:
- अधिकतम नब्बे शब्द। फ़ोन पर बात है — कम बोलें, काम की बात बोलें।
- लहजा गर्म, सहारा देने वाला, इंसानी। रोबोट की तरह या भाषण की तरह नहीं।
- सिर्फ़ कानूनी सवालों का जवाब दें। बाकी: "मैं सिर्फ़ कानूनी मामलों में मदद करता हूँ।"
- साफ़ बताएं: "ज़िला उपभोक्ता आयोग में दो साल के अंदर शिकायत दर्ज करें"
- नालसा "एक पाँच एक शून्य शून्य" हमेशा बताएं — मुफ़्त वकील मिलेगा।
- सवाल तब ही पूछें जब वाक़ई ज़रूरी हो। आम तौर पर सीधे जवाब दें।
- एक भी अंग्रेज़ी शब्द मत लिखें। पूरा जवाब देवनागरी में हो।
- केवल वही धारा और अधिनियम बताएं जो ऊपर कानूनी संदर्भ में दिए हैं।`,
};

// For other languages — template based on Hindi prompt structure
function getLangPrompt(langCode, ragContext) {
    if (LANG_PROMPTS[langCode]) return LANG_PROMPTS[langCode](ragContext);

    const langNames = {
        "bn-IN": "Bengali", "te-IN": "Telugu", "ta-IN": "Tamil", "mr-IN": "Marathi",
        "gu-IN": "Gujarati", "kn-IN": "Kannada", "ml-IN": "Malayalam",
        "pa-IN": "Punjabi", "od-IN": "Odia",
    };
    const langName = langNames[langCode] || langCode;

    return `You are NyayaSathi, India's free AI legal helpline on a LIVE phone call.

ABSOLUTE RULE: Your ENTIRE response must be in ${langName} using its native script. Do NOT write any English words, English sentences, or Roman script. Every single word must be in ${langName} native script. Legal terms must also be in ${langName} script (transliterate them). Numbers must be written as ${langName} words.

DO NOT think out loud. DO NOT start with "Okay" or "Let me" or any reasoning. Start DIRECTLY with your empathetic response in ${langName}.

STYLE: Think of yourself as a calm, caring older sibling who is also a great lawyer. The caller is scared or confused — talk to them like a human. Warm, patient, simple words. Short sentences. No heavy legal jargon. Still name the exact Acts, Sections, Courts, Forms, Helplines so they know what to do — but wrap it in plain language, never a lecture.

LEGAL REFERENCES:
${ragContext || "No specific legal reference found. Direct caller to NALSA 15100 for a free lawyer. Do not cite any section or act number."}

HOW TO RESPOND:
Start with one warm line — let them feel heard.
Then tell them clearly what the law says on their side, in simple words.
Then two or three concrete steps — where to go, what to file, what to ask for.
End with a reassuring line. Only ask a follow-up question if you truly cannot answer without more info — and when you do, make it SPECIFIC and useful, not generic. Do NOT ask a question every turn.

HARD RULES:
- MAX 90 words. Phone call — keep it tight.
- Tone: warm, comforting, human. Never robotic or lecturing.
- ENTIRE response in ${langName} native script. ZERO English words.
- ONLY Indian law questions. Non-legal: politely refuse in ${langName}.
- Be SPECIFIC: name the Court, Form, deadline, helpline.
- Cite specific Act + Section from LEGAL REFERENCES only. If unsure, direct to NALSA 15100.
- ALWAYS mention NALSA 15100 for free lawyer.
- Ask a follow-up only when truly needed — default to no question.
- NO markdown, bullets, asterisks.`;
}

// ═══════════════════════════════════════════════════════════
//  LAYER 3 — Post-AI response validator
// ═══════════════════════════════════════════════════════════
function isLegalResponse(reply) {
    const r = reply.toLowerCase();
    // If response contains clearly non-legal content
    const nonLegalSignals = [
        "cricket", "ipl", "movie", "film", "biryani", "recipe", "weather", "mausam",
        "stock market", "bitcoin", "horoscope", "kundli", "dawa", "medicine", "doctor se",
        "gym", "yoga", "diet", "exercise", "girlfriend", "boyfriend",
    ];
    for (const s of nonLegalSignals) {
        if (r.includes(s)) {
            // Only block if NO legal term present
            const legalTerms = ["section", "act", "court", "police", "fir", "nalsa", "15100", "vakeel",
                "lawyer", "petition", "bail", "complaint", "kanoon", "kanooni", "right", "haq"];
            const hasLegal = legalTerms.some(lt => r.includes(lt));
            if (!hasLegal) return false;
        }
    }
    return true;
}

// ═══════════════════════════════════════════════════════════
//  REFUSALS — natural spoken language
// ═══════════════════════════════════════════════════════════
const REFUSALS = {
    "en-IN": "I'm NyayaSathi, your legal assistant. I can only help with Indian legal matters — like property disputes, FIR filing, salary issues, fraud, divorce, or consumer complaints. Please tell me your legal problem. For a free lawyer, call NALSA on 15100.",
    "hi-IN": "मैं NyayaSathi हूँ, आपका कानूनी सहायक। मैं केवल कानूनी मामलों में मदद करता हूँ — जैसे FIR, संपत्ति, वेतन, धोखाधड़ी, तलाक, या उपभोक्ता शिकायत। कोई कानूनी समस्या बताइए। निःशुल्क वकील के लिए NALSA 15100 पर कॉल करें।",
    "bn-IN": "Aami NyayaSathi, apnar ainī sahāyak. Aami shudhu bhārater āiner bishaye sahāytā korte pāri. Āpanār āinī samasya bolan. NALSA 15100-e call korun binamūlye āinji pāben.",
    "te-IN": "Nenu NyayaSathi, mee legal sahāyakudu. Nenu kevalam India legal vishayalatho sahāyapadata. Mee legal samasya cheppandi. NALSA 15100 ki call cheyyandi free vakeel kosam.",
    "ta-IN": "Nān NyayaSathi, ungaḷ sattam udam seyvavar. Nān Indiyā satta vishayangaḷil matam udavi cheyyalām. Ungaḷ sattap piracchanai solluṅgal. Ilavasamāna vakkīlukku NALSA 15100-ai azhaikkunga.",
    "mr-IN": "Mi NyayaSathi aahe, tumcha kanooni sahayak. Mi fakt kanooni vishayavar madad karto. Tumchi kanooni samasya sanga. Muft vakil saathi NALSA 15100 var call kara.",
    "gu-IN": "Hu NyayaSathi chhu, tamaro kanooni sahayak. Hu fakt kanooni babaton maa madad kari shakhu chhu. Tamari kanooni samasya kaho. Muft vakil mate NALSA 15100 par call karo.",
    "kn-IN": "Nānu NyayaSathi, nimmā kānooni sahāyaka. Nānu kevala kānooni vishayagaḷalli sahāya māḍabahudu. Nimmā kānooni samasye heli. Muft vakīlugāgi NALSA 15100 kare māḍi.",
    "ml-IN": "Ñān NyayaSathi āṇu, ningaḷuṭe niyama sahāyakan. Ñān Bhāratīya niyama vishayaṅṅaḷil mātramaṇu sahāyikkuka. Ningaḷuṭe niyama prashnam parayan. Muft lawyer-kku NALSA 15100-il viḷikku.",
    "pa-IN": "Main NyayaSathi haan, tuhāḍā kānooni sahāyak. Main sirf kānooni māmliyān vich madad karda haan. Apni kānooni samassia dasso. Muft vakīl lait NALSA 15100 te call karo.",
    "od-IN": "Mu NyayaSathi, āpankara āini sahāyak. Mu kevala India āini vishayare sahāya kariba. Āpanka āini samasya kuhanti. Muft vakil pāin NALSA 15100 re phone karanti.",
};

function getRefusal(lang) {
    return REFUSALS[lang] || REFUSALS["hi-IN"];
}

// ═══════════════════════════════════════════════════════════
//  UTILS
// ═══════════════════════════════════════════════════════════
function sanitize(text, maxLen = 500) {
    if (typeof text !== "string") return "";
    return text.replace(/<[^>]*>/g, "").replace(/[<>"'`]/g, "").replace(/\s+/g, " ").trim().slice(0, maxLen);
}

// cleanLLMResponse replaced by applyRules() from rules-engine.js
// All post-LLM processing now goes through the unified 10-rule pipeline

// Fallback rate limiter (used only if express-rate-limit not installed)
let hasRateLimit = false;
try { require("express-rate-limit"); hasRateLimit = true; } catch {}
if (!hasRateLimit) {
    const rateMap = new Map();
    // Clean up rate map every 5 minutes to prevent memory leak
    setInterval(() => { const now = Date.now(); for (const [ip, hits] of rateMap) { const filtered = hits.filter(t => now - t < 60000); if (filtered.length === 0) rateMap.delete(ip); else rateMap.set(ip, filtered); } }, 300000);
    app.use("/api", (req, res, next) => {
        const ip = req.ip || "x"; const now = Date.now();
        const hits = (rateMap.get(ip) || []).filter(t => now - t < 60000);
        if (hits.length >= 100) return res.status(429).json({ error: "Too many requests." });
        hits.push(now); rateMap.set(ip, hits); next();
    });
}

// Fetch with timeout
async function apiFetch(url, opts, timeoutMs = 8000) {
    const ac = new AbortController();
    const t = setTimeout(() => ac.abort(), timeoutMs);
    try { const r = await fetch(url, { ...opts, signal: ac.signal }); clearTimeout(t); return r; }
    catch (e) { clearTimeout(t); throw e; }
}

// ═══════════════════════════════════════════════════════════
//  GEMINI / GEMMA — Primary LLM with Smart Model Rotation
// ═══════════════════════════════════════════════════════════
async function callGemini(systemPrompt, messages, maxTokens = 120, modelName = null) {
    if (!GK) return null;
    const MODEL = modelName || getAvailableModel();
    const IS_GEMMA = MODEL.startsWith("gemma");
    const timeout = IS_GEMMA ? 6000 : 3000;

    // Convert from OpenAI-style messages to Gemini contents format
    const contents = [];
    for (const m of messages) {
        if (m.role === "system") continue;
        contents.push({
            role: m.role === "assistant" ? "model" : "user",
            parts: [{ text: m.content }],
        });
    }
    if (contents.length && contents[0].role !== "user") {
        contents.unshift({ role: "user", parts: [{ text: "Hello" }] });
    }

    // Gemma: no systemInstruction — inject as user→model turn with constraint restating
    let finalContents = contents;
    if (systemPrompt && IS_GEMMA) {
        const ack = systemPrompt.includes("देवनागरी")
            ? "समझा। मैं NyayaSathi हूँ — Supreme Court स्तर का वकील। नियम: (1) केवल कानूनी मदद, (2) अधिकतम 90 शब्द, SPECIFIC — Act, Section, Court, deadline, helpline, (3) केवल संदर्भ के Section, (4) सहानुभूति + step-by-step, (5) सवाल तभी जब ज़रूरत हो, (6) कोई markdown नहीं।"
            : "Understood. I am NyayaSathi — a Supreme Court lawyer. Rules: (1) Legal help only, (2) Max 90 words SPECIFIC — Act, Section, Court, deadline, helpline, (3) Only cite from references, (4) Empathy + step-by-step, (5) Ask question ONLY if needed, (6) No markdown, numbers as words.";
        finalContents = [
            { role: "user", parts: [{ text: `[SYSTEM INSTRUCTIONS — follow these exactly]\n${systemPrompt}` }] },
            { role: "model", parts: [{ text: ack }] },
            ...contents,
        ];
    }

    const body = {
        contents: finalContents,
        generationConfig: { temperature: 0.3, maxOutputTokens: maxTokens },
    };
    if (systemPrompt && !IS_GEMMA) {
        body.systemInstruction = { parts: [{ text: systemPrompt }] };
    }

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`;
    console.log(`[MODEL] Using ${MODEL}`);
    const r = await apiFetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-goog-api-key": GK },
        body: JSON.stringify(body),
    }, timeout);
    if (!r.ok) {
        const errText = await r.text().catch(() => "");
        console.log(`[GEMINI] ${MODEL} ${r.status} ${errText.slice(0, 150)}`);
        if (r.status === 429) {
            modelUsage[MODEL] = { count: MODEL_LIMITS[MODEL] || 999, resetDate: getTodayStr() };
            console.log(`[QUOTA] ${MODEL} marked exhausted (429)`);
        }
        return null;
    }
    const d = await r.json();
    const text = d.candidates?.[0]?.content?.parts?.[0]?.text || null;
    if (text) recordUsage(MODEL);
    return text;
}

// callGeminiStream removed — SSE parsing was unreliable, using direct callGemini instead

// ═══════════════════════════════════════════════════════════
//  SARVAM 105B — Primary Indian LLM (FREE, 22 languages, 128K context)
// ═══════════════════════════════════════════════════════════
async function callSarvam(messages, maxTokens = 500) {
    const r = await apiFetch("https://api.sarvam.ai/v1/chat/completions", {
        method: "POST", headers: HEADERS,
        body: JSON.stringify({
            model: "sarvam-105b",
            messages,
            max_tokens: Math.max(maxTokens, 500), // 105B needs headroom for reasoning + answer
            temperature: 0.3,
            reasoning_effort: "low", // minimize thinking tokens for speed
        }),
    }, 8000);
    if (!r.ok) {
        const errBody = await r.text().catch(() => "");
        console.log(`[SARVAM-105B] ${r.status} ${errBody.slice(0, 150)}`);
        return null;
    }
    const d = await r.json();
    return d.choices?.[0]?.message?.content || null;
}

// ═══════════════════════════════════════════════════════════
//  TRANSLITERATION — Roman Hinglish → Devanagari for TTS
// ═══════════════════════════════════════════════════════════
const translitCache = new Map();
async function transliterateForTTS(text, langCode) {
    // LLM now writes in native script for all languages — no transliteration needed
    // This function is kept for backward compatibility but is a no-op
    return text;
}

// ═══════════════════════════════════════════════════════════
//  TTS CACHE — LRU with 200 entries
// ═══════════════════════════════════════════════════════════
const ttsCache = new Map();
function ttsCacheKey(text, lang, speaker) { return `${lang}:${speaker || 'default'}:${text.slice(0, 120)}`; }
function ttsCacheSet(key, buf) {
    if (ttsCache.size >= 200) ttsCache.delete(ttsCache.keys().next().value);
    ttsCache.set(key, buf);
}

// ═══════════════════════════════════════════════════════════
//  RESPONSE CACHE — LRU with 500 entries, 1hr TTL
// ═══════════════════════════════════════════════════════════
const responseCache = new Map();
const RESPONSE_CACHE_TTL = 60 * 60 * 1000; // 1 hour
const RESPONSE_CACHE_MAX = 500;

function responseCacheKey(msg, lang) {
    const tokens = msg.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).filter(Boolean).sort().join(" ");
    return `${lang}:${tokens}`;
}

function responseCacheGet(key) {
    const entry = responseCache.get(key);
    if (!entry) return null;
    if (Date.now() - entry.timestamp > RESPONSE_CACHE_TTL) {
        responseCache.delete(key);
        return null;
    }
    // Move to end (LRU)
    responseCache.delete(key);
    responseCache.set(key, entry);
    return entry;
}

function responseCacheSet(key, reply, model, ragContext) {
    if (responseCache.size >= RESPONSE_CACHE_MAX) {
        responseCache.delete(responseCache.keys().next().value);
    }
    responseCache.set(key, { reply, model, ragContext, timestamp: Date.now() });
}

// ═══════════════════════════════════════════════════════════
//  FAQ TEMPLATES — Pre-built answers for instant response
// ═══════════════════════════════════════════════════════════
const FAQ_TEMPLATES = {
    "hi-IN": [
        { patterns: ["fir", "kaise", "darj", "police", "thana", "एफआईआर", "एफ़आईआर", "एफ़", "दर्ज", "थाना", "थाने"], answer: "घबराइए मत, मैं समझ रहा हूँ। एफ़ आई आर लिखवाना आपका अधिकार है। नज़दीकी थाने जाइए, बी एन एस एस धारा एक सौ तिहत्तर के तहत पुलिस मना नहीं कर सकती। अगर मना करे तो एस पी को लिखित शिकायत भेज दीजिए, ज़ीरो एफ़ आई आर किसी भी थाने में दर्ज होती है। उच्चतम न्यायालय ने ललिता कुमारी केस में यह साफ़ कहा है। मुफ़्त वकील चाहिए तो नालसा एक पाँच एक शून्य शून्य पर फ़ोन कीजिए। आप अकेले नहीं हैं।" },
        { patterns: ["cheque", "bounce", "check", "dishonour", "चेक", "बाउंस", "बाउन्स"], answer: "चिंता मत कीजिए, चेक बाउंस में कानून आपके साथ है। बाउंस के तीस दिन के अंदर सामने वाले को कानूनी नोटिस भेजिए। पंद्रह दिन में पैसे न आएं तो मजिस्ट्रेट अदालत में शिकायत दर्ज कर दीजिए। परक्राम्य लिखत अधिनियम धारा एक सौ अड़तीस के तहत दो साल तक की सज़ा और चेक की रकम से दोगुना जुर्माना हो सकता है। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील मिलेगा। कागज़ात संभाल कर रखिए।" },
        { patterns: ["online", "fraud", "scam", "upi", "cyber", "phishing", "otp", "ऑनलाइन", "धोखाधड़ी", "धोखा", "साइबर", "फ्रॉड"], answer: "घबराइए मत, जल्दी करेंगे तो पैसे वापस आ सकते हैं। सबसे पहले एक नौ तीन शून्य पर फ़ोन कीजिए, यह राष्ट्रीय साइबर हेल्पलाइन है, पैसे फ्रीज़ हो सकते हैं। साइबरक्राइम डॉट जी ओ वी डॉट इन पर शिकायत दर्ज कीजिए। बैंक को तुरंत बताइए खाता बंद करवाइए। एफ़ आई आर आई टी अधिनियम धारा छिहत्तर डी और बी एन एस धारा तीन सौ अठारह के तहत होगी। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील।" },
        { patterns: ["domestic", "violence", "marpit", "pati", "sasural", "घरेलू", "हिंसा", "मारपीट", "पति", "ससुराल"], answer: "मैं आपकी तकलीफ़ समझ रहा हूँ। आप अकेली नहीं हैं, कानून पूरी तरह आपके साथ है। पहले महिला हेल्पलाइन एक आठ एक पर फ़ोन कीजिए, चौबीस घंटे मदद मिलेगी। घरेलू हिंसा अधिनियम दो हज़ार पाँच के तहत सुरक्षा आदेश और भरण-पोषण दोनों माँग सकती हैं, नज़दीकी संरक्षण अधिकारी से मिलिए। शेल्टर होम में रहने का भी आपका अधिकार है। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त महिला वकील मिलेगी। अभी क्या आप सुरक्षित जगह पर हैं?" },
        { patterns: ["salary", "vetan", "naukri", "job", "termination", "fired", "वेतन", "नौकरी", "तनख्वाह", "सैलरी", "निकाला", "पगार", "मिली"], answer: "तकलीफ़ समझ रहा हूँ, मेहनत का पैसा रुका है तो गुस्सा आना स्वाभाविक है। श्रम आयुक्त के पास शिकायत दीजिए, हेल्पलाइन एक चार चार तीन चार है। वेतन न मिले तो श्रम अदालत में केस कीजिए। गलत तरीके से निकाले गए हैं तो औद्योगिक विवाद अधिनियम धारा पच्चीस एफ़ के तहत हर साल के लिए पंद्रह दिन का वेतन मुआवज़ा मिलता है, पाँच साल बाद ग्रेच्युटी भी। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील।" },
        { patterns: ["consumer", "complaint", "product", "service", "refund", "उपभोक्ता", "शिकायत", "रिफंड", "सामान", "सर्विस"], answer: "चिंता मत कीजिए, उपभोक्ता संरक्षण अधिनियम दो हज़ार उन्नीस पूरी तरह आपके साथ है। मुफ़्त हेल्पलाइन एक आठ शून्य शून्य एक एक चार शून्य शून्य शून्य पर फ़ोन कीजिए। ज़िला उपभोक्ता आयोग में एक करोड़ तक की शिकायत दर्ज हो सकती है, ई-दाखिल डॉट एन आई सी डॉट इन पर ऑनलाइन भी कर सकते हैं। समय सीमा दो साल है। बिल और व्हाट्सऐप चैट जैसे सबूत संभाल कर रखिए। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील।" },
        { patterns: ["bail", "giraftari", "arrest", "jail", "ज़मानत", "जमानत", "गिरफ्तारी", "गिरफ़्तारी", "जेल"], answer: "घबराइए मत, आपके अधिकार पूरी तरह सुरक्षित हैं। ज़मानती अपराध में थाने पर ही ज़मानत का हक़ है। ग़ैर-ज़मानती अपराध हो तो सत्र अदालत या उच्च न्यायालय में अर्ज़ी दीजिए। अग्रिम ज़मानत बी एन एस एस धारा चार सौ बयासी के तहत पहले ही मिल सकती है। अगर साठ या नब्बे दिन में आरोप पत्र न दाखिल हो तो ज़मानत का पक्का अधिकार है। डी के बसु फैसले के तहत गिरफ्तारी में परिवार को बताना पुलिस की ज़िम्मेदारी है। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील तुरंत।" },
        { patterns: ["rti", "information", "suchna", "आरटीआई", "सूचना", "जानकारी"], answer: "सूचना माँगना आपका अधिकार है, डरने की कोई बात नहीं। सूचना का अधिकार अधिनियम दो हज़ार पाँच के तहत दस रुपये की फीस के साथ किसी भी सरकारी दफ़्तर से जानकारी माँग सकते हैं। तीस दिन में जवाब देना ज़रूरी है। जवाब न आए तो पहली अपील तीस दिन में, और सूचना आयोग में नब्बे दिन में। ऑनलाइन आर टी आई डॉट जी ओ वी डॉट इन पर लगा सकते हैं। बी पी एल को फीस माफ़ है। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त मदद।" },
        { patterns: ["property", "zameen", "registry", "makaan", "flat", "builder", "संपत्ति", "ज़मीन", "जमीन", "रजिस्ट्री", "मकान", "फ्लैट", "बिल्डर", "किराया", "किरायेदार"], answer: "तकलीफ़ समझ रहा हूँ, संपत्ति के मामले बड़े तनाव वाले होते हैं। बिल्डर ने देरी की है तो रेरा प्राधिकरण में शिकायत कीजिए, यहाँ बिल्डर पर सख़्ती होती है। ज़मीन का विवाद है तो राजस्व या दीवानी अदालत में जाइए। अतिक्रमण हुआ है तो एस डी एम या ज़िला कलेक्टर को लिखित शिकायत। रजिस्ट्री और नामांतरण उप-पंजीयक और तहसील कार्यालय में होते हैं। सारे कागज़ात संभाल कर रखिए। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील।" },
        { patterns: ["divorce", "talaq", "shaadi", "तलाक", "तलाक़", "शादी", "विवाह"], answer: "मैं समझ रहा हूँ, यह समय भारी होता है। हिंदू विवाह अधिनियम धारा तेरह के तहत तलाक का आपका अधिकार है। दोनों सहमत हैं तो धारा तेरह बी के तहत आपसी सहमति से तलाक छह महीने में हो जाता है। एकतरफा तलाक क्रूरता, परित्याग या सात साल से लापता होने पर मिलता है। तीन तलाक़ अब अपराध है, तीन साल की सज़ा है। भरण-पोषण का अधिकार धारा एक सौ पच्चीस में है। परिवार अदालत में याचिका दाखिल कीजिए। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील।" },
        // ─── Village / Rural Specific FAQs ───
        { patterns: ["ज़मीन", "जमीन", "कब्ज़ा", "कब्जा", "zameen", "kabza", "भूमि", "ताकतवर", "ज़बरदस्ती"], answer: "आपकी तकलीफ़ समझ रहा हूँ, ज़मीन छिनने का डर बहुत बड़ा होता है। पहले तहसीलदार या एस डी एम को लिखित शिकायत दीजिए। बी एन एस धारा तीन सौ तीस के तहत अतिक्रमण अपराध है, थाने में एफ़ आई आर दर्ज करवाइए, पुलिस मना नहीं कर सकती। साथ ही दीवानी अदालत में कब्ज़ा वापसी का दावा ठोकिए। खतौनी, रजिस्ट्री, भू नक्शा जैसे सारे कागज़ संभाल कर रखिए, यही आपकी ताक़त हैं। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील मिलेगा।" },
        { patterns: ["जाति", "जात", "दलित", "ऊँची", "छुआछूत", "भेदभाव", "मारा", "पीटा", "caste", "dalit", "atrocity"], answer: "जो हुआ वह ग़लत है, और कानून पूरी तरह आपके साथ है। अनुसूचित जाति और जनजाति अत्याचार निवारण अधिनियम के तहत पुलिस को तुरंत एफ़ आई आर दर्ज करनी ही पड़ेगी, मना करने का हक़ उन्हें नहीं है। थाने में न सुनें तो सीधे ज़िला मजिस्ट्रेट या एस पी को लिखित शिकायत भेजिए। इस कानून में पीड़ित को सरकार से मुआवज़ा भी मिलता है। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील तुरंत मिलेगा। आप अकेले नहीं हैं।" },
        { patterns: ["मनरेगा", "नरेगा", "मजदूरी", "काम", "मज़दूरी", "mnrega", "nrega", "wages", "रोज़गार"], answer: "मेहनत का पैसा रुका है तो परेशानी समझ में आती है, पर आपका हक़ साफ़ है। मनरेगा में काम माँगने का अधिकार कानूनी है। पंद्रह दिन में काम न मिले तो बेरोज़गारी भत्ता मिलना चाहिए। मज़दूरी पंद्रह दिन में खाते में आनी ज़रूरी है, देर पर ब्याज़ मिलता है। ग्राम पंचायत सचिव से लिखित में काम माँगिए, रसीद लीजिए। शिकायत ज़िला कार्यक्रम अधिकारी को या एक आठ शून्य शून्य एक एक एक शून्य शून्य पर। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील।" },
        { patterns: ["राशन", "कार्ड", "राशन कार्ड", "ration", "card", "अनाज", "गेहूँ", "चावल", "बीपीएल"], answer: "चिंता मत कीजिए, राशन आपका अधिकार है। राष्ट्रीय खाद्य सुरक्षा अधिनियम के तहत हर ग़रीब परिवार को हर व्यक्ति के हिसाब से पाँच किलो अनाज हर महीने मिलना चाहिए। कार्ड नहीं है तो खाद्य विभाग के दफ़्तर में या ऑनलाइन आवेदन कीजिए। डीलर अनाज न दे तो ज़िला खाद्य अधिकारी को शिकायत, हेल्पलाइन एक नौ शून्य शून्य या एक आठ शून्य शून्य एक एक एक शून्य शून्य पर। राशन डायरी में हर एंट्री लिखवाइए। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त मदद।" },
        { patterns: ["पंचायत", "सरपंच", "प्रधान", "ग्राम", "गाँव", "panchayat", "sarpanch", "gram"], answer: "घबराइए मत, गाँव की सत्ता के ख़िलाफ़ भी कानून पूरी तरह आपके साथ है। ज़िला पंचायत अधिकारी या ज़िला मजिस्ट्रेट को लिखित शिकायत दीजिए। सरपंच फंड में घपला कर रहा है तो भ्रष्टाचार निवारण अधिनियम के तहत शिकायत करें, जाँच होगी। आर टी आई लगा कर पंचायत के हर खर्चे का हिसाब माँग सकते हैं, यह आपका हक़ है। ज़रूरत पड़े तो अविश्वास प्रस्ताव लाकर सरपंच हटाया भी जा सकता है। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील।" },
        { patterns: ["झूठा", "झूठी", "फर्ज़ी", "false", "fake", "फँसाया", "केस"], answer: "घबराइए मत, झूठे केस से कानून आपको बचा सकता है। सबसे पहले वकील से मिल कर अग्रिम ज़मानत की अर्ज़ी तैयार करवाइए, बी एन एस एस धारा चार सौ बयासी के तहत यह सत्र अदालत या उच्च न्यायालय में लगती है। झूठी एफ़ आई आर रद्द करवाने के लिए उच्च न्यायालय में याचिका दायर कीजिए। डी के बसु फैसले के तहत गिरफ्तारी में आपके अधिकार पूरी तरह सुरक्षित हैं, परिवार को बताना पुलिस की ज़िम्मेदारी है। नालसा एक पाँच एक शून्य शून्य पर तुरंत मुफ़्त वकील।" },
        // ─── Motor Vehicle / Accident FAQs ───
        { patterns: ["एक्सीडेंट", "एक्सिडेंट", "दुर्घटना", "टक्कर", "गाड़ी", "बाइक", "कार", "ट्रक", "accident", "hit", "road"], answer: "घबराइए मत, मैं साथ हूँ। सबसे पहले अगर कोई घायल है तो एक एक दो पर तुरंत फ़ोन कीजिए। घायल को अस्पताल पहुँचाना ज़रूरी है, मोटर वाहन अधिनियम धारा एक सौ चौंतीस कहती है मदद करनी ही होगी। फिर नज़दीकी थाने में एफ़ आई आर दर्ज करवाइए। मुआवज़े के लिए मोटर दुर्घटना दावा अधिकरण में मोटर वाहन अधिनियम धारा एक सौ छियासठ के तहत अर्ज़ी दीजिए, बीमा कंपनी को भी बता दीजिए। लाइसेंस, गाड़ी के कागज़, मेडिकल रिपोर्ट संभाल कर रखिए। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील। अभी बताइए कोई घायल तो नहीं है?" },
        { patterns: ["हिट एंड रन", "भागना", "भाग गया", "hit and run", "चालान", "ट्रैफिक", "traffic"], answer: "मैं समझ रहा हूँ, हिट एंड रन में बहुत घबराहट होती है, पर कानून आपके साथ है। गाड़ी का नंबर याद हो तो तुरंत थाने में बताइए। मोटर वाहन अधिनियम धारा एक सौ इकसठ के तहत सरकार से मुआवज़ा मिलता है, मौत पर दो लाख और गंभीर चोट पर पचास हज़ार तक। सोलेशियम फंड की अर्ज़ी ज़िला मजिस्ट्रेट को दीजिए। बी एन एस धारा एक सौ छह के तहत एफ़ आई आर भी ज़रूर दर्ज कराइए। मेडिकल पेपर संभालिए। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील।" },
    ],
    "en-IN": [
        { patterns: ["fir", "police", "register", "lodge", "complaint", "file", "how"], answer: "Don't worry, I'll walk you through this. Go to your nearest police station — under BNSS Section 173, the police cannot refuse to register your FIR for a cognizable offence. If they do refuse, send a written complaint to the SP by registered post. You can also file a Zero FIR at any station, regardless of where it happened. The Supreme Court made this crystal clear in Lalita Kumari. For a free lawyer, call NALSA on 15100. You're not alone in this." },
        { patterns: ["consumer", "complaint", "defective", "product", "refund", "service"], answer: "You have strong rights here, don't worry. Call the toll-free consumer helpline 1800-11-4000 or file online at e-daakhil. The District Consumer Disputes Commission handles claims up to one crore, and the time limit is two years from the problem. The Consumer Protection Act 2019 and the Supreme Court both say that delay or deficiency in service is compensable. Keep bills and WhatsApp messages safe as evidence. Call NALSA 15100 for a free lawyer." },
        { patterns: ["cheque", "bounce", "dishonour", "check"], answer: "Don't worry, the law is firmly on your side here. Send a legal notice within 30 days of the bounce memo. If the other person doesn't pay within 15 days of the notice, file a complaint in Magistrate Court under NI Act Section 138. The punishment can be up to two years in jail plus twice the cheque amount as fine. You must file within one month after the 15-day notice period expires. Keep the cheque and bounce memo safe. Call NALSA 15100 for free legal aid." },
        { patterns: ["online", "fraud", "cyber", "scam", "upi"], answer: "Don't panic — act fast and we can recover money. Call 1930, the National Cyber Crime Helpline, right away. They can freeze funds before the fraudster withdraws. Also report on the cyber crime portal. Notify your bank immediately to block the account. An FIR can be filed under IT Act Section 66D and BNS Section 318. Save every screenshot, message, and transaction SMS — those are evidence. Call NALSA 15100 for a free lawyer. Speed matters in fraud cases." },
        { patterns: ["domestic", "violence", "husband", "abuse"], answer: "I'm so sorry you're going through this. You are not alone, and the law fully protects you. Call the Women Helpline on 181 — they're available 24 hours. Under the Domestic Violence Act 2005, you can get a Protection Order, maintenance, and residence rights from the Magistrate Court. Meet your nearest Protection Officer. You have the right to a shelter home. An FIR under BNS Section 85 is also an option. NALSA 15100 provides a free woman lawyer. Right now — are you in a safe place?" },
        { patterns: ["salary", "job", "fired", "termination", "wages", "paid", "not"], answer: "I understand — it's frustrating when hard-earned money is held back. Complain to the Labour Commissioner, helpline 14434. If your salary is pending, file in Labour Court. Wrongful termination? The Industrial Disputes Act gives you 15 days' salary per year as retrenchment compensation. After five years of service, gratuity is also your right. Keep your offer letter, salary slips, and bank statements ready as evidence. Call NALSA 15100 for a free lawyer." },
        { patterns: ["property", "land", "rent", "flat", "builder", "tenant", "problem", "issue", "deposit"], answer: "I understand — property matters are stressful. If a builder has delayed, file with the RERA authority — they are strict on builders. For land disputes, go to Revenue Court or Civil Court. For encroachment, write to the SDM or District Collector. For registration, the Sub-Registrar office handles it. Tenant or landlord issues are covered by the Rent Control Act — both sides have rights. Keep every paper safe: registry, agreement, mutation, receipts. Call NALSA 15100 for a free lawyer." },
        { patterns: ["divorce", "separation", "marriage"], answer: "I understand this is a heavy time. Under the Hindu Marriage Act Section 13, you have a clear right to file for divorce. If both agree, mutual-consent divorce under Section 13B takes about six months. Otherwise grounds include cruelty, desertion, or seven years of separation. Triple talaq is now a criminal offence with up to three years' punishment. Maintenance is protected under Section 125. Petition goes in Family Court. Call NALSA 15100 for a free lawyer — they also help with mediation." },
        // Village / Rural FAQs
        { patterns: ["land", "grab", "encroach", "kabza", "zameen", "occupy"], answer: "I understand — land disputes are terrifying. You have strong remedies. File a written complaint with the Tehsildar or SDM. Under BNS Section 330, encroachment is a criminal offence — the police must register an FIR, they cannot refuse. Also file a civil suit for possession recovery in Civil Court. Keep every document — khatauni, registry, mutation, bhu-naksha. These are your strength. Call NALSA 15100 for a free lawyer. You're not powerless here." },
        { patterns: ["caste", "dalit", "atrocity", "discrimination", "untouchability"], answer: "What happened is wrong, and the law stands firmly with you. The SC/ST Prevention of Atrocities Act forces the police to immediately register an FIR — they have no right to refuse. If the police station doesn't help, complain directly to the District Magistrate or SP. The law also gives victims compensation from the government. Call NALSA 15100 for a free lawyer, right away. You are not alone in this." },
        { patterns: ["nrega", "mnrega", "mgnrega", "wages", "work", "employment", "guarantee"], answer: "I understand — unpaid labour is painful, but your rights are clear. Under MGNREGA you have a legal right to demand work. If work isn't given in 15 days, unemployment allowance is due. Wages must reach your bank in 15 days, and delay means interest. Submit your demand in writing to the Gram Panchayat Secretary — and keep the receipt. Complain to the District Programme Officer or call 1800-111-0100. Call NALSA 15100 for a free lawyer." },
        { patterns: ["ration", "card", "food", "grain", "bpl", "antyodaya"], answer: "Don't worry — ration is your legal right. Under the National Food Security Act, every poor family gets five kilograms of grain per person per month at subsidised rates. If you don't have a card, apply at the Food Department office or online. If the dealer refuses to give your grain, complain to the District Food Officer. Helplines: 1900 or 1800-111-0100. Always make the dealer write every entry in your ration diary. Call NALSA 15100 for free help." },
        { patterns: ["panchayat", "sarpanch", "village", "gram", "pradhan"], answer: "Don't be afraid — the law protects citizens against local power too. Write to the District Panchayat Officer or District Magistrate. If the Sarpanch is misusing funds, a complaint under the Prevention of Corruption Act can trigger investigation. Use RTI to get every detail of panchayat spending — that's your right. If needed, a no-confidence motion can remove a corrupt Sarpanch. Keep all evidence safely. Call NALSA 15100 for a free lawyer." },
        { patterns: ["false", "fake", "wrongful", "framed", "trap"], answer: "Don't panic — the law has safeguards for exactly this. Contact a lawyer immediately for anticipatory bail under BNSS Section 482; the application goes to Sessions Court or High Court. To quash a false FIR, file a petition in the High Court. Under the DK Basu guidelines, your rights during arrest are fully protected — the police must inform your family. Call NALSA 15100 for a free lawyer right away." },
        // Motor Vehicle / Accident FAQs
        { patterns: ["accident", "crash", "collision", "vehicle", "car", "bike", "truck", "road"], answer: "Take a breath — I'll walk you through this. If anyone is injured, call 112 right now — that's the first priority. Motor Vehicles Act Section 134 makes it mandatory to help the injured. Then file an FIR at the nearest police station. For compensation, file a claim in the Motor Accident Claims Tribunal under Motor Vehicles Act Section 166, and notify the insurance company. Keep the driver's licence, vehicle papers, and medical reports safe. Call NALSA 15100 for a free lawyer. First — is anyone hurt?" },
        { patterns: ["hit and run", "fled", "ran away", "absconded"], answer: "I understand — hit and run is terrifying, but the law will help you. If you remember the vehicle number, tell the police immediately. Under Motor Vehicles Act Section 161, the government pays compensation to hit and run victims through the Solatium Scheme — up to two lakh for death, fifty thousand for grievous injury. File the compensation application with the District Magistrate. File an FIR under BNS Section 106 for causing death by negligence. Keep all medical papers. Call NALSA 15100 for a free lawyer." },
    ],
};

function matchFAQ(msg, lang) {
    const templates = FAQ_TEMPLATES[lang] || FAQ_TEMPLATES["hi-IN"];
    const msgLower = msg.toLowerCase();
    const words = msgLower.split(/\s+/);

    let bestMatch = null;
    let bestScore = 0;

    for (const faq of templates) {
        let score = 0;
        let devanagariHit = false;
        for (const p of faq.patterns) {
            if (words.includes(p) || msgLower.includes(p)) {
                score++;
                // Devanagari patterns (non-ASCII) are highly specific — count double
                if (/[^\x00-\x7F]/.test(p) && p.length >= 3) devanagariHit = true;
            }
        }
        // Require 2+ English pattern matches, or 1+ specific Devanagari match
        const threshold = devanagariHit ? 1 : 2;
        if (score >= threshold && score > bestScore) {
            bestScore = score;
            bestMatch = faq;
        }
    }
    return bestMatch;
}

// ═══════════════════════════════════════════════════════════
//  SESSION MEMORY — Server-side conversation tracking
//
//  ONE source of truth, used by /api/ask, /api/ask-stream, and the phone
//  flow. The web UI persists `sessionId` in localStorage so a refresh keeps
//  the conversation alive. The phone flow keys by `phone || sessionId`.
//
//  History is bounded: the most recent 6 turns (3 user + 3 assistant) live
//  verbatim in `session.history`. Anything older gets compressed into
//  `session.summary` — a one-line "what we've talked about so far" string —
//  so the prompt stays under ~1,500 tokens no matter how long the call goes.
//  This is what kept turn-3 from imploding.
// ═══════════════════════════════════════════════════════════
const webSessions = new Map(); // sessionId → { history, lang, summary, case, lastActive }
const SESSION_TTL = 30 * 60 * 1000; // 30 min
const SESSION_MAX_TURNS = 6; // 3 user + 3 assistant kept verbatim

function getOrCreateSession(sessionId, opts = {}) {
    // Hot path: in-memory hit. Don't touch disk.
    if (sessionId && webSessions.has(sessionId)) {
        const s = webSessions.get(sessionId);
        s.lastActive = Date.now();
        return { session: s, sessionId, isReturning: false };
    }
    const newId = sessionId || generateSessionId("ws");

    // Cold path: try to attach to a persisted case (refresh, callback, etc.)
    let attached = null;
    try {
        attached = caseStore.attachSessionToCase({
            sessionId: newId,
            phone: opts.phone || null,
            lang: opts.lang || "hi-IN",
        });
    } catch (e) {
        console.warn("[SESSION] case attach failed:", e.message);
    }
    const c = attached?.case || null;

    const s = {
        history: c?.history?.slice(-SESSION_MAX_TURNS * 2) || [],
        lang: c?.lang || opts.lang || "hi-IN",
        summary: c?.summary || "",
        case: c,                                  // live case object — mutated in place
        lastUserMsg: "",
        lastActive: Date.now(),
        turnCount: c?.history ? Math.floor(c.history.length / 2) : 0,
        isReturning: !!attached?.isReturning,
    };
    webSessions.set(newId, s);
    return { session: s, sessionId: newId, isReturning: s.isReturning };
}

// Compress the oldest pair of turns into the running summary string before
// they're dropped from `history`. Cheap text-only compression — no LLM call.
// Phase C will upgrade this to a Gemini-Flash JSON-mode summary.
function compressOldestPair(session) {
    const oldUser = session.history[0]?.u === 1 ? session.history[0].t : "";
    const oldAss  = session.history[1]?.u === 0 ? session.history[1].t : "";
    if (!oldUser && !oldAss) return;
    // Keep it terse — the summary is for the LLM, not the user.
    const userPart = oldUser ? `user: ${oldUser.slice(0, 140)}` : "";
    const assPart  = oldAss  ? `you advised: ${oldAss.slice(0, 160)}` : "";
    const fragment = [userPart, assPart].filter(Boolean).join(" — ");
    session.summary = session.summary
        ? `${session.summary} | ${fragment}`
        : fragment;
    // Cap the summary itself so it can't grow unboundedly on very long calls.
    if (session.summary.length > 800) {
        session.summary = "…" + session.summary.slice(-780);
    }
}

function appendToSession(session, userMsg, assistantReply, meta = {}) {
    session.history.push({ u: 1, t: userMsg });
    session.history.push({ u: 0, t: assistantReply });
    session.lastUserMsg = userMsg;
    session.turnCount = (session.turnCount || 0) + 1;
    // Trim oldest turns into the running summary, two at a time.
    while (session.history.length > SESSION_MAX_TURNS * 2) {
        compressOldestPair(session);
        session.history = session.history.slice(2);
    }

    if (!session.case) return;

    // Phase C.1: classify case type from keywords. No LLM cost. Sticky
    // (won't flip on a single ambiguous turn — see intent-classifier.js).
    try {
        const cls = intent.classify(userMsg, session.case.type);
        if (cls.type && cls.type !== "unknown" && cls.confidence >= 0.2) {
            if (cls.type !== session.case.type) {
                session.case.type = cls.type;
                console.log(`[CASE] ${session.case.id.slice(-12)} → type=${cls.type} (conf=${cls.confidence.toFixed(2)})`);
            }
        }
    } catch (e) { console.warn("[CASE] classify failed:", e.message); }

    // Phase E: lawyer-handoff trigger. Three independent signals can flip
    // needs_lawyer to true; once true, it stays true (the slot-aware system
    // prompt then nudges the LLM to offer a verified lawyer).
    try {
        const c = session.case;
        if (!c.needs_lawyer) {
            const lower = (userMsg || "").toLowerCase();
            // Signal 1: explicit user request.
            const askedForLawyer = [
                "वकील", "वकेल", "lawyer", "advocate", "legal aid",
                "मुझे वकील", "वकील चाहिए", "किसी वकील", "वकील से बात",
                "i want a lawyer", "find me a lawyer", "talk to a lawyer",
            ].some(t => lower.includes(t.toLowerCase()));
            // Signal 2: case became complex enough (3+ user turns + a real type).
            const conversationDeep = (c.history?.filter(m => m.u === 1)?.length || 0) >= 3
                && c.type && c.type !== "unknown";
            // Signal 3: high urgency (already set by extractor or distress).
            const highUrgency = c.urgency === "critical" || c.urgency === "high";
            if (askedForLawyer || conversationDeep || highUrgency) {
                c.needs_lawyer = true;
                console.log(`[CASE] ${c.id.slice(-12)} needs_lawyer=true (asked=${askedForLawyer} deep=${conversationDeep} urg=${highUrgency})`);
            }
        }
    } catch (e) { console.warn("[CASE] lawyer-trigger failed:", e.message); }

    // Persist the user-facing transcript synchronously. Cheap (file write).
    // NOTE: do NOT overwrite case.summary from session.summary — they serve
    // different purposes. case.summary is owned by the entity extractor
    // (LLM-curated topical summary). session.summary is the cheap text
    // compression of trimmed-out turns. The extractor's work would be lost
    // every turn if we copied session.summary onto it.
    session.case.lang = session.lang || session.case.lang;
    try {
        caseStore.persistTurn(session.case, {
            userMsg,
            assistantReply,
            model: meta.model,
        });
    } catch (e) { console.warn("[SESSION] persistTurn failed:", e.message); }

    // Phase C.3: fire-and-forget Gemini JSON extractor. Updates entities,
    // facts, summary, urgency, needs_lawyer in the live case object. The
    // *next* turn's prompt will include whatever this learned.
    // Skip for trivial messages (≤2 words) — not worth the API call.
    if (userMsg && userMsg.trim().split(/\s+/).length >= 2) {
        const liveCase = session.case;
        const recent = session.history.slice(-8);
        const tag = liveCase.id.slice(-12);
        extractAndUpdate(liveCase, { userMsg, assistantReply, recentHistory: recent })
            .then(({ ok, changed }) => {
                if (ok && changed) {
                    try { caseStore.saveCase(liveCase); } catch { }
                    console.log(`[EXTRACT] ${tag} ✓ entities=${Object.keys(liveCase.entities||{}).length} urg=${liveCase.urgency} lawyer=${liveCase.needs_lawyer}`);
                } else if (ok && !changed) {
                    console.log(`[EXTRACT] ${tag} no-change`);
                } else {
                    console.log(`[EXTRACT] ${tag} skipped (gemini failed/aborted)`);
                }
            })
            .catch(e => console.warn(`[EXTRACT] ${tag} ERROR ${e.message}`));
    }
}

// Auto-cleanup expired sessions every 5 min
setInterval(() => {
    const now = Date.now();
    for (const [id, s] of webSessions) {
        if (now - s.lastActive > SESSION_TTL) webSessions.delete(id);
    }
}, 5 * 60 * 1000);

// ═══════════════════════════════════════════════════════════
//  SMART FALLBACK — never serve generic "तकनीकी समस्या" again
//
//  When all LLM providers fail, instead of the cold technical-error string:
//   1. Try to match the user's last message to an FAQ template (this is the
//      same matcher the hot path uses, just at the bottom of the cascade).
//   2. If nothing matches, return a warm "tell me a bit more" prompt that
//      asks the right next question instead of apologising.
//  The user shouldn't be able to tell the LLM failed.
// ═══════════════════════════════════════════════════════════
const SOFT_FALLBACK = {
    "hi-IN": "मैं आपकी बात पूरी तरह समझना चाहता हूँ — थोड़ा और बताइए, क्या मामला एफ़ आई आर का है, किसी एक्सीडेंट का, पैसे या नौकरी का, या परिवार का? जो भी हो, मैं आपके साथ हूँ।",
    "en-IN": "I want to make sure I understand you fully — tell me a little more. Is this about an FIR, an accident, money or your job, or family? Whatever it is, I'm here with you.",
    "bn-IN": "আমি আপনাকে ঠিকমতো বুঝতে চাই — একটু বিস্তারিত বলুন। কি ব্যাপারটা FIR, দুর্ঘটনা, টাকা বা চাকরি, নাকি পরিবার সংক্রান্ত? যা-ই হোক, আমি আপনার পাশে আছি।",
    "te-IN": "మీరు చెప్పేది నాకు పూర్తిగా అర్థం కావాలి — కొంచెం ఎక్కువ చెప్పండి. విషయం FIR, ప్రమాదం, డబ్బు లేదా ఉద్యోగం, లేక కుటుంబానికి సంబంధించినదా? ఏదైనా సరే, నేను మీతో ఉన్నాను.",
    "ta-IN": "நீங்கள் சொல்வதை முழுமையாகப் புரிந்துகொள்ள விரும்புகிறேன் — கொஞ்சம் கூடுதலாகச் சொல்லுங்கள். விஷயம் FIR, விபத்து, பணம் அல்லது வேலை, அல்லது குடும்பம் தொடர்பானதா? எதுவாக இருந்தாலும், நான் உங்களுடன் இருக்கிறேன்.",
    "mr-IN": "मला तुमचं नीट समजून घ्यायचं आहे — थोडं अजून सांगा. प्रकरण FIR, अपघात, पैसे किंवा नोकरी, की कुटुंबाशी संबंधित आहे? काहीही असो, मी तुमच्यासोबत आहे.",
    "gu-IN": "મારે તમારી વાત બરાબર સમજવી છે — થોડું વધારે કહો. વાત FIR, અકસ્માત, પૈસા કે નોકરી, કે પરિવારની છે? ગમે તે હોય, હું તમારી સાથે છું.",
    "kn-IN": "ನಾನು ನಿಮ್ಮನ್ನು ಸಂಪೂರ್ಣವಾಗಿ ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ಬಯಸುತ್ತೇನೆ — ಸ್ವಲ್ಪ ಇನ್ನಷ್ಟು ಹೇಳಿ. ವಿಷಯ FIR, ಅಪಘಾತ, ಹಣ ಅಥವಾ ಕೆಲಸ, ಅಥವಾ ಕುಟುಂಬದ ಬಗ್ಗೆಯೇ? ಏನಾದರೂ ಸರಿ, ನಾನು ನಿಮ್ಮ ಜೊತೆಯಲ್ಲಿದ್ದೇನೆ.",
    "ml-IN": "എനിക്ക് നിങ്ങളെ ശരിക്കും മനസിലാക്കണം — കുറച്ച് കൂടി പറയൂ. കാര്യം FIR, അപകടം, പണം അല്ലെങ്കിൽ ജോലി, അതോ കുടുംബമോ? എന്തായാലും, ഞാൻ നിങ്ങളോടൊപ്പമുണ്ട്.",
    "pa-IN": "ਮੈਂ ਤੁਹਾਡੀ ਗੱਲ ਚੰਗੀ ਤਰ੍ਹਾਂ ਸਮਝਣਾ ਚਾਹੁੰਦਾ ਹਾਂ — ਥੋੜ੍ਹਾ ਹੋਰ ਦੱਸੋ। ਮਾਮਲਾ FIR, ਹਾਦਸਾ, ਪੈਸੇ ਜਾਂ ਨੌਕਰੀ, ਜਾਂ ਪਰਿਵਾਰ ਨਾਲ ਸਬੰਧਤ ਹੈ? ਜੋ ਵੀ ਹੋਵੇ, ਮੈਂ ਤੁਹਾਡੇ ਨਾਲ ਹਾਂ।",
    "od-IN": "ମୁଁ ଆପଣଙ୍କୁ ଭଲ ଭାବରେ ବୁଝିବାକୁ ଚାହୁଁଛି — ଅଳ୍ପ ଅଧିକ କୁହନ୍ତୁ। କଥା FIR, ଦୁର୍ଘଟଣା, ଟଙ୍କା କିମ୍ବା ଚାକିରି, କିମ୍ବା ପରିବାର ସଙ୍ଗେ? ଯାହା ବି ହେଉ, ମୁଁ ଆପଣଙ୍କ ସଙ୍ଗରେ ଅଛି।",
};

/**
 * Build the system prompt for an LLM turn. Combines the language-specific
 * base prompt with (a) the running summary of older turns, and (b) the
 * slot-aware "Case so far / Still need to learn" block from the case state.
 * Centralized so /api/ask and /api/ask-stream stay in sync.
 */
function buildSystemPrompt(lang, ragContext, session) {
    let content = getLangPrompt(lang, ragContext || "");
    // Prefer the LLM-curated case summary when available; fall back to the
    // cheap text-compression summary from session.summary (used for very
    // long calls where the extractor hasn't caught up).
    const caseSummary = session?.case?.summary;
    const txtSummary = session?.summary;
    const summary = (caseSummary && caseSummary.length >= 8) ? caseSummary : txtSummary;
    if (summary) {
        content += `\n\nEarlier in this conversation: ${summary}\nReference these facts naturally — do NOT re-ask what you already know.`;
    }
    if (session?.case) {
        const c = session.case;
        const hasState = c.type || (c.entities && Object.keys(c.entities).length);
        if (hasState) {
            content += `\n\n${slots.renderCaseContext(c.type || "unknown", c.entities || {})}`;
            if (c.urgency && c.urgency !== "medium") {
                content += `\nUrgency: ${c.urgency}.`;
            }
            if (c.needs_lawyer) {
                content += `\nThe caller has signaled they want a human lawyer — when natural, offer to connect them to a verified one.`;
            }
            console.log(`[PROMPT] case=${c.id?.slice(-12)} type=${c.type} slots=${Object.keys(c.entities||{}).length} sum=${c.summary?.length||0}c`);
        }
        // Phase F: returning-caller note. Only meaningful on the first turn
        // of a resumed session (turnCount === 0 here = no new turns yet on
        // *this* session, but the case has prior history from a past one).
        const priorTurns = Math.floor((c.history?.length || 0) / 2);
        if (session.isReturning && priorTurns >= 1) {
            content += `\n\nReturning caller: this person has talked with you before about this case. Greet them warmly by acknowledging the context (e.g. "अच्छा हुआ आपने वापस फ़ोन किया — पिछली बार हमने ${c.type || "आपके मामले"} पर बात की थी, क्या उसके बाद कुछ हुआ?"). Do not re-ask everything; pick up where you left off.`;
        }
    }
    return content;
}

function getSmartFallback(lang, userMsg, session) {
    // First pass: try the FAQ matcher on the current user message.
    // It already covers FIR, accident, fraud, salary, property, divorce, DV,
    // bail, RTI, ration, panchayat, false case, caste atrocity, MNREGA, etc.
    if (userMsg) {
        const m = matchFAQ(userMsg, lang);
        if (m) return m.answer;
    }
    // Second pass: walk the conversation history backwards. This handles
    // mid-conversation follow-ups like "kya karein?", "haan", "kaunse kagaz
    // chahiye?" — where the topic was set turns ago. We scan every prior
    // user turn so the case context survives short responses.
    if (session?.history?.length) {
        for (let i = session.history.length - 1; i >= 0; i--) {
            const m = session.history[i];
            if (m.u !== 1 || !m.t || m.t === userMsg) continue;
            const hit = matchFAQ(m.t, lang);
            if (hit) return hit.answer;
        }
    }
    // Also try the running summary string — it concatenates older user msgs.
    if (session?.summary) {
        const hit = matchFAQ(session.summary, lang);
        if (hit) return hit.answer;
    }
    // Last resort: warm prompt that opens the conversation rather than closing it.
    return SOFT_FALLBACK[lang] || SOFT_FALLBACK["hi-IN"];
}

// ═══════════════════════════════════════════════════════════
//  TTS HELPER — with voice engine normalization
// ═══════════════════════════════════════════════════════════
async function generateTTS(text, langCode, customSpeaker, _retries = 0) {
    const gender = (customSpeaker === "female") ? "female" : undefined;
    const vp = getVoiceParams(langCode, gender);
    const speaker = vp.speaker;
    const normalized = normalizeTTS(text, langCode);
    const ck = ttsCacheKey(normalized, langCode, speaker);
    if (ttsCache.has(ck)) return { audio: ttsCache.get(ck).toString("base64"), text };

    const r = await apiFetch("https://api.sarvam.ai/text-to-speech", {
        method: "POST", headers: HEADERS,
        body: JSON.stringify({
            text: normalized,
            target_language_code: langCode,
            speaker,
            model: vp.model || "bulbul:v3",
            pace: vp.pace,
            speech_sample_rate: 24000,
            temperature: vp.temperature || 0.5,
        }),
    }, 4000);
    if (r.ok) {
        const d = await r.json();
        if (d.audios?.[0]) {
            const buf = Buffer.from(d.audios[0], "base64");
            ttsCacheSet(ck, buf);
            return { audio: d.audios[0], text };
        }
    } else {
        const errBody = await r.text().catch(() => "");
        // Retry once on 429 with backoff
        if (r.status === 429 && _retries < 1) {
            await new Promise(res => setTimeout(res, 500 + Math.random() * 500));
            return generateTTS(text, langCode, customSpeaker, _retries + 1);
        }
        console.log(`[TTS] Failed: ${r.status} ${errBody.slice(0, 100)}`);
    }
    return null;
}

// ═══════════════════════════════════════════════════════════
//  1. STT — Speech to Text
// ═══════════════════════════════════════════════════════════
app.post("/api/stt", upload.single("audio"), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: "No audio" });
    if (!SK) return res.status(500).json({ error: "No API key" });
    if (req.file.size < 300) return res.json({ transcript: "", confident: false });
    const t0 = Date.now();
    const langCode = req.body?.lang || "hi-IN";
    try {
        const fd = new FormData();
        // Always send as octet-stream — Sarvam rejects webm but accepts octet-stream
        fd.append("file", new File([req.file.buffer], "audio.wav", { type: "application/octet-stream" }));
        fd.append("model", "saarika:v2.5");
        fd.append("language_code", langCode === "unknown" ? "hi-IN" : langCode);
        fd.append("mode", "transcribe");
        const r = await apiFetch("https://api.sarvam.ai/speech-to-text", {
            method: "POST", headers: { "api-subscription-key": SK }, body: fd,
        }, 5000);
        if (!r.ok) {
            const errText = await r.text().catch(() => "");
            console.log(`[STT] ${r.status} ${errText.slice(0, 200)}`);
            return res.status(r.status).json({ error: `STT ${r.status}: ${errText.slice(0, 100)}` });
        }
        const d = await r.json();
        const transcript = sanitize(d.transcript || "", 1000);
        const { confident, reason } = scoreTranscript(transcript, langCode);
        console.log(`[STT] ${Date.now() - t0}ms lang:${langCode} conf:${confident} "${transcript.slice(0, 50)}"`);

        // Self-correction: if not confident, return clarification prompt
        if (!confident) {
            const clarification = getClarificationPrompt(langCode, reason);
            return res.json({ transcript: "", confident: false, clarification });
        }
        res.json({ transcript, confident: true });
    } catch (e) {
        console.log("[STT] FAIL:", e.message);
        res.status(500).json({ error: "STT failed" });
    }
});

// ═══════════════════════════════════════════════════════════
//  2. MAIN /api/ask — RAG + AI + Parallel TTS
// ═══════════════════════════════════════════════════════════
app.post("/api/ask", async (req, res) => {
    const { message, lang = "hi-IN", speaker, sessionId: reqSessionId, phone: reqPhone } = req.body;
    const { session, sessionId, isReturning } = getOrCreateSession(reqSessionId, { phone: reqPhone, lang });
    session.lang = lang;
    const msg = sanitize(message, 500);
    if (!msg) return res.status(400).json({ error: "No message" });
    if (!SK) return res.status(500).json({ error: "No API key" });
    const t0 = Date.now();
    const refusal = getRefusal(lang);

    // LAYER 1 GUARDRAIL
    const { allow, reason: guardReason } = isLegalQuery(msg);
    if (!allow) {
        console.log(`[ASK] L1_BLOCK [${guardReason}]: "${msg.slice(0, 50)}"`);
        // Still generate TTS for the refusal
        let audioChunks = [];
        try {
            const chunks = segmentForTTS(normalizeTTS(refusal, lang));
            audioChunks = (await Promise.allSettled(chunks.map(c => generateTTS(c, lang, speaker))))
                .filter(r => r.status === "fulfilled" && r.value).map(r => r.value);
        } catch { }
        return res.json({ reply: refusal, model: "guardrail", blocked: true, audioChunks, ms: 0 });
    }

    // DISTRESS DETECTION (Phase D) — runs before FAQ/cache/LLM. On a critical
    // signal (suicide ideation, active violence, child danger, threat to life)
    // we bypass the entire LLM cascade and serve a pre-written safety message
    // with the right helpline. Sub-millisecond. The user can never wait on
    // Gemini for "वो मार रहा है" — that fails too often.
    const distressCheck = distress.detect(msg, lang);
    if (distressCheck.level === "critical") {
        console.log(`[SAFETY] ${distressCheck.reason} matched="${distressCheck.matched}" → bypass`);
        const reply = distressCheck.prepend;
        // Persist into the case so urgency / needs_lawyer flip immediately.
        if (session.case) {
            session.case.urgency = "critical";
            session.case.needs_lawyer = true;
            session.case.distress_level = Math.max(session.case.distress_level || 0, distress.levelToScore("critical"));
        }
        appendToSession(session, msg, reply, { model: "safety-bypass" });
        const ttsT0 = Date.now();
        const segments = segmentForTTS(reply);
        const ttsResults = await Promise.allSettled(segments.map(chunk => generateTTS(chunk, lang, speaker)));
        const audioChunks = ttsResults.filter(r => r.status === "fulfilled" && r.value).map(r => r.value);
        return res.json({
            reply, model: "safety-bypass", sessionId, audioChunks,
            ms: Date.now() - t0, aiMs: 0, ttsMs: Date.now() - ttsT0,
            segments: segments.length, safety: distressCheck.reason,
            helplines: distressCheck.helplines,
            caseId: session.case?.id, isReturning,
        });
    }
    if (distressCheck.level === "high" && session.case) {
        // High distress doesn't bypass the LLM, but it bumps the case state
        // so the slot-aware prompt can soften the tone.
        session.case.distress_level = Math.max(session.case.distress_level || 0, distress.levelToScore("high"));
    }

    // FAQ CHECK — instant pre-built answer, skip LLM entirely
    const faqMatch = matchFAQ(msg, lang);
    if (faqMatch) {
        console.log(`[ASK] FAQ HIT: "${faqMatch.answer.slice(0, 60)}"`);
        const ttsT0 = Date.now();
        const segments = segmentForTTS(faqMatch.answer);
        const ttsResults = await Promise.allSettled(segments.map(chunk => generateTTS(chunk, lang, speaker)));
        const audioChunks = ttsResults.filter(r => r.status === "fulfilled" && r.value).map(r => r.value);
        appendToSession(session, msg, faqMatch.answer, { model: "faq-template" });
        return res.json({
            reply: faqMatch.answer, model: "faq-template", sessionId, audioChunks,
            ms: Date.now() - t0, aiMs: 0, ttsMs: Date.now() - ttsT0,
            ragContext: true, segments: segments.length, faq: true,
            caseId: session.case?.id, isReturning,
        });
    }

    // RESPONSE CACHE CHECK — skip LLM entirely if cached
    const cacheKey = responseCacheKey(msg, lang);
    const cached = responseCacheGet(cacheKey);
    if (cached) {
        console.log(`[ASK] CACHE HIT: "${cached.reply.slice(0, 60)}"`);
        const ttsT0 = Date.now();
        const segments = segmentForTTS(cached.reply);
        const ttsResults = await Promise.allSettled(segments.map(chunk => generateTTS(chunk, lang, speaker)));
        const audioChunks = ttsResults.filter(r => r.status === "fulfilled" && r.value).map(r => r.value);
        appendToSession(session, msg, cached.reply, { model: cached.model });
        return res.json({
            reply: cached.reply, model: cached.model + " (cached)", sessionId, audioChunks,
            ms: Date.now() - t0, aiMs: 0, ttsMs: Date.now() - ttsT0,
            ragContext: cached.ragContext, segments: segments.length, cached: true,
            caseId: session.case?.id, isReturning,
        });
    }

    // RAG RETRIEVAL
    const { contextString: ragContext, chunks: ragChunks } = buildContext(msg);
    const ragSnippet = ragContext || "";

    // Build conversation — server-side history is the single source of truth.
    // Client `history` parameter is intentionally ignored to prevent prompt bloat
    // (the bug that killed turn 3). Older turns live compressed in session.summary,
    // and known slot values are injected via buildSystemPrompt → renderCaseContext.
    const systemContent = buildSystemPrompt(lang, ragSnippet, session);
    const messages = [{ role: "system", content: systemContent }];
    let lastRole = "system";
    for (const m of session.history.slice(-SESSION_MAX_TURNS * 2)) {
        const role = m.u ? "user" : "assistant";
        if (role === lastRole) continue;
        messages.push({ role, content: sanitize(m.t, 300) });
        lastRole = role;
    }
    if (messages[messages.length - 1]?.role === "user") {
        messages.push({ role: "assistant", content: lang === "en-IN" ? "Yes, please go ahead." : "जी, बताइए।" });
    }
    messages.push({ role: "user", content: msg });

    // AI CALL — Smart model rotation with concurrent Gemini + Sarvam race
    const aiT0 = Date.now();
    let reply = "";
    let model = getAvailableModel();
    // Use the enriched prompt (RAG + summary + case context) — same as `messages[0]`.
    const systemPrompt = systemContent;

    // SPEED: Race primary model against Sarvam fallback concurrently — no delay
    try {
        const primaryPromise = callGemini(systemPrompt, messages, 512, model)
            .then(r => r ? { reply: stripMarkdown(r), model } : Promise.reject(new Error("empty")))
            .catch(e => Promise.reject(e));
        const sarvamPromise = callSarvam(messages, 300)
            .then(r => r ? { reply: stripMarkdown(r), model: "sarvam-105b" } : Promise.reject(new Error("empty")))
            .catch(e => Promise.reject(e));
        // Promise.any — first to RESOLVE SUCCESSFULLY wins; rejections don't prematurely end the race
        const result = await Promise.race([
            Promise.any([primaryPromise, sarvamPromise]).catch(() => null),
            new Promise(resolve => setTimeout(() => resolve(null), 8000)),
        ]);
        if (result && result.reply) {
            reply = result.reply;
            model = result.model;
            console.log(`[ASK] ${model} OK: "${reply.slice(0, 60)}"`);
        }
    } catch (e) { console.log(`[ASK-RACE] FAIL:`, e.message); }

    // If race failed entirely, try next available Gemini model
    if (!reply) {
        const fallbackModel = getAvailableModel();
        if (fallbackModel !== model) {
            try {
                const retryReply = await callGemini(systemPrompt, messages, 512, fallbackModel);
                if (retryReply) {
                    reply = stripMarkdown(retryReply);
                    model = fallbackModel;
                    console.log(`[ASK] ${fallbackModel} fallback OK: "${reply.slice(0, 60)}"`);
                }
            } catch (e) { console.log(`[ASK-${fallbackModel}] FAIL:`, e.message); }
        }
    }

    // Last resort: direct Sarvam LLM (if race didn't finish)
    if (!reply) {
        try {
            const sarvamReply = await callSarvam(messages, 300);
            if (sarvamReply) {
                reply = stripMarkdown(sarvamReply);
                model = "sarvam-105b";
                console.log(`[ASK] Sarvam direct: "${reply.slice(0, 60)}"`);
            }
        } catch (e) { console.log("[ASK-SARVAM] FAIL:", e.message); }
    }

    const aiMs = Date.now() - aiT0;

    // Unified post-processing through rules engine (handles validation, language, citations, emergency, NALSA)
    if (reply) {
        reply = applyRules(reply, { lang, ragChunks, isPhone: false, originalQuery: msg, refusal });
    }

    let isFallback = false;
    if (!reply) {
        // Case-aware soft fallback — never the cold "तकनीकी समस्या" string.
        // Picks the most relevant FAQ template, or a warm "tell me more" prompt.
        reply = getSmartFallback(lang, msg, session);
        model = "smart-fallback";
        isFallback = true;
    }

    // Cache ONLY successful, non-fallback, non-refusal responses
    const allFallbackValues = Object.values(FALLBACK_ERROR).concat(Object.values(SOFT_FALLBACK));
    if (reply && reply !== refusal && !isFallback && !allFallbackValues.includes(reply) && model !== "fallback" && model !== "smart-fallback") {
        responseCacheSet(cacheKey, reply, model, !!ragContext);
    }

    // TTS — segment, normalize, generate in PARALLEL
    const ttsT0 = Date.now();
    const segments = segmentForTTS(reply);
    const ttsResults = await Promise.allSettled(
        segments.map(chunk => generateTTS(chunk, lang, speaker))
    );
    const audioChunks = ttsResults
        .filter(r => r.status === "fulfilled" && r.value)
        .map(r => r.value);
    const ttsMs = Date.now() - ttsT0;

    console.log(`[ASK] ${Date.now() - t0}ms (AI:${aiMs} TTS:${ttsMs}) segs:${segments.length} rag:${!!ragContext} "${reply.slice(0, 60)}"`);

    // Save to session memory + disk (case file)
    appendToSession(session, msg, reply, { model });

    res.json({
        reply,
        model,
        sessionId,
        audioChunks,
        ms: Date.now() - t0,
        aiMs,
        ttsMs,
        ragContext: !!ragContext,
        segments: segments.length,
        caseId: session.case?.id,
        isReturning,
    });
});

// ═══════════════════════════════════════════════════════════
//  3. TTS endpoint (for greetings, one-off)
// ═══════════════════════════════════════════════════════════
app.post("/api/tts", async (req, res) => {
    const { text, lang = "hi-IN", speaker } = req.body;
    const ct = sanitize(text, 500);
    if (!ct || !SK) return res.status(400).json({ error: "No text" });
    try {
        const result = await generateTTS(ct, lang, speaker);
        if (result) {
            const buf = Buffer.from(result.audio, "base64");
            return res.set("Content-Type", "audio/wav").send(buf);
        }
    } catch (e) { console.log("[TTS] Error:", e.message); }
    res.status(500).json({ error: "TTS failed" });
});

// ═══════════════════════════════════════════════════════════
//  4. FILLER AUDIO
// ═══════════════════════════════════════════════════════════
const FALLBACK_ERROR = {
    "en-IN": "Sorry, a technical issue occurred. Call NALSA on 15100 for a free lawyer.",
    "hi-IN": "थोड़ी तकनीकी समस्या हुई। NALSA 15100 पर कॉल करें — निःशुल्क वकील मिलेगा।",
    "bn-IN": "দুঃখিত, একটি প্রযুক্তিগত সমস্যা হয়েছে। বিনামূল্যে আইনজীবীর জন্য NALSA 15100-এ কল করুন।",
    "te-IN": "క్షమించండి, సాంకేతిక సమస్య జరిగింది. ఉచిత లాయర్ కోసం NALSA 15100కు కాల్ చేయండి.",
    "ta-IN": "மன்னிக்கவும், தொழில்நுட்ப சிக்கல் ஏற்பட்டது. இலவச வழக்கறிஞருக்கு NALSA 15100-ஐ அழைக்கவும்.",
    "mr-IN": "माफ करा, तांत्रिक समस्या आली. मोफत वकिलासाठी NALSA 15100 वर कॉल करा.",
    "gu-IN": "માફ કરજો, ટેકનિકલ સમસ્યા આવી. મફત વકીલ માટે NALSA 15100 પર કૉલ કરો.",
    "kn-IN": "ಕ್ಷಮಿಸಿ, ತಾಂತ್ರಿಕ ಸಮಸ್ಯೆ ಉಂಟಾಯಿತು. ಉಚಿತ ವಕೀಲರಿಗಾಗಿ NALSA 15100ಗೆ ಕರೆ ಮಾಡಿ.",
    "ml-IN": "ക്ഷമിക്കണം, ഒരു സാങ്കേതിക പ്രശ്നം ഉണ്ടായി. സൗജന്യ അഭിഭാഷകനെ ലഭിക്കാൻ NALSA 15100-ൽ വിളിക്കൂ.",
    "pa-IN": "ਮਾਫ਼ ਕਰਨਾ, ਤਕਨੀਕੀ ਸਮੱਸਿਆ ਆਈ। ਮੁਫ਼ਤ ਵਕੀਲ ਲਈ NALSA 15100 ਤੇ ਕਾਲ ਕਰੋ।",
    "od-IN": "କ୍ଷମା କରନ୍ତୁ, ଏକ ଟେକ୍ନିକାଲ ସମସ୍ୟା ଘଟିଲା। ମାଗଣା ଓକିଲ ପାଇଁ NALSA 15100ରେ କଲ କରନ୍ତୁ।",
};

const FILLERS = {
    "hi-IN": ["अच्छा, देखते हैं...", "समझ रहा हूँ, रुकिए एक पल...", "जी हाँ, सोच रहा हूँ...", "ठीक है, बिल्कुल..."],
    "en-IN": ["Let me think about this for a moment...", "One moment please...", "Sure, looking into your situation...", "Let me check that for you..."],
    "bn-IN": ["বুঝতে পারছি, একটু অপেক্ষা করুন...", "দেখছি...", "হ্যাঁ, বুঝেছি..."],
    "te-IN": ["అర్థమవుతుంది, ఒక్క నిమిషం...", "చూస్తున్నాను...", "సరే, ఆలోచిస్తున్నా..."],
    "ta-IN": ["புரிகிறேன், ஒரு கணம்...", "பார்க்கிறேன்...", "சரி, யோசித்துப் பார்க்கிறேன்..."],
    "mr-IN": ["समजलं, एक पल...", "पाहतो...", "ठीक आहे, विचार करतो..."],
    "gu-IN": ["સમજુ છું, એક પળ...", "જોઉં છું...", "ઠીક છે, વિચાર કરું છું..."],
    "kn-IN": ["ಅರ್ಥಮಾಗುತ್ತಿದೆ, ಒಂದು ನಿಮಿಷ...", "ನೋಡುತ್ತೇನೆ...", "ಸರಿ, ಆಲೋಚಿಸುತ್ತೇನೆ..."],
    "ml-IN": ["മനസ്സിലാക്കുന്നുണ്ട്, ഒരു നിമിഷം...", "നോക്കുന്നുണ്ട്...", "ശരി, ആലോചിക്കുന്നു..."],
    "pa-IN": ["ਸਮਝ ਗਿਆ, ਇੱਕ ਪਲ...", "ਦੇਖ ਰਿਹਾ ਹਾਂ...", "ਠੀਕ ਹੈ, ਸੋਚਦਾ ਹਾਂ..."],
    "od-IN": ["ବୁଝିଲି, ଗୋଟେ ମିନିଟ...", "ଦେଖୁଅଛି...", "ଠିକ, ଭାବୁଛି..."],
};
const fillerCache = new Map();

async function warmFillers() {
    if (!SK) return;
    console.log("  [Filler] Warming all languages (male + female)...");
    const langs = Object.keys(FILLERS);
    // Process ONE language at a time, sequential phrases, to respect Sarvam rate limits
    for (const lc of langs) {
        const phrases = FILLERS[lc] || FILLERS["hi-IN"];
        for (const sp of [undefined, "female"]) {
            const bufs = [];
            const cacheKey = sp === "female" ? `${lc}:female` : lc;
            for (const ph of phrases) {
                try {
                    const result = await generateTTS(ph, lc, sp);
                    if (result) bufs.push(Buffer.from(result.audio, "base64"));
                } catch { }
                await new Promise(r => setTimeout(r, 150)); // 150ms gap between calls
            }
            if (bufs.length) fillerCache.set(cacheKey, bufs);
        }
    }
    const warmed = [...fillerCache.entries()].map(([k, v]) => `${k}:${v.length}`).join(", ");
    console.log(`  [Filler] OK — ${warmed}`);
}

app.get("/api/filler", async (req, res) => {
    const lc = req.query.lang || "hi-IN";
    const sp = req.query.speaker || undefined; // "female" or undefined (male default)
    const cacheKey = sp === "female" ? `${lc}:female` : lc;
    const cached = fillerCache.get(cacheKey);
    if (cached?.length) return res.set("Content-Type", "audio/wav").send(cached[Math.floor(Math.random() * cached.length)]);
    const phrases = FILLERS[lc] || FILLERS["hi-IN"];
    const ph = phrases[Math.floor(Math.random() * phrases.length)];
    try {
        const result = await generateTTS(ph, lc, sp);
        if (result) {
            const buf = Buffer.from(result.audio, "base64");
            if (!fillerCache.has(cacheKey)) fillerCache.set(cacheKey, []);
            fillerCache.get(cacheKey).push(buf);
            return res.set("Content-Type", "audio/wav").send(buf);
        }
    } catch { }
    res.status(204).end();
});

// ═══════════════════════════════════════════════════════════
//  5. IVR PROMPT
// ═══════════════════════════════════════════════════════════
let ivrBuf = null;

// Multi-segment IVR — each language option spoken clearly with pauses
// Format: "Welcome... For Hindi press 1... For English press 2..." etc.
// Spoken in Hindi (primary) with clear number pronunciation
const IVR_SEGMENTS = [
    { text: "न्यायसाथी में आपका स्वागत है।  भारत का मुफ़्त ए आई कानूनी सहायक।  अपनी भाषा चुनने के लिए नंबर दबाएं।", lang: "hi-IN" },
    { text: "हिंदी के लिए, एक दबाएं।", lang: "hi-IN" },
    { text: "For English, press two.", lang: "en-IN" },
    { text: "বাংলার জন্য, তিন টিপুন।", lang: "bn-IN" },
    { text: "తెలుగు కోసం, నాలుగు నొక్కండి.", lang: "te-IN" },
    { text: "தமிழுக்கு, ஐந்து அழுத்தவும்.", lang: "ta-IN" },
    { text: "मराठीसाठी, सहा दाबा.", lang: "mr-IN" },
    { text: "ગુજરાતી માટે, સાત દબાવો.", lang: "gu-IN" },
    { text: "ಕನ್ನಡಕ್ಕಾಗಿ, ಎಂಟು ಒತ್ತಿ.", lang: "kn-IN" },
    { text: "മലയാളത്തിന്, ഒമ്പത് അമർത്തുക.", lang: "ml-IN" },
    { text: "ਪੰਜਾਬੀ ਲਈ, ਸਟਾਰ ਦਬਾਓ.", lang: "pa-IN" },
];

// Build combined IVR audio from all segments (pre-generated on startup)
async function buildIVRAudio() {
    const pcmChunks = [];
    let ok = 0;
    // Sequential with delay to respect Sarvam rate limits (shared with filler warmup)
    for (let i = 0; i < IVR_SEGMENTS.length; i++) {
        const seg = IVR_SEGMENTS[i];
        try {
            const result = await generateTTS(seg.text, seg.lang);
            if (result?.audio) {
                const wav = Buffer.from(result.audio, "base64");
                pcmChunks.push(wav.slice(44));
                const silenceBytes = Math.round(24000 * 0.4) * 2;
                pcmChunks.push(Buffer.alloc(silenceBytes));
                ok++;
            } else {
                console.log(`[IVR] Failed segment ${seg.lang}: no audio returned`);
            }
        } catch (e) {
            console.log(`[IVR] Failed segment ${seg.lang}: ${e.message}`);
        }
        await new Promise(r => setTimeout(r, 200)); // 200ms gap
    }
    console.log(`[IVR] Segments: ${ok}/${IVR_SEGMENTS.length} succeeded`);
    if (pcmChunks.length > 0) {
        const pcm = Buffer.concat(pcmChunks);
        ivrBuf = pcmToWav(pcm, 24000);
        const durSec = (pcm.length / (24000 * 2)).toFixed(1);
        console.log(`[IVR] Built multi-language IVR: ${(ivrBuf.length / 1024).toFixed(0)}KB, ${durSec}s`);
    }
}

// PCM → WAV helper (16-bit mono)
function pcmToWav(pcmData, sampleRate) {
    const header = Buffer.alloc(44);
    const dataSize = pcmData.length;
    const fileSize = dataSize + 36;
    header.write("RIFF", 0);
    header.writeUInt32LE(fileSize, 4);
    header.write("WAVE", 8);
    header.write("fmt ", 12);
    header.writeUInt32LE(16, 16);       // fmt chunk size
    header.writeUInt16LE(1, 20);        // PCM format
    header.writeUInt16LE(1, 22);        // mono
    header.writeUInt32LE(sampleRate, 24);
    header.writeUInt32LE(sampleRate * 2, 28); // byte rate
    header.writeUInt16LE(2, 32);        // block align
    header.writeUInt16LE(16, 34);       // bits per sample
    header.write("data", 36);
    header.writeUInt32LE(dataSize, 40);
    return Buffer.concat([header, pcmData]);
}

// Fallback single-language IVR text (if multi-segment build fails)
const IVR_TEXT = "न्यायसाथी में आपका स्वागत है। भारत का मुफ़्त ए आई कानूनी सहायक। हिंदी के लिए एक दबाएं। इंग्लिश के लिए दो। बंगाली के लिए तीन। तेलुगु के लिए चार। तमिल के लिए पाँच। मराठी के लिए छह। गुजराती के लिए सात। कन्नड़ के लिए आठ। मलयालम के लिए नौ। पंजाबी के लिए स्टार दबाएं।";

app.get("/api/ivr-prompt", async (_, res) => {
    if (ivrBuf) return res.set("Content-Type", "audio/wav").send(ivrBuf);
    try {
        // Try multi-language IVR
        await buildIVRAudio();
        if (ivrBuf) return res.set("Content-Type", "audio/wav").send(ivrBuf);
        // Fallback single Hindi
        const result = await generateTTS(IVR_TEXT, "hi-IN");
        if (result) { ivrBuf = Buffer.from(result.audio, "base64"); return res.set("Content-Type", "audio/wav").send(ivrBuf); }
    } catch { }
    res.status(204).end();
});

// ═══════════════════════════════════════════════════════════
//  6. PHONE DIAL-IN — Sarvam Telephony Webhook
//  When caller dials, Sarvam sends audio chunks → we process and respond
// ═══════════════════════════════════════════════════════════
const phoneSessions = new Map(); // sessionId → { lang, history, step }

// Auto-cleanup expired phone sessions every 5 min (prevents memory leak from dropped calls)
setInterval(() => {
    const now = Date.now();
    for (const [id, s] of phoneSessions) {
        if (now - (s.lastActive || s.created || 0) > SESSION_TTL) {
            phoneSessions.delete(id);
            console.log(`[PHONE] Session expired: ${id}`);
        }
    }
}, 5 * 60 * 1000);

app.post("/api/phone/call-start", async (req, res) => {
    // Sarvam telephony calls this when a call begins
    const sessionId = req.body?.session_id || `ph_${Date.now()}`;
    phoneSessions.set(sessionId, { lang: "hi-IN", history: [], step: "ivr", created: Date.now(), lastActive: Date.now() });
    console.log(`[PHONE] New call: ${sessionId}`);

    // Respond with IVR prompt audio
    if (!ivrBuf) {
        try { const r = await generateTTS(IVR_TEXT, "hi-IN"); if (r) ivrBuf = Buffer.from(r.audio, "base64"); } catch { }
    }
    res.json({
        session_id: sessionId,
        action: "play_and_collect_dtmf",
        audio_base64: ivrBuf ? ivrBuf.toString("base64") : null,
        max_digits: 1,
        timeout_seconds: 10,
    });
});

app.post("/api/phone/dtmf", async (req, res) => {
    const { session_id, digit } = req.body;
    const session = phoneSessions.get(session_id);
    if (!session) return res.status(404).json({ error: "Session not found" });

    const DTMF_LANGS = {
        "1": "hi-IN", "2": "en-IN", "3": "bn-IN", "4": "te-IN", "5": "ta-IN",
        "6": "mr-IN", "7": "gu-IN", "8": "kn-IN", "9": "ml-IN", "0": "pa-IN"
    };
    const lang = DTMF_LANGS[digit] || "hi-IN";
    session.lang = lang;
    session.step = "active";

    const greeting = GREETINGS[lang] || GREETINGS["hi-IN"];
    const result = await generateTTS(greeting, lang);

    res.json({
        session_id,
        action: "play_and_listen",
        audio_base64: result?.audio || null,
        max_record_seconds: 15,
        silence_threshold_seconds: 2,
    });
});

app.post("/api/phone/audio", upload.single("audio"), async (req, res) => {
    const sessionId = req.body?.session_id;
    const session = phoneSessions.get(sessionId);
    if (!session || !req.file) return res.status(400).json({ error: "Invalid request" });
    session.lastActive = Date.now();

    const lang = session.lang || "hi-IN";

    // STT
    let transcript = "";
    try {
        const fd = new FormData();
        fd.append("file", new File([req.file.buffer], "audio.wav", { type: "application/octet-stream" }));
        fd.append("model", "saarika:v2.5");
        fd.append("language_code", lang);
        fd.append("mode", "transcribe");
        const r = await apiFetch("https://api.sarvam.ai/speech-to-text", {
            method: "POST", headers: { "api-subscription-key": SK }, body: fd,
        }, 6000);
        if (r.ok) { const d = await r.json(); transcript = sanitize(d.transcript || "", 500); }
    } catch (e) { console.log("[PHONE-STT]", e.message); }

    // Self-correction check
    const { confident, reason } = scoreTranscript(transcript, lang);
    if (!confident) {
        const clarification = getClarificationPrompt(lang, reason);
        const clarResult = await generateTTS(clarification, lang);
        return res.json({
            session_id: sessionId,
            action: "play_and_listen",
            audio_base64: clarResult?.audio || null,
            transcript: "",
            max_record_seconds: 15,
        });
    }

    // Guardrail + RAG + Gemini AI
    const { allow } = isLegalQuery(transcript);
    let reply = "";
    let audioChunks = [];

    if (!allow) {
        reply = getRefusal(lang);
    } else {
        const { contextString: ragContext, chunks: ragChunks } = buildContext(transcript);
        const systemPrompt = getLangPrompt(lang, ragContext);
        const phoneMessages = [
            { role: "system", content: systemPrompt },
            ...session.history.slice(-6).map(m => ({ role: m.u ? "user" : "assistant", content: m.t })),
            { role: "user", content: transcript },
        ];
        // Smart model rotation, Sarvam fallback
        let phoneModel = getAvailableModel();
        try {
            reply = await callGemini(systemPrompt, phoneMessages, 512, phoneModel);
        } catch (e) { console.log(`[PHONE-${phoneModel}]`, e.message); }
        if (!reply) {
            const fb = getAvailableModel();
            if (fb !== phoneModel) {
                try { reply = await callGemini(systemPrompt, phoneMessages, 512, fb); }
                catch (e) { console.log(`[PHONE-${fb}]`, e.message); }
            }
        }
        if (!reply) {
            try { reply = await callSarvam(phoneMessages, 300); }
            catch (e) { console.log("[PHONE-SARVAM]", e.message); }
        }
        if (reply) {
            reply = applyRules(reply, { lang, ragChunks, isPhone: true, originalQuery: transcript, refusal: getRefusal(lang) });
        }
        if (!reply) reply = getRefusal(lang);
    }

    // Update session history
    session.history.push({ u: 1, t: transcript });
    session.history.push({ u: 0, t: reply });
    if (session.history.length > 20) session.history = session.history.slice(-20);

    // TTS
    const segments = segmentForTTS(reply);
    const ttsResults = await Promise.allSettled(segments.map(c => generateTTS(c, lang)));
    const combinedAudio = ttsResults
        .filter(r => r.status === "fulfilled" && r.value)
        .map(r => Buffer.from(r.value.audio, "base64"));

    // Concatenate WAV buffers (simple append — works for PCM WAV)
    const combinedBuf = combinedAudio.length > 0
        ? Buffer.concat(combinedAudio)
        : null;

    console.log(`[PHONE] ${sessionId} "${transcript.slice(0, 40)}" → "${reply.slice(0, 40)}"`);

    res.json({
        session_id: sessionId,
        action: "play_and_listen",
        audio_base64: combinedBuf ? combinedBuf.toString("base64") : null,
        transcript,
        reply,
        max_record_seconds: 15,
    });
});

app.post("/api/phone/call-end", (req, res) => {
    const { session_id } = req.body;
    if (session_id) phoneSessions.delete(session_id);
    console.log(`[PHONE] Call ended: ${session_id}`);
    res.json({ status: "ok" });
});

// Phone info endpoint
app.get("/api/phone/info", (_, res) => {
    res.json({
        webhook_base: `${process.env.PUBLIC_URL || "http://localhost:" + (process.env.PORT || 3000)}/api/phone`,
        endpoints: {
            call_start: "POST /api/phone/call-start",
            dtmf: "POST /api/phone/dtmf",
            audio: "POST /api/phone/audio (multipart/form-data, field: audio)",
            call_end: "POST /api/phone/call-end",
        },
        active_sessions: phoneSessions.size,
        languages: Object.keys(GREETINGS),
        note: "Configure your Sarvam telephony number to POST to these webhook URLs",
    });
});

// ═══════════════════════════════════════════════════════════
//  6b. EXOTEL TOLL-FREE — Indian 1800 number integration
//  Exotel calls our webhooks with ExoML (XML) responses
//  Flow: Call → IVR greeting → DTMF lang select → Record → STT → AI → TTS → Play
// ═══════════════════════════════════════════════════════════

// Audio store — serve TTS audio files for Exotel <Play> URLs
const audioStore = new Map(); // id → { buf, created }
const AUDIO_STORE_TTL = 10 * 60 * 1000; // 10 min
const AUDIO_STORE_MAX = 200;

function storeAudio(buf) {
    // Evict expired entries
    const now = Date.now();
    for (const [k, v] of audioStore) {
        if (now - v.created > AUDIO_STORE_TTL) audioStore.delete(k);
    }
    // Evict oldest if full
    if (audioStore.size >= AUDIO_STORE_MAX) {
        const oldest = audioStore.keys().next().value;
        audioStore.delete(oldest);
    }
    const id = `a_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    audioStore.set(id, { buf, created: now });
    return id;
}

// Serve stored audio files — Exotel <Play> points here
app.get("/audio/:id", (req, res) => {
    const entry = audioStore.get(req.params.id);
    if (!entry) return res.status(404).send("Audio not found");
    res.set("Content-Type", "audio/wav");
    res.set("Cache-Control", "no-cache");
    res.send(entry.buf);
});

// Helper: get public base URL for audio files
function getPublicUrl() {
    return process.env.PUBLIC_URL || `http://localhost:${process.env.PORT || 3000}`;
}

// Helper: generate ExoML XML response
function exoml(body) {
    return `<?xml version="1.0" encoding="UTF-8"?>\n<Response>\n${body}\n</Response>`;
}

// Store and get URL for TTS audio
async function ttsToUrl(text, lang, speaker) {
    const result = await generateTTS(text, lang, speaker);
    if (!result) return null;
    const buf = Buffer.from(result.audio, "base64");
    const id = storeAudio(buf);
    return `${getPublicUrl()}/audio/${id}`;
}

// Store and get URL for combined TTS segments (supports gender-based custom speaker)
async function combinedTtsToUrl(text, lang, speaker) {
    const segments = segmentForTTS(text);
    const results = await Promise.allSettled(segments.map(c => generateTTS(c, lang, speaker)));
    const buffers = results
        .filter(r => r.status === "fulfilled" && r.value)
        .map(r => Buffer.from(r.value.audio, "base64"));
    if (buffers.length === 0) return null;
    const combined = Buffer.concat(buffers);
    const id = storeAudio(combined);
    return `${getPublicUrl()}/audio/${id}`;
}

// Pre-generate IVR audio on startup
let ivrAudioUrl = null;
async function warmExotelIVR() {
    try {
        // Try multi-language IVR first (each option in its own language)
        if (!ivrBuf) {
            console.log("[IVR] Building multi-language IVR...");
            await buildIVRAudio();
        }
        // Fallback: single Hindi TTS
        if (!ivrBuf) {
            console.log("[IVR] Multi-lang failed, using single Hindi fallback");
            const r = await generateTTS(IVR_TEXT, "hi-IN");
            if (r) ivrBuf = Buffer.from(r.audio, "base64");
        }
        if (ivrBuf) {
            const id = storeAudio(ivrBuf);
            ivrAudioUrl = `${getPublicUrl()}/audio/${id}`;
            console.log(`[IVR] Ready: ${ivrAudioUrl}`);
        }
    } catch (e) { console.log("[EXOTEL] IVR warmup failed:", e.message); }
}

/**
 * Detect caller gender from WAV audio using zero-crossing rate.
 * Higher ZCR → higher pitch → female voice. Threshold: 0.12
 * Returns: "female" | "male" | "unknown"
 */
function detectGender(wavBuffer) {
    if (!wavBuffer || wavBuffer.length < 200) return "unknown";
    try {
        const start = 44; // Skip WAV header
        let crossings = 0;
        let prev = 0;
        const limit = Math.min(start + 16000, wavBuffer.length - 1); // ~0.5s at 16kHz 16-bit
        let samples = 0;
        for (let i = start; i < limit - 1; i += 2) {
            const curr = wavBuffer.readInt16LE(i);
            if ((curr >= 0) !== (prev >= 0)) crossings++;
            prev = curr;
            samples++;
        }
        if (samples < 100) return "unknown";
        const zcr = crossings / samples;
        return zcr > 0.12 ? "female" : "male";
    } catch { return "unknown"; }
}

// 1. Incoming call — play IVR greeting, collect DTMF
app.post("/api/phone/exotel/call-start", verifyWebhookToken, async (req, res) => {
    const callSid = req.body?.CallSid || req.query?.CallSid || `exo_${Date.now()}`;
    console.log(`[EXOTEL] New call: ${callSid} from ${req.body?.From || req.query?.From || "unknown"}`);

    // Create session with silence tracking + call duration
    phoneSessions.set(callSid, {
        lang: "hi-IN",
        history: [],
        step: "ivr",
        provider: "exotel",
        silenceCount: 0,        // consecutive silent recordings
        startTime: Date.now(),  // for 5-minute call duration limit
        lastActive: Date.now(), // for TTL cleanup
        created: Date.now(),
        speakerGender: null,    // detected from first voice recording
        customSpeaker: null,    // female speaker override if female detected
    });

    // Ensure IVR audio is ready
    if (!ivrAudioUrl) await warmExotelIVR();

    const gatherUrl = appendWebhookToken(`${getPublicUrl()}/api/phone/exotel/gather?CallSid=${encodeURIComponent(callSid)}`);

    if (ivrAudioUrl) {
        res.set("Content-Type", "application/xml");
        res.send(exoml(`  <Play>${ivrAudioUrl}</Play>\n  <Gather action="${gatherUrl}" method="POST" numDigits="1" timeout="10" />`));
    } else {
        // Fallback: just gather without audio
        res.set("Content-Type", "application/xml");
        res.send(exoml(`  <Say>Welcome to NyayaSathi. Press 1 for Hindi. Press 2 for English.</Say>\n  <Gather action="${gatherUrl}" method="POST" numDigits="1" timeout="10" />`));
    }
});

// 2. DTMF digit received — set language, play greeting, start recording
app.post("/api/phone/exotel/gather", verifyWebhookToken, async (req, res) => {
    const callSid = req.body?.CallSid || req.query?.CallSid;
    const digits = req.body?.Digits || req.body?.digits || req.query?.Digits || "1";
    const session = phoneSessions.get(callSid);

    if (!session) {
        res.set("Content-Type", "application/xml");
        return res.send(exoml(`  <Say>Session expired. Please call again.</Say>\n  <Hangup />`));
    }

    const DTMF_LANGS = {
        "1": "hi-IN", "2": "en-IN", "3": "bn-IN", "4": "te-IN", "5": "ta-IN",
        "6": "mr-IN", "7": "gu-IN", "8": "kn-IN", "9": "ml-IN", "0": "pa-IN"
    };
    session.lang = DTMF_LANGS[digits] || "hi-IN";
    session.step = "active";

    // Generate greeting audio
    const greeting = GREETINGS[session.lang] || GREETINGS["hi-IN"];
    const greetingUrl = await ttsToUrl(greeting, session.lang);

    const recordUrl = appendWebhookToken(`${getPublicUrl()}/api/phone/exotel/audio?CallSid=${encodeURIComponent(callSid)}`);

    res.set("Content-Type", "application/xml");
    if (greetingUrl) {
        res.send(exoml(`  <Play>${greetingUrl}</Play>\n  <Record action="${recordUrl}" method="POST" maxLength="15" finishOnKey="#" timeout="3" />`));
    } else {
        res.send(exoml(`  <Say>${greeting}</Say>\n  <Record action="${recordUrl}" method="POST" maxLength="15" finishOnKey="#" timeout="3" />`));
    }
});

// 3. Audio recording received — STT → AI → TTS → play response
app.post("/api/phone/exotel/audio", verifyWebhookToken, async (req, res) => {
    const callSid = req.body?.CallSid || req.query?.CallSid;
    const recordingUrl = req.body?.RecordingUrl || req.body?.recording_url;
    const session = phoneSessions.get(callSid);

    if (!session) {
        res.set("Content-Type", "application/xml");
        return res.send(exoml(`  <Say>Session expired. Please call again.</Say>\n  <Hangup />`));
    }
    session.lastActive = Date.now();

    const lang = session.lang || "hi-IN";
    let reply = getRefusal(lang);
    const recordUrl = appendWebhookToken(`${getPublicUrl()}/api/phone/exotel/audio?CallSid=${encodeURIComponent(callSid)}`);

    // ── Call duration limit: 5 minutes ──────────────────────────────
    const callAge = Date.now() - (session.startTime || Date.now());
    if (callAge > 5 * 60 * 1000) {
        const bye = GOODBYES[lang] || GOODBYES["hi-IN"];
        const byeUrl = await ttsToUrl(bye, lang);
        phoneSessions.delete(callSid);
        res.set("Content-Type", "application/xml");
        return res.send(exoml((byeUrl ? `  <Play>${byeUrl}</Play>` : `  <Say>${bye}</Say>`) + "\n  <Hangup />"));
    }

    try {
        const tCall = timer("EXOTEL-TOTAL");
        // Download recording from Exotel (with SSRF protection)
        let audioBuffer;
        if (recordingUrl) {
            if (!isAllowedRecordingUrl(recordingUrl)) {
                console.log(`[SECURITY] SSRF blocked: ${recordingUrl}`);
                res.set("Content-Type", "application/xml");
                return res.send(exoml(`  <Say>Security error.</Say>\n  <Hangup />`));
            }
            const tDownload = timer("RECORDING-DOWNLOAD");
            const authHeader = process.env.EXOTEL_API_KEY && process.env.EXOTEL_API_TOKEN
                ? "Basic " + Buffer.from(`${process.env.EXOTEL_API_KEY}:${process.env.EXOTEL_API_TOKEN}`).toString("base64")
                : null;
            const headers = authHeader ? { Authorization: authHeader } : {};
            const audioRes = await apiFetch(recordingUrl, { headers }, 10000);
            if (audioRes.ok) {
                audioBuffer = Buffer.from(await audioRes.arrayBuffer());
            }
            tDownload.end();
        }

        // ── Silence / empty audio detection ──────────────────────────
        if (!audioBuffer || audioBuffer.length < 300) {
            session.silenceCount = (session.silenceCount || 0) + 1;

            if (session.silenceCount >= 2) {
                // Second silence → goodbye + hangup
                const bye = GOODBYES[lang] || GOODBYES["hi-IN"];
                const byeUrl = await ttsToUrl(bye, lang);
                phoneSessions.delete(callSid);
                res.set("Content-Type", "application/xml");
                return res.send(exoml((byeUrl ? `  <Play>${byeUrl}</Play>` : `  <Say>${bye}</Say>`) + "\n  <Hangup />"));
            }

            // First silence → warn caller, give 10 more seconds
            const warn = SILENCE_WARNINGS[lang] || SILENCE_WARNINGS["hi-IN"];
            const warnUrl = await ttsToUrl(warn, lang);
            res.set("Content-Type", "application/xml");
            return res.send(exoml(
                (warnUrl ? `  <Play>${warnUrl}</Play>` : `  <Say>${warn}</Say>`) +
                `\n  <Record action="${recordUrl}" method="POST" maxLength="15" finishOnKey="#" timeout="10" />`
            ));
        }

        // Valid audio — reset silence counter
        session.silenceCount = 0;

        // ── Gender detection (first recording only) ──────────────────
        if (!session.speakerGender) {
            const gender = detectGender(audioBuffer);
            session.speakerGender = gender;
            if (gender === "female") {
                const { LANG_VOICE_FEMALE } = require("./voice-engine");
                session.customSpeaker = LANG_VOICE_FEMALE[lang]?.speaker || null;
                console.log(`[EXOTEL] ${callSid} gender=female → speaker=${session.customSpeaker}`);
            }
        }

        // STT
        const tSTT = timer("STT");
        let transcript = "";
        const fd = new FormData();
        fd.append("file", new File([audioBuffer], "audio.wav", { type: "application/octet-stream" }));
        fd.append("model", "saarika:v2.5");
        fd.append("language_code", lang);
        fd.append("mode", "transcribe");
        const sttRes = await apiFetch("https://api.sarvam.ai/speech-to-text", {
            method: "POST", headers: { "api-subscription-key": SK }, body: fd,
        }, 6000);
        if (sttRes.ok) {
            const d = await sttRes.json();
            transcript = sanitizeForLLM(sanitize(d.transcript || "", 500));
        } else {
            const errBody = await sttRes.text().catch(() => "");
            console.log(`[EXOTEL-STT] Failed: ${sttRes.status} ${errBody.slice(0, 200)}`);
        }
        tSTT.end();

        // Confidence check
        const { confident, reason } = scoreTranscript(transcript, lang);
        if (!confident) {
            const clarification = getClarificationPrompt(lang, reason);
            const clarUrl = await ttsToUrl(clarification, lang);
            const recordUrl2 = appendWebhookToken(`${getPublicUrl()}/api/phone/exotel/audio?CallSid=${encodeURIComponent(callSid)}`);
            res.set("Content-Type", "application/xml");
            return res.send(exoml(
                (clarUrl ? `  <Play>${clarUrl}</Play>` : `  <Say>${clarification}</Say>`) +
                `\n  <Record action="${recordUrl2}" method="POST" maxLength="15" finishOnKey="#" timeout="3" />`
            ));
        }

        // Guardrail + RAG + AI (with Promise.race for speed)
        const { allow } = isLegalQuery(transcript);
        if (!allow) {
            reply = getRefusal(lang);
        } else {
            const tRAG = timer("RAG");
            const { contextString: ragContext, chunks: ragChunks } = buildContext(transcript);
            tRAG.end();
            const systemPrompt = getLangPrompt(lang, ragContext);
            const phoneMessages = [
                { role: "system", content: systemPrompt },
                ...session.history.slice(-6).map(m => ({ role: m.u ? "user" : "assistant", content: m.t })),
                { role: "user", content: transcript },
            ];
            let phoneModel = getAvailableModel();
            const refusal = getRefusal(lang);

            // SPEED: Race primary Gemini vs delayed Sarvam (like /api/ask)
            const tLLM = timer("LLM");
            try {
                const primaryPromise = callGemini(systemPrompt, phoneMessages, 512, phoneModel)
                    .then(r => r ? { reply: r, model: phoneModel } : null)
                    .catch(() => null);
                const sarvamPromise = new Promise(resolve =>
                    setTimeout(() => {
                        callSarvam(phoneMessages, 300)
                            .then(r => r ? resolve({ reply: r, model: "sarvam-105b" }) : resolve(null))
                            .catch(() => resolve(null));
                    }, 400)
                );
                const result = await Promise.race([
                    primaryPromise,
                    sarvamPromise,
                    new Promise(resolve => setTimeout(() => resolve(null), 6000)),
                ]);
                if (result) {
                    reply = result.reply;
                    phoneModel = result.model;
                }
            } catch (e) { console.log("[EXOTEL-RACE]", e.message); }

            // Fallback if race failed
            if (!reply || reply === refusal) {
                const fb = getAvailableModel();
                if (fb !== phoneModel) {
                    try { const r = await callGemini(systemPrompt, phoneMessages, 512, fb); if (r) { reply = r; phoneModel = fb; } }
                    catch (e) { console.log(`[EXOTEL-${fb}]`, e.message); }
                }
            }
            if (!reply) {
                try { const r = await callSarvam(phoneMessages, 300); if (r) { reply = r; phoneModel = "sarvam-105b"; } }
                catch (e) { console.log("[EXOTEL-SARVAM]", e.message); }
            }
            tLLM.end();

            // Apply rules engine (replaces scattered post-processing)
            if (reply) {
                reply = applyRules(reply, {
                    lang, ragChunks, isPhone: true,
                    originalQuery: transcript, refusal,
                });
            }
            if (!reply) reply = refusal;
        }

        // Update history
        session.history.push({ u: 1, t: transcript });
        session.history.push({ u: 0, t: reply });
        if (session.history.length > 20) session.history = session.history.slice(-20);

        console.log(`[EXOTEL] ${callSid} "${transcript.slice(0, 40)}" → "${reply.slice(0, 40)}"`);

        tCall.end();
    } catch (e) {
        console.log("[EXOTEL-ERROR]", e.message);
    }

    // Generate response audio — use gender-appropriate speaker if detected
    const tTTS = timer("TTS");
    const replyUrl = await combinedTtsToUrl(reply, lang, session.customSpeaker || undefined);
    tTTS.end();

    res.set("Content-Type", "application/xml");
    res.send(exoml(
        (replyUrl ? `  <Play>${replyUrl}</Play>` : `  <Say>${reply}</Say>`) +
        `\n  <Record action="${recordUrl}" method="POST" maxLength="15" finishOnKey="#" timeout="3" />`
    ));
});

// 4. Call status update / call end
app.post("/api/phone/exotel/status", verifyWebhookToken, (req, res) => {
    const callSid = req.body?.CallSid || req.query?.CallSid;
    const status = req.body?.Status || req.body?.status;
    console.log(`[EXOTEL] Call ${callSid} status: ${status}`);
    if (status === "completed" || status === "failed" || status === "no-answer") {
        phoneSessions.delete(callSid);
    }
    res.json({ status: "ok" });
});

// 5. Outbound call — trigger a call to test the agent
app.post("/api/phone/exotel/call", async (req, res) => {
    const { to, from } = req.body;
    const apiKey = process.env.EXOTEL_API_KEY;
    const apiToken = process.env.EXOTEL_API_TOKEN;
    const sid = process.env.EXOTEL_SID || process.env.EXOTEL_ACCOUNT_SID;

    if (!apiKey || !apiToken || !sid) {
        return res.status(400).json({
            error: "Missing Exotel credentials",
            required: ["EXOTEL_API_KEY", "EXOTEL_API_TOKEN", "EXOTEL_SID"],
            help: "Add these to your .env file from your Exotel dashboard",
        });
    }
    if (!to) return res.status(400).json({ error: "Provide 'to' phone number (e.g., +919XXXXXXXXX)" });

    const callbackUrl = appendWebhookToken(`${getPublicUrl()}/api/phone/exotel/call-start`);
    const statusUrl = appendWebhookToken(`${getPublicUrl()}/api/phone/exotel/status`);

    try {
        const authHeader = "Basic " + Buffer.from(`${apiKey}:${apiToken}`).toString("base64");
        const callFrom = from || process.env.EXOTEL_CALLER_ID || process.env.EXOTEL_NUMBER;
        if (!callFrom) return res.status(400).json({ error: "Set EXOTEL_CALLER_ID or EXOTEL_NUMBER in .env, or pass 'from' in request body" });

        // Exotel REST API v1 — make outbound call
        const formBody = new URLSearchParams({
            From: to,
            CallerId: callFrom,
            Url: callbackUrl,
            StatusCallback: statusUrl,
            StatusCallbackEvents: "terminal",
        });
        const apiUrl = `https://api.exotel.com/v1/Accounts/${sid}/Calls/connect.json`;
        console.log(`[EXOTEL] Outbound call to ${to} via ${callFrom}`);
        const r = await apiFetch(apiUrl, {
            method: "POST",
            headers: { Authorization: authHeader, "Content-Type": "application/x-www-form-urlencoded" },
            body: formBody.toString(),
        }, 15000);
        const d = await r.json().catch(() => ({}));
        if (r.ok) {
            console.log(`[EXOTEL] Call initiated: ${d.Call?.Sid || "unknown"}`);
            res.json({ success: true, callSid: d.Call?.Sid, status: d.Call?.Status, to, from: callFrom });
        } else {
            console.log(`[EXOTEL] Call failed: ${r.status}`, JSON.stringify(d).slice(0, 200));
            res.status(r.status).json({ error: "Exotel API error", details: d });
        }
    } catch (e) {
        console.log("[EXOTEL-CALL]", e.message);
        res.status(500).json({ error: e.message });
    }
});

// Exotel setup info endpoint
app.get("/api/phone/exotel/info", (_, res) => {
    const hasKey = !!process.env.EXOTEL_API_KEY;
    const hasToken = !!process.env.EXOTEL_API_TOKEN;
    const hasSid = !!(process.env.EXOTEL_SID || process.env.EXOTEL_ACCOUNT_SID);
    const hasPublicUrl = !!process.env.PUBLIC_URL;
    const hasCallerId = !!(process.env.EXOTEL_CALLER_ID || process.env.EXOTEL_NUMBER);

    res.json({
        provider: "Exotel",
        status: {
            api_key: hasKey ? "✓ configured" : "✗ missing",
            api_token: hasToken ? "✓ configured" : "✗ missing",
            account_sid: hasSid ? "✓ configured" : "✗ missing",
            public_url: hasPublicUrl ? `✓ ${process.env.PUBLIC_URL}` : "✗ missing (use ngrok)",
            caller_id: hasCallerId ? "✓ configured" : "✗ missing (needed for outbound calls)",
            ready: hasKey && hasToken && hasSid && hasPublicUrl,
        },
        webhook_base: `${getPublicUrl()}/api/phone/exotel`,
        setup_instructions: {
            step1: "Create Exotel account at exotel.com",
            step2: "Purchase a 1800 toll-free number or use existing ExoPhone",
            step3: "Run: ngrok http 3000  (copy the https URL)",
            step4: "Add to .env: PUBLIC_URL=https://xxxx.ngrok.io",
            step5: "Add to .env: EXOTEL_API_KEY, EXOTEL_API_TOKEN, EXOTEL_SID (from Exotel dashboard Settings → API)",
            step6: "Add to .env: EXOTEL_CALLER_ID=your-exophone-number (or EXOTEL_NUMBER)",
            step7: `In Exotel dashboard → ExoPhone → set incoming call webhook to: ${getPublicUrl()}/api/phone/exotel/call-start`,
            step8: "Test: POST /api/phone/exotel/call with {\"to\": \"+919XXXXXXXXX\"} to trigger outbound test call",
        },
        endpoints: {
            call_start: "POST /api/phone/exotel/call-start — Incoming call webhook (set in Exotel)",
            gather: "POST /api/phone/exotel/gather — DTMF handling (auto-linked)",
            audio: "POST /api/phone/exotel/audio — Recording handler (auto-linked)",
            status: "POST /api/phone/exotel/status — Call status updates",
            outbound: "POST /api/phone/exotel/call — {to, from?} Trigger outbound test call",
        },
        env_required: {
            EXOTEL_API_KEY: "API key from Exotel dashboard",
            EXOTEL_API_TOKEN: "API token from Exotel dashboard",
            EXOTEL_SID: "Account SID from Exotel dashboard",
            PUBLIC_URL: "Your public URL (ngrok https URL)",
            EXOTEL_CALLER_ID: "Your ExoPhone number (for outbound calls)",
        },
        active_sessions: [...phoneSessions.entries()].filter(([, v]) => v.provider === "exotel").length,
        languages: Object.keys(GREETINGS),
    });
});

// ═══════════════════════════════════════════════════════════
//  7. EVAL — trigger test suite
// ═══════════════════════════════════════════════════════════
app.get("/api/eval/status", (_, res) => {
    const fs = require("fs");
    try {
        const report = JSON.parse(fs.readFileSync("eval-report.json", "utf8"));
        res.json({ hasReport: true, ...report });
    } catch {
        res.json({ hasReport: false, message: "Run: node eval-engine.js" });
    }
});

// ═══════════════════════════════════════════════════════════
//  8. HEALTH
// ═══════════════════════════════════════════════════════════
app.get("/api/health", async (_, res) => {
    if (!SK) return res.json({ stt: false, ai: false, gemini: false, tts: false, rag: false });
    const h = { stt: false, ai: false, gemini: false, tts: false, rag: false };
    await Promise.allSettled([
        // Test Gemini/Gemma (primary brain) — uses smart rotation
        GK ? callGemini("Reply OK", [{ role: "user", content: "test" }], 5)
            .then(r => { h.gemini = !!r; h.ai = !!r; })
            .catch(() => { }) : Promise.resolve(),
        // Test Sarvam TTS + STT
        apiFetch("https://api.sarvam.ai/text-to-speech", {
            method: "POST", headers: HEADERS,
            body: JSON.stringify({ text: "test", target_language_code: "hi-IN", speaker: "shubh", model: "bulbul:v3" })
        }, 8000).then(r => { h.tts = r.ok; h.stt = r.ok; }),
        // Test Sarvam LLM (always available as fallback)
        apiFetch("https://api.sarvam.ai/v1/chat/completions", {
            method: "POST", headers: HEADERS,
            body: JSON.stringify({ model: "sarvam-105b", messages: [{ role: "user", content: "test" }], max_tokens: 3 })
        }, 8000).then(r => { if (r.ok) h.ai = true; }),
    ]);
    h.rag = true;
    const activeModel = getAvailableModel();
    res.json({ ...h, version: "12.0", brain: activeModel, models: getModelStatus(), fillersCached: fillerCache.size, ttsCached: ttsCache.size, phoneSessions: phoneSessions.size, publicUrl: process.env.PUBLIC_URL || null });
});

// Model status endpoint
app.get("/api/models", (_, res) => {
    const today = getTodayStr();
    const status = MODEL_PRIORITY.map(m => {
        const usage = modelUsage[m];
        const used = (usage && usage.resetDate === today) ? usage.count : 0;
        const limit = MODEL_LIMITS[m];
        return { model: m, used, limit: limit === Infinity ? "unlimited" : limit, available: limit === Infinity || used < limit };
    });
    res.json({ active: getAvailableModel(), models: status, date: today });
});

// ═══════════════════════════════════════════════════════════
//  GREETINGS — per language (for call start)
// ═══════════════════════════════════════════════════════════
const GREETINGS = {
    "hi-IN": "नमस्ते! मैं न्यायसाथी हूँ — भारत का मुफ़्त AI कानूनी सहायक। अपनी कानूनी समस्या बताइए, मैं पूरी मदद करूँगा।",
    "en-IN": "Hello! I'm NyayaSathi — India's free AI legal assistant. Tell me your legal problem, I'll do my best to help you.",
    "bn-IN": "নমস্কার! আমি ন্যায়সাথী — ভারতের বিনামূল্যে AI আইনি সহায়ক। আপনার আইনি সমস্যা বলুন, আমি পুরো সাহায্য করব।",
    "te-IN": "నమస్కారం! నేను న్యాయసాథి — భారత్ యొక్క ఉచిత AI న్యాయ సహాయకుడు। మీ చట్టపరమైన సమస్య చెప్పండి, నేను పూర్తిగా సహాయం చేస్తాను.",
    "ta-IN": "வணக்கம்! நான் நியாயசாதி — இந்தியாவின் இலவச AI சட்ட உதவியாளர். உங்கள் சட்ட பிரச்சனை சொல்லுங்கள், நான் முழுமையாக உதவுகிறேன்.",
    "mr-IN": "नमस्कार! मी न्यायसाथी — भारताचा मोफत AI कायदेशीर सहाय्यक. तुमची कायदेशीर समस्या सांगा, मी पूर्ण मदत करतो.",
    "gu-IN": "નમસ્તે! હું ન્યાયસાથી છું — ભારતનો મફત AI કાનૂની સહાયક. તમારી કાનૂની સમસ્યા કહો, હું પૂરી મદદ કરીશ.",
    "kn-IN": "ನಮಸ್ಕಾರ! ನಾನು ನ್ಯಾಯಸಾಥಿ — ಭಾರತದ ಉಚಿತ AI ಕಾನೂನು ಸಹಾಯಕ. ನಿಮ್ಮ ಕಾನೂನು ಸಮಸ್ಯೆ ಹೇಳಿ, ನಾನು ಪೂರ್ಣವಾಗಿ ಸಹಾಯ ಮಾಡುತ್ತೇನೆ.",
    "ml-IN": "നമസ്കാരം! ഞാൻ ന്യായസാഥി — ഭാരതത്തിന്റെ സൗജന്യ AI നിയമ സഹായി. നിങ്ങളുടെ നിയമ പ്രശ്നം പറയൂ, ഞാൻ പൂർണ്ണമായി സഹായിക്കാം.",
    "pa-IN": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਨਿਆਂਸਾਥੀ ਹਾਂ — ਭਾਰਤ ਦਾ ਮੁਫ਼ਤ AI ਕਾਨੂੰਨੀ ਸਹਾਇਕ। ਆਪਣੀ ਕਾਨੂੰਨੀ ਸਮੱਸਿਆ ਦੱਸੋ, ਮੈਂ ਪੂਰੀ ਮਦਦ ਕਰਾਂਗਾ।",
    "od-IN": "ନମସ୍କାର! ମୁଁ ନ୍ୟାୟସାଥୀ — ଭାରତର ମୁଫ୍ତ AI ଆଇନ ସହାୟକ। ଆପଣଙ୍କ ଆଇନ ସମସ୍ୟା କୁହନ୍ତୁ, ମୁଁ ସମ୍ପୂର୍ଣ୍ଣ ସାହାଯ୍ୟ କରିବି।",
};

// Silence warning — played when caller is quiet for too long
const SILENCE_WARNINGS = {
    "hi-IN": "क्या आप अभी भी वहाँ हैं? कृपया अपनी बात बोलिए।",
    "en-IN": "Are you still there? Please speak your question.",
    "bn-IN": "আপনি কি এখনও আছেন? অনুগ্রহ করে আপনার কথা বলুন।",
    "te-IN": "మీరు ఇంకా అక్కడ ఉన్నారా? దయచేసి మీ సమస్య చెప్పండి।",
    "ta-IN": "நீங்கள் இன்னும் இருக்கிறீர்களா? தயவுசெய்து பேசுங்கள்.",
    "mr-IN": "तुम्ही अजूनही आहात का? कृपया तुमची समस्या सांगा.",
    "gu-IN": "શું તમે હજુ ત્યાં છો? કૃपया તમારી સમસ્યા જણાવો.",
    "kn-IN": "ನೀವು ಇನ್ನೂ ಇದ್ದೀರಾ? ದಯವಿಟ್ಟು ನಿಮ್ಮ ಸಮಸ್ಯೆ ಹೇಳಿ.",
    "ml-IN": "നിങ്ങൾ ഇനിയും ഉണ്ടോ? ദയവായി നിങ്ങളുടെ കാര്യം പറയൂ.",
    "pa-IN": "ਕੀ ਤੁਸੀਂ ਅਜੇ ਵੀ ਉੱਥੇ ਹੋ? ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੀ ਸਮੱਸਿਆ ਦੱਸੋ.",
    "od-IN": "ଆପଣ ଏଖନ ବି ଅଛନ୍ତି? ଦୟାକରି ଆପଣଙ୍କ ସମସ୍ୟା କୁହନ୍ତୁ।",
};

// Goodbye messages — played before hanging up
const GOODBYES = {
    "hi-IN": "ठीक है। अगर आपको कोई कानूनी मदद चाहिए तो दोबारा फ़ोन करें। धन्यवाद।",
    "en-IN": "Alright. Please call back anytime you need legal help. Thank you. Goodbye.",
    "bn-IN": "ঠিক আছে। যেকোনো সময় আইনি সাহায্যের জন্য আবার ফোন করুন। ধন্যবাদ।",
    "te-IN": "సరే. ఏ సమయంలోనైనా చట్టపరమైన సహాయం కావాలంటే తిరిగి ఫోన్ చేయండి. ధన్యవాదాలు.",
    "ta-IN": "சரி. எப்போது வேண்டுமானாலும் சட்ட உதவிக்கு மீண்டும் அழைக்கவும். நன்றி.",
    "mr-IN": "ठीक आहे. कधीही कायदेशीर मदतीसाठी परत फोन करा. धन्यवाद.",
    "gu-IN": "ઠીક છે. ગમે ત્યારે કાયદાકીય મદદ માટે ફરી ફોન કરો. આભાર.",
    "kn-IN": "ಸರಿ. ಯಾವಾಗ ಬೇಕಾದರೂ ಕಾನೂನು ಸಹಾಯಕ್ಕಾಗಿ ಮತ್ತೆ ಕರೆ ಮಾಡಿ. ಧನ್ಯವಾದ.",
    "ml-IN": "ശരി. ഏത് സമയത്തും നിയമ സഹായത്തിനായി വിളിക്കൂ. നന്ദി.",
    "pa-IN": "ਠੀਕ ਹੈ। ਕਿਸੇ ਵੀ ਸਮੇਂ ਕਾਨੂੰਨੀ ਮਦਦ ਲਈ ਦੁਬਾਰਾ ਫ਼ੋਨ ਕਰੋ। ਧੰਨਵਾਦ।",
    "od-IN": "ଠିକ ଅଛି। ଯେ କୌଣସି ସମୟରେ ଆଇନ ସାହାଯ୍ୟ ପାଇଁ ପୁଣି ଫୋନ କରନ୍ତୁ। ଧନ୍ୟବାଦ।",
};

// ═══════════════════════════════════════════════════════════
//  2a. CASE ENDPOINTS — read-only inspection of persisted cases
//
//  These exist so the web UI can render "case so far" badges, and so a
//  callback flow (or future lawyer dashboard) can look up history. Writes
//  always go through appendToSession → caseStore.persistTurn; nothing here
//  mutates state.
// ═══════════════════════════════════════════════════════════
app.get("/api/cases/:id", (req, res) => {
    const id = String(req.params.id || "");
    if (!/^case_[a-zA-Z0-9_-]+$/.test(id)) return res.status(400).json({ error: "bad id" });
    const c = caseStore.loadCase(id);
    if (!c) return res.status(404).json({ error: "not found" });
    // Strip phone PII unless the request includes a matching sessionId.
    // Tightens later when we add real auth.
    const { phone, ...safe } = c;
    res.json({ ...safe, phone: phone ? phone.slice(-4).padStart(phone.length, "*") : null });
});

app.get("/api/cases/by-phone/:phone", (req, res) => {
    const cases = caseStore.findCasesByPhone(req.params.phone);
    res.json({
        count: cases.length,
        cases: cases.map(c => ({
            id: c.id,
            type: c.type,
            urgency: c.urgency,
            status: c.status,
            summary: c.summary || (c.history?.[0]?.t || "").slice(0, 120),
            updatedAt: c.updatedAt,
            createdAt: c.createdAt,
            needs_lawyer: c.needs_lawyer,
            turn_count: Math.floor((c.history?.length || 0) / 2),
        })),
    });
});

// ═══════════════════════════════════════════════════════════
//  Lawyer match (Phase E) — score & rank lawyers for a case.
//  Writes a handoff record to the case file when the user proceeds. The
//  outbound call / SMS to the lawyer is a later phase; for now we just
//  capture intent, so a future job can fan it out.
// ═══════════════════════════════════════════════════════════
app.post("/api/lawyers/match", async (req, res) => {
    const { caseId, limit, preferFemale, preferLanguage, preferCity } = req.body || {};
    if (!caseId || typeof caseId !== "string") return res.status(400).json({ error: "caseId required" });
    const c = caseStore.loadCase(caseId);
    if (!c) return res.status(404).json({ error: "case not found" });

    const matches = lawyerMatch.matchLawyers(c, {
        limit: Math.max(1, Math.min(10, limit || 3)),
        preferFemale,
        preferLanguage,
        preferCity,
    });

    // Record the match request on the case file. Don't include lawyer phone
    // numbers in the handoff record yet — only persist the lawyer ids until
    // the user explicitly chooses one (a later "accept" endpoint, Phase E
    // follow-up).
    c.lawyer_offered = true;
    c.lawyer_handoffs = c.lawyer_handoffs || [];
    c.lawyer_handoffs.push({
        ts: Date.now(),
        type: "match-presented",
        candidates: matches.map(m => ({ id: m.lawyer.id, score: m.score })),
    });
    try { caseStore.saveCase(c); } catch (e) { console.warn("[LAWYER] persist failed:", e.message); }

    res.json({
        caseId: c.id,
        matches,
        count: matches.length,
    });
});

// Accept a specific lawyer match — exposes the contact details (phone /
// email) that were withheld in /api/lawyers/match. Records the acceptance
// on the case file. The actual outbound dialer is later phase work.
app.post("/api/lawyers/accept", async (req, res) => {
    const { caseId, lawyerId } = req.body || {};
    if (!caseId || !lawyerId) return res.status(400).json({ error: "caseId + lawyerId required" });
    const c = caseStore.loadCase(caseId);
    if (!c) return res.status(404).json({ error: "case not found" });
    const full = lawyerMatch.getLawyerById(lawyerId);
    if (!full) return res.status(404).json({ error: "lawyer not found" });

    c.lawyer_handoffs = c.lawyer_handoffs || [];
    c.lawyer_handoffs.push({
        ts: Date.now(),
        type: "accepted",
        lawyer_id: lawyerId,
        lawyer_name: full.name,
    });
    try { caseStore.saveCase(c); } catch { }

    res.json({
        caseId: c.id,
        lawyer: {
            id: full.id, name: full.name, city: full.city, state: full.state,
            phone: full.phone, email: full.email,
            specializations: full.specializations, languages: full.languages,
            rating: full.rating, years_experience: full.years_experience,
            bar_council: full.bar_council, fee_first_consult: full.fee_first_consult,
        },
        next_step: "We've shared your case summary with the lawyer. They will reach out within 24 hours.",
    });
});

// ═══════════════════════════════════════════════════════════
//  2b. STREAMING /api/ask-stream — AI + TTS chunks streamed as ready
// ═══════════════════════════════════════════════════════════
app.post("/api/ask-stream", async (req, res) => {
    const { message, lang = "hi-IN", speaker, sessionId: reqSessionId, phone: reqPhone } = req.body;
    // Server-side session is the single source of truth — same as /api/ask.
    // The client previously passed its own `history` array here, which let the
    // prompt bloat past 2,000 tokens by turn 3 and triggered the cascade timeout.
    const { session, sessionId, isReturning } = getOrCreateSession(reqSessionId, { phone: reqPhone, lang });
    session.lang = lang;
    const msg = sanitize(message, 500);
    if (!msg) return res.status(400).json({ error: "No message" });
    if (!SK) return res.status(500).json({ error: "No API key" });
    const t0 = Date.now();
    const refusal = getRefusal(lang);

    // Set up NDJSON streaming
    res.setHeader("Content-Type", "application/x-ndjson");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("X-Accel-Buffering", "no");
    const sendChunk = (obj) => { try { res.write(JSON.stringify(obj) + "\n"); } catch { } };

    // LAYER 1 GUARDRAIL
    const { allow, reason: guardReason } = isLegalQuery(msg);
    if (!allow) {
        console.log(`[STREAM] L1_BLOCK [${guardReason}]: "${msg.slice(0, 50)}"`);
        sendChunk({ type: "reply", reply: refusal, model: "guardrail", blocked: true, sessionId });
        try {
            const result = await generateTTS(refusal, lang, speaker);
            if (result) sendChunk({ type: "audio", audio: result.audio, index: 0 });
        } catch { }
        sendChunk({ type: "done", ms: Date.now() - t0, sessionId });
        return res.end();
    }

    // DISTRESS DETECTION (Phase D) — bypass everything for critical signals.
    // First chunk on the wire goes out in <50ms; the user gets a helpline
    // number before the LLM would have even started thinking.
    const distressCheck = distress.detect(msg, lang);
    if (distressCheck.level === "critical") {
        console.log(`[STREAM-SAFETY] ${distressCheck.reason} matched="${distressCheck.matched}" → bypass`);
        const reply = distressCheck.prepend;
        if (session.case) {
            session.case.urgency = "critical";
            session.case.needs_lawyer = true;
            session.case.distress_level = Math.max(session.case.distress_level || 0, distress.levelToScore("critical"));
        }
        sendChunk({
            type: "reply", reply, model: "safety-bypass", sessionId,
            safety: distressCheck.reason, helplines: distressCheck.helplines,
            caseId: session.case?.id, isReturning,
        });
        appendToSession(session, msg, reply, { model: "safety-bypass" });
        const segments = segmentForTTS(reply);
        await Promise.allSettled(segments.map((chunk, i) =>
            generateTTS(chunk, lang, speaker).then(result => {
                if (result) sendChunk({ type: "audio", audio: result.audio, index: i, text: chunk });
            }).catch(() => { })
        ));
        sendChunk({ type: "done", ms: Date.now() - t0, aiMs: 0, segments: segments.length, sessionId, safety: true });
        return res.end();
    }
    if (distressCheck.level === "high" && session.case) {
        session.case.distress_level = Math.max(session.case.distress_level || 0, distress.levelToScore("high"));
    }

    // FAQ CHECK — instant pre-built answer, skip LLM entirely
    const faqMatch = matchFAQ(msg, lang);
    if (faqMatch) {
        console.log(`[STREAM] FAQ HIT: "${faqMatch.answer.slice(0, 60)}"`);
        sendChunk({ type: "reply", reply: faqMatch.answer, model: "faq-template", faq: true, sessionId, caseId: session.case?.id, isReturning });
        appendToSession(session, msg, faqMatch.answer, { model: "faq-template" });
        const segments = segmentForTTS(faqMatch.answer);
        await Promise.allSettled(segments.map((chunk, i) =>
            generateTTS(chunk, lang, speaker).then(result => {
                if (result) sendChunk({ type: "audio", audio: result.audio, index: i, text: chunk });
            }).catch(() => { })
        ));
        sendChunk({ type: "done", ms: Date.now() - t0, aiMs: 0, ttsMs: Date.now() - t0, segments: segments.length, sessionId });
        console.log(`[STREAM] ${Date.now() - t0}ms FAQ segs:${segments.length}`);
        return res.end();
    }

    // RESPONSE CACHE CHECK
    const cacheKey = responseCacheKey(msg, lang);
    const cached = responseCacheGet(cacheKey);
    if (cached) {
        console.log(`[STREAM] CACHE HIT: "${cached.reply.slice(0, 60)}"`);
        sendChunk({ type: "reply", reply: cached.reply, model: cached.model + " (cached)", cached: true, sessionId, caseId: session.case?.id, isReturning });
        appendToSession(session, msg, cached.reply, { model: cached.model });
        const segments = segmentForTTS(cached.reply);
        await Promise.allSettled(segments.map((chunk, i) =>
            generateTTS(chunk, lang, speaker).then(result => {
                if (result) sendChunk({ type: "audio", audio: result.audio, index: i, text: chunk });
            }).catch(() => { })
        ));
        sendChunk({ type: "done", ms: Date.now() - t0, aiMs: 0, ttsMs: Date.now() - t0, segments: segments.length, sessionId });
        console.log(`[STREAM] ${Date.now() - t0}ms CACHE segs:${segments.length}`);
        return res.end();
    }

    // RAG RETRIEVAL
    const { contextString: ragContext, chunks: ragChunks } = buildContext(msg);

    // Build conversation — server-side history only. Older turns live compressed
    // in session.summary so the prompt stays bounded across long calls.
    // Slot-aware case context (Phase C) gets injected via buildSystemPrompt.
    const systemContent = buildSystemPrompt(lang, ragContext || "", session);
    const messages = [{ role: "system", content: systemContent }];
    let lastRole = "system";
    for (const m of session.history.slice(-SESSION_MAX_TURNS * 2)) {
        const role = m.u ? "user" : "assistant";
        if (role === lastRole) continue;
        messages.push({ role, content: sanitize(m.t, 300) });
        lastRole = role;
    }
    if (messages[messages.length - 1]?.role === "user") {
        messages.push({ role: "assistant", content: lang === "en-IN" ? "Yes, please go ahead." : "जी, बताइए।" });
    }
    messages.push({ role: "user", content: msg });

    // AI CALL — Direct Gemini (fast, proven) with Sarvam fallback only if Gemini fails
    const aiT0 = Date.now();
    let reply = "";
    let model = getAvailableModel();
    const systemPrompt = systemContent;

    // Helper: TTS a sentence and stream audio chunk
    function ttsSentence(sentence, idx) {
        return generateTTS(sentence, lang, speaker).then(result => {
            if (result) sendChunk({ type: "audio", audio: result.audio, index: idx, text: sentence });
        }).catch(() => { });
    }

    // SPEED: Race Gemini vs Sarvam 105B — first SUCCESS wins (Promise.any),
    // wrapped in an 8s wall-clock so the user never waits forever.
    // Promise.any (not Promise.race) so a fast rejection from one provider
    // doesn't kill the whole race when the other is still working.
    try {
        const primaryPromise = callGemini(systemPrompt, messages, 512, model)
            .then(r => r ? { reply: stripMarkdown(r), model } : Promise.reject(new Error("empty")))
            .catch(e => Promise.reject(e));
        const sarvamPromise = callSarvam(messages, 500)
            .then(r => r ? { reply: stripMarkdown(r), model: "sarvam-105b" } : Promise.reject(new Error("empty")))
            .catch(e => Promise.reject(e));
        const result = await Promise.race([
            Promise.any([primaryPromise, sarvamPromise]).catch(() => null),
            new Promise(resolve => setTimeout(() => resolve(null), 8000)),
        ]);
        if (result && result.reply) {
            reply = result.reply;
            model = result.model;
        }
    } catch (e) { console.log(`[STREAM-RACE] FAIL:`, e.message); }

    // Fallback if race failed: try the next available Gemini key
    if (!reply) {
        const fallbackModel = getAvailableModel();
        if (fallbackModel !== model) {
            try {
                const retryReply = await callGemini(systemPrompt, messages, 512, fallbackModel);
                if (retryReply) { reply = stripMarkdown(retryReply); model = fallbackModel; }
            } catch (e) { console.log(`[STREAM-${fallbackModel}] FAIL:`, e.message); }
        }
    }
    if (!reply) {
        try {
            const sarvamReply = await callSarvam(messages, 300);
            if (sarvamReply) { reply = stripMarkdown(sarvamReply); model = "sarvam-105b"; }
        } catch (e) { console.log("[STREAM-SARVAM] FAIL:", e.message); }
    }

    const aiMs = Date.now() - aiT0;

    // Post-processing through rules engine
    if (reply) {
        reply = applyRules(reply, { lang, ragChunks, isPhone: false, originalQuery: msg, refusal });
    }

    let isFallback = false;
    if (!reply) {
        // Case-aware soft fallback — never the cold "तकनीकी समस्या" string.
        reply = getSmartFallback(lang, msg, session);
        model = "smart-fallback";
        isFallback = true;
    }

    // Cache only successful, non-fallback, non-refusal responses.
    const allFallbackValues = Object.values(FALLBACK_ERROR).concat(Object.values(SOFT_FALLBACK));
    if (reply && reply !== refusal && !isFallback && !allFallbackValues.includes(reply) && model !== "fallback" && model !== "smart-fallback") {
        responseCacheSet(cacheKey, reply, model, !!ragContext);
    }

    // Persist this turn into the server-side session + case file BEFORE TTS,
    // so even if TTS hangs, the next turn already sees the new history.
    appendToSession(session, msg, reply, { model });

    // Send reply text so frontend can display it
    sendChunk({ type: "reply", reply, model, aiMs, rag: !!ragContext, sessionId, caseId: session.case?.id, isReturning });

    // Parallel TTS for all segments
    const ttsT0 = Date.now();
    const segments = segmentForTTS(reply);
    const ttsPromises = segments.map((chunk, i) => ttsSentence(chunk, i));
    const audioIndex = segments.length;

    await Promise.allSettled(ttsPromises);
    const ttsMs = Date.now() - ttsT0;

    sendChunk({ type: "done", ms: Date.now() - t0, aiMs, ttsMs, segments: audioIndex, sessionId });
    console.log(`[STREAM] ${Date.now() - t0}ms (AI:${aiMs} TTS:${ttsMs}) segs:${audioIndex} sid:${sessionId.slice(-6)} "${reply.slice(0, 60)}"`);
    res.end();
});

// ═══════════════════════════════════════════════════════════
//  LEGACY /api/ai endpoint (backward compat)
// ═══════════════════════════════════════════════════════════
app.post("/api/ai", async (req, res) => {
    const { history = [], message, lang = "hi-IN" } = req.body;
    const msg = sanitize(message, 500);
    if (!msg) return res.status(400).json({ error: "No message" });
    if (!SK) return res.status(500).json({ error: "No API key" });
    const refusal = getRefusal(lang);
    const { allow } = isLegalQuery(msg);
    if (!allow) return res.json({ reply: refusal, model: "guardrail", blocked: true });

    const { contextString: ragContext } = buildContext(msg);
    const systemPrompt = getLangPrompt(lang, ragContext);
    const messages = [
        { role: "system", content: systemPrompt },
        ...history.slice(-6).map(m => ({ role: m.u ? "user" : "assistant", content: sanitize(m.t, 200) })),
        { role: "user", content: msg },
    ];
    const t0 = Date.now();
    let reply = "";
    try { reply = await callGemini(systemPrompt, messages, 256); } catch { }
    if (!reply) { try { reply = await callSarvam(messages, 150); } catch { } }
    if (reply) reply = applyRules(reply, { lang, ragChunks: [], isPhone: false, originalQuery: msg, refusal });
    if (!reply) reply = refusal;
    res.json({ reply, model: reply === refusal ? "fallback" : "gemini", ms: Date.now() - t0 });
});

// ═══════════════════════════════════════════════════════════
//  ERROR HANDLING
// ═══════════════════════════════════════════════════════════
process.on("uncaughtException", e => console.error("[CRASH]", e.message));
process.on("unhandledRejection", e => console.error("[UNHANDLED]", e?.message || e));

// ═══════════════════════════════════════════════════════════
//  START
// ═══════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════
//  NGROK AUTO-DETECTION — find public URL automatically
// ═══════════════════════════════════════════════════════════
async function detectNgrok() {
    if (process.env.PUBLIC_URL) return process.env.PUBLIC_URL;
    try {
        const r = await fetch("http://127.0.0.1:4040/api/tunnels", { signal: AbortSignal.timeout(2000) });
        if (r.ok) {
            const d = await r.json();
            const https = d.tunnels?.find(t => t.proto === "https");
            if (https?.public_url) {
                process.env.PUBLIC_URL = https.public_url;
                console.log(`  [NGROK] Auto-detected: ${https.public_url}`);
                return https.public_url;
            }
        }
    } catch { }
    return null;
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, async () => {
    console.log(`\n  ╔══════════════════════════════════════════════════╗`);
    console.log(`  ║  NyayaSathi v12.0 — India's AI Legal Helpline   ║`);
    console.log(`  ║  Brain: Smart Race (Flash+Sarvam 105B) + RAG    ║`);
    console.log(`  ║  Court: 50+ SC Judgments + BNS/IPC Mapping      ║`);
    console.log(`  ║  Voice: Sarvam Bulbul v3 TTS (11 languages)     ║`);
    console.log(`  ║  Phone: Exotel 1800 Toll-Free + Outbound API    ║`);
    console.log(`  ║  http://localhost:${PORT}                           ║`);
    console.log(`  ╚══════════════════════════════════════════════════╝\n`);
    if (!SK) { console.log("  ⚠  Set SARVAM_API_KEY in .env\n"); return; }
    if (!GK) { console.log("  ⚠  Set GEMINI_API_KEY in .env (using Sarvam LLM fallback)\n"); }

    // Auto-detect ngrok tunnel
    const ngrokUrl = await detectNgrok();

    console.log(`  Brain: ${getAvailableModel()} STT:✓ TTS:✓ RAG:✓`);
    console.log(`  Sarvam Phone:  GET /api/phone/info`);
    console.log(`  Exotel Phone:  GET /api/phone/exotel/info`);
    if (ngrokUrl) {
        console.log(`  Exotel Webhook: ${ngrokUrl}/api/phone/exotel/call-start`);
        console.log(`  Test Call:      POST /api/phone/exotel/call {"to":"+919XXXXXXXXX"}`);
    } else {
        console.log(`  ⚠  No PUBLIC_URL — run "ngrok http ${PORT}" for Exotel webhooks`);
    }
    console.log(`  Eval:          node eval-engine.js\n`);
    // Warm fillers first, then IVR — sequential to avoid Sarvam 429 rate limits
    warmFillers().then(() => warmExotelIVR()).catch(() => { });
});

