/**
 * NyayaSathi v12 — India's Advanced AI Legal Agent
 * ════════════════════════════════════════════════
 * Features:
 *  - RAG: BM25 retrieval over 25-chunk Indian legal corpus
 *  - Self-correction: transcript confidence + response validation
 *  - Phone dial-in: Sarvam telephony webhook (/api/phone/*)
 *  - Advanced guardrails: 3-layer (pre-AI, LLM system, post-AI)
 *  - All 11 Sarvam languages with optimal voice + pace per language
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

const app = express();
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

CRITICAL: Your output goes DIRECTLY to a Text-to-Speech engine. Write ONLY plain spoken English sentences. NO Devanagari. NO markdown. NO bullets. NO asterisks. NO numbered lists. NO digits — write all numbers as words (say "one hundred thirty-eight" not "138").

STYLE: You are a brilliant, experienced Supreme Court lawyer who genuinely cares. You speak with authority AND warmth. You give SPECIFIC, ACTIONABLE advice — not vague generalities. You name exact Acts, Sections, Courts, Forms, Helplines. You tell them the exact steps like a GPS for their legal journey.

LEGAL REFERENCES:
${rag || "No specific legal reference found. Direct caller to NALSA helpline fifteen-one-hundred for a free lawyer. Do not cite any section or act number."}

HOW TO RESPOND:
First, one line of empathy — show you understand their exact pain.
Then give the STRONGEST legal weapon they have — name the Act, Section, and the landmark Supreme Court judgment if you know it.
Then give them a STEP-BY-STEP action plan: (a) where to go physically, (b) what document to file, (c) any critical deadline they must know, (d) the helpline or website.
If the caller's situation is unclear or you need more details to give proper advice, ask ONE specific follow-up question. But if they already explained their full problem clearly, give the complete answer directly — do NOT force a question at the end.

HARD RULES:
- MAX 90 words. This is a phone call — pack maximum value in minimum words.
- ONLY Indian law questions. Non-legal: "I can only help with legal matters. Please tell me your legal problem."
- Be SPECIFIC: say "file Form A at District Consumer Commission within two years" not "you can complain."
- Cite the specific Act, Section, AND the relevant Court or Authority.
- ALWAYS mention NALSA fifteen-one-hundred for free lawyer.
- Do NOT ask unnecessary questions. If the person has given you enough context, just answer fully.
- NEVER use Devanagari script or Hindi words.
- Write numbers as words: "Section one hundred thirty-eight" not "Section 138".
- CRITICAL: ONLY cite Acts and Sections from the LEGAL REFERENCES above. If the reference doesn't cover their issue, say so honestly and direct them to NALSA.`,

    "hi-IN": (rag) => `आप न्यायसाथी हैं — भारत की मुफ़्त कानूनी हेल्पलाइन। आप फ़ोन पर बात कर रहे हैं।

सबसे ज़रूरी बात: आपका जवाब सीधे बोलने वाली मशीन में जाएगा। इसलिए:
- पूरा जवाब शुद्ध हिंदी देवनागरी में लिखें। कोई भी अंग्रेज़ी शब्द मत लिखें।
- कानूनी शब्द हिंदी में लिखें: "धारा" (Section), "अधिनियम" (Act), "अदालत" (Court), "उच्चतम न्यायालय" (Supreme Court), "ज़िला न्यायालय" (District Court), "उपभोक्ता आयोग" (Consumer Commission), "प्रथम सूचना रिपोर्ट" (FIR), "ज़मानत" (Bail), "याचिका" (Petition)
- सभी अंक हिंदी शब्दों में लिखें: "एक सौ अड़तीस" (138), "तीस दिन" (30 days), "पंद्रह सौ" (1500)
- फ़ोन नंबर भी हिंदी में: "एक पाँच एक शून्य शून्य" (15100)
- कोई मार्कडाउन, बुलेट, तारा चिह्न, या सूची नहीं। सीधे बोलचाल वाले वाक्य लिखें।

अंदाज़: आप एक अनुभवी वकील हैं जो गाँव के लोगों से बात कर रहे हैं। सरल, साफ़ हिंदी बोलें। छोटे वाक्य। हर बात आसान शब्दों में। जैसे कोई बड़ा भाई समझा रहा हो।

कानूनी संदर्भ:
${rag || "कोई विशेष कानूनी संदर्भ नहीं मिला। कॉलर को नालसा हेल्पलाइन एक पाँच एक शून्य शून्य पर फ़ोन करने को बोलें। कोई धारा या अधिनियम न बताएं।"}

जवाब कैसे दें:
पहले एक लाइन सहानुभूति — "मैं आपकी परेशानी समझ रहा हूँ।"
फिर सबसे ज़रूरी कानूनी जानकारी — कौन सा अधिनियम, कौन सी धारा।
फिर क्या करना है — कहाँ जाना है, क्या लिखवाना है, कितने दिन में करना है।
आख़िर में नालसा हेल्पलाइन "एक पाँच एक शून्य शून्य" ज़रूर बताएं।
अगर बात पूरी तरह साफ़ नहीं है तो एक सवाल पूछें। लेकिन अगर बात साफ़ है तो सीधे जवाब दें।

सख्त नियम:
- अधिकतम नब्बे शब्द। फ़ोन पर बात है — कम बोलें, काम की बात बोलें।
- सिर्फ़ कानूनी सवालों का जवाब दें। बाकी: "मैं सिर्फ़ कानूनी मामलों में मदद करता हूँ।"
- साफ़ बताएं: "ज़िला उपभोक्ता आयोग में दो साल के अंदर शिकायत दर्ज करें"
- नालसा "एक पाँच एक शून्य शून्य" हमेशा बताएं — मुफ़्त वकील मिलेगा।
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

CRITICAL: Output goes DIRECTLY to TTS. Write ONLY in spoken ${langName} using the native script. Keep legal terms (Act, Section, Court) in English. NO markdown, bullets, asterisks, or numbered lists. Write numbers as words.

STYLE: You are a brilliant Supreme Court lawyer who genuinely cares. Give SPECIFIC, ACTIONABLE advice — name exact Acts, Sections, Courts, Forms, Helplines, and deadlines. Not vague advice.

LEGAL REFERENCES:
${ragContext || "No specific legal reference found. Direct caller to NALSA 15100 for a free lawyer. Do not cite any section or act number."}

HOW TO RESPOND:
First empathize — one line showing you understand their pain.
Then give the STRONGEST legal weapon — Act, Section, landmark judgment.
Then STEP-BY-STEP action plan: where to go, what to file, deadline, helpline/website.
If the situation needs clarification, ask ONE follow-up question. If they already explained clearly, give the full answer directly without forcing a question.

HARD RULES:
- MAX 90 words. Phone call — pack maximum value in minimum words.
- ONLY Indian law questions. Non-legal: "I can only help with legal matters."
- Be SPECIFIC: name the Court, Form, deadline, helpline — not just "complain."
- Cite specific Act + Section from LEGAL REFERENCES only. If unsure, direct to NALSA 15100.
- ALWAYS mention NALSA 15100 for free lawyer.
- NO markdown, bullets, asterisks.
- Only ask a question if you truly need more information.`;
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

function cleanLLMResponse(text, lang) {
    if (!text || typeof text !== "string") return getRefusal(lang);
    let clean = text;
    // Strip thinking tags
    if (clean.includes("</think>")) clean = clean.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
    else if (clean.includes("<think>")) clean = clean.replace(/<think>\s*/gi, "").trim();
    // Strip markdown
    clean = clean.replace(/<[^>]*>/g, "").replace(/\*{1,3}/g, "").replace(/#{1,6}\s*/g, "")
        .replace(/```[\s\S]*?```/g, "").replace(/`[^`]*`/g, "")
        .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
        .replace(/^\s*[-*•]\s+/gm, "")
        .replace(/^\s*\d+\.\s+/gm, "")
        .replace(/\n+/g, " ").replace(/\s{2,}/g, " ").trim();

    // For Hindi: replace any English words that slipped through with Hindi equivalents
    if (lang !== "en-IN") {
        const hindiReplace = {
            "Section": "धारा", "section": "धारा",
            "Act": "अधिनियम", "act": "अधिनियम",
            "Court": "अदालत", "court": "अदालत",
            "Supreme Court": "उच्चतम न्यायालय", "supreme court": "उच्चतम न्यायालय",
            "High Court": "उच्च न्यायालय", "high court": "उच्च न्यायालय",
            "District Court": "ज़िला अदालत", "district court": "ज़िला अदालत",
            "Consumer Commission": "उपभोक्ता आयोग", "consumer commission": "उपभोक्ता आयोग",
            "Consumer Forum": "उपभोक्ता मंच", "consumer forum": "उपभोक्ता मंच",
            "FIR": "एफ़ आई आर", "F.I.R.": "एफ़ आई आर", "F.I.R": "एफ़ आई आर",
            "Bail": "ज़मानत", "bail": "ज़मानत",
            "Petition": "याचिका", "petition": "याचिका",
            "Police": "पुलिस", "police": "पुलिस",
            "Lawyer": "वकील", "lawyer": "वकील",
            "Complaint": "शिकायत", "complaint": "शिकायत",
            "Magistrate": "मजिस्ट्रेट",
            "NALSA": "नालसा", "Nalsa": "नालसा",
            "Legal Services Authority": "विधिक सेवा प्राधिकरण",
            "Domestic Violence": "घरेलू हिंसा", "domestic violence": "घरेलू हिंसा",
            "Negotiable Instruments": "परक्राम्य लिखत", "negotiable instruments": "परक्राम्य लिखत",
            "Indian Penal Code": "भारतीय दंड संहिता",
            "IPC": "आई पी सी", "BNS": "बी एन एस", "CrPC": "सी आर पी सी", "BNSS": "बी एन एस एस",
            "Protection of Women": "महिला सुरक्षा",
            "Cheque": "चेक", "cheque": "चेक",
            "Notice": "नोटिस", "notice": "नोटिस",
            "Helpline": "हेल्पलाइन", "helpline": "हेल्पलाइन",
            "Form": "फ़ॉर्म", "form": "फ़ॉर्म",
            "Call": "फ़ोन करें", "call": "फ़ोन करें",
        };
        // Replace longer phrases first to avoid partial replacements
        const sorted = Object.entries(hindiReplace).sort((a, b) => b[0].length - a[0].length);
        for (const [eng, hin] of sorted) {
            clean = clean.replace(new RegExp(`\\b${eng.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, "g"), hin);
        }
        // Replace remaining digits with Hindi words
        clean = clean.replace(/\b(\d{1,6})\b/g, (_, n) => {
            const num = parseInt(n);
            if (num > 0 && num <= 999999) {
                const { numToHindi } = require("./voice-engine");
                return numToHindi(num);
            }
            return n;
        });

        // ── Hard Devanagari enforcer: strip any remaining Roman/Latin words ──
        // Keep only: NyayaSathi, NALSA, RERA, RTI, PIL, IPC, BNS (no Hindi equivalent)
        const ALLOWED_ACRONYMS = new Set(["NyayaSathi", "NALSA", "RERA", "RTI", "PIL", "NCLT", "NCLAT", "HC", "SC"]);
        clean = clean.replace(/\b[A-Za-z]{3,}\b/g, (match) => {
            if (ALLOWED_ACRONYMS.has(match) || ALLOWED_ACRONYMS.has(match.toUpperCase())) return match;
            return ""; // strip unrecognized English words
        });
        // Collapse extra whitespace after stripping
        clean = clean.replace(/\s{2,}/g, " ").trim();
    }

    // Enforce max length — keep concise but allow richer responses (90 words ≈ 580 chars)
    if (clean.length > 580) {
        clean = clean.slice(0, 580);
        const lastPunct = Math.max(clean.lastIndexOf("."), clean.lastIndexOf("!"), clean.lastIndexOf("?"), clean.lastIndexOf("।"));
        if (lastPunct > 300) clean = clean.slice(0, lastPunct + 1);
        clean += lang === "en-IN" ? " Call NALSA 15100." : " नालसा एक पाँच एक शून्य शून्य पर फ़ोन करें।";
    }
    if (clean.length < 5) return getRefusal(lang);
    return clean;
}

// Rate limiter
const rateMap = new Map();
function rateLimit(req, res, next) {
    const ip = req.ip || "x"; const now = Date.now();
    const hits = (rateMap.get(ip) || []).filter(t => now - t < 60000);
    if (hits.length >= 100) return res.status(429).json({ error: "Too many requests." });
    hits.push(now); rateMap.set(ip, hits); next();
}
app.use("/api", rateLimit);

// Fetch with timeout
async function apiFetch(url, opts, timeoutMs = 12000) {
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
    const timeout = IS_GEMMA ? 8000 : 3500;

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

// ═══════════════════════════════════════════════════════════
//  SARVAM LLM — Fallback
// ═══════════════════════════════════════════════════════════
async function callSarvam(messages, maxTokens = 120) {
    const r = await apiFetch("https://api.sarvam.ai/v1/chat/completions", {
        method: "POST", headers: HEADERS,
        body: JSON.stringify({ model: "sarvam-m", messages, max_tokens: maxTokens, temperature: 0.3 }),
    }, 8000);
    if (!r.ok) return null;
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
function ttsCacheKey(text, lang) { return `${lang}:${text.slice(0, 120)}`; }
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
        { patterns: ["fir", "kaise", "darj", "police", "thana"], answer: "एफ़ आई आर दर्ज करने के लिए नज़दीकी थाने में जाइए। बी एन एस एस धारा एक सौ तिहत्तर के तहत पुलिस एफ़ आई आर दर्ज करने से मना नहीं कर सकती। अगर मना करे तो एस पी को लिखित शिकायत भेजें। ज़ीरो एफ़ आई आर किसी भी थाने में दर्ज हो सकती है। ललिता कुमारी बनाम उत्तर प्रदेश में उच्चतम न्यायालय ने एफ़ आई आर अनिवार्य बताई। नालसा एक पाँच एक शून्य शून्य पर फ़ोन करें, मुफ़्त वकील मिलेगा। क्या आपकी एफ़ आई आर मना की गई है?" },
        { patterns: ["cheque", "bounce", "check", "dishonour"], answer: "चेक बाउंस पर परक्राम्य लिखत अधिनियम की धारा एक सौ अड़तीस लागू होती है। बाउंस के बाद तीस दिन के अंदर कानूनी नोटिस भेजिए। पंद्रह दिन में पैसे न आएं तो मजिस्ट्रेट अदालत में शिकायत दर्ज करें। दो साल तक की सज़ा और चेक की रकम से दोगुना जुर्माना हो सकता है। शिकायत बाउंस के तीस दिन बाद और एक महीने के अंदर करनी होती है। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील मिलेगा। चेक कितने का है?" },
        { patterns: ["online", "fraud", "scam", "upi", "cyber", "phishing", "otp"], answer: "ऑनलाइन धोखाधड़ी में तुरंत एक नौ तीन शून्य पर फ़ोन करें, यह राष्ट्रीय साइबर अपराध हेल्पलाइन है, पैसे फ्रीज़ हो सकते हैं। साइबरक्राइम डॉट जी ओ वी डॉट इन पर शिकायत दर्ज करें। आई टी अधिनियम धारा छिहत्तर डी और बी एन एस धारा तीन सौ अठारह के तहत एफ़ आई आर दर्ज करें। बैंक को तुरंत सूचित करें। कितने पैसे गए हैं?" },
        { patterns: ["domestic", "violence", "marpit", "pati", "sasural"], answer: "घरेलू हिंसा में सबसे पहले महिला हेल्पलाइन एक आठ एक पर फ़ोन करें। घरेलू हिंसा अधिनियम दो हज़ार पाँच और बी एन एस धारा पचासी के तहत आपको सुरक्षा मिलेगी। नज़दीकी संरक्षण अधिकारी से मिलें। मजिस्ट्रेट अदालत में सुरक्षा आदेश और भरण-पोषण की अर्ज़ी दें। एफ़ आई आर भी दर्ज करवाएं। आपको शेल्टर होम में रहने का अधिकार है। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील मिलेगा। क्या आप सुरक्षित जगह पर हैं?" },
        { patterns: ["salary", "vetan", "naukri", "job", "termination", "fired"], answer: "वेतन या नौकरी की समस्या के लिए श्रम आयुक्त को शिकायत दें, हेल्पलाइन एक चार चार तीन चार पर फ़ोन करें। वेतन न मिले तो श्रम अदालत में केस करें। गलत तरीके से निकाला गया है तो औद्योगिक विवाद अधिनियम धारा पच्चीस एफ़ के तहत हर साल के लिए पंद्रह दिन का वेतन मुआवज़ा मिलेगा। पाँच साल की नौकरी के बाद ग्रेच्युटी का अधिकार है। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील। कितने दिन से वेतन नहीं मिला?" },
        { patterns: ["consumer", "complaint", "product", "service", "refund"], answer: "उपभोक्ता शिकायत के लिए उपभोक्ता संरक्षण अधिनियम दो हज़ार उन्नीस लागू होता है। एक आठ शून्य शून्य एक एक चार शून्य शून्य शून्य पर फ़ोन करें, यह मुफ़्त हेल्पलाइन है। ज़िला उपभोक्ता आयोग में एक करोड़ तक की शिकायत दर्ज हो सकती है। ई-दाखिल डॉट एन आई सी डॉट इन पर ऑनलाइन शिकायत करें। दो साल के अंदर शिकायत करनी होती है। शिकायत किस बारे में है?" },
        { patterns: ["bail", "giraftari", "arrest", "jail"], answer: "ज़मानत और गिरफ्तारी के बारे में बताता हूँ। ज़मानती अपराध में थाने पर ही ज़मानत का अधिकार है। ग़ैर-ज़मानती अपराध में सत्र अदालत या उच्च न्यायालय में अर्ज़ी दें। अग्रिम ज़मानत बी एन एस एस धारा चार सौ बयासी के तहत मिलती है। अगर साठ या नब्बे दिन में आरोप पत्र न दाखिल हो तो ज़मानत का अधिकार है। गिरफ्तारी में परिवार को सूचित करना पुलिस की ज़िम्मेदारी है। नालसा एक पाँच एक शून्य शून्य पर फ़ोन करें। किस मामले में गिरफ्तारी हुई?" },
        { patterns: ["rti", "information", "suchna"], answer: "सूचना का अधिकार अधिनियम दो हज़ार पाँच के तहत आप किसी भी सरकारी कार्यालय से जानकारी माँग सकते हैं। दस रुपये की फीस के साथ आवेदन दें। तीस दिन में जवाब अनिवार्य है। जवाब न मिले तो पहली अपील तीस दिन में और सूचना आयोग में नब्बे दिन में करें। ऑनलाइन आर टी आई डॉट जी ओ वी डॉट इन पर लगा सकते हैं। गरीबी रेखा से नीचे वालों को फीस माफ़ है। किस विभाग से जानकारी चाहिए?" },
        { patterns: ["property", "zameen", "registry", "makaan", "flat", "builder"], answer: "संपत्ति विवाद में बिल्डर ने देरी की है तो रेरा प्राधिकरण में शिकायत करें। ज़मीन का विवाद है तो राजस्व अदालत या दीवानी अदालत में केस करें। रजिस्ट्री के लिए उप-पंजीयक कार्यालय जाएं। नामांतरण के लिए तहसील कार्यालय में आवेदन दें। अतिक्रमण है तो एस डी एम या ज़िला कलेक्टर को शिकायत करें। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील मिलेगा। किस तरह का संपत्ति विवाद है?" },
        { patterns: ["divorce", "talaq", "shaadi"], answer: "तलाक के लिए हिंदू विवाह अधिनियम धारा तेरह लागू होती है। आपसी सहमति से तलाक धारा तेरह बी के तहत होता है, छह महीने का इंतज़ार करना पड़ता है। एकतरफा तलाक क्रूरता, परित्याग या सात साल से लापता होने पर मिलता है। तीन तलाक़ अब अपराध है, तीन साल की सज़ा है। भरण-पोषण का अधिकार धारा एक सौ पच्चीस के तहत है। परिवार अदालत में याचिका दाखिल करें। नालसा एक पाँच एक शून्य शून्य पर मुफ़्त वकील। क्या दोनों पक्ष सहमत हैं?" },
    ],
    "en-IN": [
        { patterns: ["fir", "police", "register", "lodge"], answer: "FIR Filing: Under BNSS Section 173, police MUST register FIR for cognizable offences — they cannot refuse. Steps: (1) Go to nearest police station. (2) If refused, send written complaint to SP by registered post. (3) Zero FIR can be filed at ANY police station. Supreme Court in Lalita Kumari v. UP (2013) made FIR registration mandatory. Call NALSA 15100 for free lawyer. Was your FIR refused?" },
        { patterns: ["consumer", "complaint", "defective", "product", "refund", "service"], answer: "Consumer Complaint — Consumer Protection Act 2019: (1) File online at consumerhelpline.gov.in or call 1800-11-4000 (toll-free). (2) File at District Consumer Disputes Redressal Commission for claims up to 1 crore. (3) E-filing at edaakhil.nic.in. (4) Limitation period: 2 years from cause of action. (5) Lucknow Development Authority v. MK Gupta — delay/deficiency in service is compensable. NALSA 15100 for free lawyer. What product or service is the complaint about?" },
        { patterns: ["cheque", "bounce", "dishonour"], answer: "Cheque Bounce — NI Act Section 138: (1) Send legal notice within 30 days of bounce memo. (2) If not paid within 15 days of notice, file complaint in Magistrate Court. (3) Punishment: up to 2 years jail + twice the cheque amount. (4) File within 1 month after 15-day notice period expires. NALSA 15100 for free legal aid. What is the cheque amount?" },
        { patterns: ["online", "fraud", "cyber", "scam", "upi"], answer: "Online Fraud: Act immediately: (1) Call 1930 — National Cyber Crime Helpline — money can be frozen. (2) Report at cybercrime.gov.in. (3) FIR under IT Act Section 66D + BNS Section 318. (4) Notify your bank immediately. DK Basu guidelines protect against wrongful arrest. How much money was lost?" },
        { patterns: ["domestic", "violence", "husband", "abuse"], answer: "Domestic Violence — DV Act 2005 + BNS Section 85: (1) Call Women Helpline 181. (2) Meet nearest Protection Officer. (3) File for Protection Order + maintenance in Magistrate Court. (4) FIR under BNS 85 (cruelty). (5) Right to shelter home. NALSA 15100 for free lawyer. Are you in a safe place?" },
        { patterns: ["salary", "job", "fired", "termination"], answer: "Salary/Job issues — Code on Wages 2019: (1) Complain to Labour Commissioner — helpline 14434. (2) Non-payment — file case in Labour Court. (3) Wrongful termination — retrenchment compensation under ID Act Section 25F (15 days salary per year). (4) Gratuity after 5 years under Payment of Gratuity Act. NALSA 15100. How long has salary been pending?" },
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
        for (const p of faq.patterns) {
            if (words.includes(p) || msgLower.includes(p)) score++;
        }
        // Require at least 2 pattern matches for FAQ hit
        if (score >= 2 && score > bestScore) {
            bestScore = score;
            bestMatch = faq;
        }
    }
    return bestMatch;
}

// ═══════════════════════════════════════════════════════════
//  SESSION MEMORY — Server-side conversation tracking
// ═══════════════════════════════════════════════════════════
const webSessions = new Map(); // sessionId → { history: [], lang, lastActive }
const SESSION_TTL = 30 * 60 * 1000; // 30 min
const SESSION_MAX_TURNS = 6; // 3 user + 3 assistant

function getOrCreateSession(sessionId) {
    if (sessionId && webSessions.has(sessionId)) {
        const s = webSessions.get(sessionId);
        s.lastActive = Date.now();
        return { session: s, sessionId };
    }
    const newId = `ws_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    const s = { history: [], lang: "hi-IN", lastActive: Date.now() };
    webSessions.set(newId, s);
    return { session: s, sessionId: newId };
}

function appendToSession(session, userMsg, assistantReply) {
    session.history.push({ u: 1, t: userMsg });
    session.history.push({ u: 0, t: assistantReply });
    if (session.history.length > SESSION_MAX_TURNS * 2) {
        session.history = session.history.slice(-SESSION_MAX_TURNS * 2);
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
//  TTS HELPER — with voice engine normalization
// ═══════════════════════════════════════════════════════════
async function generateTTS(text, langCode, customSpeaker) {
    const vp = getVoiceParams(langCode);
    const speaker = customSpeaker || vp.speaker;
    const normalized = normalizeTTS(text, langCode);
    const ck = ttsCacheKey(normalized, langCode);
    if (ttsCache.has(ck)) return { audio: ttsCache.get(ck).toString("base64"), text };

    // Transliterate Roman Hinglish → Devanagari for better Hindi TTS pronunciation
    const ttsText = await transliterateForTTS(normalized, langCode);

    const r = await apiFetch("https://api.sarvam.ai/text-to-speech", {
        method: "POST", headers: HEADERS,
        body: JSON.stringify({
            text: ttsText,
            target_language_code: langCode,
            speaker,
            model: vp.model || "bulbul:v3",
            pace: vp.pace,
            speech_sample_rate: 24000,
            temperature: vp.temperature || 0.65,
        }),
    }, 6000);
    if (r.ok) {
        const d = await r.json();
        if (d.audios?.[0]) {
            const buf = Buffer.from(d.audios[0], "base64");
            ttsCacheSet(ck, buf);
            return { audio: d.audios[0], text };
        }
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
        }, 6000);
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
    const { history = [], message, lang = "hi-IN", speaker, sessionId: reqSessionId } = req.body;
    const { session, sessionId } = getOrCreateSession(reqSessionId);
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

    // FAQ CHECK — instant pre-built answer, skip LLM entirely
    const faqMatch = matchFAQ(msg, lang);
    if (faqMatch) {
        console.log(`[ASK] FAQ HIT: "${faqMatch.answer.slice(0, 60)}"`);
        const ttsT0 = Date.now();
        const segments = segmentForTTS(faqMatch.answer);
        const ttsResults = await Promise.allSettled(segments.map(chunk => generateTTS(chunk, lang, speaker)));
        const audioChunks = ttsResults.filter(r => r.status === "fulfilled" && r.value).map(r => r.value);
        appendToSession(session, msg, faqMatch.answer);
        return res.json({
            reply: faqMatch.answer, model: "faq-template", sessionId, audioChunks,
            ms: Date.now() - t0, aiMs: 0, ttsMs: Date.now() - ttsT0,
            ragContext: true, segments: segments.length, faq: true,
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
        appendToSession(session, msg, cached.reply);
        return res.json({
            reply: cached.reply, model: cached.model + " (cached)", sessionId, audioChunks,
            ms: Date.now() - t0, aiMs: 0, ttsMs: Date.now() - ttsT0,
            ragContext: cached.ragContext, segments: segments.length, cached: true,
        });
    }

    // RAG RETRIEVAL
    const { contextString: ragContext, chunks: ragChunks } = buildContext(msg);
    const ragSnippet = ragContext || "";

    // Build conversation history — merge client + server-side session
    const messages = [{ role: "system", content: getLangPrompt(lang, ragSnippet) }];
    const combinedHistory = [...session.history, ...history].slice(-8);
    let lastRole = "system";
    for (const m of combinedHistory) {
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
    const systemPrompt = getLangPrompt(lang, ragSnippet);

    // SPEED: Race primary model against Sarvam fallback concurrently
    // Whichever returns first with a valid response wins
    try {
        const primaryPromise = callGemini(systemPrompt, messages, 1024, model)
            .then(r => r ? { reply: cleanLLMResponse(r, lang), model } : null)
            .catch(() => null);
        const sarvamPromise = new Promise(resolve =>
            setTimeout(() => {
                // Only start Sarvam after 2s if primary is slow
                callSarvam(messages, 300).then(r => r ? resolve({ reply: cleanLLMResponse(r, lang), model: "sarvam-m" }) : resolve(null)).catch(() => resolve(null));
            }, 2000)
        );
        const result = await Promise.race([
            primaryPromise.then(r => r || new Promise(resolve => setTimeout(() => resolve(null), 6000))),
            sarvamPromise,
        ]);
        if (result) {
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
                const retryReply = await callGemini(systemPrompt, messages, 1024, fallbackModel);
                if (retryReply) {
                    reply = cleanLLMResponse(retryReply, lang);
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
                reply = cleanLLMResponse(sarvamReply, lang);
                model = "sarvam-m";
                console.log(`[ASK] Sarvam direct: "${reply.slice(0, 60)}"`);
            }
        } catch (e) { console.log("[ASK-SARVAM] FAIL:", e.message); }
    }

    const aiMs = Date.now() - aiT0;

    // RESPONSE SELF-CORRECTION
    if (reply) {
        const { valid, hasLegal } = validateResponse(reply, lang);
        if (!valid) {
            console.log(`[ASK] RESPONSE_INVALID: "${reply.slice(0, 50)}"`);
            reply = refusal;
        }
    }

    // LAYER 3 POST-AI GUARDRAIL
    if (reply && !isLegalResponse(reply)) {
        console.log(`[ASK] L3_BLOCK: "${reply.slice(0, 50)}"`);
        reply = refusal;
    }

    // LAYER 3b — Citation grounding check
    if (reply && reply !== refusal && ragChunks?.length > 0) {
        const { score } = computeGroundingScore(reply, ragChunks);
        console.log(`[GROUNDING] score:${score.toFixed(2)} "${reply.slice(0, 40)}"`);
        if (score < 0.6) {
            reply = sanitizeResponse(reply, ragChunks, lang, refusal);
            console.log(`[GROUNDING] Sanitized: "${reply.slice(0, 40)}"`);
        }
    }

    if (!reply) {
        reply = lang === "en-IN"
            ? "Sorry, a technical issue occurred. Call NALSA on 15100 for a free lawyer."
            : "थोड़ी तकनीकी समस्या हुई। NALSA 15100 पर कॉल करें — निःशुल्क वकील मिलेगा।";
    }

    // Cache successful non-refusal responses
    if (reply && reply !== refusal) {
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

    // Save to session memory
    appendToSession(session, msg, reply);

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
    } catch { }
    res.status(500).json({ error: "TTS failed" });
});

// ═══════════════════════════════════════════════════════════
//  4. FILLER AUDIO
// ═══════════════════════════════════════════════════════════
const FILLERS = {
    "hi-IN": ["अच्छा, देखते हैं...", "समझ रहा हूँ, रुकिए एक पल...", "जी हाँ, सोच रहा हूँ...", "ठीक है, बिल्कुल..."],
    "en-IN": ["Let me think about this for a moment...", "One moment please...", "Sure, looking into your situation...", "Let me check that for you..."],
    "bn-IN": ["Bujhte pārchi, ektu opekkha korun...", "Dekhchi...", "Haan, bujhchi..."],
    "te-IN": ["Arthamavutundi, okka nimesha...", "Chusthanu...", "Sare, alochisthunna..."],
    "ta-IN": ["Purikiṟēn, oru khaṇam...", "Pārkkiṟēn...", "Sari, yosithu pārkkiṟēn..."],
    "mr-IN": ["Samajh gelo, ek pal...", "Pāhto...", "Thik aahe, vichar karto..."],
    "gu-IN": ["Samaju chhu, ek pal...", "Jou chhu...", "Theek chhe, vichar karu chhu..."],
    "kn-IN": ["Arthamāguttide, oru nimesha...", "Nōḍuttēne...", "Sari, alocisuttēne..."],
    "ml-IN": ["Manasiḷakkunnuṇṭu, oru khaṇam...", "Nōkkuuṇṭu...", "Sar, alocikkunnu..."],
    "pa-IN": ["Samajh gaya, ek pal...", "Dekh reha haan...", "Theek haan, soch da haan..."],
    "od-IN": ["Bujhilani, gote khana...", "Dekhuachhi...", "Theek, bhābuchhi..."],
};
const fillerCache = new Map();

async function warmFillers() {
    if (!SK) return;
    console.log("  [Filler] Warming all languages...");
    const langs = Object.keys(FILLERS);
    await Promise.allSettled(langs.map(async lc => {
        const phrases = FILLERS[lc] || FILLERS["hi-IN"];
        const bufs = [];
        await Promise.allSettled(phrases.map(async (ph) => {
            try {
                const result = await generateTTS(ph, lc);
                if (result) bufs.push(Buffer.from(result.audio, "base64"));
            } catch { }
        }));
        if (bufs.length) fillerCache.set(lc, bufs);
    }));
    const warmed = [...fillerCache.entries()].map(([k, v]) => `${k}:${v.length}`).join(", ");
    console.log(`  [Filler] OK — ${warmed}`);
}

app.get("/api/filler", async (req, res) => {
    const lc = req.query.lang || "hi-IN";
    const cached = fillerCache.get(lc);
    if (cached?.length) return res.set("Content-Type", "audio/wav").send(cached[Math.floor(Math.random() * cached.length)]);
    const ph = (FILLERS[lc] || FILLERS["hi-IN"])[Math.floor(Math.random() * 4)];
    try {
        const result = await generateTTS(ph, lc);
        if (result) {
            const buf = Buffer.from(result.audio, "base64");
            if (!fillerCache.has(lc)) fillerCache.set(lc, []);
            fillerCache.get(lc).push(buf);
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
    for (const seg of IVR_SEGMENTS) {
        try {
            const result = await generateTTS(seg.text, seg.lang);
            if (result?.audio) {
                const wav = Buffer.from(result.audio, "base64");
                // Strip 44-byte WAV header → raw PCM
                const pcm = wav.slice(44);
                pcmChunks.push(pcm);
                // Add 400ms silence between segments for clear separation
                const silenceBytes = Math.round(24000 * 0.4) * 2; // 400ms at 24kHz, 16-bit
                pcmChunks.push(Buffer.alloc(silenceBytes));
            }
        } catch (e) {
            console.log(`[IVR] Failed segment ${seg.lang}:`, e.message);
        }
    }
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

app.post("/api/phone/call-start", async (req, res) => {
    // Sarvam telephony calls this when a call begins
    const sessionId = req.body?.session_id || `ph_${Date.now()}`;
    phoneSessions.set(sessionId, { lang: "hi-IN", history: [], step: "ivr" });
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
        }, 10000);
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
            reply = await callGemini(systemPrompt, phoneMessages, 1024, phoneModel);
            if (reply) reply = cleanLLMResponse(reply, lang);
        } catch (e) { console.log(`[PHONE-${phoneModel}]`, e.message); }
        if (!reply) {
            const fb = getAvailableModel();
            if (fb !== phoneModel) {
                try { reply = await callGemini(systemPrompt, phoneMessages, 1024, fb); if (reply) reply = cleanLLMResponse(reply, lang); }
                catch (e) { console.log(`[PHONE-${fb}]`, e.message); }
            }
        }
        if (!reply) {
            try {
                reply = await callSarvam(phoneMessages, 300);
                if (reply) reply = cleanLLMResponse(reply, lang);
            } catch (e) { console.log("[PHONE-SARVAM]", e.message); }
        }
        if (!reply) reply = getRefusal(lang);
        // Post-AI guard
        if (!isLegalResponse(reply)) reply = getRefusal(lang);
        // Citation grounding check
        if (reply && reply !== getRefusal(lang) && ragChunks?.length > 0) {
            const { score } = computeGroundingScore(reply, ragChunks);
            if (score < 0.6) {
                reply = sanitizeResponse(reply, ragChunks, lang, getRefusal(lang));
            }
        }
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
app.post("/api/phone/exotel/call-start", async (req, res) => {
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
        speakerGender: null,    // detected from first voice recording
        customSpeaker: null,    // female speaker override if female detected
    });

    // Ensure IVR audio is ready
    if (!ivrAudioUrl) await warmExotelIVR();

    const gatherUrl = `${getPublicUrl()}/api/phone/exotel/gather?CallSid=${callSid}`;

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
app.post("/api/phone/exotel/gather", async (req, res) => {
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

    const recordUrl = `${getPublicUrl()}/api/phone/exotel/audio?CallSid=${callSid}`;

    res.set("Content-Type", "application/xml");
    if (greetingUrl) {
        res.send(exoml(`  <Play>${greetingUrl}</Play>\n  <Record action="${recordUrl}" method="POST" maxLength="15" finishOnKey="#" timeout="3" />`));
    } else {
        res.send(exoml(`  <Say>${greeting}</Say>\n  <Record action="${recordUrl}" method="POST" maxLength="15" finishOnKey="#" timeout="3" />`));
    }
});

// 3. Audio recording received — STT → AI → TTS → play response
app.post("/api/phone/exotel/audio", async (req, res) => {
    const callSid = req.body?.CallSid || req.query?.CallSid;
    const recordingUrl = req.body?.RecordingUrl || req.body?.recording_url;
    const session = phoneSessions.get(callSid);

    if (!session) {
        res.set("Content-Type", "application/xml");
        return res.send(exoml(`  <Say>Session expired. Please call again.</Say>\n  <Hangup />`));
    }

    const lang = session.lang || "hi-IN";
    let reply = getRefusal(lang);
    const recordUrl = `${getPublicUrl()}/api/phone/exotel/audio?CallSid=${callSid}`;

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
        // Download recording from Exotel
        let audioBuffer;
        if (recordingUrl) {
            const authHeader = process.env.EXOTEL_API_KEY && process.env.EXOTEL_API_TOKEN
                ? "Basic " + Buffer.from(`${process.env.EXOTEL_API_KEY}:${process.env.EXOTEL_API_TOKEN}`).toString("base64")
                : null;
            const headers = authHeader ? { Authorization: authHeader } : {};
            const audioRes = await apiFetch(recordingUrl, { headers }, 10000);
            if (audioRes.ok) {
                audioBuffer = Buffer.from(await audioRes.arrayBuffer());
            }
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
        let transcript = "";
        const fd = new FormData();
        fd.append("file", new File([audioBuffer], "audio.wav", { type: "application/octet-stream" }));
        fd.append("model", "saarika:v2.5");
        fd.append("language_code", lang);
        fd.append("mode", "transcribe");
        const sttRes = await apiFetch("https://api.sarvam.ai/speech-to-text", {
            method: "POST", headers: { "api-subscription-key": SK }, body: fd,
        }, 10000);
        if (sttRes.ok) {
            const d = await sttRes.json();
            transcript = sanitize(d.transcript || "", 500);
        }

        // Confidence check
        const { confident, reason } = scoreTranscript(transcript, lang);
        if (!confident) {
            const clarification = getClarificationPrompt(lang, reason);
            const clarUrl = await ttsToUrl(clarification, lang);
            const recordUrl = `${getPublicUrl()}/api/phone/exotel/audio?CallSid=${callSid}`;
            res.set("Content-Type", "application/xml");
            return res.send(exoml(
                (clarUrl ? `  <Play>${clarUrl}</Play>` : `  <Say>${clarification}</Say>`) +
                `\n  <Record action="${recordUrl}" method="POST" maxLength="15" finishOnKey="#" timeout="3" />`
            ));
        }

        // Guardrail + RAG + AI (same as Sarvam phone endpoint)
        const { allow } = isLegalQuery(transcript);
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
            let phoneModel = getAvailableModel();
            try {
                reply = await callGemini(systemPrompt, phoneMessages, 1024, phoneModel);
                if (reply) reply = cleanLLMResponse(reply, lang);
            } catch (e) { console.log(`[EXOTEL-${phoneModel}]`, e.message); }
            if (!reply) {
                const fb = getAvailableModel();
                if (fb !== phoneModel) {
                    try { reply = await callGemini(systemPrompt, phoneMessages, 1024, fb); if (reply) reply = cleanLLMResponse(reply, lang); }
                    catch (e) { console.log(`[EXOTEL-${fb}]`, e.message); }
                }
            }
            if (!reply) {
                try { reply = await callSarvam(phoneMessages, 300); if (reply) reply = cleanLLMResponse(reply, lang); }
                catch (e) { console.log("[EXOTEL-SARVAM]", e.message); }
            }
            if (!reply) reply = getRefusal(lang);
            if (!isLegalResponse(reply)) reply = getRefusal(lang);
            if (reply && reply !== getRefusal(lang) && ragChunks?.length > 0) {
                const { score } = computeGroundingScore(reply, ragChunks);
                if (score < 0.6) reply = sanitizeResponse(reply, ragChunks, lang, getRefusal(lang));
            }
        }

        // Update history
        session.history.push({ u: 1, t: transcript });
        session.history.push({ u: 0, t: reply });
        if (session.history.length > 20) session.history = session.history.slice(-20);

        console.log(`[EXOTEL] ${callSid} "${transcript.slice(0, 40)}" → "${reply.slice(0, 40)}"`);

    } catch (e) {
        console.log("[EXOTEL-ERROR]", e.message);
    }

    // Generate response audio — use gender-appropriate speaker if detected
    const replyUrl = await combinedTtsToUrl(reply, lang, session.customSpeaker || undefined);

    res.set("Content-Type", "application/xml");
    res.send(exoml(
        (replyUrl ? `  <Play>${replyUrl}</Play>` : `  <Say>${reply}</Say>`) +
        `\n  <Record action="${recordUrl}" method="POST" maxLength="15" finishOnKey="#" timeout="3" />`
    ));
});

// 4. Call status update / call end
app.post("/api/phone/exotel/status", (req, res) => {
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

    const callbackUrl = `${getPublicUrl()}/api/phone/exotel/call-start`;
    const statusUrl = `${getPublicUrl()}/api/phone/exotel/status`;

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
            body: JSON.stringify({ text: "test", target_language_code: "hi-IN", speaker: "karun", model: "bulbul:v2" })
        }, 8000).then(r => { h.tts = r.ok; h.stt = r.ok; }),
        // Test Sarvam LLM (always available as fallback)
        apiFetch("https://api.sarvam.ai/v1/chat/completions", {
            method: "POST", headers: HEADERS,
            body: JSON.stringify({ model: "sarvam-m", messages: [{ role: "user", content: "test" }], max_tokens: 3 })
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
    "hi-IN": "नमस्ते! मैं न्यायसाथी हूँ — भारत का मुफ़्त कानूनी सहायक। बताइए, आपकी क्या कानूनी परेशानी है? मैं पूरी मदद करूँगा।",
    "en-IN": "Hello! I'm NyayaSathi, India's free AI legal assistant. Please tell me your legal problem and I'll do my best to help you.",
    "bn-IN": "নমস্কার! আমি ন্যায়সাথী — ভারতের বিনামূল্যে আইনি সহায়ক। আপনার আইনি সমস্যা বলুন, আমি সাহায্য করব।",
    "te-IN": "నమస్కారం! నేను న్యాయసాథి — భారత్ యొక్క ఉచిత న్యాయ సహాయకుడు। మీ చట్టపరమైన సమస్య చెప్పండి, నేను సహాయం చేస్తాను।",
    "ta-IN": "வணக்கம்! நான் நியாயசாதி — இந்தியாவின் இலவச சட்ட உதவியாளர். உங்கள் சட்ட பிரச்சனை சொல்லுங்கள், நான் உதவுகிறேன்.",
    "mr-IN": "नमस्कार! मी न्यायसाथी — भारताचा मोफत कायदेशीर सहाय्यक. तुमची कायदेशीर समस्या सांगा, मी मदत करतो.",
    "gu-IN": "નમસ્તે! હું ન્યાયસાથી છું — ભારતનો મફત કાનૂની સહાયક. તમારી કાનૂની સમસ્યા કહો, હું મદદ કરીશ.",
    "kn-IN": "ನಮಸ್ಕಾರ! ನಾನು ನ್ಯಾಯಸಾಥಿ — ಭಾರತದ ಉಚಿತ ಕಾನೂನು ಸಹಾಯಕ. ನಿಮ್ಮ ಕಾನೂನು ಸಮಸ್ಯೆ ಹೇಳಿ, ನಾನು ಸಹಾಯ ಮಾಡುತ್ತೇನೆ.",
    "ml-IN": "നമസ്കാരം! ഞാൻ ന്യായസാഥി — ഭാരതത്തിന്റെ സൗജന്യ നിയമ സഹായി. നിങ്ങളുടെ നിയമ പ്രശ്നം പറയൂ, ഞാൻ സഹായിക്കാം.",
    "pa-IN": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਨਿਆਂਸਾਥੀ ਹਾਂ — ਭਾਰਤ ਦਾ ਮੁਫ਼ਤ ਕਾਨੂੰਨੀ ਸਹਾਇਕ। ਆਪਣੀ ਕਾਨੂੰਨੀ ਸਮੱਸਿਆ ਦੱਸੋ, ਮੈਂ ਮਦਦ ਕਰਾਂਗਾ।",
    "od-IN": "ନମସ୍କାର! ମୁଁ ନ୍ୟାୟସାଥୀ — ଭାରତର ମୁଫ୍ତ ଆଇନ ସହାୟକ। ଆପଣଙ୍କ ଆଇନ ସମସ୍ୟା କୁହନ୍ତୁ, ମୁଁ ସାହାଯ୍ୟ କରିବି।",
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
//  2b. STREAMING /api/ask-stream — AI + TTS chunks streamed as ready
// ═══════════════════════════════════════════════════════════
app.post("/api/ask-stream", async (req, res) => {
    const { history = [], message, lang = "hi-IN", speaker } = req.body;
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
        sendChunk({ type: "reply", reply: refusal, model: "guardrail", blocked: true });
        // Generate TTS for refusal
        try {
            const result = await generateTTS(refusal, lang, speaker);
            if (result) sendChunk({ type: "audio", audio: result.audio, index: 0 });
        } catch { }
        sendChunk({ type: "done", ms: Date.now() - t0 });
        return res.end();
    }

    // RAG RETRIEVAL
    const { contextString: ragContext, chunks: ragChunks } = buildContext(msg);

    // Build conversation
    const messages = [{ role: "system", content: getLangPrompt(lang, ragContext || "") }];
    const recent = history.slice(-8);
    let lastRole = "system";
    for (const m of recent) {
        const role = m.u ? "user" : "assistant";
        if (role === lastRole) continue;
        messages.push({ role, content: sanitize(m.t, 300) });
        lastRole = role;
    }
    if (messages[messages.length - 1]?.role === "user") {
        messages.push({ role: "assistant", content: lang === "en-IN" ? "Yes, please go ahead." : "जी, बताइए।" });
    }
    messages.push({ role: "user", content: msg });

    // AI CALL — Race primary model against Sarvam for speed
    const aiT0 = Date.now();
    let reply = "";
    let model = getAvailableModel();
    const systemPrompt = getLangPrompt(lang, ragContext || "");

    try {
        const primaryPromise = callGemini(systemPrompt, messages, 1024, model)
            .then(r => r ? { reply: cleanLLMResponse(r, lang), model } : null)
            .catch(() => null);
        const sarvamPromise = new Promise(resolve =>
            setTimeout(() => {
                callSarvam(messages, 300).then(r => r ? resolve({ reply: cleanLLMResponse(r, lang), model: "sarvam-m" }) : resolve(null)).catch(() => resolve(null));
            }, 2000)
        );
        const result = await Promise.race([
            primaryPromise.then(r => r || new Promise(resolve => setTimeout(() => resolve(null), 6000))),
            sarvamPromise,
        ]);
        if (result) { reply = result.reply; model = result.model; }
    } catch (e) { console.log(`[STREAM-RACE] FAIL:`, e.message); }

    if (!reply) {
        try {
            const sarvamReply = await callSarvam(messages, 300);
            if (sarvamReply) { reply = cleanLLMResponse(sarvamReply, lang); model = "sarvam-m"; }
        } catch (e) { console.log("[STREAM-SARVAM] FAIL:", e.message); }
    }
    const aiMs = Date.now() - aiT0;

    // Validate response
    if (reply) {
        const { valid } = validateResponse(reply, lang);
        if (!valid) reply = refusal;
    }
    if (reply && !isLegalResponse(reply)) reply = refusal;

    // LAYER 3b — Citation grounding check
    if (reply && reply !== refusal && ragChunks?.length > 0) {
        const { score } = computeGroundingScore(reply, ragChunks);
        if (score < 0.6) {
            reply = sanitizeResponse(reply, ragChunks, lang, refusal);
        }
    }

    if (!reply) {
        reply = lang === "en-IN"
            ? "Sorry, a technical issue occurred. Call NALSA on 15100 for a free lawyer."
            : "थोड़ी तकनीकी समस्या हुई। NALSA 15100 पर कॉल करें — निःशुल्क वकील मिलेगा।";
    }

    // Send reply text immediately so frontend can display it
    sendChunk({ type: "reply", reply, model, aiMs, rag: !!ragContext });

    // TTS — stream each chunk as it's ready
    const ttsT0 = Date.now();
    const segments = segmentForTTS(reply);

    // Fire all TTS requests in parallel, stream each as it resolves
    const ttsPromises = segments.map((chunk, i) =>
        generateTTS(chunk, lang, speaker).then(result => {
            if (result) sendChunk({ type: "audio", audio: result.audio, index: i, text: chunk });
        }).catch(() => { })
    );
    await Promise.allSettled(ttsPromises);
    const ttsMs = Date.now() - ttsT0;

    sendChunk({ type: "done", ms: Date.now() - t0, aiMs, ttsMs, segments: segments.length });
    console.log(`[STREAM] ${Date.now() - t0}ms (AI:${aiMs} TTS:${ttsMs}) segs:${segments.length} "${reply.slice(0, 60)}"`);
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
    try { reply = await callGemini(systemPrompt, messages, 256); if (reply) reply = cleanLLMResponse(reply, lang); } catch { }
    if (!reply) { try { reply = await callSarvam(messages, 150); if (reply) reply = cleanLLMResponse(reply, lang); } catch { } }
    if (!reply) reply = refusal;
    if (!isLegalResponse(reply)) reply = refusal;
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
    console.log(`  ║  Brain: Smart Race (Flash+Sarvam) + RAG         ║`);
    console.log(`  ║  Court: 50+ SC Judgments + BNS/IPC Mapping      ║`);
    console.log(`  ║  Voice: Sarvam Bulbul v3 TTS (11 languages)     ║`);
    console.log(`  ║  Phone: Exotel 1800 Toll-Free + Outbound API    ║`);
    console.log(`  ║  http://localhost:${PORT}                           ║`);
    console.log(`  ╚══════════════════════════════════════════════════╝\n`);
    if (!SK) { console.log("  ⚠  Set SARVAM_API_KEY in .env\n"); return; }
    if (!GK) { console.log("  ⚠  Set GEMINI_API_KEY in .env (using Sarvam LLM fallback)\n"); }

    // Auto-detect ngrok tunnel
    const ngrokUrl = await detectNgrok();

    try {
        const h = await (await fetch(`http://localhost:${PORT}/api/health`)).json();
        console.log(`  Brain: ${h.brain} ${h.gemini ? "✓" : "✗"}  STT:${h.stt ? "✓" : "✗"} TTS:${h.tts ? "✓" : "✗"} RAG:${h.rag ? "✓" : "✗"}`);
        console.log(`  Sarvam Phone:  GET /api/phone/info`);
        console.log(`  Exotel Phone:  GET /api/phone/exotel/info`);
        if (ngrokUrl) {
            console.log(`  Exotel Webhook: ${ngrokUrl}/api/phone/exotel/call-start`);
            console.log(`  Test Call:      POST /api/phone/exotel/call {"to":"+919XXXXXXXXX"}`);
        } else {
            console.log(`  ⚠  No PUBLIC_URL — run "ngrok http ${PORT}" for Exotel webhooks`);
        }
        console.log(`  Eval:          node eval-engine.js\n`);
    } catch { }
    warmFillers().catch(() => { });
    warmExotelIVR().catch(() => { });
});

