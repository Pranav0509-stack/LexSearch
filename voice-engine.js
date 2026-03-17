/**
 * NyayaSathi Voice Engine
 * - Number/section expansion for natural TTS speech
 * - Self-correction pipeline: transcription confidence + retry
 * - Speaking rate optimizer per language
 * - Sentence segmentation tuned for Indian legal speech
 */

// ═══════════════════════════════════════════════════════════════════════
//  NUMBER → WORDS  (so TTS reads "1 lakh 50 thousand" not "150000")
// ═══════════════════════════════════════════════════════════════════════
const ONES_HI = ["", "ek", "do", "teen", "chaar", "paanch", "chhah", "saat", "aath", "nau", "das",
    "gyarah", "barah", "terah", "chaudah", "pandrah", "solah", "satrah", "atharah", "unnis", "bees",
    "ikees", "baais", "teis", "chaubis", "pachchees", "chhabbees", "sattaais", "atthaais", "untees", "tees",
    "ikattees", "battees", "taintees", "chautees", "paintees", "chhattees", "saintees", "artees", "untalis", "chalis",
    "ikattalis", "bayalis", "taintalis", "chavvalis", "paintalis", "chhiyalis", "saintalis", "artalis", "unchaas", "pachaas",
    "ikyaavan", "baavan", "tirpan", "chauvan", "pachpan", "chhappan", "sattaavan", "athhavan", "unsath", "saath",
    "iksath", "baasath", "tirsath", "chausath", "painsath", "chhiyasath", "sarsath", "arsath", "unhattar", "sattar",
    "ikyattar", "bahattar", "tihattar", "chauhattar", "pachhattar", "chhiyattar", "sathattar", "aththattar", "unnyaasi", "assi",
    "ikyaasi", "baasi", "tiraasi", "chaurasi", "pachaasi", "chhiyaasi", "sataasi", "athhasi", "navaasi", "navve",
    "ikyaanave", "baanave", "tiranave", "chauranave", "pachaanave", "chhiyanave", "sataanave", "athaanave", "ninyaanave"];
const ONES_EN = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS_EN = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

function numToHindi(n) {
    if (n < 0) return "minus " + numToHindi(-n);
    if (n === 0) return "shoonya";
    if (n <= 99) return ONES_HI[n] || (ONES_HI[Math.floor(n / 10) * 10] || (TENS_EN[Math.floor(n / 10)] + " ")) + ONES_HI[n % 10];
    if (n < 1000) return ONES_HI[Math.floor(n / 100)] + " sau " + (n % 100 ? " " + numToHindi(n % 100) : "");
    if (n < 100000) return numToHindi(Math.floor(n / 1000)) + " hazaar" + (n % 1000 ? " " + numToHindi(n % 1000) : "");
    if (n < 10000000) return numToHindi(Math.floor(n / 100000)) + " lakh" + (n % 100000 ? " " + numToHindi(n % 100000) : "");
    return numToHindi(Math.floor(n / 10000000)) + " crore" + (n % 10000000 ? " " + numToHindi(n % 10000000) : "");
}

function numToEnglish(n) {
    if (n === 0) return "zero";
    if (n < 0) return "minus " + numToEnglish(-n);
    if (n < 20) return ONES_EN[n];
    if (n < 100) return TENS_EN[Math.floor(n / 10)] + (n % 10 ? "-" + ONES_EN[n % 10] : "");
    if (n < 1000) return ONES_EN[Math.floor(n / 100)] + " hundred" + (n % 100 ? " and " + numToEnglish(n % 100) : "");
    if (n < 100000) return numToEnglish(Math.floor(n / 1000)) + " thousand" + (n % 1000 ? " " + numToEnglish(n % 1000) : "");
    if (n < 10000000) return numToEnglish(Math.floor(n / 100000)) + " lakh" + (n % 100000 ? " " + numToEnglish(n % 100000) : "");
    return numToEnglish(Math.floor(n / 10000000)) + " crore" + (n % 10000000 ? " " + numToEnglish(n % 10000000) : "");
}

// ═══════════════════════════════════════════════════════════════════════
//  TTS TEXT NORMALIZER — make text speak naturally
// ═══════════════════════════════════════════════════════════════════════

// Legal section references → spoken form (full number words, not digit-by-digit)
const SECTION_PATTERNS = [
    // "Section 138" → "Section one hundred thirty-eight"
    { rx: /[Ss]ection\s+(\d{1,4})([A-Z]?)/g, fn: (_, n, suffix) => {
        return "Section " + numToEnglish(parseInt(n)) + (suffix ? " " + suffix : "");
    }},
    // "Sec. 138" or "Sec 138"
    { rx: /Sec\.?\s+(\d{1,4})([A-Z]?)/g, fn: (_, n, suffix) => {
        return "Section " + numToEnglish(parseInt(n)) + (suffix ? " " + suffix : "");
    }},
    // "S. 138"
    { rx: /S\.\s*(\d{1,4})/g, fn: (_, n) => "Section " + numToEnglish(parseInt(n)) },
    // "Article 21" → "Article twenty-one"
    { rx: /[Aa]rticle\s+(\d{1,3})([A-Z]?)/g, fn: (_, n, suffix) => {
        return "Article " + numToEnglish(parseInt(n)) + (suffix ? " " + suffix : "");
    }},
    // "IPC 420" → "IPC four hundred twenty"
    { rx: /\b(IPC|BNS|BNSS|CrPC|IEA|BSA)\s+(\d+)/g, fn: (_, act, n) => act + " " + numToEnglish(parseInt(n)) },
];

// Phone numbers → digit by digit
const PHONE_RX = /\b(15100|1516|1915|181|112|1930|1098|1800[\-\s]?\d{3}[\-\s]?\d{4}|\d{10})\b/g;

// Currency → spoken
function expandCurrency(text, isHindi) {
    return text.replace(/₹\s*(\d[\d,]*(?:\.\d+)?)/g, (_, amt) => {
        const n = parseFloat(amt.replace(/,/g, ""));
        const words = isHindi ? numToHindi(Math.round(n)) : numToEnglish(Math.round(n));
        return (isHindi ? "rupay " : "rupees ") + words;
    });
}

// Percentages
function expandPercent(text) {
    return text.replace(/(\d+(?:\.\d+)?)\s*%/g, (_, n) => n + " percent");
}

// Years → spoken naturally ("2023" → "do hazaar teis" in Hindi, "two thousand twenty-three" in EN)
function expandYear(text, isHindi) {
    return text.replace(/\b(19\d\d|20\d\d)\b/g, (_, yr) => {
        const n = parseInt(yr);
        return isHindi ? numToHindi(n) : numToEnglish(n);
    });
}

// Phone number expansion — digit by digit with clear spacing
function expandPhone(text, isHindi) {
    return text.replace(PHONE_RX, (match) => {
        const digits = match.replace(/[\-\s]/g, "");
        if (isHindi) {
            const HINDI_DIGITS = ["shoonya", "ek", "do", "teen", "chaar", "paanch", "chheh", "saat", "aath", "nau"];
            // Group in pairs/triples for natural reading: "15100" → "ek paanch, ek shoonya shoonya"
            const parts = [];
            for (let i = 0; i < digits.length; i++) {
                parts.push(HINDI_DIGITS[parseInt(digits[i])]);
                // Add comma pause after every 2-3 digits for clarity
                if ((i === 1 || i === 3) && i < digits.length - 1) parts.push(",");
            }
            return parts.join(" ") + ".";
        }
        // English: group with commas for clarity
        const parts = [];
        for (let i = 0; i < digits.length; i++) {
            parts.push(digits[i]);
            if ((i === 1 || i === 3) && i < digits.length - 1) parts.push(",");
        }
        return parts.join(" ") + ".";
    });
}

// Abbreviations
const ABBR_EXPAND = {
    "FIR": "F I R", "RTI": "R T I", "PF": "P F", "ESI": "E S I",
    "MACT": "M A C T", "RERA": "R E R A", "DV": "D V",
    "PIL": "P I L", "CPC": "C P C", "IPC": "I P C",
    "BNS": "B N S", "BNSS": "B N S S", "BSA": "B S A",
    "POCSO": "P O C S O", "POSH": "P O S H",
    "NALSA": "N A L S A", "DLSA": "D L S A",
    "NI": "N I", "SC": "S C", "ST": "S T", "OBC": "O B C",
    "EWS": "E W S", "NGT": "N G T", "DRT": "D R T",
};

function expandAbbreviations(text) {
    return text.replace(/\b([A-Z]{2,8})\b/g, (match) => ABBR_EXPAND[match] || match);
}

/**
 * Normalize text for TTS — makes numbers, sections, phones, currencies
 * all speak correctly in the target language
 */
function normalizeTTS(text, langCode) {
    const isHindi = !["en-IN"].includes(langCode);
    let t = text;

    // Section patterns first (before generic number expansion)
    for (const { rx, fn } of SECTION_PATTERNS) {
        t = t.replace(rx, fn);
    }

    // Phone numbers
    t = expandPhone(t, isHindi);

    // Currency
    t = expandCurrency(t, isHindi);

    // Percentage
    t = expandPercent(t);

    // Years (before generic numbers so "2023" doesn't become "two thousand twenty three" unintentionally)
    // Only expand years in Hindi for more natural speech; English TTS handles years well
    if (isHindi) t = expandYear(t, isHindi);

    // Abbreviations — expand only in English context
    if (!isHindi) t = expandAbbreviations(t);

    // Expand remaining standalone numbers (e.g., "30 days", "2 saal", "15 din")
    // Skip if already expanded (years, phones, sections handled above)
    t = t.replace(/\b(\d{1,6})\b/g, (match) => {
        const num = parseInt(match);
        if (num > 0 && num <= 999999) {
            return isHindi ? numToHindi(num) : numToEnglish(num);
        }
        return match;
    });

    // Remove markdown artifacts
    t = t.replace(/\*{1,3}/g, "").replace(/#{1,6}\s*/g, "")
        .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
        .replace(/[-•]\s+/g, ". ")
        .replace(/\(\d+\)\s*/g, ". ")  // "(1) do this" → ". do this"
        .replace(/\n+/g, " ").replace(/\s{2,}/g, " ").trim();

    return t;
}

// ═══════════════════════════════════════════════════════════════════════
//  SENTENCE SEGMENTATION — optimal chunk sizes for TTS
// ═══════════════════════════════════════════════════════════════════════

/**
 * Split text into TTS-optimal chunks:
 * - Natural sentence breaks (., !, ?, ।)
 * - Clause breaks (,) only if sentence is long
 * - Max ~120 chars per chunk for fastest TTS
 * - Min ~20 chars to avoid tiny clips
 */
function segmentForTTS(text, maxChunkLen = 120) {
    // Primary split: sentence endings
    const sentences = text.split(/(?<=[.!?।])\s+/);
    const chunks = [];
    let current = "";

    for (const sent of sentences) {
        if (!sent.trim()) continue;

        if ((current + " " + sent).length <= maxChunkLen) {
            current = current ? current + " " + sent : sent;
        } else {
            // Current chunk is full, flush it
            if (current) chunks.push(current.trim());
            // If single sentence is too long, split on commas
            if (sent.length > maxChunkLen) {
                const parts = sent.split(/(?<=,)\s+/);
                let sub = "";
                for (const p of parts) {
                    if ((sub + " " + p).length <= maxChunkLen) {
                        sub = sub ? sub + " " + p : p;
                    } else {
                        if (sub) chunks.push(sub.trim());
                        sub = p;
                    }
                }
                current = sub;
            } else {
                current = sent;
            }
        }
    }
    if (current) chunks.push(current.trim());
    return chunks.filter(c => c.length >= 5);
}

// ═══════════════════════════════════════════════════════════════════════
//  LANGUAGE → TTS SPEAKER + PACE MAP
// ═══════════════════════════════════════════════════════════════════════
// bulbul:v2 speakers: anushka, abhilash, manisha, vidya, arya, karun, hitesh
// Speaker selection rationale:
//   karun  — warm Indian male voice, best for Hindi/Punjabi (authoritative, trustworthy)
//   anushka — clear Indian female voice, best for South Indian scripts (less accent bleed)
//   arya   — neutral female multilingual, good for Bengali/Marathi/Gujarati
//   manisha — warm female, good for Odia
// Optimized for CLARITY — rural users need clear pronunciation, proper breaks
// Lower temperature (0.4-0.5) = consistent, clear diction, less mumbling
// Pace 1.05-1.10 = natural speed, not rushed (rural users need time to understand)
const LANG_VOICE = {
    "hi-IN": { speaker: "rahul",     pace: 1.0,  model: "bulbul:v3", temperature: 0.3 },   // Male, clearest Hindi — natural diction, no accent
    "en-IN": { speaker: "shubh",     pace: 1.10, model: "bulbul:v3", temperature: 0.4 },   // Male, Indian English — clear & steady
    "bn-IN": { speaker: "simran",    pace: 1.0,  model: "bulbul:v3", temperature: 0.45 },  // Female, clear Bengali
    "te-IN": { speaker: "priya",     pace: 1.0,  model: "bulbul:v3", temperature: 0.45 },  // Female, clear Telugu — fastest response
    "ta-IN": { speaker: "kavitha",   pace: 0.95, model: "bulbul:v3", temperature: 0.45 },  // Female, clear Tamil — slightly slower for clarity
    "mr-IN": { speaker: "ritu",      pace: 1.0,  model: "bulbul:v3", temperature: 0.45 },  // Female, clear Marathi
    "gu-IN": { speaker: "anand",     pace: 1.0,  model: "bulbul:v3", temperature: 0.45 },  // Male, clear Gujarati
    "kn-IN": { speaker: "kavya",     pace: 1.0,  model: "bulbul:v3", temperature: 0.45 },  // Female, clear Kannada
    "ml-IN": { speaker: "kavitha",   pace: 0.95, model: "bulbul:v3", temperature: 0.45 },  // Female, clear Malayalam — slower for clarity
    "pa-IN": { speaker: "anand",     pace: 1.0,  model: "bulbul:v3", temperature: 0.45 },  // Male, clear Punjabi
    "od-IN": { speaker: "pooja",     pace: 1.0,  model: "bulbul:v3", temperature: 0.45 },  // Female, clear Odia
};

function getVoiceParams(langCode) {
    return LANG_VOICE[langCode] || LANG_VOICE["hi-IN"];
}

// ═══════════════════════════════════════════════════════════════════════
//  SELF-CORRECTION — transcript confidence scoring
// ═══════════════════════════════════════════════════════════════════════

/**
 * Score transcript quality and flag for re-ask
 * Returns: { confident: bool, reason: string }
 */
function scoreTranscript(transcript, langCode) {
    if (!transcript || transcript.length < 3) {
        return { confident: false, reason: "too_short" };
    }

    // Too many non-alpha characters suggest poor transcription
    const alphaRatio = (transcript.match(/[a-zA-Z\u0900-\u0D7F]/g) || []).length / transcript.length;
    if (alphaRatio < 0.4) {
        return { confident: false, reason: "low_alpha_ratio" };
    }

    // Check for common STT artifacts
    const artifacts = ["...", "hmm hmm", "uh uh", "[inaudible]", "[noise]", "undefined"];
    for (const a of artifacts) {
        if (transcript.toLowerCase().includes(a)) {
            return { confident: false, reason: "artifact_detected" };
        }
    }

    // Very short transcript with no real words
    const words = transcript.trim().split(/\s+/).filter(w => w.length > 1);
    if (words.length < 2 && transcript.length < 10) {
        return { confident: false, reason: "too_few_words" };
    }

    return { confident: true, reason: "ok" };
}

/**
 * Generate a clarification prompt when transcript is unclear
 */
function getClarificationPrompt(langCode, reason) {
    const prompts = {
        "hi-IN": {
            too_short: "माफ़ कीजिए, आपकी बात समझ नहीं आई। क्या आप थोड़ा और विस्तार से बता सकते हैं?",
            too_few_words: "कृपया दोबारा बोलें, मैं सुन रहा हूँ।",
            default: "क्या आप अपनी बात दोबारा बोल सकते हैं? मैं ध्यान से सुनना चाहता हूँ।",
        },
        "en-IN": {
            too_short: "I'm sorry, I couldn't quite hear that. Could you please say that again?",
            too_few_words: "Please say that again, I'm listening.",
            default: "Could you repeat that? I want to make sure I understand correctly.",
        },
    };
    const langPrompts = prompts[langCode] || prompts["hi-IN"];
    return langPrompts[reason] || langPrompts.default;
}

// ═══════════════════════════════════════════════════════════════════════
//  RESPONSE SELF-CORRECTION — post-LLM validation
// ═══════════════════════════════════════════════════════════════════════

/**
 * Check if LLM response is well-formed and legal
 * @param {string} text - Response text
 * @param {string} langCode - Language code
 * @param {number|null} groundingScore - Optional citation grounding score (0–1)
 */
function validateResponse(text, langCode, groundingScore = null) {
    if (!text || text.length < 15) return { valid: false, reason: "too_short" };

    // If grounding score is very low, flag as invalid immediately
    if (groundingScore !== null && groundingScore < 0.4) {
        return { valid: false, reason: "low_grounding" };
    }

    // Check it contains at least one legal signal
    const legalSignals = [
        "act", "section", "court", "police", "fir", "nalsa", "15100", "vakeel",
        "lawyer", "petition", "bail", "complaint", "right", "law",
        "kanoon", "adalat", "thana", "haq", "adhikar", "article",
    ];
    const tl = text.toLowerCase();
    const hasLegal = legalSignals.some(s => tl.includes(s));

    // Sanity: response shouldn't be all punctuation or garbled
    const wordCount = text.split(/\s+/).filter(w => w.length > 1).length;
    if (wordCount < 5) return { valid: false, reason: "too_few_words" };

    return { valid: true, hasLegal };
}

module.exports = {
    normalizeTTS,
    segmentForTTS,
    getVoiceParams,
    scoreTranscript,
    getClarificationPrompt,
    validateResponse,
    numToHindi,
    numToEnglish,
};