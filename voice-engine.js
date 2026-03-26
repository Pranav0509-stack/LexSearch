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
// Devanagari number words — Sarvam TTS requires native script for natural Hindi speech
const ONES_HI = ["", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस",
    "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस",
    "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस",
    "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस", "चालीस",
    "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास", "पचास",
    "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ", "साठ",
    "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर", "सत्तर",
    "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी", "अस्सी",
    "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सतासी", "अठासी", "नवासी", "नब्बे",
    "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सतानवे", "अठानवे", "निन्यानवे"];
const ONES_EN = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS_EN = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

function numToHindi(n) {
    if (n < 0) return "माइनस " + numToHindi(-n);
    if (n === 0) return "शून्य";
    if (n <= 99) return ONES_HI[n] || (ONES_HI[Math.floor(n / 10) * 10] || "") + " " + ONES_HI[n % 10];
    if (n < 1000) return ONES_HI[Math.floor(n / 100)] + " सौ" + (n % 100 ? " " + numToHindi(n % 100) : "");
    if (n < 100000) return numToHindi(Math.floor(n / 1000)) + " हज़ार" + (n % 1000 ? " " + numToHindi(n % 1000) : "");
    if (n < 10000000) return numToHindi(Math.floor(n / 100000)) + " लाख" + (n % 100000 ? " " + numToHindi(n % 100000) : "");
    return numToHindi(Math.floor(n / 10000000)) + " करोड़" + (n % 10000000 ? " " + numToHindi(n % 10000000) : "");
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
// Section patterns are language-aware — built dynamically in normalizeTTS
function buildSectionPatterns(isHindi) {
    const numFn = isHindi ? numToHindi : numToEnglish;
    const secWord = isHindi ? "धारा" : "Section";
    const artWord = isHindi ? "अनुच्छेद" : "Article";
    return [
        { rx: /[Ss]ection\s+(\d{1,4})([A-Z]?)/g, fn: (_, n, suffix) => secWord + " " + numFn(parseInt(n)) + (suffix ? " " + suffix : "") },
        { rx: /Sec\.?\s+(\d{1,4})([A-Z]?)/g, fn: (_, n, suffix) => secWord + " " + numFn(parseInt(n)) + (suffix ? " " + suffix : "") },
        { rx: /S\.\s*(\d{1,4})/g, fn: (_, n) => secWord + " " + numFn(parseInt(n)) },
        { rx: /धारा\s+(\d{1,4})([A-Z]?)?/g, fn: (_, n, suffix) => "धारा " + numToHindi(parseInt(n)) + (suffix ? " " + suffix : "") },
        { rx: /[Aa]rticle\s+(\d{1,3})([A-Z]?)/g, fn: (_, n, suffix) => artWord + " " + numFn(parseInt(n)) + (suffix ? " " + suffix : "") },
        { rx: /अनुच्छेद\s+(\d{1,3})([A-Z]?)?/g, fn: (_, n, suffix) => "अनुच्छेद " + numToHindi(parseInt(n)) + (suffix ? " " + suffix : "") },
        { rx: /\b(IPC|BNS|BNSS|CrPC|IEA|BSA)\s+(\d+)/g, fn: (_, act, n) => act + " " + numFn(parseInt(n)) },
    ];
}

// Phone numbers → digit by digit
const PHONE_RX = /\b(15100|1516|1915|181|112|1930|1098|14434|1800[\-\s]?\d{2,3}[\-\s]?\d{3,4}|\d{10})\b/g;

// Currency → spoken
function expandCurrency(text, isHindi) {
    return text.replace(/₹\s*(\d[\d,]*(?:\.\d+)?)/g, (_, amt) => {
        const n = parseFloat(amt.replace(/,/g, ""));
        const words = isHindi ? numToHindi(Math.round(n)) : numToEnglish(Math.round(n));
        return (isHindi ? "रुपये " : "rupees ") + words;
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
            const HINDI_DIGITS = ["शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ"];
            return digits.split("").map(d => HINDI_DIGITS[parseInt(d)]).join(" ");
        }
        // English: digit words, grouped with pauses for clarity
        const EN_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
        const words = digits.split("").map(d => EN_DIGITS[parseInt(d)]);
        // Group in chunks of 3-4 digits with comma pauses
        const grouped = [];
        for (let i = 0; i < words.length; i += 3) {
            grouped.push(words.slice(i, Math.min(i + 3, words.length)).join(" "));
        }
        return grouped.join(", ");
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

// State/common abbreviations that should NOT be letter-spaced
const ABBR_SKIP = new Set(["UP", "MP", "AP", "HP", "UK", "JK", "WB", "TN", "KL", "MH", "RJ", "GJ", "HR", "PB", "OR", "GA", "IT", "ID", "IF", "IN", "IS", "AT", "TO", "OR", "AN", "AS", "OF", "ON", "BY", "NO"]);

function expandAbbreviations(text) {
    return text.replace(/\b([A-Z]{2,8})\b/g, (match) => {
        if (ABBR_SKIP.has(match)) return match;
        return ABBR_EXPAND[match] || match;
    });
}

/**
 * Normalize text for TTS — makes numbers, sections, phones, currencies
 * all speak correctly in the target language
 */
function normalizeTTS(text, langCode) {
    const isHindi = !["en-IN"].includes(langCode);
    let t = text;

    // Section patterns first (before generic number expansion) — language-aware
    const sectionPatterns = buildSectionPatterns(isHindi);
    for (const { rx, fn } of sectionPatterns) {
        rx.lastIndex = 0;
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

    // Abbreviations — expand for ALL languages (BNS, IPC etc. must be letter-spaced for clear TTS)
    t = expandAbbreviations(t);

    // URLs → readable form (remove for TTS, or read as "website")
    t = t.replace(/https?:\/\/[^\s,.)]+/g, "");
    t = t.replace(/\b\w+\.(gov|nic|org|com|in)\.\w{2,4}\b/g, "the government website");
    t = t.replace(/\b\w+\.(gov|nic|org|com)\b/g, "the government website");

    // Case citations: "v." or "vs." or "vs" → "versus"
    t = t.replace(/\bv\.\s*/g, "versus ");
    t = t.replace(/\bvs\.?\s*/g, "versus ");

    // Common legal pronunciation fixes
    t = t.replace(/\bSec\.\s*/g, "Section ");
    t = t.replace(/\bArt\.\s*/g, "Article ");
    t = t.replace(/\bNo\.\s*/g, "Number ");
    t = t.replace(/\bi\.e\.\s*/gi, "that is, ");
    t = t.replace(/\be\.g\.\s*/gi, "for example, ");
    t = t.replace(/\betc\.\s*/gi, "and so on. ");
    t = t.replace(/\bw\.r\.t\.?\s*/gi, "with respect to ");

    // Slash between words → "or" (e.g., "husband/wife")
    t = t.replace(/(\w)\s*\/\s*(\w)/g, "$1 or $2");

    // Plus sign → "and"
    t = t.replace(/\s*\+\s*/g, " and ");

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
function segmentForTTS(text, maxChunkLen = 150) {
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
                // Word-boundary fallback for lines with no commas
                if (sub.length > maxChunkLen) {
                    const words = sub.split(/\s+/);
                    let wordChunk = "";
                    for (const w of words) {
                        if ((wordChunk + " " + w).length > maxChunkLen) {
                            if (wordChunk) chunks.push(wordChunk.trim());
                            wordChunk = w;
                        } else {
                            wordChunk = wordChunk ? wordChunk + " " + w : w;
                        }
                    }
                    current = wordChunk;
                } else {
                    current = sub;
                }
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
// ── Bulbul v3 Speaker Selection (39 voices available) ──
// All v3 speakers support ALL 11 languages.
// Shubh = v3 default, most polished male — natural conversational tone
// Kavya = warm, expressive female — excellent across Indic languages
// Temperature 0.5 = balanced: natural prosody + consistent enough for production
// Pace tuned per language for natural fluent cadence
// Sample rate: 24000 Hz (v3 default, best quality for web)
const LANG_VOICE = {
    "hi-IN": { speaker: "shubh",     pace: 1.0,  model: "bulbul:v3", temperature: 0.5 },
    "en-IN": { speaker: "shubh",     pace: 1.05, model: "bulbul:v3", temperature: 0.5 },
    "bn-IN": { speaker: "shubh",     pace: 0.95, model: "bulbul:v3", temperature: 0.5 },
    "te-IN": { speaker: "shubh",     pace: 0.95, model: "bulbul:v3", temperature: 0.5 },
    "ta-IN": { speaker: "shubh",     pace: 0.9,  model: "bulbul:v3", temperature: 0.5 },
    "mr-IN": { speaker: "shubh",     pace: 0.95, model: "bulbul:v3", temperature: 0.5 },
    "gu-IN": { speaker: "shubh",     pace: 0.95, model: "bulbul:v3", temperature: 0.5 },
    "kn-IN": { speaker: "shubh",     pace: 0.9,  model: "bulbul:v3", temperature: 0.5 },
    "ml-IN": { speaker: "shubh",     pace: 0.85, model: "bulbul:v3", temperature: 0.5 },
    "pa-IN": { speaker: "shubh",     pace: 0.95, model: "bulbul:v3", temperature: 0.5 },
    "od-IN": { speaker: "shubh",     pace: 0.95, model: "bulbul:v3", temperature: 0.5 },
};

// Female speaker — Kavya: warm, expressive, fluent across all Indic languages
const LANG_VOICE_FEMALE = {
    "hi-IN": { speaker: "kavya",     pace: 1.0,  model: "bulbul:v3", temperature: 0.5 },
    "en-IN": { speaker: "kavya",     pace: 1.05, model: "bulbul:v3", temperature: 0.5 },
    "bn-IN": { speaker: "kavya",     pace: 0.95, model: "bulbul:v3", temperature: 0.5 },
    "te-IN": { speaker: "kavya",     pace: 0.95, model: "bulbul:v3", temperature: 0.5 },
    "ta-IN": { speaker: "kavya",     pace: 0.9,  model: "bulbul:v3", temperature: 0.5 },
    "mr-IN": { speaker: "kavya",     pace: 0.95, model: "bulbul:v3", temperature: 0.5 },
    "gu-IN": { speaker: "kavya",     pace: 0.95, model: "bulbul:v3", temperature: 0.5 },
    "kn-IN": { speaker: "kavya",     pace: 0.9,  model: "bulbul:v3", temperature: 0.5 },
    "ml-IN": { speaker: "kavya",     pace: 0.85, model: "bulbul:v3", temperature: 0.5 },
    "pa-IN": { speaker: "kavya",     pace: 0.95, model: "bulbul:v3", temperature: 0.5 },
    "od-IN": { speaker: "kavya",     pace: 0.95, model: "bulbul:v3", temperature: 0.5 },
};

function getVoiceParams(langCode, gender) {
    if (gender === "female" && LANG_VOICE_FEMALE[langCode]) {
        return LANG_VOICE_FEMALE[langCode];
    }
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
    if (!transcript || transcript.length < 2) {
        return { confident: false, reason: "too_short" };
    }

    // Check for common STT artifacts — only reject if ENTIRE transcript is artifacts
    const artifacts = ["[inaudible]", "[noise]", "undefined", "thank you", "thanks"];
    const tl = transcript.toLowerCase().trim();
    for (const a of artifacts) {
        if (tl === a || tl === "..." || tl === "hmm hmm" || tl === "uh uh") {
            return { confident: false, reason: "artifact_detected" };
        }
    }

    // Single character noise artifacts
    if (tl.length === 1) {
        return { confident: false, reason: "artifact_detected" };
    }

    // Repeated single syllable (e.g., "ha ha ha", "na na na")
    const words = tl.split(/\s+/);
    if (words.length >= 2 && new Set(words).size === 1 && words[0].length <= 3) {
        return { confident: false, reason: "artifact_detected" };
    }

    // Very short non-word (under 3 chars) — allow valid short responses
    if (tl.length <= 3 && !/^(yes|no|ok|haa|ji|nhi|han|hā|नहीं|हाँ|हां)$/i.test(tl)) {
        return { confident: false, reason: "too_short" };
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
    LANG_VOICE,
    LANG_VOICE_FEMALE,
};