/**
 * intent-classifier.js — fast, no-LLM case-type classifier.
 *
 * The whole point: Phase C needs to inject "Case so far / Still need to learn"
 * into the prompt, but the slot template is keyed on case type. Doing an LLM
 * call just to figure out "is this an accident or a divorce?" wastes 1-2s and
 * a tier-1 quota slot. Almost every user message in this domain has obvious
 * keywords (FIR, accident, salary, divorce, etc.) — keyword scoring works.
 *
 * Returns: { type, confidence, matchedKeywords[] }
 *   type ∈ {bail, accident, dv, property, fraud, salary, fir, consumer,
 *           divorce, land, false_case, vehicle, caste_atrocity, mnrega,
 *           ration, panchayat, rti, cheque_bounce, unknown}
 *   confidence: 0..1 (number of keyword matches normalized)
 */

const TYPES = {
    accident: {
        keywords: [
            // English
            "accident", "crash", "collision", "vehicle", "hit and run",
            "bike", "car", "truck", "motor", "macT", "insurance claim", "compensation",
            // Devanagari
            "एक्सीडेंट", "एक्सिडेंट", "दुर्घटना", "टक्कर", "हादसा", "गाड़ी", "बाइक",
            "मुआवज़ा", "मुआवजा", "बीमा",
        ],
    },
    fir: {
        keywords: [
            "fir", "complaint", "police", "thana", "lodge", "register",
            "एफआईआर", "एफ़आईआर", "एफ़", "थाना", "थाने", "पुलिस", "दर्ज",
        ],
    },
    bail: {
        keywords: [
            "bail", "arrest", "jail", "custody", "remand", "anticipatory",
            "ज़मानत", "जमानत", "गिरफ्तारी", "गिरफ़्तारी", "जेल", "हिरासत",
        ],
    },
    dv: {
        keywords: [
            "domestic violence", "husband", "abuse", "beat", "torture", "marital",
            "घरेलू हिंसा", "पति", "मारपीट", "ससुराल", "मार रहा", "धमकी", "दहेज",
        ],
    },
    fraud: {
        keywords: [
            "fraud", "scam", "cyber", "upi", "phishing", "otp", "online cheat",
            "धोखाधड़ी", "धोखा", "ऑनलाइन", "साइबर", "ठगी", "फ्रॉड", "फ़्रॉड",
        ],
    },
    salary: {
        keywords: [
            "salary", "wages", "fired", "termination", "boss", "employer", "job",
            "वेतन", "तनख्वाह", "तनख़्वाह", "सैलरी", "नौकरी", "पगार", "निकाला",
        ],
    },
    property: {
        keywords: [
            "property", "flat", "builder", "rera", "registry", "rent", "tenant", "landlord",
            "संपत्ति", "रजिस्ट्री", "मकान", "फ्लैट", "बिल्डर", "किराया", "किरायेदार",
        ],
    },
    land: {
        keywords: [
            "land", "encroach", "kabza", "occupy", "boundary",
            "ज़मीन", "जमीन", "कब्ज़ा", "कब्जा", "खेत", "भूमि", "तहसील",
        ],
    },
    divorce: {
        keywords: [
            "divorce", "separation", "marriage", "talaq", "custody", "maintenance",
            "तलाक", "तलाक़", "शादी", "विवाह", "गुज़ारा", "भरण-पोषण",
        ],
    },
    consumer: {
        keywords: [
            "consumer", "refund", "defective", "service", "ecommerce", "delivery",
            "उपभोक्ता", "रिफंड", "रिफ़ंड", "सामान", "सर्विस",
        ],
    },
    cheque_bounce: {
        keywords: [
            "cheque", "check bounce", "ni act", "138",
            "चेक", "बाउंस", "बाउन्स",
        ],
    },
    rti: {
        keywords: ["rti", "right to information", "आरटीआई", "सूचना का अधिकार"],
    },
    caste_atrocity: {
        keywords: [
            "caste", "dalit", "atrocity", "untouchability", "sc/st",
            "जाति", "जात", "दलित", "छुआछूत", "अनुसूचित",
        ],
    },
    mnrega: {
        keywords: ["mnrega", "nrega", "mgnrega", "मनरेगा", "नरेगा"],
    },
    ration: {
        keywords: ["ration", "ration card", "pds", "राशन", "राशन कार्ड", "अनाज"],
    },
    panchayat: {
        keywords: ["panchayat", "sarpanch", "pradhan", "पंचायत", "सरपंच", "प्रधान"],
    },
    false_case: {
        keywords: [
            "false case", "fake case", "framed", "wrongful", "trapped",
            "झूठा केस", "फ़र्ज़ी केस", "फँसाया", "झूठी",
        ],
    },
};

/** Lowercase + collapse whitespace. Keeps Devanagari intact. */
function normalize(s) {
    return (s || "").toLowerCase().replace(/\s+/g, " ").trim();
}

/**
 * Classify a user message. Pass `priorType` if you want to weight in favor
 * of stickiness — case-type changes are rare, so once we've classified a
 * conversation as "accident" we shouldn't flip on a single ambiguous turn.
 */
function classify(userMsg, priorType = null) {
    const text = normalize(userMsg);
    if (!text) return { type: priorType || "unknown", confidence: 0, matchedKeywords: [] };

    const scores = {};
    const matched = {};

    for (const [type, def] of Object.entries(TYPES)) {
        let score = 0;
        const hits = [];
        for (const kw of def.keywords) {
            const k = normalize(kw);
            if (text.includes(k)) {
                score += k.length >= 5 ? 2 : 1; // longer keywords are more specific
                hits.push(kw);
            }
        }
        if (score > 0) {
            scores[type] = score;
            matched[type] = hits;
        }
    }

    // Stickiness: prior type gets a +1 bump so trivial messages don't flip it.
    if (priorType && scores[priorType] !== undefined) {
        scores[priorType] += 1;
    }

    let bestType = priorType || "unknown";
    let bestScore = 0;
    for (const [type, score] of Object.entries(scores)) {
        if (score > bestScore) { bestScore = score; bestType = type; }
    }

    // Normalize confidence to 0..1. Cap at 6 hits (very strong signal).
    const confidence = Math.min(bestScore / 6, 1);

    return {
        type: bestType,
        confidence,
        matchedKeywords: matched[bestType] || [],
    };
}

/**
 * Run classify against the full conversation history (user turns only).
 * Useful when the latest turn is short ("ok", "haan") but earlier turns
 * had clear topic signals.
 */
function classifyFromHistory(history = [], priorType = null) {
    const userText = history.filter(m => m.u === 1).map(m => m.t).join(" ");
    if (!userText) return { type: priorType || "unknown", confidence: 0, matchedKeywords: [] };
    return classify(userText, priorType);
}

module.exports = { classify, classifyFromHistory, TYPES };
