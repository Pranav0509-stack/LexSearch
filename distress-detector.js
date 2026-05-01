/**
 * distress-detector.js — sub-millisecond safety triage.
 *
 * Why keyword scan, not LLM? Two reasons:
 *  1. Latency. A user saying "वो मुझे मार रहा है, बच्चे को भी मारेगा" cannot
 *     wait 5–8 seconds for Gemini to produce a thoughtful response. They
 *     need 181 / 112 / 1098 numbers in their face NOW.
 *  2. Reliability. The LLM cascade fails sometimes. The safety path must
 *     never depend on it.
 *
 * Returns:
 *   { level: "critical"|"high"|"none", reason, helplines[], prepend }
 *
 * - level: critical → bypass LLM entirely, send helpline message immediately.
 * - level: high     → run LLM but soften prompt (set in case.distress_level).
 * - level: none     → normal path.
 *
 * Helplines surfaced:
 *   181  — All India Women Helpline (24×7)
 *   112  — Pan-India Emergency
 *   1098 — Childline
 *   100  — Police
 *   1930 — Cyber Crime
 *   15100 — NALSA Free Legal Aid
 *   9152987821 — iCall mental health (suicide ideation)
 */

// Critical: immediate physical danger / self-harm / child safety.
// Match must trigger LLM-bypass and surface a helpline number in the first
// sentence of the reply. False positives here are FAR less costly than false
// negatives.
const CRITICAL = [
    // Self-harm / suicide — all variants
    { kw: ["suicide", "kill myself", "end my life", "end it all", "want to die", "no point living"], reason: "suicide_ideation", helplines: ["9152987821", "112"] },
    { kw: ["आत्महत्या", "ख़ुदकुशी", "खुदकुशी", "जान देना", "मरना चाहता", "मरना चाहती", "जीना नहीं चाहता", "जीना नहीं चाहती"], reason: "suicide_ideation", helplines: ["9152987821", "112"] },

    // Active physical violence happening NOW
    { kw: ["he is beating me", "he is hitting me", "beating me right now", "pls help me", "save me"], reason: "active_violence", helplines: ["181", "112"] },
    { kw: ["मार रहा है", "मार रहे हैं", "पीट रहा है", "पीट रहे हैं", "जान का खतरा", "जान को खतरा", "बचाओ"], reason: "active_violence", helplines: ["181", "112"] },

    // Child in danger
    { kw: ["child being beaten", "kidnap", "kidnapped", "child abuse", "trafficking"], reason: "child_danger", helplines: ["1098", "112"] },
    { kw: ["बच्चे को मार", "बच्चे को मारेगा", "बच्चा गायब", "बच्चे का अपहरण", "बच्चे को छीन"], reason: "child_danger", helplines: ["1098", "112"] },

    // Imminent acid attack / threat to life
    { kw: ["acid attack", "kill me", "going to kill", "threatening to kill"], reason: "threat_to_life", helplines: ["112", "100"] },
    { kw: ["तेज़ाब", "तेजाब", "जान से मारेगा", "जान से मारने की धमकी"], reason: "threat_to_life", helplines: ["112", "100"] },
];

// High distress: emotional anguish, fear, crying — no immediate physical
// danger, but the AI should soften its tone, drop legal jargon, lead with
// empathy. Doesn't bypass the LLM.
const HIGH = [
    { kw: ["scared", "terrified", "afraid", "crying", "shaking", "panic"] },
    { kw: ["डर लग रहा", "डर रहा", "डर रही", "रो रही", "रो रहा", "घबराहट", "बहुत डर"] },
    { kw: ["helpless", "hopeless", "alone", "no one to help", "depressed"] },
    { kw: ["कोई नहीं है", "अकेला हूँ", "अकेली हूँ", "क्या करूँ", "बहुत परेशान", "टेंशन में"] },
];

const HELPLINE_TEXT = {
    "hi-IN": {
        suicide_ideation: "रुकिए, आप अकेले नहीं हैं। अभी iCall पर 9152987821 पर फ़ोन कीजिए — तुरंत बात मिलेगी, हिंदी में, मुफ़्त। अगर अभी ख़तरे में हैं तो 112 दबाइए।",
        active_violence:  "आप सुरक्षित जगह जाइए — पड़ोसी, माँ-बाप का घर, कहीं भी। फिर तुरंत 181 दबाइए (महिला हेल्पलाइन, 24 घंटे), या 112 (पुलिस आपातकाल)। मैं यहाँ हूँ, आप बताइए कहाँ हैं।",
        child_danger:     "बच्चे की सुरक्षा सबसे पहले। 1098 दबाइए — चाइल्डलाइन, 24 घंटे, मुफ़्त। 112 भी दबाइए। बच्चे को अपने पास रखिए, घर से बाहर ले जाइए अगर ज़रूरी हो।",
        threat_to_life:   "ये गंभीर है। 112 दबाइए — अभी, बिना देर। पुलिस को सब बताइए, धमकी देने वाले का नाम, फ़ोन नंबर, पता। आप अकेले नहीं हैं।",
    },
    "en-IN": {
        suicide_ideation: "Please stop and listen — you are not alone. Call iCall on 9152987821 right now. They speak Hindi and English, it's free, available 24×7. If you're in immediate danger, dial 112.",
        active_violence:  "Get to a safe place first — a neighbour, parents, anywhere away from the person. Then dial 181 (Women Helpline, 24×7) or 112 (police emergency). I'm here, tell me where you are.",
        child_danger:     "The child's safety is the priority. Dial 1098 (Childline, 24×7, free). Also dial 112. Keep the child with you and leave the location if you can.",
        threat_to_life:   "This is serious. Dial 112 right now, without delay. Tell the police everything — the name, phone number, address of whoever is threatening. You're not alone.",
    },
};

// Languages that don't yet have explicit copy fall back to Hindi.
const SUPPORTED_LANGS = new Set(["hi-IN", "en-IN"]);

function normalize(s) {
    return (s || "").toLowerCase();
}

/**
 * Run the scan. Returns the highest-priority match. O(N×M) where N is the
 * keyword count (~80) and M is the message length — sub-millisecond.
 */
function detect(userMsg, lang = "hi-IN") {
    const text = normalize(userMsg);
    if (!text) return { level: "none", reason: null, helplines: [], prepend: "" };

    // Critical scan first — earliest exit.
    for (const rule of CRITICAL) {
        for (const kw of rule.kw) {
            if (text.includes(normalize(kw))) {
                const langKey = SUPPORTED_LANGS.has(lang) ? lang : "hi-IN";
                const prepend = HELPLINE_TEXT[langKey]?.[rule.reason] || HELPLINE_TEXT["hi-IN"][rule.reason] || "";
                return {
                    level: "critical",
                    reason: rule.reason,
                    helplines: rule.helplines,
                    matched: kw,
                    prepend,
                };
            }
        }
    }

    // High scan — emotional but no physical-danger keyword hit.
    for (const rule of HIGH) {
        for (const kw of rule.kw) {
            if (text.includes(normalize(kw))) {
                return { level: "high", reason: "emotional_distress", helplines: [], matched: kw, prepend: "" };
            }
        }
    }

    return { level: "none", reason: null, helplines: [], prepend: "" };
}

/**
 * Convert a level to a 0..1 distress_level score for case state.
 * critical=0.9, high=0.6, none=0. Monotonic — caller can max() with prior.
 */
function levelToScore(level) {
    if (level === "critical") return 0.9;
    if (level === "high") return 0.6;
    return 0;
}

module.exports = { detect, levelToScore, HELPLINE_TEXT };
