"use strict";

/**
 * rules-engine.js — NyayaSathi 10-Rule Post-LLM Pipeline
 *
 * Every LLM response passes through these rules before reaching the user.
 * Rules are applied in order — each one transforms or validates the response.
 *
 * Pipeline: LLM Response → [Rule 1..10] → Clean, verified, safe response
 */

const { extractCitations, verifyCitation, computeGroundingScore } = require("./citation-guard");
const { numToHindi, numToEnglish } = require("./voice-engine");

// ═══════════════════════════════════════════════════════════
//  RULE 1: Strip Markdown & Thinking Tags
// ═══════════════════════════════════════════════════════════

function stripMarkdown(text) {
    let clean = text;
    // Strip <think>...</think> tags
    if (clean.includes("</think>")) clean = clean.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
    else if (clean.includes("<think>")) clean = clean.replace(/<think>\s*/gi, "").trim();
    // Strip LLM reasoning/thinking prefixes that leak into responses
    clean = clean.replace(/^(?:Okay,?\s*(?:the user|so|let me|I (?:need|will|should|have|can|want)))[^.।]*[.।]\s*/i, "");
    clean = clean.replace(/^(?:Let me\s|I (?:need to|will|should|want to|have to)\s)[^.।]*[.।]\s*/i, "");
    clean = clean.replace(/^(?:Here'?s?\s|The (?:user|caller|person)\s)[^.।]*[.।]\s*/i, "");
    // Strip HTML, markdown formatting
    clean = clean.replace(/<[^>]*>/g, "")
        .replace(/\*{1,3}/g, "")
        .replace(/#{1,6}\s*/g, "")
        .replace(/```[\s\S]*?```/g, "")
        .replace(/`[^`]*`/g, "")
        .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
        .replace(/^\s*[-*•]\s+/gm, "")
        .replace(/^\s*\d+\.\s+/gm, "")
        .replace(/\n+/g, " ")
        .replace(/\s{2,}/g, " ")
        .trim();
    return clean;
}

// ═══════════════════════════════════════════════════════════
//  RULE 2: Enforce Language (Hindi→Devanagari, English terms)
// ═══════════════════════════════════════════════════════════

const HINDI_REPLACE = {
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

// Legal English words/acronyms to KEEP in Hindi responses (not strip)
const LEGAL_KEEP = new Set([
    // Acronyms & abbreviations
    "NyayaSathi", "NALSA", "RERA", "RTI", "PIL", "NCLT", "NCLAT", "HC", "SC",
    "BNS", "BNSS", "IPC", "CrPC", "CPC", "BSA", "IEA", "NI",
    "POSH", "POCSO", "DV", "FIR", "MACT", "DLSA", "SDM", "CAT", "NGT",
    "FSSAI", "DRT", "SARFAESI", "PMLA", "NDPS", "SEBI", "NCDRC", "MOU",
    "EWS", "OBC", "PF", "ESI", "EPFO", "ESIC", "RBI", "IRDAI",
    "GST", "CGST", "TDS", "ITR", "PAN", "UGC", "AICTE", "NMC",
    "ADR", "RFA", "OA", "SLP", "ICC", "ICJ", "NCW", "NHRC",
    // Common legal English words that appear in Hindi legal text
    "Act", "Section", "Article", "Form", "Rule", "Order",
]);

function enforceLanguage(text, lang) {
    if (lang === "en-IN") return text;
    let clean = text;

    // Replace English→Hindi terms FIRST (longest first)
    const sorted = Object.entries(HINDI_REPLACE).sort((a, b) => b[0].length - a[0].length);
    for (const [eng, hin] of sorted) {
        clean = clean.replace(new RegExp(`\\b${eng.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "g"), hin);
    }

    // Replace digits with Hindi words
    clean = clean.replace(/\b(\d{1,6})\b/g, (_, n) => {
        const num = parseInt(n);
        if (num > 0 && num <= 999999) return numToHindi(num);
        return n;
    });

    // Strip English words that aren't legal acronyms or known terms
    clean = clean.replace(/\b[A-Za-z]{2,}\b/g, (match) => {
        if (LEGAL_KEEP.has(match) || LEGAL_KEEP.has(match.toUpperCase())) return match;
        if (/^[A-Z]{2,}$/.test(match)) return match;
        return "";
    });

    // Clean up artifacts from word removal
    clean = clean.replace(/\s*[,;:]\s*[,;:]\s*/g, ", ")
        .replace(/\.\s*\./g, ".")
        .replace(/\s{2,}/g, " ")
        .trim();

    // POST-CLEANUP CHECK: if barely any native script content remains, response was junk
    const nonLatinChars = (clean.match(/[^\x00-\x7F]/g) || []).length;
    if (nonLatinChars < 15) {
        console.log(`[LANG] After cleanup, only ${nonLatinChars} native chars remain for ${lang}`);
        return null;
    }
    return clean;
}

// ═══════════════════════════════════════════════════════════
//  RULE 3: Verify Citations (only cite from RAG context)
// ═══════════════════════════════════════════════════════════

function verifyCitations(text, ragChunks) {
    if (!ragChunks || ragChunks.length === 0) return text;
    const { score } = computeGroundingScore(text, ragChunks);
    if (score >= 0.6) return text;

    // Try to salvage: remove sentences with unverified section numbers
    const citations = extractCitations(text);
    const unverified = citations.filter(c =>
        (c.type === "section" || c.type === "article") && !verifyCitation(c, ragChunks).verified
    );
    if (unverified.length === 0) return text;

    const sentences = text.split(/(?<=[.!?।])\s+/);
    const kept = sentences.filter(sent => {
        const sentLower = sent.toLowerCase();
        return !unverified.some(u => sentLower.includes(u.raw.toLowerCase()));
    });
    return kept.length > 0 ? kept.join(" ") : text;
}

// ═══════════════════════════════════════════════════════════
//  RULE 4: Detect Emergency — add helpline numbers
// ═══════════════════════════════════════════════════════════

const EMERGENCY_PATTERNS = [
    { pattern: /(?:domestic\s*violence|घरेलू\s*हिंसा|पत्नी.*(?:मारता|पीटता)|husband.*(?:beat|hit|violence))/i, helpline: "181", label: "Women Helpline" },
    { pattern: /(?:suicide|आत्महत्या|खुदकुशी|self[\s-]*harm|जान\s*देना)/i, helpline: "9152987821", label: "iCall Mental Health" },
    { pattern: /(?:threat|जान.*(?:खतर|मार)|life.*danger|death.*threat|धमकी)/i, helpline: "112", label: "Emergency" },
    { pattern: /(?:child\s*abuse|बच्च.*(?:शोषण|यौन)|minor.*(?:abuse|assault)|POCSO)/i, helpline: "1098", label: "Childline" },
    { pattern: /(?:dowry|दहेज|bride.*burn|जलाना)/i, helpline: "181", label: "Women Helpline" },
    { pattern: /(?:rape|बलात्कार|sexual\s*assault|यौन.*(?:हमला|उत्पीड़न))/i, helpline: "1091", label: "Women Helpline" },
    { pattern: /(?:cyber\s*(?:crime|fraud)|ऑनलाइन.*(?:धोखा|ठगी)|digital\s*arrest)/i, helpline: "1930", label: "Cyber Crime" },
    { pattern: /(?:accident|दुर्घटना|road.*(?:accident|crash)|गाड़ी.*टक्कर)/i, helpline: "112", label: "Emergency" },
];

function detectEmergency(text, originalQuery, lang) {
    const combined = (originalQuery || "") + " " + text;
    const helplines = [];
    for (const { pattern, helpline, label } of EMERGENCY_PATTERNS) {
        pattern.lastIndex = 0;
        if (pattern.test(combined) && !text.includes(helpline)) {
            helplines.push({ helpline, label });
        }
    }
    if (helplines.length === 0) return text;

    // Append emergency helplines
    const additions = helplines.map(h =>
        lang === "en-IN"
            ? `${h.label}: ${h.helpline}.`
            : `${h.label}: ${h.helpline}।`
    ).join(" ");
    return text + " " + additions;
}

// ═══════════════════════════════════════════════════════════
//  RULE 5: Ensure NALSA 15100 is always mentioned
// ═══════════════════════════════════════════════════════════

function ensureNALSA(text, lang) {
    if (text.includes("15100") || text.includes("NALSA") || text.includes("नालसा")) return text;
    const suffix = lang === "en-IN"
        ? " For free legal aid, call NALSA 15100."
        : " मुफ़्त कानूनी सहायता के लिए नालसा 15100 पर कॉल करें।";
    return text + suffix;
}

// ═══════════════════════════════════════════════════════════
//  RULE 6: Enforce Word Limit
// ═══════════════════════════════════════════════════════════

function enforceWordLimit(text, lang, isPhone) {
    const maxChars = isPhone ? 580 : 950; // ~90 words phone, ~150 words web
    if (text.length <= maxChars) return text;

    let trimmed = text.slice(0, maxChars);
    const lastPunct = Math.max(
        trimmed.lastIndexOf("."), trimmed.lastIndexOf("!"),
        trimmed.lastIndexOf("?"), trimmed.lastIndexOf("।")
    );
    if (lastPunct > maxChars * 0.5) trimmed = trimmed.slice(0, lastPunct + 1);
    trimmed += lang === "en-IN" ? " Call NALSA 15100." : " नालसा एक पाँच एक शून्य शून्य पर फ़ोन करें।";
    return trimmed;
}

// ═══════════════════════════════════════════════════════════
//  RULE 7: Filter Non-Legal Responses
// ═══════════════════════════════════════════════════════════

const NON_LEGAL_PATTERNS = [
    /\b(?:recipe|cricket|movie|song|poem|weather|horoscope|astrology)\b/i,
    /\b(?:film|actor|actress|bollywood|hollywood|music)\b/i,
    /\b(?:joke|funny|meme|game|sport|match|score)\b/i,
    /\b(?:cook|bake|ingredient|dish|food|restaurant)\b/i,
];

function filterNonLegal(text, refusal) {
    const lower = text.toLowerCase();
    // Check for legal signals first — if present, it's legal content
    const legalSignals = [
        "act", "section", "court", "police", "fir", "nalsa", "15100", "lawyer",
        "petition", "bail", "complaint", "right", "law", "kanoon", "adalat",
        "vakeel", "haq", "adhikar", "article", "धारा", "अदालत", "कानून",
        "पुलिस", "वकील", "ज़मानत", "शिकायत", "अधिकार",
    ];
    if (legalSignals.some(s => lower.includes(s))) return text;

    // No legal signals — check for non-legal content
    if (NON_LEGAL_PATTERNS.some(p => p.test(text))) return refusal;
    return text;
}

// ═══════════════════════════════════════════════════════════
//  RULE 8: Convert Numbers for TTS (digits → words)
// ═══════════════════════════════════════════════════════════

function convertNumbers(text, lang, isPhone) {
    if (!isPhone) return text; // Only for phone/TTS
    return text.replace(/\b(\d{1,6})\b/g, (_, n) => {
        const num = parseInt(n);
        if (num > 0 && num <= 999999) {
            return lang === "en-IN" ? numToEnglish(num) : numToHindi(num);
        }
        return n;
    });
}

// ═══════════════════════════════════════════════════════════
//  RULE 9: Profanity Filter
// ═══════════════════════════════════════════════════════════

const PROFANITY_PATTERNS = [
    // Hindi abusive words
    /\b(?:madarchod|behenchod|chutiya|gaand|bhosdike?|harami|randi|saala|kamina|kutta|kutti)\b/gi,
    // English abusive words
    /\b(?:fuck|shit|bitch|asshole|bastard|damn|crap|dick|pussy)\b/gi,
];

function filterProfanity(text) {
    let clean = text;
    for (const pattern of PROFANITY_PATTERNS) {
        pattern.lastIndex = 0;
        clean = clean.replace(pattern, "***");
    }
    return clean;
}

// ═══════════════════════════════════════════════════════════
//  RULE 10: Hallucination Guard — validate Act+Section ranges
// ═══════════════════════════════════════════════════════════

const ACT_SECTION_RANGES = {
    "BNS": { min: 1, max: 395 },
    "IPC": { min: 1, max: 511 },
    "BNSS": { min: 1, max: 531 },
    "CrPC": { min: 1, max: 484 },
    "CPC": { min: 1, max: 158 },
    "BSA": { min: 1, max: 170 },
    "IEA": { min: 1, max: 167 },
    "POCSO": { min: 1, max: 46 },
    "POSH": { min: 1, max: 30 },
    "RERA": { min: 1, max: 92 },
    "NI": { min: 1, max: 147 },
    "IT Act": { min: 1, max: 90 },
    "RTI": { min: 1, max: 31 },
    "Consumer Protection Act": { min: 1, max: 107 },
    "HMA": { min: 1, max: 30 },
    "Hindu Marriage Act": { min: 1, max: 30 },
    "MVA": { min: 1, max: 217 },
    "Motor Vehicles Act": { min: 1, max: 217 },
    "Companies Act": { min: 1, max: 484 },
    "DV Act": { min: 1, max: 37 },
    "NDPS": { min: 1, max: 83 },
    "SC/ST Act": { min: 1, max: 22 },
    "Land Acquisition Act": { min: 1, max: 114 },
    "RFCTLARR": { min: 1, max: 114 },
    "TPA": { min: 1, max: 137 },
    "Transfer of Property Act": { min: 1, max: 137 },
    "Indian Contract Act": { min: 1, max: 238 },
    "Arbitration Act": { min: 1, max: 87 },
    "Arms Act": { min: 1, max: 41 },
    "HSA": { min: 1, max: 40 },
    "Hindu Succession Act": { min: 1, max: 40 },
    "Maternity Benefit Act": { min: 1, max: 28 },
};

function guardHallucination(text) {
    // Check for Act+Section combinations and validate ranges
    const actSectionRx = /\b(BNS|IPC|BNSS|CrPC|CPC|BSA|IEA|POCSO|POSH|RERA|NI|IT Act|RTI|HMA|Hindu Marriage Act|MVA|Motor Vehicles Act|Companies Act|DV Act|NDPS|SC\/ST Act|Land Acquisition Act|RFCTLARR|TPA|Transfer of Property Act|Indian Contract Act|Arbitration Act|Arms Act|HSA|Hindu Succession Act|Maternity Benefit Act|Consumer Protection Act)\b[^.]*?(?:Section|Sec\.?|S\.?|धारा)\s*(\d{1,4})/gi;
    let match;
    const invalidSections = [];
    actSectionRx.lastIndex = 0;
    while ((match = actSectionRx.exec(text)) !== null) {
        const act = match[1].toUpperCase();
        const sec = parseInt(match[2]);
        const range = ACT_SECTION_RANGES[act] || ACT_SECTION_RANGES[match[1]];
        if (range && (sec < range.min || sec > range.max)) {
            invalidSections.push({ act, sec, raw: match[0] });
        }
    }

    // Also check reverse: "Section X of BNS/IPC..."
    const sectionActRx = /(?:Section|Sec\.?|S\.?|धारा)\s*(\d{1,4})[^.]*?\b(BNS|IPC|BNSS|CrPC|CPC|BSA|IEA|POCSO|POSH|RERA|HMA|MVA|DV Act|NDPS|TPA|HSA|RFCTLARR|Arms Act|Arbitration Act|Companies Act|Consumer Protection Act)\b/gi;
    sectionActRx.lastIndex = 0;
    while ((match = sectionActRx.exec(text)) !== null) {
        const sec = parseInt(match[1]);
        const act = match[2].toUpperCase();
        const range = ACT_SECTION_RANGES[act];
        if (range && (sec < range.min || sec > range.max)) {
            const key = `${act}:${sec}`;
            if (!invalidSections.some(i => `${i.act}:${i.sec}` === key)) {
                invalidSections.push({ act, sec, raw: match[0] });
            }
        }
    }

    if (invalidSections.length === 0) return text;

    // Remove sentences containing invalid section references
    console.log(`[HALLUCINATION] Blocked invalid sections: ${invalidSections.map(i => `${i.act} ${i.sec}`).join(", ")}`);
    const sentences = text.split(/(?<=[.!?।])\s+/);
    const kept = sentences.filter(sent => {
        const sentLower = sent.toLowerCase();
        return !invalidSections.some(i => {
            // Check for "Act Section N" pattern in the sentence
            return sentLower.includes(i.act.toLowerCase()) && sentLower.includes(String(i.sec));
        });
    });
    if (kept.length > 0 && kept.length < sentences.length) return kept.join(" ");
    // If all sentences had invalid sections or it's a single sentence, strip the bad references inline
    let cleaned = text;
    for (const inv of invalidSections) {
        // Remove "BNS Section 999" style references
        cleaned = cleaned.replace(new RegExp(`\\b${inv.act}\\s+(?:Section|Sec\\.?|S\\.?)\\s*${inv.sec}\\b`, "gi"), `${inv.act}`);
        cleaned = cleaned.replace(new RegExp(`(?:Section|Sec\\.?|S\\.?|धारा)\\s*${inv.sec}\\s+(?:of\\s+)?${inv.act}`, "gi"), inv.act);
    }
    return cleaned;
}

// ═══════════════════════════════════════════════════════════
//  MAIN: Apply All Rules
// ═══════════════════════════════════════════════════════════

/**
 * Apply the full 10-rule pipeline to an LLM response.
 * @param {string} text - Raw LLM response
 * @param {object} opts - Options
 * @param {string} opts.lang - Language code (e.g., "hi-IN")
 * @param {string[]} opts.ragChunks - Retrieved RAG chunks for citation verification
 * @param {boolean} opts.isPhone - Whether this is a phone call (stricter limits)
 * @param {string} opts.originalQuery - Original user query (for emergency detection)
 * @param {string} opts.refusal - Refusal text for non-legal content
 * @returns {string} Clean, verified, safe response
 */
function applyRules(text, opts = {}) {
    const {
        lang = "hi-IN",
        ragChunks = [],
        isPhone = false,
        originalQuery = "",
        refusal = "",
    } = opts;

    if (!text || typeof text !== "string" || text.length < 5) return refusal || text || "";

    let result = text;

    // Rule 1: Strip markdown & thinking tags
    result = stripMarkdown(result);
    if (result.length < 5) return refusal || result;

    // Rule 2: Enforce language (Hindi→Devanagari, etc.)
    result = enforceLanguage(result, lang);
    if (result === null) {
        console.log(`[LANG] Response was predominantly English for ${lang}, using refusal`);
        return refusal || "मैं सिर्फ़ कानूनी मामलों में मदद करता हूँ। कृपया अपनी समस्या बताइए।";
    }

    // Rule 3: Verify citations against RAG context
    result = verifyCitations(result, ragChunks);

    // Rule 4: Detect emergency situations, add helplines
    result = detectEmergency(result, originalQuery, lang);

    // Rule 5: Ensure NALSA 15100 is always mentioned
    result = ensureNALSA(result, lang);

    // Rule 6: Enforce word limit
    result = enforceWordLimit(result, lang, isPhone);

    // Rule 7: Filter non-legal responses
    result = filterNonLegal(result, refusal);

    // Rule 8: Convert numbers for TTS
    result = convertNumbers(result, lang, isPhone);

    // Rule 9: Filter profanity
    result = filterProfanity(result);

    // Rule 10: Hallucination guard — validate Act+Section ranges
    result = guardHallucination(result);

    // Final cleanup
    result = result.replace(/\s{2,}/g, " ").trim();
    if (result.length < 5) return refusal || result;

    return result;
}

module.exports = {
    applyRules,
    stripMarkdown,
    enforceLanguage,
    verifyCitations,
    detectEmergency,
    ensureNALSA,
    enforceWordLimit,
    filterNonLegal,
    convertNumbers,
    filterProfanity,
    guardHallucination,
};
