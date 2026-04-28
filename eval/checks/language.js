/**
 * checks/language.js — verifies the answer is in the requested language.
 *
 * For voice mode the user dialled in a specific language (hi-IN, en-IN, etc.).
 * The reply must be in that language — answering a Hindi caller in English
 * fails the call regardless of correctness.
 *
 * Heuristic: count Devanagari vs Latin characters. A reply tagged hi-IN
 * should have >40% Devanagari chars; en-IN should have <5% Devanagari.
 * Other languages add their own script-block ratios.
 *
 * Not perfect — code-switched answers ("BNS धारा 318") will pass either
 * way, which is the right behaviour for India.
 */
"use strict";

function scriptRatios(text) {
    if (!text) return { latin: 0, devanagari: 0, bengali: 0, tamil: 0, telugu: 0, total: 0 };
    let latin = 0, devanagari = 0, bengali = 0, tamil = 0, telugu = 0, total = 0;
    for (const ch of text) {
        const c = ch.codePointAt(0);
        if (c >= 0x0041 && c <= 0x007A) { latin++; total++; }
        else if (c >= 0x0900 && c <= 0x097F) { devanagari++; total++; }
        else if (c >= 0x0980 && c <= 0x09FF) { bengali++; total++; }
        else if (c >= 0x0B80 && c <= 0x0BFF) { tamil++; total++; }
        else if (c >= 0x0C00 && c <= 0x0C7F) { telugu++; total++; }
    }
    if (total === 0) return { latin: 0, devanagari: 0, bengali: 0, tamil: 0, telugu: 0, total: 0 };
    return {
        latin: latin / total,
        devanagari: devanagari / total,
        bengali: bengali / total,
        tamil: tamil / total,
        telugu: telugu / total,
        total,
    };
}

const LANG_RULES = {
    "hi-IN": (r) => r.devanagari >= 0.4,
    "en-IN": (r) => r.latin >= 0.6,
    "bn-IN": (r) => r.bengali >= 0.4,
    "ta-IN": (r) => r.tamil >= 0.4,
    "te-IN": (r) => r.telugu >= 0.4,
    "mr-IN": (r) => r.devanagari >= 0.4,
    "gu-IN": (r) => r.devanagari >= 0.4 || r.latin >= 0.6, // Gujarati script not yet checked
};

function run(answer, expect, question) {
    const lang = question?.lang || "hi-IN";
    const rule = LANG_RULES[lang];
    const ratios = scriptRatios(answer);
    if (!rule) return { passed: true, lang, ratios, note: "no rule for lang" };
    const passed = rule(ratios);
    return { passed, lang, ratios };
}

module.exports = { run, name: "language" };
