/**
 * checks/citations.js — verifies the answer cites every required section.
 *
 * For each question, expect.must_cite_section is a list of canonical refs
 * like "BNSS 173", "NI Act 138", "DV Act 2005", "Lalita Kumari". We do
 * a normalized substring match — case-insensitive, whitespace-collapsed,
 * with common variants (BNSS/B N S S, Act/अधिनियम, etc.) accepted.
 *
 * Returns { passed, missing, found } so the harness can print which
 * citations were missing on a fail. Hallucination guarding (G5) is a
 * separate check — this one only checks for absence, not fabrication.
 */
"use strict";

function normalize(s) {
    return (s || "")
        .toLowerCase()
        .replace(/[\s ]+/g, " ")
        .replace(/[.,;:]/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

// Section name aliases. Each canonical → list of equivalent forms we accept.
const ALIASES = {
    "bnss 173": ["bnss 173", "bnss section 173", "बीएनएसएस 173", "धारा 173"],
    "bnss 482": ["bnss 482", "bnss section 482", "बीएनएसएस 482", "धारा 482"],
    "bnss 479": ["bnss 479", "bnss section 479", "धारा 479"],
    "bnss 144": ["bnss 144", "bnss section 144", "धारा 144", "crpc 125"],
    "bns 318": ["bns 318", "bns section 318", "धारा 318", "ipc 420"],
    "bns 303": ["bns 303", "bns section 303", "धारा 303"],
    "bns 330": ["bns 330", "bns section 330"],
    "bns 85": ["bns 85", "bns section 85", "498a"],
    "bns 281": ["bns 281", "bns section 281"],
    "bns 106": ["bns 106", "bns section 106"],
    "ni act 138": ["ni act 138", "section 138", "negotiable instruments act 138", "138 ni"],
    "it act 66d": ["it act 66d", "section 66d", "information technology act 66d"],
    "dv act 2005": ["dv act", "domestic violence act", "घरेलू हिंसा"],
    "hindu marriage act 13": ["hindu marriage act 13", "section 13 hindu", "13b"],
    "hindu marriage act 10": ["hindu marriage act 10", "section 10 hindu", "judicial separation"],
    "hindu succession act": ["hindu succession act", "succession act"],
    "industrial disputes act": ["industrial disputes act", "id act"],
    "industrial disputes act 25f": ["industrial disputes act 25f", "section 25f", "25f"],
    "rera": ["rera", "real estate regulatory authority"],
    "rent control act": ["rent control act", "किराया अधिनियम"],
    "guardians and wards act": ["guardians and wards act", "guardian"],
    "lalita kumari": ["lalita kumari"],
    "sc/st": ["sc/st", "scheduled caste", "atrocities act", "अनुसूचित"],
    "dowry prohibition act": ["dowry prohibition act", "dowry act", "498a", "दहेज"],
    "crpc 125": ["crpc 125", "section 125", "section 125 crpc"],
};

function matches(answerText, ref) {
    const norm = normalize(answerText);
    const key = normalize(ref);
    const aliases = ALIASES[key] || [key];
    return aliases.some(a => norm.includes(normalize(a)));
}

/**
 * Run the citations check.
 * @param {string} answer - the AI's reply text
 * @param {object} expect - expect.must_cite_section: string[]
 * @returns {{passed:boolean, missing:string[], found:string[]}}
 */
function run(answer, expect) {
    const required = expect?.must_cite_section || [];
    if (required.length === 0) return { passed: true, missing: [], found: [] };
    const missing = [];
    const found = [];
    for (const ref of required) {
        if (matches(answer, ref)) found.push(ref);
        else missing.push(ref);
    }
    return { passed: missing.length === 0, missing, found };
}

module.exports = { run, name: "citations" };
