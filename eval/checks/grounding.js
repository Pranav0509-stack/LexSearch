/**
 * checks/grounding.js — verifies must-mention terms + word-count budget.
 *
 * For NyayaSathi voice mode, two soft constraints turn into hard fails:
 *   1. expect.must_mention[] — every term must appear (case-insensitive)
 *      Allows multilingual terms (will accept any one of a comma-separated
 *      group, which is encoded as | in the term: e.g. "FIR|एफआईआर").
 *   2. expect.max_words — answer must not exceed this. Voice mode is 90–160
 *      words depending on case complexity. Long answers are useless on phone.
 *
 * Returns granular failure info so the harness can print which terms missed
 * and what the actual word count was.
 */
"use strict";

function normalize(s) {
    return (s || "").toLowerCase().replace(/\s+/g, " ").trim();
}

function termFound(answer, term) {
    const norm = normalize(answer);
    // Allow alternatives separated by | — any one match satisfies.
    const alternatives = term.split("|").map(t => normalize(t));
    return alternatives.some(t => t && norm.includes(t));
}

function wordCount(text) {
    if (!text) return 0;
    // Count whitespace-separated tokens (works for Devanagari + Latin).
    return text.trim().split(/\s+/).length;
}

function run(answer, expect) {
    const result = { passed: true, missing_terms: [], word_count: wordCount(answer), max_words: expect?.max_words };

    // 1. Must-mention terms
    const required = expect?.must_mention || [];
    for (const term of required) {
        if (!termFound(answer, term)) result.missing_terms.push(term);
    }
    if (result.missing_terms.length > 0) result.passed = false;

    // 2. Word count budget
    if (expect?.max_words && result.word_count > expect.max_words) {
        result.passed = false;
        result.over_budget_by = result.word_count - expect.max_words;
    }

    return result;
}

module.exports = { run, name: "grounding" };
