/**
 * checks/banned_phrases.js — Sanhita G3 + G7 + G9 ported.
 *
 * Three categories of banned language:
 *
 *  1. Hedge / hallucination tells: "I think", "in my opinion", "I'm not sure",
 *     "as an AI" — these mean the model is guessing or breaking persona.
 *  2. Judgment phrases (R1): "you should", "I recommend", "your best move is" —
 *     AI must not render judgment.
 *  3. Accountability phrases (R3): "I'll handle it", "leave it to me" — AI must
 *     not take ownership of the user's decision.
 *
 * Per-question `expect.forbid_phrases` adds question-specific bans on top.
 * Returns a single failure list with category + matched phrase.
 */
"use strict";

const HEDGE = [
    /\bas an ai\b/i,
    /\bi (?:think|believe|feel|guess)\b/i,
    /\bin my opinion\b/i,
    /\bi'?m not (?:sure|certain)\b/i,
    /\bbased on the retrieved\b/i,
    /\bit is (?:widely|generally|commonly) (?:held|believed|known)\b/i,
];

const JUDGMENT = [
    /\byou should\b/i,
    /\bi recommend\b/i,
    /\bi advise you to\b/i,
    /\byour best (?:move|option|bet) is\b/i,
    /\bthe right (?:thing|move|step) is\b/i,
];

const ACCOUNTABILITY = [
    /\bi'?ll (?:handle|take care of|manage) (?:this|it|that)\b/i,
    /\bleave it to me\b/i,
    /\bi'?ve got this\b/i,
];

function scan(answer, patterns, category) {
    const violations = [];
    for (const re of patterns) {
        const m = re.exec(answer);
        if (m) violations.push({ category, phrase: m[0] });
    }
    return violations;
}

function run(answer, expect) {
    const text = answer || "";
    const violations = [
        ...scan(text, HEDGE, "hedge"),
        ...scan(text, JUDGMENT, "judgment"),
        ...scan(text, ACCOUNTABILITY, "accountability"),
    ];
    // Question-specific forbids (`expect.forbid_phrases` is array of strings).
    const extra = expect?.forbid_phrases || [];
    for (const p of extra) {
        const re = new RegExp("\\b" + p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\b", "i");
        const m = re.exec(text);
        if (m) violations.push({ category: "question_specific", phrase: m[0] });
    }
    return { passed: violations.length === 0, violations };
}

module.exports = { run, name: "banned_phrases" };
