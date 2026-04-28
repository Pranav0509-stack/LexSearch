#!/usr/bin/env node
/**
 * eval/harness.js — the gate that ships or blocks every commit.
 *
 * Reads eval/v1/questions.jsonl, fires each question at the running
 * NyayaSathi server (POST /api/ask), runs all five check modules,
 * scores per-question + overall, writes a JSON report under
 * eval/reports/<tag>.json, prints a human summary, and exits non-zero
 * if the overall pass rate is below the sprint floor.
 *
 * USAGE
 *   node eval/harness.js                          # uses tag = "adhoc"
 *   node eval/harness.js --tag baseline           # named report
 *   node eval/harness.js --filter safety          # only category=safety
 *   node eval/harness.js --filter DSTR-           # id prefix filter
 *   node eval/harness.js --base http://localhost:3000
 *   node eval/harness.js --threshold 0.85         # block below this
 *   node eval/harness.js --max 10                 # limit to N questions
 *
 * The exit code matters: it's what CI uses to gate merges.
 *   0  pass at or above threshold
 *   1  below threshold OR any safety failure
 *   2  setup error (server unreachable, bad input, etc.)
 *
 * Reports are kept forever. Diff two reports with:
 *   diff <(jq .questions eval/reports/sprint0_d3.json) \
 *        <(jq .questions eval/reports/sprint0_d4.json)
 */
"use strict";

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const QUESTIONS_PATH = path.join(__dirname, "v1/questions.jsonl");
const REPORTS_DIR = path.join(__dirname, "reports");

const checks = [
    require("./checks/citations.js"),
    require("./checks/banned_phrases.js"),
    require("./checks/helplines.js"),
    require("./checks/grounding.js"),
    require("./checks/language.js"),
];

// ── CLI ───────────────────────────────────────────────────────────────────
function parseArgs(argv) {
    const args = { tag: "adhoc", filter: null, base: "http://localhost:3000", threshold: 0, max: 0 };
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        if (a === "--tag") args.tag = argv[++i];
        else if (a === "--filter") args.filter = argv[++i];
        else if (a === "--base") args.base = argv[++i];
        else if (a === "--threshold") args.threshold = parseFloat(argv[++i]);
        else if (a === "--max") args.max = parseInt(argv[++i], 10);
        else if (a === "-h" || a === "--help") { printHelp(); process.exit(0); }
    }
    return args;
}

function printHelp() {
    console.log(`Usage: node eval/harness.js [--tag NAME] [--filter STR] [--base URL] [--threshold 0..1] [--max N]`);
}

// ── Question loader ───────────────────────────────────────────────────────
function loadQuestions(filterStr) {
    if (!fs.existsSync(QUESTIONS_PATH)) {
        console.error(`✗ no questions at ${QUESTIONS_PATH}`);
        process.exit(2);
    }
    const lines = fs.readFileSync(QUESTIONS_PATH, "utf8").split("\n").filter(l => l.trim());
    const questions = [];
    for (const line of lines) {
        try {
            const q = JSON.parse(line);
            if (filterStr) {
                const hay = (q.id + " " + q.category + " " + q.text).toLowerCase();
                if (!hay.includes(filterStr.toLowerCase())) continue;
            }
            questions.push(q);
        } catch (e) {
            console.warn(`! skipped malformed line: ${line.slice(0, 60)}…`);
        }
    }
    return questions;
}

// ── Server caller ─────────────────────────────────────────────────────────
async function askServer(base, q, sessionId) {
    const t0 = Date.now();
    const resp = await fetch(`${base}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            sessionId,
            message: q.text,
            lang: q.lang || "hi-IN",
        }),
    });
    const latencyMs = Date.now() - t0;
    if (!resp.ok) {
        const txt = await resp.text().catch(() => "");
        return { ok: false, status: resp.status, error: txt.slice(0, 300), latencyMs };
    }
    const body = await resp.json();
    return { ok: true, body, latencyMs };
}

// ── Score one question ────────────────────────────────────────────────────
function scoreOne(q, answer) {
    const results = {};
    let passed = true;
    for (const check of checks) {
        const r = check.run(answer, q.expect || {}, q);
        results[check.name] = r;
        if (!r.passed) passed = false;
    }
    return { passed, results };
}

function isSafetyQuestion(q) {
    return q.category === "distress" || q.expect?.is_critical === true;
}

// ── Pretty-print ──────────────────────────────────────────────────────────
function colour(text, code) {
    return process.stdout.isTTY ? `\x1b[${code}m${text}\x1b[0m` : text;
}
const green = s => colour(s, "32");
const red = s => colour(s, "31");
const yellow = s => colour(s, "33");
const dim = s => colour(s, "2");

function summarizeFailure(res) {
    const r = res.results;
    const bits = [];
    if (!r.citations.passed) bits.push(`cites missing: ${r.citations.missing.join(", ")}`);
    if (!r.banned_phrases.passed) bits.push(`banned: ${r.banned_phrases.violations.map(v => v.phrase).join(", ")}`);
    if (!r.helplines.passed) bits.push(`helplines missing: ${(r.helplines.missing || []).join(", ")}${r.helplines.reason ? " (" + r.helplines.reason + ")" : ""}`);
    if (!r.grounding.passed) {
        const g = r.grounding;
        if (g.missing_terms.length) bits.push(`terms missing: ${g.missing_terms.join(", ")}`);
        if (g.over_budget_by) bits.push(`over budget by ${g.over_budget_by} words`);
    }
    if (!r.language.passed) bits.push(`wrong lang (${r.language.lang}, ratios=${JSON.stringify(r.language.ratios).slice(0, 80)})`);
    return bits.join(" | ");
}

// ── Main ──────────────────────────────────────────────────────────────────
async function main() {
    const args = parseArgs(process.argv);
    const all = loadQuestions(args.filter);
    const questions = args.max > 0 ? all.slice(0, args.max) : all;
    if (questions.length === 0) {
        console.error("✗ no questions to run");
        process.exit(2);
    }

    // Probe server
    try {
        const probe = await fetch(args.base + "/", { method: "GET" });
        if (!probe.ok && probe.status !== 404) throw new Error(`probe ${probe.status}`);
    } catch (e) {
        console.error(`✗ server unreachable at ${args.base}: ${e.message}`);
        console.error(`  start it with: node server.js`);
        process.exit(2);
    }

    console.log(`\n${dim("eval/harness")}  tag=${args.tag}  base=${args.base}  questions=${questions.length}\n`);

    const startedAt = new Date().toISOString();
    const out = [];
    let pass = 0, fail = 0, safetyFail = 0;

    for (const [i, q] of questions.entries()) {
        const sessionId = `eval_${args.tag}_${q.id}_${Date.now()}`;
        let answer = "";
        let serverResult;
        try {
            serverResult = await askServer(args.base, q, sessionId);
            if (!serverResult.ok) {
                console.log(`${red("✗")} [${i + 1}/${questions.length}] ${q.id} — server error ${serverResult.status}: ${serverResult.error.slice(0, 80)}`);
                fail++;
                if (isSafetyQuestion(q)) safetyFail++;
                out.push({ id: q.id, category: q.category, passed: false, server_error: serverResult.error, latency_ms: serverResult.latencyMs });
                continue;
            }
            answer = serverResult.body.reply || "";
        } catch (e) {
            console.log(`${red("✗")} [${i + 1}/${questions.length}] ${q.id} — exception: ${e.message}`);
            fail++;
            if (isSafetyQuestion(q)) safetyFail++;
            out.push({ id: q.id, category: q.category, passed: false, exception: e.message });
            continue;
        }

        const score = scoreOne(q, answer);
        const safety = isSafetyQuestion(q);

        if (score.passed) {
            pass++;
            console.log(`${green("✓")} [${i + 1}/${questions.length}] ${q.id} ${dim("(" + serverResult.latencyMs + "ms)")} ${safety ? yellow("[safety]") : ""}`);
        } else {
            fail++;
            if (safety) safetyFail++;
            const summary = summarizeFailure(score);
            console.log(`${red("✗")} [${i + 1}/${questions.length}] ${q.id} ${dim("(" + serverResult.latencyMs + "ms)")} ${safety ? yellow("[SAFETY]") : ""} — ${summary}`);
        }
        out.push({
            id: q.id,
            category: q.category,
            lang: q.lang,
            text: q.text,
            answer,
            passed: score.passed,
            checks: score.results,
            latency_ms: serverResult.latencyMs,
            model: serverResult.body.model,
        });
    }

    const total = pass + fail;
    const passRate = total > 0 ? pass / total : 0;

    // Categorical breakdown
    const byCategory = {};
    for (const r of out) {
        const c = r.category || "unknown";
        if (!byCategory[c]) byCategory[c] = { pass: 0, fail: 0 };
        r.passed ? byCategory[c].pass++ : byCategory[c].fail++;
    }

    // Build report
    const report = {
        tag: args.tag,
        started_at: startedAt,
        finished_at: new Date().toISOString(),
        base: args.base,
        questions_path: path.relative(ROOT, QUESTIONS_PATH),
        total, pass, fail,
        pass_rate: +passRate.toFixed(4),
        safety_fail_count: safetyFail,
        by_category: byCategory,
        threshold: args.threshold || null,
        questions: out,
    };

    fs.mkdirSync(REPORTS_DIR, { recursive: true });
    const reportPath = path.join(REPORTS_DIR, `${args.tag}.json`);
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

    // ── Summary ───────────────────────────────────────────────────────────
    console.log(`\n${dim("─".repeat(60))}`);
    console.log(`${dim("Pass rate:")}     ${pass}/${total} = ${(passRate * 100).toFixed(1)}%`);
    console.log(`${dim("Safety fails:")}  ${safetyFail === 0 ? green("0") : red(String(safetyFail))}`);
    console.log(`${dim("By category:")}`);
    for (const [cat, counts] of Object.entries(byCategory).sort()) {
        const t = counts.pass + counts.fail;
        const r = t > 0 ? counts.pass / t : 0;
        const dot = r === 1 ? green("●") : r >= 0.6 ? yellow("●") : red("●");
        console.log(`  ${dot} ${cat.padEnd(16)} ${counts.pass}/${t} = ${(r * 100).toFixed(0)}%`);
    }
    console.log(`${dim("Report:")}        ${path.relative(ROOT, reportPath)}`);
    console.log(`${dim("─".repeat(60))}\n`);

    // ── Block rules ───────────────────────────────────────────────────────
    if (safetyFail > 0) {
        console.log(red(`✗ HARD FAIL: ${safetyFail} safety question(s) failed. Safety subset must be 100%.`));
        process.exit(1);
    }
    if (args.threshold > 0 && passRate < args.threshold) {
        console.log(red(`✗ HARD FAIL: ${(passRate * 100).toFixed(1)}% < ${(args.threshold * 100).toFixed(0)}% threshold.`));
        process.exit(1);
    }
    process.exit(0);
}

main().catch(e => {
    console.error("eval crashed:", e);
    process.exit(2);
});
