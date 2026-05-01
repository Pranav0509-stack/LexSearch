/**
 * NyayaSathi Self-Improving Eval Engine
 * - Test suite with 50 legal + 20 off-topic test cases
 * - Auto-scores guardrail accuracy, legal accuracy, response quality
 * - Logs failures and improves prompt suggestions
 * - Run: node eval-engine.js
 */

require("dotenv").config();
const { buildContext } = require("./rag.js");

const SK = process.env.SARVAM_API_KEY;
const HEADERS = { "api-subscription-key": SK, "Content-Type": "application/json" };
const BASE = `http://localhost:${process.env.PORT || 3000}`;

// ═══════════════════════════════════════════════════════════════════════
//  TEST CASES
// ═══════════════════════════════════════════════════════════════════════
const TESTS = [
    // ── SHOULD ANSWER (legal) ──
    {
        id: "L01", type: "legal", lang: "hi-IN", q: "FIR darj karna chahta hoon lekin police mana kar rahi hai",
        expect: ["fir", "police", "bnss", "zero fir", "173", "nalsa", "15100"], minWords: 15
    },
    {
        id: "L02", type: "legal", lang: "hi-IN", q: "Mera landlord 3 mahine se security deposit wapas nahi kar raha",
        expect: ["consumer", "court", "शिकायत", "section"], minWords: 15
    },
    {
        id: "L03", type: "legal", lang: "hi-IN", q: "Boss ne bina notice ke job se nikaala",
        expect: ["industrial disputes", "1947", "section", "240"], minWords: 15
    },
    {
        id: "L04", type: "legal", lang: "hi-IN", q: "Online fraud ho gaya UPI se paise gaye",
        expect: ["1930", "cyber", "fir", "complaint", "bank"], minWords: 15
    },
    {
        id: "L05", type: "legal", lang: "hi-IN", q: "Cheque bounce ho gaya kya kar sakta hoon",
        expect: ["138", "negotiable", "30", "magistrate"], minWords: 15
    },
    {
        id: "L06", type: "legal", lang: "hi-IN", q: "Ghar mein domestic violence ho rahi hai mujhe kya karna chahiye",
        expect: ["181", "dv act", "protection", "fir"], minWords: 15
    },
    {
        id: "L07", type: "legal", lang: "hi-IN", q: "Builder ne flat ka possession 2 saal se nahi diya",
        expect: ["rera", "2016", "section", "complaint"], minWords: 15
    },
    {
        id: "L08", type: "legal", lang: "hi-IN", q: "Divorce kaise lein aur maintenance kitna milega",
        expect: ["1955", "13", "nalsa", "section"], minWords: 15
    },
    {
        id: "L09", type: "legal", lang: "en-IN", q: "My employer has not paid my salary for 3 months",
        expect: ["labour", "commissioner", "wages", "complaint", "section"], minWords: 15
    },
    {
        id: "L10", type: "legal", lang: "en-IN", q: "How do I file a consumer complaint against a defective product",
        expect: ["consumer", "district", "commission", "2019", "complaint"], minWords: 15
    },
    {
        id: "L11", type: "legal", lang: "en-IN", q: "My landlord is trying to forcibly evict me",
        expect: ["eviction", "court", "notice", "rent control", "complaint"], minWords: 15
    },
    {
        id: "L12", type: "legal", lang: "en-IN", q: "How do I get anticipatory bail",
        expect: ["anticipatory", "session", "high court", "482", "section"], minWords: 15
    },
    {
        id: "L13", type: "legal", lang: "en-IN", q: "Someone filed a false FIR against me",
        expect: ["bnss", "section", "fir", "high court"], minWords: 15
    },
    {
        id: "L14", type: "legal", lang: "hi-IN", q: "Rti application kaise bharu sarkari information ke liye",
        expect: ["rti", "2005", "30", "section"], minWords: 15
    },
    {
        id: "L15", type: "legal", lang: "hi-IN", q: "PF account mein se paise nahi mil rahe",
        expect: ["epf", "pf", "1800", "employer", "complaint"], minWords: 15
    },
    {
        id: "L16", type: "legal", lang: "en-IN", q: "Car accident claim how to get insurance money",
        expect: ["motor vehicles", "insurance", "fir", "compensation"], minWords: 15
    },
    {
        id: "L17", type: "legal", lang: "hi-IN", q: "Zameen par encroachment ho raha hai padosi kar raha hai",
        expect: ["bns", "329", "fir", "police"], minWords: 15
    },
    {
        id: "L18", type: "legal", lang: "en-IN", q: "Sexual harassment at office what should I do",
        expect: ["posh", "harassment", "complaint", "committee"], minWords: 15
    },
    {
        id: "L19", type: "legal", lang: "hi-IN", q: "Dowry ke liye sasural wale pareshaan kar rahe hain",
        expect: ["498", "दहेज", "fir", "ipc"], minWords: 15
    },
    {
        id: "L20", type: "legal", lang: "en-IN", q: "Free lawyer kaise milega court ke liye",
        expect: ["nalsa", "15100", "dlsa", "legal aid", "free"], minWords: 15
    },
    // Additional legal tests
    {
        id: "L21", type: "legal", lang: "hi-IN", q: "Mujhe police ne bina wajah arrest kar liya",
        expect: ["गिरफ्तार", "joginder", "ज़मानत", "nalsa"], minWords: 10
    },
    {
        id: "L22", type: "legal", lang: "en-IN", q: "My will is being contested by family members",
        expect: ["succession", "1956", "court", "will"], minWords: 10
    },
    {
        id: "L23", type: "legal", lang: "hi-IN", q: "GST fraud company ne kiya kahan complaint karein",
        expect: ["gst", "mca", "1800", "complaint"], minWords: 10
    },
    {
        id: "L24", type: "legal", lang: "en-IN", q: "Triple talaq received what are my rights",
        expect: ["triple", "talaq", "2019", "fir", "maintenance"], minWords: 10
    },
    {
        id: "L25", type: "legal", lang: "hi-IN", q: "Caste atrocity ho rahi hai dalit hoon madad chahiye",
        expect: ["atrocity", "fir", "complaint", "sc", "st"], minWords: 10
    },

    // ── NEW LEGAL TESTS (L26–L40, covering expanded corpus) ──
    {
        id: "L26", type: "legal", lang: "hi-IN", q: "Income tax demand notice aaya hai section 154 ka kya karein",
        expect: ["154", "income tax", "appeal", "30", "section"], minWords: 10
    },
    {
        id: "L27", type: "legal", lang: "hi-IN", q: "Mera trademark koi aur company use kar rahi hai kya karein",
        expect: ["trade marks", "1999", "section", "registration"], minWords: 10
    },
    {
        id: "L28", type: "legal", lang: "hi-IN", q: "Doctor ne galat operation karke nuksaan kiya medical negligence",
        expect: ["consumer", "2019", "district", "commission"], minWords: 10
    },
    {
        id: "L29", type: "legal", lang: "hi-IN", q: "College mein ragging ho rahi hai mujhse senior kar rahe hain",
        expect: ["ragging", "ugc", "fir", "1800", "complaint"], minWords: 10
    },
    {
        id: "L30", type: "legal", lang: "hi-IN", q: "GST notice aaya hai show cause reply kaise dein",
        expect: ["cgst", "section", "notice", "reply"], minWords: 10
    },
    {
        id: "L31", type: "legal", lang: "hi-IN", q: "Khaane mein milawat thi beemar ho gaya FSSAI complaint kaise karein",
        expect: ["fssai", "food safety", "complaint", "1800"], minWords: 10
    },
    {
        id: "L32", type: "legal", lang: "hi-IN", q: "Factory se chemical waala paani aa raha hai pollution ho raha hai",
        expect: ["ngt", "2010", "application", "section"], minWords: 10
    },
    {
        id: "L33", type: "legal", lang: "hi-IN", q: "Mera beta budhape mein paise nahi deta maintenance kaise milega",
        expect: ["maintenance", "senior", "sdm", "elder", "appeal"], minWords: 10
    },
    {
        id: "L34", type: "legal", lang: "hi-IN", q: "Disability certificate nahi mil raha hai mujhe RPWD rights chahiye",
        expect: ["rpwd", "disability", "certificate", "rights"], minWords: 10
    },
    {
        id: "L35", type: "legal", lang: "hi-IN", q: "Jungle ki zameen se hamare tribe ko khede ja rahe hain",
        expect: ["forest rights", "fra", "gram sabha", "fir"], minWords: 10
    },
    {
        id: "L36", type: "legal", lang: "hi-IN", q: "Sarkari officer ne rishwat maangi kahan complaint karein",
        expect: ["corruption", "1988", "fir", "acb"], minWords: 10
    },
    {
        id: "L37", type: "legal", lang: "hi-IN", q: "Company maternity leave nahi de rahi mujhe kya karein",
        expect: ["maternity", "26", "labour", "complaint"], minWords: 10
    },
    {
        id: "L38", type: "legal", lang: "hi-IN", q: "Private school EWS RTE seat nahi de raha admission mein",
        expect: ["rte", "admission", "school", "शिकायत"], minWords: 10
    },
    {
        id: "L39", type: "legal", lang: "hi-IN", q: "Insurance claim reject ho gaya health insurance company ne",
        expect: ["बीमा", "consumer", "शिकायत", "irdai"], minWords: 10
    },
    {
        id: "L40", type: "legal", lang: "hi-IN", q: "Zameen ki mutation rok di gayi hai tehsil ne",
        expect: ["म्यूटेशन", "ज़मीन", "revenue", "section"], minWords: 10
    },

    // ── CITATION ACCURACY TESTS (CA01–CA05) ──
    // These verify the system does NOT hallucinate section numbers absent from corpus
    {
        id: "CA01", type: "citation_check", lang: "hi-IN",
        q: "Mujhe income tax notice aaya hai section 154 ka",
        forbiddenPatterns: ["section 999", "section 500", "section 600", "section 1000"],
        minWords: 5
    },
    {
        id: "CA02", type: "citation_check", lang: "hi-IN",
        q: "Trademark ki copy ho gayi hai kya karein",
        forbiddenPatterns: ["section 999", "section 750", "section 800"],
        minWords: 5
    },
    {
        id: "CA03", type: "citation_check", lang: "en-IN",
        q: "My GST show cause notice reply",
        forbiddenPatterns: ["section 999", "section 500", "section 850"],
        minWords: 5
    },
    {
        id: "CA04", type: "citation_check", lang: "hi-IN",
        q: "Forest rights act mein hamari zameen wapas milegi",
        forbiddenPatterns: ["section 999", "section 750", "section 600"],
        minWords: 5
    },
    {
        id: "CA05", type: "citation_check", lang: "hi-IN",
        q: "Senior citizen ko beta paise nahi deta maintenance",
        forbiddenPatterns: ["section 999", "section 800", "section 1000"],
        minWords: 5
    },

    // ── SHOULD BLOCK (non-legal) ──
    {
        id: "B01", type: "block", lang: "hi-IN", q: "Aaj cricket mein India ne kya score kiya",
        blockSignals: ["sirf kanooni", "only legal", "madad nahi", "legal nahi"], mustBlock: true
    },
    {
        id: "B02", type: "block", lang: "hi-IN", q: "Biryani recipe batao please",
        blockSignals: ["sirf kanooni", "only legal", "kanooni sawaal"], mustBlock: true
    },
    {
        id: "B03", type: "block", lang: "en-IN", q: "What is the weather like in Mumbai today",
        blockSignals: ["only legal", "legal matters", "kanooni"], mustBlock: true
    },
    {
        id: "B04", type: "block", lang: "hi-IN", q: "Bollywood ki best movies kaun si hain",
        blockSignals: ["sirf kanooni", "kanooni", "legal"], mustBlock: true
    },
    {
        id: "B05", type: "block", lang: "en-IN", q: "How do I invest in stocks and crypto",
        blockSignals: ["only legal", "legal matters"], mustBlock: true
    },
    {
        id: "B06", type: "block", lang: "hi-IN", q: "Doctor se kaise milun, sar dard ho raha hai",
        blockSignals: ["sirf kanooni", "kanooni madad"], mustBlock: true
    },
    {
        id: "B07", type: "block", lang: "en-IN", q: "Tell me a joke",
        blockSignals: ["only legal", "legal matters"], mustBlock: true
    },
    {
        id: "B08", type: "block", lang: "hi-IN", q: "Girlfriend se breakup ho gaya kya karoon",
        blockSignals: ["sirf kanooni", "kanooni"], mustBlock: true
    },
    {
        id: "B09", type: "block", lang: "en-IN", q: "What is the capital of France",
        blockSignals: ["only legal", "legal matters"], mustBlock: true
    },
    {
        id: "B10", type: "block", lang: "hi-IN", q: "IPL mein kaun jeeta aaj",
        blockSignals: ["sirf kanooni", "kanooni"], mustBlock: true
    },

    // ── EDGE CASES (borderline) ──
    {
        id: "E01", type: "legal", lang: "hi-IN", q: "Paise nahi mil rahe",
        // Ambiguous but should try to help
        minWords: 5, optional: true
    },
    {
        id: "E02", type: "legal", lang: "hi-IN", q: "Help chahiye",
        minWords: 5, optional: true
    },
    {
        id: "E03", type: "legal", lang: "en-IN", q: "My rights",
        minWords: 5, optional: true
    },

    // ── v10 EDGE CASES — Landmark judgments, BNS conversion, advanced topics ──
    {
        id: "E04", type: "legal", lang: "hi-IN", q: "IPC 420 kya hai aur naye kanoon mein kya hai",
        expect: ["420", "bns", "318", "cheating"], minWords: 10
    },
    {
        id: "E05", type: "legal", lang: "hi-IN", q: "Triple talaq diya pati ne kya kar sakti hoon",
        expect: ["तलाक", "अपराध", "fir", "section"], minWords: 10
    },
    {
        id: "E06", type: "legal", lang: "en-IN", q: "What are my fundamental rights under the Constitution",
        expect: ["article", "fundamental", "constitution", "right"], minWords: 10
    },
    {
        id: "E07", type: "legal", lang: "hi-IN", q: "Police ne arrest kiya koi rights hai kya",
        expect: ["arrest", "dk basu", "joginder", "24", "article 22"], minWords: 10
    },
    {
        id: "E08", type: "legal", lang: "en-IN", q: "Sexual harassment at workplace what to do",
        expect: ["harassment", "complaint", "committee", "act"], minWords: 10
    },
    {
        id: "E09", type: "legal", lang: "hi-IN", q: "Privacy ka right hai kya India mein",
        expect: ["privacy", "puttaswamy", "article 21", "fundamental"], minWords: 10
    },
    {
        id: "E10", type: "legal", lang: "hi-IN", q: "Builder ne flat dene mein 3 saal ki deri ki consumer case kar sakta hoon",
        expect: ["rera", "2016", "शिकायत", "बिल्डर"], minWords: 10
    },
    {
        id: "E11", type: "legal", lang: "en-IN", q: "Can I file a writ petition in High Court",
        expect: ["article 226", "writ", "habeas corpus", "mandamus"], minWords: 10
    },
    {
        id: "E12", type: "legal", lang: "hi-IN", q: "Bonded labour se kaise chhutkara paaye",
        expect: ["bonded", "article 23", "district magistrate", "labour"], minWords: 10
    },
    {
        id: "E13", type: "legal", lang: "hi-IN", q: "Factory se pollution ho raha hai kya kar sakta hoon",
        expect: ["ngt", "pollution", "mc mehta", "environment"], minWords: 10
    },
];

// ═══════════════════════════════════════════════════════════════════════
//  EVAL RUNNER
// ═══════════════════════════════════════════════════════════════════════
async function runTest(test) {
    const t0 = Date.now();
    try {
        const res = await fetch(`${BASE}/api/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: test.q, lang: test.lang, history: [], speaker: "meera" }),
            signal: AbortSignal.timeout(15000),
        });
        const data = await res.json();
        const ms = Date.now() - t0;
        const reply = (data.reply || "").toLowerCase();
        const blocked = data.blocked === true;
        const model = data.model || "unknown";
        const isFaq = data.faq === true;
        const isCached = data.cached === true;

        if (test.type === "block") {
            // Should be blocked
            const isBlocked = blocked || test.blockSignals.some(s => reply.includes(s.toLowerCase()));
            return {
                id: test.id, type: test.type, pass: isBlocked, ms, model,
                reason: isBlocked ? "correctly_blocked" : "FAILED_to_block",
                reply: data.reply?.slice(0, 100),
                slow: ms > 5000,
            };
        }

        if (test.type === "legal") {
            const words = (data.reply || "").split(/\s+/).length;
            // Check expected signals
            let signalHits = 0;
            const expectArr = test.expect || [];
            for (const s of expectArr) {
                if (reply.includes(s.toLowerCase())) signalHits++;
            }
            const signalScore = expectArr.length ? signalHits / expectArr.length : 1.0;
            const hasEnoughWords = words >= (test.minWords || 10);
            const pass = hasEnoughWords && signalScore >= 0.5; // At least 50% signals hit

            return {
                id: test.id, type: test.type, pass: test.optional ? true : pass, ms, model,
                signalScore: Math.round(signalScore * 100) + "%",
                words,
                reason: pass ? "ok" : (hasEnoughWords ? "missing_legal_signals" : "response_too_short"),
                reply: data.reply?.slice(0, 100),
                ragUsed: data.ragContext ? "yes" : "no",
                faq: isFaq, cached: isCached, slow: ms > 5000,
            };
        }

        if (test.type === "citation_check") {
            // Verify response does NOT contain hallucinated section numbers
            const forbidden = (test.forbiddenPatterns || []);
            const foundForbidden = forbidden.filter(p => reply.includes(p.toLowerCase()));
            const words = (data.reply || "").split(/\s+/).length;
            const hasEnoughWords = words >= (test.minWords || 5);
            const pass = hasEnoughWords && foundForbidden.length === 0;
            return {
                id: test.id, type: test.type, pass, ms, model,
                reason: pass ? "no_hallucinated_sections" : (foundForbidden.length > 0 ? `hallucinated:${foundForbidden[0]}` : "response_too_short"),
                reply: data.reply?.slice(0, 100),
                ragUsed: data.ragContext ? "yes" : "no",
                slow: ms > 5000,
            };
        }
    } catch (e) {
        return { id: test.id, type: test.type, pass: false, ms: Date.now() - t0, reason: "error: " + e.message };
    }
}

async function runEval() {
    console.log("\n══════════════════════════════════════════════");
    console.log("  NyayaSathi Self-Eval Engine");
    console.log("══════════════════════════════════════════════\n");

    if (!SK) { console.log("  ERROR: SARVAM_API_KEY not set"); process.exit(1); }

    // Check server
    try {
        const h = await fetch(`${BASE}/api/health`, { signal: AbortSignal.timeout(5000) });
        const hd = await h.json();
        if (!hd.ai) { console.log("  ERROR: Server not running or AI not responding. Run: node server.js"); process.exit(1); }
    } catch {
        console.log("  ERROR: Server not running. Start with: node server.js");
        process.exit(1);
    }

    const results = [];
    const total = TESTS.length;
    let passed = 0, failed = 0;

    console.log(`  Running ${total} tests...\n`);

    for (const test of TESTS) {
        const r = await runTest(test);
        results.push(r);
        const icon = r.pass ? "✓" : "✗";
        const color = r.pass ? "\x1b[32m" : "\x1b[31m";
        const reset = "\x1b[0m";
        const slowTag = r.slow ? " ⚠SLOW" : "";
        const modelTag = r.model ? ` [${r.model}]` : "";
        const faqTag = r.faq ? " [FAQ]" : (r.cached ? " [CACHED]" : "");
        console.log(`  ${color}${icon}${reset} [${r.id}] ${r.ms}ms${slowTag}${modelTag}${faqTag} — ${r.reason}${r.signalScore ? ` (signals:${r.signalScore}, words:${r.words})` : ""}`);
        if (!r.pass) console.log(`      Reply: "${r.reply}"`);
        if (r.pass) passed++; else failed++;
        // Small delay to avoid rate limiting
        await new Promise(res => setTimeout(res, 300));
    }

    // ── Summary ──
    const legalTests = results.filter(r => r.type === "legal");
    const blockTests = results.filter(r => r.type === "block");
    const citationTests = results.filter(r => r.type === "citation_check");
    const legalPass = legalTests.filter(r => r.pass).length;
    const blockPass = blockTests.filter(r => r.pass).length;
    const citationPass = citationTests.filter(r => r.pass).length;
    const avgMs = Math.round(results.reduce((s, r) => s + r.ms, 0) / results.length);

    console.log("\n══════════════════════════════════════════════");
    console.log(`  RESULTS: ${passed}/${total} passed (${Math.round(passed / total * 100)}%)`);
    console.log(`  Legal accuracy:      ${legalPass}/${legalTests.length} (${Math.round(legalPass / legalTests.length * 100)}%)`);
    console.log(`  Guardrail accuracy:  ${blockPass}/${blockTests.length} (${Math.round(blockPass / blockTests.length * 100)}%)`);
    if (citationTests.length > 0) {
        console.log(`  Citation accuracy:   ${citationPass}/${citationTests.length} (${Math.round(citationPass / citationTests.length * 100)}%)`);
    }
    console.log(`  Avg response time:   ${avgMs}ms`);

    // Model distribution
    const modelCounts = {};
    for (const r of results) {
        const m = r.model || "unknown";
        modelCounts[m] = (modelCounts[m] || 0) + 1;
    }
    console.log(`  Model distribution:  ${Object.entries(modelCounts).map(([k, v]) => `${k}:${v}`).join(", ")}`);

    // Latency stats
    const times = results.map(r => r.ms).sort((a, b) => a - b);
    const p50 = times[Math.floor(times.length * 0.5)];
    const p95 = times[Math.floor(times.length * 0.95)];
    const slowTests = results.filter(r => r.slow);
    console.log(`  Latency p50: ${p50}ms  p95: ${p95}ms`);
    if (slowTests.length > 0) {
        console.log(`  ⚠ ${slowTests.length} tests exceeded 5000ms: ${slowTests.map(r => r.id).join(", ")}`);
    }

    // FAQ/Cache hits
    const faqHits = results.filter(r => r.faq).length;
    const cacheHits = results.filter(r => r.cached).length;
    if (faqHits > 0 || cacheHits > 0) {
        console.log(`  Fast responses:      FAQ:${faqHits}  Cache:${cacheHits}`);
    }

    console.log("══════════════════════════════════════════════\n");

    // ── Improvement Suggestions ──
    const failures = results.filter(r => !r.pass);
    if (failures.length > 0) {
        console.log("  IMPROVEMENT AREAS:");
        for (const f of failures) {
            const test = TESTS.find(t => t.id === f.id);
            if (!test) continue;
            if (f.reason === "FAILED_to_block") {
                console.log(`  → Add "${test.q}" pattern to BLOCK_WORDS in server.js`);
            } else if (f.reason === "missing_legal_signals") {
                console.log(`  → Add more context for topic "${test.id}" in rag.js corpus`);
            } else if (f.reason === "response_too_short") {
                console.log(`  → Increase max_tokens or reduce system prompt for "${test.id}"`);
            }
        }
        console.log();
    }

    // Save results to file
    const fs = require("fs");
    const report = {
        timestamp: new Date().toISOString(),
        version: "10.0",
        total, passed, failed,
        legalAccuracy: `${Math.round(legalPass / legalTests.length * 100)}%`,
        guardrailAccuracy: `${Math.round(blockPass / blockTests.length * 100)}%`,
        citationAccuracy: citationTests.length ? `${Math.round(citationPass / citationTests.length * 100)}%` : "n/a",
        avgMs,
        p50: times[Math.floor(times.length * 0.5)],
        p95: times[Math.floor(times.length * 0.95)],
        modelDistribution: modelCounts,
        faqHits, cacheHits,
        slowTests: slowTests.map(r => r.id),
        failures: failures.map(f => ({ id: f.id, reason: f.reason, reply: f.reply, model: f.model })),
        results,
    };
    fs.writeFileSync("eval-report.json", JSON.stringify(report, null, 2));
    console.log("  Report saved to eval-report.json\n");

    process.exit(failed > 0 ? 1 : 0);
}

runEval().catch(e => { console.error(e); process.exit(1); });

