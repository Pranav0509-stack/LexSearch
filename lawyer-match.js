/**
 * lawyer-match.js — score and rank lawyers against a case.
 *
 * Score components (additive, max ~10 points):
 *   specialization match  : +4 (case.type ∈ lawyer.specializations)
 *                           +1 per *related* specialization match
 *   language match        : +2 (case.lang ∈ lawyer.languages)
 *   location match        : +2 same city, +1 same state
 *   rating                : +0..1 (rating - 4.0)
 *   experience            : +0..1 (years_experience / 20, capped at 1)
 *   gender preference     : +1 if a female lawyer is preferred (DV/divorce
 *                            cases default to female-preferred)
 *   availability          : hard filter (unavailable lawyers dropped)
 *
 * Future hooks (not implemented yet — out of scope for this phase):
 *   - capacity-aware scoring (lawyer's case load this week)
 *   - rank decay if recently matched but no follow-up
 *   - explicit user preference (gender, language, fee)
 */

const path = require("path");
const fs = require("fs");

const SEED_PATH = path.join(__dirname, "lawyers.json");

let _lawyersCache = null;
function loadLawyers() {
    if (_lawyersCache) return _lawyersCache;
    try {
        const raw = fs.readFileSync(SEED_PATH, "utf8");
        const data = JSON.parse(raw);
        _lawyersCache = Array.isArray(data) ? data : (data.lawyers || []);
    } catch (e) {
        console.warn("[lawyer-match] failed to load lawyers.json:", e.message);
        _lawyersCache = [];
    }
    return _lawyersCache;
}

// City → state index, used so a Pune case prefers Mumbai over Delhi.
// Hand-coded for the seed cities; trivial to extend later.
const CITY_TO_STATE = {
    "pune": "MH", "mumbai": "MH", "thane": "MH", "nagpur": "MH",
    "delhi": "DL", "noida": "UP", "gurgaon": "HR", "gurugram": "HR",
    "bengaluru": "KA", "bangalore": "KA", "mysuru": "KA",
    "patna": "BR", "gaya": "BR",
    "hyderabad": "TS", "secunderabad": "TS",
    "jaipur": "RJ", "udaipur": "RJ",
    "chennai": "TN", "coimbatore": "TN",
    "kolkata": "WB", "howrah": "WB",
    "lucknow": "UP", "kanpur": "UP", "varanasi": "UP",
};

function inferState(loc) {
    if (!loc) return null;
    const k = loc.toLowerCase().trim();
    if (CITY_TO_STATE[k]) return CITY_TO_STATE[k];
    // Already a state code?
    if (/^[A-Z]{2}$/.test(loc.trim())) return loc.trim();
    return null;
}

// Case types that map onto each other — used for the "related specialization"
// bonus. Light touch: if the case is "fraud", a "consumer" lawyer is a
// reasonable fallback, but get half-credit only.
const RELATED = {
    accident: ["vehicle", "consumer"],
    vehicle: ["accident"],
    bail: ["false_case", "fir"],
    false_case: ["bail", "fir"],
    fir: ["false_case", "bail"],
    fraud: ["consumer", "cheque_bounce"],
    consumer: ["fraud"],
    cheque_bounce: ["consumer", "fraud"],
    dv: ["divorce"],
    divorce: ["dv"],
    land: ["property", "panchayat"],
    property: ["land", "consumer"],
    panchayat: ["land", "rti"],
    caste_atrocity: ["fir", "false_case"],
    salary: ["consumer"],
};

// Cases that default to a female-lawyer preference. The user can override
// this in the request body, but the default reduces friction.
const FEMALE_PREFERRED = new Set(["dv", "divorce"]);

function scoreLawyer(c, lawyer, opts = {}) {
    if (!lawyer.available) return null;
    let score = 0;
    const reasons = [];

    // Specialization
    if (lawyer.specializations?.includes(c.type)) {
        score += 4;
        reasons.push(`specialty:${c.type}`);
    } else if (RELATED[c.type]?.some(r => lawyer.specializations?.includes(r))) {
        score += 1.5;
        reasons.push("specialty:related");
    }

    // Language
    if (c.lang && lawyer.languages?.includes(c.lang)) {
        score += 2;
        reasons.push(`lang:${c.lang}`);
    }

    // Location — prefer city match, fall back to state.
    const caseLoc = c.entities?.location || c.entities?.city || null;
    if (caseLoc && lawyer.city && caseLoc.toLowerCase() === lawyer.city.toLowerCase()) {
        score += 2;
        reasons.push("city:match");
    } else {
        const caseState = inferState(caseLoc);
        if (caseState && caseState === lawyer.state) {
            score += 1;
            reasons.push(`state:${caseState}`);
        }
    }

    // Rating / experience
    const ratingBonus = Math.max(0, Math.min(1, (lawyer.rating || 4) - 4.0));
    score += ratingBonus;
    if (ratingBonus) reasons.push(`rating:+${ratingBonus.toFixed(1)}`);
    const expBonus = Math.min(1, (lawyer.years_experience || 0) / 20);
    score += expBonus;
    if (expBonus >= 0.3) reasons.push(`exp:+${expBonus.toFixed(1)}`);

    // Gender preference
    const wantFemale = opts.preferFemale ?? FEMALE_PREFERRED.has(c.type);
    if (wantFemale && lawyer.gender === "female") {
        score += 1;
        reasons.push("gender:female");
    }

    return { score: +score.toFixed(2), reasons };
}

/**
 * Public: rank lawyers for a case. Returns up to `limit` matches, sorted
 * by score descending. Each entry: { lawyer (safe subset), score, reasons }.
 */
function matchLawyers(c, opts = {}) {
    if (!c) return [];
    const limit = Number.isFinite(opts.limit) ? opts.limit : 3;
    const lawyers = loadLawyers();
    const scored = [];
    for (const lawyer of lawyers) {
        const r = scoreLawyer(c, lawyer, opts);
        if (!r) continue;
        if (r.score < 1) continue; // skip nothing-matched entries
        // Build a safe public view of the lawyer — strip phone/email until
        // the user explicitly accepts the handoff.
        const safe = {
            id: lawyer.id,
            name: lawyer.name,
            city: lawyer.city,
            state: lawyer.state,
            specializations: lawyer.specializations,
            languages: lawyer.languages,
            rating: lawyer.rating,
            years_experience: lawyer.years_experience,
            fee_first_consult: lawyer.fee_first_consult,
            gender: lawyer.gender,
        };
        scored.push({ lawyer: safe, score: r.score, reasons: r.reasons });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit);
}

/** Look up a lawyer's full record (including contact details) by id. */
function getLawyerById(id) {
    return loadLawyers().find(l => l.id === id) || null;
}

module.exports = { matchLawyers, getLawyerById, loadLawyers };
