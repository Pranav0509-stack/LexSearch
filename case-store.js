/**
 * case-store.js — file-backed persistence for legal cases and sessions.
 *
 * Why a flat-file store? At local-dev / early-prod scale (a few hundred cases
 * per day), one JSON file per case beats Postgres on dev simplicity, and reads
 * are still O(1) by ID. The phone→case index lives in a single JSON file that
 * we rewrite atomically. When we cross ~10k active cases, this gets swapped
 * for Postgres + Redis (the "Production hardening" plan).
 *
 * On-disk layout:
 *   data/cases/<caseId>.json       — full case dump (history, entities, summary, handoffs)
 *   data/sessions/<sessionId>.json — pointer { sessionId, caseId, lang, createdAt }
 *   data/phone-index.json          — { "<phoneE164>": ["caseId1", "caseId2", ...] }
 *
 * Phone numbers are normalized to E.164 (+91XXXXXXXXXX) before any lookup.
 * Two callers from the same number always resolve to the same chain of cases.
 */

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const DATA_DIR = path.join(__dirname, "data");
const CASES_DIR = path.join(DATA_DIR, "cases");
const SESSIONS_DIR = path.join(DATA_DIR, "sessions");
const PHONE_INDEX_PATH = path.join(DATA_DIR, "phone-index.json");

// Make sure dirs exist on import — first-run safety.
for (const d of [DATA_DIR, CASES_DIR, SESSIONS_DIR]) {
    try { fs.mkdirSync(d, { recursive: true }); } catch { }
}

// ───────────────────────────────────────────────────────────
//  ID + phone helpers
// ───────────────────────────────────────────────────────────

function newCaseId() {
    return "case_" + crypto.randomBytes(6).toString("hex") + "_" + Date.now().toString(36);
}

/**
 * Normalize an Indian phone number to E.164 (+91XXXXXXXXXX).
 * Accepts: "9876543210", "09876543210", "+919876543210", "919876543210", etc.
 * Returns null if the input doesn't look like an Indian mobile number.
 */
function normalizePhone(raw) {
    if (!raw || typeof raw !== "string") return null;
    let digits = raw.replace(/[^0-9]/g, "");
    if (!digits) return null;
    // Strip leading 0
    if (digits.startsWith("0")) digits = digits.slice(1);
    // Strip 91 country code if present and we still have 10 left
    if (digits.length === 12 && digits.startsWith("91")) digits = digits.slice(2);
    if (digits.length !== 10) return null;
    // Indian mobile numbers start with 6/7/8/9
    if (!/^[6-9]/.test(digits)) return null;
    return "+91" + digits;
}

// ───────────────────────────────────────────────────────────
//  Atomic JSON I/O
// ───────────────────────────────────────────────────────────

function readJsonSafe(filePath, fallback) {
    try {
        if (!fs.existsSync(filePath)) return fallback;
        const raw = fs.readFileSync(filePath, "utf8");
        if (!raw) return fallback;
        return JSON.parse(raw);
    } catch (e) {
        console.warn(`[case-store] read failed for ${path.basename(filePath)}: ${e.message}`);
        return fallback;
    }
}

function writeJsonAtomic(filePath, obj) {
    try {
        const tmp = filePath + ".tmp." + process.pid;
        fs.writeFileSync(tmp, JSON.stringify(obj, null, 2));
        fs.renameSync(tmp, filePath);
        return true;
    } catch (e) {
        console.warn(`[case-store] write failed for ${path.basename(filePath)}: ${e.message}`);
        return false;
    }
}

// ───────────────────────────────────────────────────────────
//  Case CRUD
// ───────────────────────────────────────────────────────────

/**
 * Build a fresh case object. The schema is intentionally rich — Phase C will
 * fill `entities`, Phase D writes `distress_level` + `urgency`, Phase E pushes
 * onto `lawyer_handoffs`. For now most fields stay empty defaults.
 */
function makeNewCase({ phone = null, lang = "hi-IN", sessionId = null } = {}) {
    return {
        id: newCaseId(),
        phone: phone ? normalizePhone(phone) : null,
        sessionIds: sessionId ? [sessionId] : [],
        lang,
        type: null,                  // bail | accident | dv | property | fraud | salary | fir | consumer | divorce | land | false_case | vehicle | unknown
        urgency: "medium",           // critical | high | medium | low
        distress_level: 0,           // 0..1
        status: "open",              // open | resolved | abandoned
        entities: {},                // free-form slot bag — Phase C populates
        facts: [],                   // ["bike accident at 8pm", "no FIR yet"]
        actions_advised: [],         // ["call 112", "file FIR", ...]
        next_steps_pending: [],      // unfinished asks
        needs_lawyer: false,
        lawyer_offered: false,
        lawyer_handoffs: [],         // Phase E appends { lawyerId, ts, status }
        summary: "",                 // running compressed summary
        history: [],                 // full transcript, last 30 turns
        createdAt: Date.now(),
        updatedAt: Date.now(),
        lastTurnAt: Date.now(),
    };
}

function casePath(caseId) {
    return path.join(CASES_DIR, `${caseId}.json`);
}

function loadCase(caseId) {
    if (!caseId) return null;
    return readJsonSafe(casePath(caseId), null);
}

function saveCase(c) {
    if (!c?.id) return false;
    c.updatedAt = Date.now();
    return writeJsonAtomic(casePath(c.id), c);
}

// ───────────────────────────────────────────────────────────
//  Session pointer (sessionId → caseId)
//  This is a thin lookup so multiple sessionIds (e.g. browser + a later
//  phone callback) can attach to the same case.
// ───────────────────────────────────────────────────────────

function sessionPath(sessionId) {
    // Sanitize sessionId so it's safe to use as a filename.
    const safe = String(sessionId).replace(/[^a-zA-Z0-9_-]/g, "_");
    return path.join(SESSIONS_DIR, `${safe}.json`);
}

function loadSessionPointer(sessionId) {
    if (!sessionId) return null;
    return readJsonSafe(sessionPath(sessionId), null);
}

function saveSessionPointer(sessionId, caseId, extra = {}) {
    if (!sessionId || !caseId) return false;
    return writeJsonAtomic(sessionPath(sessionId), {
        sessionId, caseId, ...extra, updatedAt: Date.now(),
    });
}

// ───────────────────────────────────────────────────────────
//  Phone index (phone → caseIds[])
// ───────────────────────────────────────────────────────────

let _phoneIndexCache = null;
function loadPhoneIndex() {
    if (_phoneIndexCache) return _phoneIndexCache;
    _phoneIndexCache = readJsonSafe(PHONE_INDEX_PATH, {});
    return _phoneIndexCache;
}

function savePhoneIndex() {
    if (!_phoneIndexCache) return;
    writeJsonAtomic(PHONE_INDEX_PATH, _phoneIndexCache);
}

function indexPhoneToCase(phone, caseId) {
    const e164 = normalizePhone(phone);
    if (!e164 || !caseId) return;
    const idx = loadPhoneIndex();
    if (!idx[e164]) idx[e164] = [];
    if (!idx[e164].includes(caseId)) {
        idx[e164].unshift(caseId);             // newest first
        if (idx[e164].length > 20) idx[e164] = idx[e164].slice(0, 20);
        savePhoneIndex();
    }
}

function findCasesByPhone(phone) {
    const e164 = normalizePhone(phone);
    if (!e164) return [];
    const idx = loadPhoneIndex();
    const ids = idx[e164] || [];
    return ids.map(id => loadCase(id)).filter(Boolean);
}

/**
 * Find the most recent open case for this phone, OR null. Used by the phone
 * flow on incoming call: if the same number called yesterday and we haven't
 * marked the case resolved, resume it.
 */
function findOpenCaseForPhone(phone) {
    const cases = findCasesByPhone(phone);
    return cases.find(c => c?.status === "open") || null;
}

// ───────────────────────────────────────────────────────────
//  Public attach helper used by server.js
// ───────────────────────────────────────────────────────────

/**
 * Resolve a session+phone pair to a case. Order of resolution:
 *   1. If sessionPointer(sessionId) exists → load that case (web refresh path).
 *   2. Else if phone has an open case → attach this session to it (callback).
 *   3. Else create a fresh case and bind both sessionId + phone to it.
 *
 * Always returns { case, isReturning }. `isReturning` lets the caller
 * personalize the greeting ("Welcome back — last time we discussed X").
 */
function attachSessionToCase({ sessionId, phone = null, lang = "hi-IN" } = {}) {
    // 1. Existing session pointer
    const ptr = loadSessionPointer(sessionId);
    if (ptr?.caseId) {
        const existing = loadCase(ptr.caseId);
        if (existing) return { case: existing, isReturning: false, attached: "session" };
    }

    // 2. Phone-based callback
    if (phone) {
        const open = findOpenCaseForPhone(phone);
        if (open) {
            if (!open.sessionIds.includes(sessionId)) open.sessionIds.push(sessionId);
            saveCase(open);
            saveSessionPointer(sessionId, open.id, { phone: normalizePhone(phone), lang });
            return { case: open, isReturning: true, attached: "phone-callback" };
        }
    }

    // 3. Fresh case
    const fresh = makeNewCase({ phone, lang, sessionId });
    saveCase(fresh);
    saveSessionPointer(sessionId, fresh.id, { phone: normalizePhone(phone), lang });
    if (fresh.phone) indexPhoneToCase(fresh.phone, fresh.id);
    return { case: fresh, isReturning: false, attached: "fresh" };
}

/**
 * Update the case after a turn completes. Keeps the in-session view and the
 * on-disk file in sync. We bound history at 30 turns on disk too — older
 * stuff is already compressed into `summary`.
 */
function persistTurn(c, { userMsg, assistantReply, model } = {}) {
    if (!c) return;
    c.lastTurnAt = Date.now();
    if (userMsg) c.history.push({ u: 1, t: userMsg, ts: Date.now() });
    if (assistantReply) c.history.push({ u: 0, t: assistantReply, model, ts: Date.now() });
    if (c.history.length > 60) c.history = c.history.slice(-60); // 30 turns
    saveCase(c);
}

module.exports = {
    // Constants / paths
    DATA_DIR, CASES_DIR, SESSIONS_DIR, PHONE_INDEX_PATH,
    // Helpers
    newCaseId, normalizePhone, makeNewCase,
    // CRUD
    loadCase, saveCase,
    loadSessionPointer, saveSessionPointer,
    indexPhoneToCase, findCasesByPhone, findOpenCaseForPhone,
    // Higher-level
    attachSessionToCase, persistTurn,
};
