"use strict";

/**
 * security.js — NyayaSathi Security Module
 * SSRF protection, webhook authentication, cryptographic sessions, prompt injection defense
 */

const crypto = require("crypto");

// ═══════════════════════════════════════════════════════════
//  SSRF Protection — validate recording URLs from webhooks
// ═══════════════════════════════════════════════════════════

const ALLOWED_RECORDING_HOSTS = [
    ".exotel.com",
    ".exotel.in",
    ".amazonaws.com",
    ".cloudfront.net",
];

/**
 * Validate that a recording URL is from an allowed domain.
 * Blocks SSRF attacks where attacker sends recordingUrl=http://127.0.0.1:6379
 */
function isAllowedRecordingUrl(url) {
    if (!url || typeof url !== "string") return false;
    try {
        const parsed = new URL(url);
        if (parsed.protocol !== "https:" && parsed.protocol !== "http:") return false;
        // Block private/internal IPs
        const host = parsed.hostname;
        if (host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0" ||
            host.startsWith("10.") || host.startsWith("172.") || host.startsWith("192.168.") ||
            host === "::1" || host === "[::1]" || host.startsWith("169.254.")) {
            return false;
        }
        // Must match an allowed domain
        return ALLOWED_RECORDING_HOSTS.some(suffix => host.endsWith(suffix));
    } catch {
        return false;
    }
}

// ═══════════════════════════════════════════════════════════
//  Webhook Authentication — shared token verification
// ═══════════════════════════════════════════════════════════

/**
 * Express middleware to verify Exotel webhook token.
 * If EXOTEL_WEBHOOK_TOKEN is set, requires ?token=<TOKEN> on all webhook requests.
 * If not set, passes through (development mode).
 */
function verifyWebhookToken(req, res, next) {
    const expectedToken = process.env.EXOTEL_WEBHOOK_TOKEN;
    if (!expectedToken) return next(); // No token configured = dev mode
    const token = req.query?.token || req.body?.token;
    if (!token || token !== expectedToken) {
        console.log(`[SECURITY] Webhook auth failed from ${req.ip}`);
        return res.status(403).send("Forbidden");
    }
    next();
}

/**
 * Append webhook token to a URL if configured.
 */
function appendWebhookToken(url) {
    const token = process.env.EXOTEL_WEBHOOK_TOKEN;
    if (!token) return url;
    const sep = url.includes("?") ? "&" : "?";
    return `${url}${sep}token=${encodeURIComponent(token)}`;
}

// ═══════════════════════════════════════════════════════════
//  Cryptographic Session IDs
// ═══════════════════════════════════════════════════════════

/**
 * Generate a cryptographically secure session ID.
 * Format: ws_<32 hex chars> (128 bits of entropy)
 */
function generateSessionId(prefix = "ws") {
    return `${prefix}_${crypto.randomBytes(16).toString("hex")}`;
}

// ═══════════════════════════════════════════════════════════
//  Prompt Injection Defense
// ═══════════════════════════════════════════════════════════

const INJECTION_PATTERNS = [
    /ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?/gi,
    /you\s+are\s+(now|no\s+longer|actually)/gi,
    /system\s*:?\s*prompt/gi,
    /forget\s+(everything|all|your)\s+(you|instructions?|rules?)/gi,
    /new\s+instructions?\s*:/gi,
    /override\s+(system|safety|rules?)/gi,
    /pretend\s+(you\s+are|to\s+be)/gi,
    /roleplay\s+as/gi,
    /jailbreak/gi,
    /\bDAN\b.*mode/gi,
];

/**
 * Sanitize user transcript before sending to LLM.
 * Strips prompt injection attempts and limits length.
 */
function sanitizeForLLM(text) {
    if (!text || typeof text !== "string") return "";
    let clean = text;
    for (const pattern of INJECTION_PATTERNS) {
        pattern.lastIndex = 0;
        clean = clean.replace(pattern, "[blocked]");
    }
    return clean.slice(0, 500);
}

// ═══════════════════════════════════════════════════════════
//  Performance Timer
// ═══════════════════════════════════════════════════════════

/**
 * Simple timer for latency instrumentation.
 * Usage: const t = timer("STT"); ... const ms = t.end();
 */
function timer(label) {
    const t0 = Date.now();
    return {
        end: () => {
            const ms = Date.now() - t0;
            console.log(`[PERF] ${label}: ${ms}ms`);
            return ms;
        },
    };
}

module.exports = {
    isAllowedRecordingUrl,
    verifyWebhookToken,
    appendWebhookToken,
    generateSessionId,
    sanitizeForLLM,
    timer,
};
