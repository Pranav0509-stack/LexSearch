/**
 * entity-extractor.js — background Gemini JSON-mode case state updater.
 *
 * Runs AFTER the user-facing reply is sent. Doesn't block the hot path.
 * Reads the latest turn + prior case state, returns updated entities + a
 * compressed summary. Result is merged into the case file so the *next*
 * turn's system prompt includes the new facts.
 *
 * We use Gemini Flash with `responseMimeType: application/json` and a strict
 * schema — no markdown parsing, no regex, no LLM hallucination wrapping.
 *
 * Cost: one Flash call per user turn ≈ 1.5K input tokens + 200 output. At
 * Vertex pricing that's ~$0.00015/turn — basically free.
 */

const { getSlots } = require("./slot-templates.js");

const GK = process.env.GEMINI_API_KEY;
const MODEL = "gemini-2.5-flash";
const URL = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`;
// 8s budget — generous because this runs after the user reply is sent.
// JSON-mode generation tends to be slower than free-form on Flash.
const TIMEOUT_MS = 8000;

/**
 * Build a JSON schema for the slot keys. Gemini honors `responseSchema` when
 * `responseMimeType` is application/json — the model output is guaranteed
 * parseable JSON shaped like our schema.
 */
function buildSchema(caseType) {
    const slots = getSlots(caseType);
    const properties = {};
    for (const s of slots) {
        properties[s.key] = {
            type: "string",
            description: `${s.ask}${s.hint ? ` (e.g. ${s.hint})` : ""}. Use empty string if not mentioned.`,
        };
    }
    return {
        type: "object",
        properties: {
            entities: {
                type: "object",
                description: "Slot values extracted from the conversation. Empty string for unknown.",
                properties,
            },
            new_facts: {
                type: "array",
                items: { type: "string" },
                description: "Short factual statements newly learned this turn (≤8 words each).",
            },
            actions_advised: {
                type: "array",
                items: { type: "string" },
                description: "Concrete actions the assistant just told the user to take.",
            },
            summary: {
                type: "string",
                description: "One-sentence English summary of the case so far. Replace prior summary.",
            },
            urgency: {
                type: "string",
                enum: ["critical", "high", "medium", "low"],
                description: "How urgent the situation is.",
            },
            needs_lawyer: {
                type: "boolean",
                description: "True if the user has indicated they want a human lawyer, or if the matter clearly cannot be self-handled.",
            },
        },
        required: ["entities", "summary", "urgency", "needs_lawyer"],
    };
}

/**
 * Build the user prompt. Include only the last 4 turns + the prior case
 * snapshot. Keeping this short matters — extraction quality drops fast on
 * long inputs and we don't want a runaway token bill.
 */
function buildExtractionPrompt(caseType, priorEntities, priorSummary, recentHistory, userMsg, assistantReply) {
    const histText = recentHistory.slice(-6).map(m => `${m.u ? "USER" : "AI"}: ${m.t}`).join("\n");
    const priorJson = JSON.stringify(priorEntities || {}, null, 0);
    return [
        `You are extracting structured facts from an Indian legal helpline conversation. Case type: ${caseType}.`,
        ``,
        `Prior known entities: ${priorJson}`,
        priorSummary ? `Prior summary: ${priorSummary}` : ``,
        ``,
        `Recent conversation:`,
        histText,
        `USER: ${userMsg}`,
        `AI: ${assistantReply}`,
        ``,
        `Update the entities. Preserve any prior values that are still correct; overwrite only when the conversation contradicts or refines them. Empty string if a slot was not discussed. Output JSON matching the schema.`,
    ].filter(Boolean).join("\n");
}

/**
 * Call Gemini in JSON mode. Returns parsed object or null on any failure.
 * Hard-times-out after TIMEOUT_MS — never blocks the next user turn.
 */
async function callGeminiJson(systemInstruction, userText, schema) {
    if (!GK) return null;
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    try {
        const r = await fetch(URL, {
            method: "POST",
            headers: { "Content-Type": "application/json", "x-goog-api-key": GK },
            signal: ctrl.signal,
            body: JSON.stringify({
                systemInstruction: { parts: [{ text: systemInstruction }] },
                contents: [{ role: "user", parts: [{ text: userText }] }],
                generationConfig: {
                    temperature: 0.1,
                    maxOutputTokens: 800,
                    responseMimeType: "application/json",
                    responseSchema: schema,
                },
            }),
        });
        clearTimeout(t);
        if (!r.ok) {
            console.warn(`[extractor] gemini ${r.status}`);
            return null;
        }
        const d = await r.json();
        const text = d.candidates?.[0]?.content?.parts?.[0]?.text;
        if (!text) return null;
        try { return JSON.parse(text); } catch { return null; }
    } catch (e) {
        clearTimeout(t);
        if (e.name !== "AbortError") console.warn(`[extractor] failed: ${e.message}`);
        return null;
    }
}

/**
 * Public entry: extract for one turn. `c` is the live case object — this
 * function mutates it in place (entities, summary, urgency, needs_lawyer,
 * facts, actions_advised) on success. Returns { ok, changed } so the caller
 * can decide whether to persist.
 */
async function extractAndUpdate(c, { userMsg, assistantReply, recentHistory } = {}) {
    if (!c) return { ok: false, changed: false };
    const caseType = c.type || "unknown";
    const schema = buildSchema(caseType);
    const sys = "Extract structured facts. Be conservative — never invent details. Only output JSON.";
    const prompt = buildExtractionPrompt(
        caseType,
        c.entities || {},
        c.summary || "",
        recentHistory || [],
        userMsg || "",
        assistantReply || "",
    );

    const result = await callGeminiJson(sys, prompt, schema);
    if (!result) return { ok: false, changed: false };

    let changed = false;

    // Merge entities — preserve prior, overwrite with new non-empty values.
    const nextEntities = { ...(c.entities || {}) };
    for (const [k, v] of Object.entries(result.entities || {})) {
        if (v === undefined || v === null) continue;
        const trimmed = String(v).trim();
        if (!trimmed) continue;
        if (nextEntities[k] !== trimmed) {
            nextEntities[k] = trimmed;
            changed = true;
        }
    }
    c.entities = nextEntities;

    // Append new facts (dedupe).
    if (Array.isArray(result.new_facts) && result.new_facts.length) {
        const existing = new Set((c.facts || []).map(f => f.toLowerCase()));
        for (const f of result.new_facts) {
            if (typeof f !== "string") continue;
            const fT = f.trim();
            if (!fT) continue;
            if (!existing.has(fT.toLowerCase())) {
                c.facts = c.facts || [];
                c.facts.push(fT);
                changed = true;
            }
        }
        if (c.facts && c.facts.length > 30) c.facts = c.facts.slice(-30);
    }

    // Append advised actions (dedupe).
    if (Array.isArray(result.actions_advised) && result.actions_advised.length) {
        const existing = new Set((c.actions_advised || []).map(a => a.toLowerCase()));
        for (const a of result.actions_advised) {
            if (typeof a !== "string") continue;
            const aT = a.trim();
            if (!aT) continue;
            if (!existing.has(aT.toLowerCase())) {
                c.actions_advised = c.actions_advised || [];
                c.actions_advised.push(aT);
                changed = true;
            }
        }
        if (c.actions_advised && c.actions_advised.length > 20) c.actions_advised = c.actions_advised.slice(-20);
    }

    // Replace summary on improvement. Accept any non-empty string from the
    // model — even short ones are useful for the next turn's prompt.
    if (typeof result.summary === "string") {
        const s = result.summary.trim();
        if (s && s.length >= 8 && s !== c.summary) {
            c.summary = s.slice(0, 600);
            changed = true;
        }
    }

    // Urgency upgrade only — never silently downgrade a critical case.
    const URGENCY_RANK = { low: 0, medium: 1, high: 2, critical: 3 };
    const proposed = result.urgency;
    if (proposed && URGENCY_RANK[proposed] !== undefined) {
        const cur = URGENCY_RANK[c.urgency] ?? 1;
        const next = URGENCY_RANK[proposed];
        if (next > cur) { c.urgency = proposed; changed = true; }
    }

    // needs_lawyer: monotonic — once true, stays true.
    if (result.needs_lawyer === true && !c.needs_lawyer) {
        c.needs_lawyer = true;
        changed = true;
    }

    return { ok: true, changed };
}

module.exports = { extractAndUpdate };
