/**
 * slot-templates.js — required slots per case type.
 *
 * The point of slots: turn the LLM from a chat partner into a case-builder.
 * For each case type, there's a known set of facts a lawyer would need to
 * know before they can draft anything. The system prompt is augmented with
 * "Case so far: ..." (filled slots) and "Still need to learn: ..." (unfilled).
 * The model then naturally pursues the gaps without the user re-stating
 * what they already said.
 *
 * Each slot has:
 *   key      — machine name (used in case.entities)
 *   ask      — short Hindi+English label shown to the LLM (not the user)
 *   priority — 1 (must-have) | 2 (helpful) | 3 (nice-to-have)
 *   hint     — example value, helps the extractor
 */

const SLOTS = {
    // ─── Road accident / motor vehicle ───
    accident: [
        { key: "date",            ask: "When did the accident happen",         priority: 1, hint: "2025-04-22 or '3 days ago'" },
        { key: "location",        ask: "Where it happened (city/road)",        priority: 1, hint: "Pune, NH-48" },
        { key: "injured",         ask: "Was anyone injured?",                  priority: 1, hint: "yes/no" },
        { key: "fir_filed",       ask: "Has an FIR been filed?",              priority: 1, hint: "yes/no" },
        { key: "vehicle_number",  ask: "Other vehicle registration number",    priority: 2, hint: "MH 12 AB 1234" },
        { key: "insurance_status",ask: "Is the vehicle insured / claim filed", priority: 2, hint: "yes/no/unknown" },
        { key: "driver_known",    ask: "Is the other driver identified",       priority: 2, hint: "yes/no" },
        { key: "medical_papers",  ask: "Medical reports available",            priority: 3, hint: "yes/no" },
    ],
    // ─── Bail / arrest ───
    bail: [
        { key: "person_name",     ask: "Name of the arrested person",          priority: 1 },
        { key: "relation",        ask: "Relationship to caller",               priority: 1, hint: "brother, husband, son" },
        { key: "arrest_date",     ask: "Date of arrest",                       priority: 1 },
        { key: "sections",        ask: "BNS/IPC sections charged under",       priority: 1, hint: "BNS 318, IPC 420" },
        { key: "police_station",  ask: "Police station name + city",           priority: 1 },
        { key: "bailable",        ask: "Bailable or non-bailable offence",     priority: 2 },
        { key: "court_stage",     ask: "Has charge-sheet been filed?",        priority: 2 },
        { key: "prior_bail",      ask: "Was bail applied for already?",       priority: 3 },
    ],
    // ─── Domestic violence ───
    dv: [
        { key: "perpetrator",     ask: "Who is the abuser",                    priority: 1, hint: "husband, in-laws" },
        { key: "abuse_type",      ask: "Type of abuse",                        priority: 1, hint: "physical/verbal/financial/sexual" },
        { key: "duration",        ask: "How long this has been happening",     priority: 1 },
        { key: "children",        ask: "Are children involved?",               priority: 1, hint: "ages, count" },
        { key: "current_safety",  ask: "Is the caller currently safe?",        priority: 1 },
        { key: "fir_filed",       ask: "Has 498A or DV Act case been filed",   priority: 2 },
        { key: "shelter_needed",  ask: "Does she need shelter home?",         priority: 2 },
        { key: "income_source",   ask: "Caller's own income / dependence",     priority: 3 },
    ],
    // ─── Fraud / cyber ───
    fraud: [
        { key: "amount",          ask: "Amount lost",                          priority: 1 },
        { key: "method",          ask: "How the fraud happened",               priority: 1, hint: "UPI link/OTP/fake job/phishing" },
        { key: "date",            ask: "When it happened",                     priority: 1 },
        { key: "transaction_id",  ask: "Transaction reference / UPI ref",      priority: 1 },
        { key: "bank_informed",   ask: "Bank already notified?",              priority: 1 },
        { key: "1930_called",     ask: "Did caller call 1930 cyber helpline?", priority: 1 },
        { key: "fir_filed",       ask: "Cyber crime FIR filed?",              priority: 2 },
        { key: "evidence",        ask: "Screenshots/SMS preserved",            priority: 2 },
    ],
    // ─── Salary / employment ───
    salary: [
        { key: "employer",        ask: "Employer name",                        priority: 1 },
        { key: "amount",          ask: "Amount unpaid",                        priority: 1 },
        { key: "duration",        ask: "How many months pending",              priority: 1 },
        { key: "issue_type",      ask: "Type of dispute",                      priority: 1, hint: "unpaid salary / wrongful termination / PF" },
        { key: "tenure",          ask: "How long was caller employed",         priority: 2 },
        { key: "notice_given",    ask: "Was notice period served",             priority: 2 },
        { key: "appointment_letter", ask: "Appointment letter available",      priority: 3 },
    ],
    // ─── FIR ───
    fir: [
        { key: "incident_type",   ask: "What incident",                        priority: 1, hint: "theft, assault, missing person" },
        { key: "date",            ask: "When the incident happened",           priority: 1 },
        { key: "location",        ask: "Where it happened",                    priority: 1 },
        { key: "police_station",  ask: "Which police station",                 priority: 1 },
        { key: "police_refused",  ask: "Did police refuse to register?",       priority: 1 },
        { key: "cognizable",      ask: "Cognizable or non-cognizable",         priority: 2 },
        { key: "evidence",        ask: "Evidence available",                   priority: 2 },
    ],
    // ─── Property / RERA / rent ───
    property: [
        { key: "property_type",   ask: "Type",                                  priority: 1, hint: "flat / land / rental" },
        { key: "location",        ask: "Property location",                    priority: 1 },
        { key: "issue",           ask: "What's the dispute",                   priority: 1, hint: "builder delay, eviction, deposit" },
        { key: "amount",          ask: "Amount at stake",                      priority: 2 },
        { key: "agreement_signed",ask: "Is there a written agreement",         priority: 2 },
        { key: "rera_registered", ask: "Is the project RERA-registered",       priority: 3 },
    ],
    // ─── Land / encroachment ───
    land: [
        { key: "location",        ask: "Village + tehsil",                     priority: 1 },
        { key: "area",            ask: "How much land",                        priority: 1, hint: "in bigha/acre" },
        { key: "occupier",        ask: "Who is occupying it",                  priority: 1 },
        { key: "papers_have",     ask: "Do you have khatauni / registry",      priority: 1 },
        { key: "duration",        ask: "Since when occupied",                  priority: 2 },
        { key: "fir_filed",       ask: "Encroachment FIR filed",               priority: 2 },
    ],
    // ─── Divorce / family ───
    divorce: [
        { key: "marriage_date",   ask: "Date of marriage",                     priority: 1 },
        { key: "issue",           ask: "Reason for separation",                priority: 1, hint: "cruelty, mutual, desertion" },
        { key: "children",        ask: "Children + ages",                      priority: 1 },
        { key: "religion",        ask: "Marriage Act applicable",              priority: 1, hint: "Hindu / Muslim / Special / Christian" },
        { key: "maintenance",     ask: "Is maintenance being claimed",         priority: 2 },
        { key: "case_filed",      ask: "Any case already filed",              priority: 2 },
    ],
    // ─── Consumer ───
    consumer: [
        { key: "vendor",          ask: "Company / shop name",                  priority: 1 },
        { key: "amount",          ask: "Amount or value",                      priority: 1 },
        { key: "issue",           ask: "Defect / non-delivery / refund",       priority: 1 },
        { key: "purchase_date",   ask: "Date of purchase",                     priority: 1 },
        { key: "complaint_made",  ask: "Did caller complain to vendor first",  priority: 2 },
        { key: "evidence",        ask: "Bills / chats / emails preserved",     priority: 2 },
    ],
    // ─── Cheque bounce ───
    cheque_bounce: [
        { key: "amount",          ask: "Cheque amount",                        priority: 1 },
        { key: "bounce_date",     ask: "Date of bounce",                       priority: 1 },
        { key: "drawer",          ask: "Who issued the cheque",                priority: 1 },
        { key: "notice_sent",     ask: "Has 30-day legal notice been sent",    priority: 1 },
        { key: "reason",          ask: "Reason for bounce on memo",            priority: 2 },
    ],
    // ─── False case / framing ───
    false_case: [
        { key: "fir_number",      ask: "FIR number + police station",          priority: 1 },
        { key: "sections",        ask: "Sections charged under",               priority: 1 },
        { key: "complainant",     ask: "Who filed the case",                   priority: 1 },
        { key: "arrest_status",   ask: "Has caller been arrested",             priority: 1 },
        { key: "anticipatory_bail", ask: "Anticipatory bail filed?",          priority: 2 },
    ],
    // ─── Caste atrocity ───
    caste_atrocity: [
        { key: "incident",        ask: "What happened",                        priority: 1 },
        { key: "perpetrators",    ask: "Who did it",                           priority: 1 },
        { key: "date",            ask: "When",                                 priority: 1 },
        { key: "fir_filed",       ask: "SC/ST Act FIR registered",             priority: 1 },
        { key: "witnesses",       ask: "Are there witnesses",                  priority: 2 },
    ],
    // Generic / unknown — minimal probe to figure out the type first.
    unknown: [
        { key: "case_topic",      ask: "What broadly is the issue",            priority: 1, hint: "FIR, accident, money, family, property" },
        { key: "urgency",         ask: "How urgent — happened now or earlier", priority: 1 },
        { key: "location",        ask: "Where (city/state)",                   priority: 2 },
    ],
};

/** Get the slot template for a case type. Falls back to `unknown`. */
function getSlots(type) {
    return SLOTS[type] || SLOTS.unknown;
}

/**
 * Given filled entity values + slot template, partition into
 * { collected, missing }. `missing` is sorted by priority.
 */
function partitionSlots(type, entities = {}) {
    const slots = getSlots(type);
    const collected = [];
    const missing = [];
    for (const s of slots) {
        const v = entities[s.key];
        if (v !== undefined && v !== null && v !== "" && v !== "unknown") {
            collected.push({ ...s, value: v });
        } else {
            missing.push(s);
        }
    }
    missing.sort((a, b) => a.priority - b.priority);
    return { collected, missing };
}

/**
 * Render a system-prompt fragment that the LLM reads. The format is tuned
 * for token efficiency (~80–150 tokens) and aimed at the model, not the
 * user — short labels, no fluff.
 */
function renderCaseContext(type, entities = {}) {
    const { collected, missing } = partitionSlots(type, entities);
    const lines = [];
    lines.push(`Case type: ${type}`);
    if (collected.length) {
        lines.push("Case so far:");
        for (const c of collected) lines.push(`  • ${c.ask}: ${c.value}`);
    }
    if (missing.length) {
        // Only surface priority-1 missing slots; deeper ones come later.
        const top = missing.filter(m => m.priority === 1).slice(0, 4);
        if (top.length) {
            lines.push("Still need to learn (ask naturally only if relevant to the user's current question):");
            for (const m of top) lines.push(`  • ${m.ask}`);
        }
    }
    lines.push("DO NOT re-ask anything already in 'Case so far'. Reference it naturally instead.");
    return lines.join("\n");
}

module.exports = { SLOTS, getSlots, partitionSlots, renderCaseContext };
