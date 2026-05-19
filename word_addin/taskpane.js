/**
 * Sanhita Word Task-Pane logic.
 *
 * Talks to the local Sanhita FastAPI backend (default :8080) and writes
 * results back into the open Word document via Office.js.
 *
 * Endpoints used:
 *    GET  /api/contract/templates                  — list
 *    GET  /api/contract/templates/{id}             — slots + anchors
 *    POST /api/contract/draft                      — generate
 *    POST /api/contract/compliance                 — 8 plug-ins
 *    POST /api/contract/quick-edit                 — polish / shorten / cite
 *    POST /api/cases/smart-search                  — hybrid case search
 */

const BACKEND = "http://localhost:8080";

// ── Tab switching ─────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".pane").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`pane-${btn.dataset.pane}`).classList.add("active");
  });
});

// ── Helpers ───────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const escape = (s) => String(s ?? "").replace(/[<>&"]/g, (c) =>
  ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" }[c]));

const showErr = (el, msg) => el.innerHTML = `<div class="err">${escape(msg)}</div>`;
const showOk  = (el, msg) => el.innerHTML = `<div class="ok">${escape(msg)}</div>`;
const showSpin = (el, txt = "Working…") =>
  el.innerHTML = `<div class="ok"><span class="spin"></span>${escape(txt)}</div>`;

async function api(path, init = {}) {
  const r = await fetch(`${BACKEND}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0,160)}`);
  return r.json();
}

// Office.js: insert markdown-ish text into the document body
async function insertIntoDoc(md) {
  // Word doesn't render Markdown — split into paragraphs + apply bold runs.
  return Word.run(async (ctx) => {
    const sel = ctx.document.getSelection();
    const lines = md.split(/\n/);
    for (const raw of lines) {
      const line = raw.replace(/\r/g, "");
      if (/^#{1,6}\s/.test(line)) {
        const level = (line.match(/^#+/) || ["#"])[0].length;
        const text  = line.replace(/^#+\s*/, "");
        const p = sel.insertParagraph(text, "After");
        p.styleBuiltIn = level === 1 ? "Title" : level === 2 ? "Heading1" : "Heading2";
        continue;
      }
      if (line.trim() === "---") {
        sel.insertParagraph("", "After");
        continue;
      }
      // Plain paragraph with **bold** runs honoured
      const para = sel.insertParagraph("", "After");
      const parts = line.split(/(\*\*[^*]+\*\*)/g);
      for (const part of parts) {
        if (!part) continue;
        if (part.startsWith("**") && part.endsWith("**")) {
          const r = para.insertText(part.slice(2, -2), "End");
          r.font.bold = true;
        } else {
          para.insertText(part, "End");
        }
      }
    }
    await ctx.sync();
  });
}

async function readSelection() {
  return Word.run(async (ctx) => {
    const sel = ctx.document.getSelection();
    sel.load("text");
    await ctx.sync();
    return sel.text || "";
  });
}

async function replaceSelection(newText) {
  return Word.run(async (ctx) => {
    const sel = ctx.document.getSelection();
    sel.insertText(newText, "Replace");
    await ctx.sync();
  });
}

async function readDocBody() {
  return Word.run(async (ctx) => {
    const body = ctx.document.body;
    body.load("text");
    await ctx.sync();
    return body.text || "";
  });
}


// ── Tab: DRAFT ────────────────────────────────────────────────────────
let templateCache = [];
let selectedSlotSchema = [];

Office.onReady(async () => {
  $("backend-url").textContent = BACKEND;
  try {
    templateCache = await api("/api/contract/templates");
    const sel = $("template-select");
    sel.innerHTML = `<option value="">— pick a template —</option>` +
      templateCache.map((t) =>
        `<option value="${escape(t.id)}">${escape(t.title)}</option>`
      ).join("");
    sel.addEventListener("change", onTemplateChange);
  } catch (e) {
    showErr($("draft-out"), `Couldn't load templates: ${e.message}`);
  }
});

async function onTemplateChange() {
  const id = $("template-select").value;
  $("btn-draft").disabled = !id;
  if (!id) { $("slot-form").innerHTML = ""; $("template-meta").textContent = ""; return; }
  try {
    const t = await api(`/api/contract/templates/${encodeURIComponent(id)}`);
    selectedSlotSchema = t.slots || [];
    $("template-meta").textContent =
      `${t.slots?.length || 0} fields · ${t.clause_count} clauses · ${t.anchors?.statutes?.length || 0} statutes`;
    const required = selectedSlotSchema.filter((s) => s.required);
    $("slot-form").innerHTML = required.map(renderSlot).join("");
  } catch (e) {
    showErr($("draft-out"), e.message);
  }
}

function renderSlot(s) {
  const key = `slot-${escape(s.name)}`;
  const label = `<label for="${key}">${escape(s.name.replace(/_/g, " "))}${s.required ? " *" : ""}</label>`;
  if (s.type === "enum" && Array.isArray(s.options)) {
    return label + `<select id="${key}" data-slot="${escape(s.name)}"><option value="">—</option>` +
      s.options.map((o) => `<option value="${escape(o)}">${escape(o)}</option>`).join("") +
      `</select>`;
  }
  if (s.type === "text") {
    return label + `<textarea id="${key}" rows="3" data-slot="${escape(s.name)}" placeholder="${escape(s.hint || "")}"></textarea>`;
  }
  if (s.type === "integer" || s.type === "number") {
    return label + `<input id="${key}" type="number" data-slot="${escape(s.name)}" placeholder="${escape(s.hint || "")}" value="${s.default ?? ""}"/>`;
  }
  if (s.type === "bool") {
    return label + `<input id="${key}" type="checkbox" data-slot="${escape(s.name)}" ${s.default ? "checked" : ""}/>`;
  }
  return label + `<input id="${key}" data-slot="${escape(s.name)}" placeholder="${escape(s.hint || "")}" value="${escape(s.default || "")}"/>`;
}

$("btn-draft").addEventListener("click", async () => {
  const id = $("template-select").value;
  const out = $("draft-out");
  const slots = {};
  document.querySelectorAll("#slot-form [data-slot]").forEach((el) => {
    const k = el.dataset.slot;
    const v = el.type === "checkbox" ? el.checked
            : el.type === "number"   ? (el.value === "" ? null : Number(el.value))
            : el.value;
    if (v !== "" && v !== null) slots[k] = v;
  });
  showSpin(out, "Generating draft…");
  try {
    const d = await api("/api/contract/draft", {
      method: "POST",
      body: JSON.stringify({ template_id: id, slots, mode: "deterministic_only" }),
    });
    await insertIntoDoc(d.body_md);
    showOk(out, `Inserted ${d.word_count.toLocaleString()} words (draft ${d.draft_id.slice(2,10)}, risk ${d.risk_score}).`);
  } catch (e) {
    showErr(out, e.message);
  }
});


// ── Tab: SEARCH ───────────────────────────────────────────────────────
document.querySelectorAll(".chip").forEach((c) =>
  c.addEventListener("click", () => { $("search-q").value = c.dataset.q; $("btn-search").click(); }));

$("btn-search").addEventListener("click", async () => {
  const q = $("search-q").value.trim();
  const out = $("search-out");
  if (!q) return;
  showSpin(out, "Searching 70M+ Indian court records…");
  try {
    const r = await api("/api/cases/smart-search", {
      method: "POST",
      body: JSON.stringify({ q, mode: $("search-mode").value, limit: 10 }),
    });
    const hits = r.hits || [];
    if (!hits.length) { showOk(out, "No hits."); return; }
    out.innerHTML = `<div class="result">` + hits.map((h) => `
      <div class="item">
        <div class="title">${escape((h.title || "(untitled)").slice(0,160))}</div>
        <div class="meta">${escape(h.court || "—")} · ${escape(h.year || "")} · <code style="font-size:10px;">${escape(h.source_table || "")}</code> ${h.citation ? "· "+escape(h.citation) : ""}</div>
        ${h.snippet ? `<div class="snip">"${escape(h.snippet.slice(0,260))}…"</div>` : ""}
        <button class="secondary" style="margin-top:6px;" data-cite='${escape(h.title || "")} (${escape(h.court || "")}, ${escape(h.year || "")})'>Insert citation</button>
      </div>
    `).join("") + `</div>`;
    out.querySelectorAll("button[data-cite]").forEach((b) =>
      b.addEventListener("click", async () => {
        await Word.run(async (ctx) => {
          ctx.document.getSelection().insertText(` (*${b.dataset.cite}*)`, "End");
          await ctx.sync();
        });
      }));
  } catch (e) {
    showErr(out, e.message);
  }
});


// ── Tab: COMPLIANCE ───────────────────────────────────────────────────
$("btn-comply").addEventListener("click", async () => {
  const out = $("comply-out");
  showSpin(out, "Reading document + running 8 plug-ins…");
  try {
    const body = await readDocBody();
    if (body.length < 50) { showErr(out, "Document is too short to audit."); return; }
    const r = await api("/api/contract/compliance", {
      method: "POST",
      body: JSON.stringify({ body_md: body, doc_type: $("comply-doctype").value || "" }),
    });
    const findings = r.findings || [];
    if (!findings.length) { showOk(out, "✓ All 8 compliance plug-ins passed."); return; }
    out.innerHTML = `<div class="result">` + findings.map((f) => `
      <div class="finding">
        <span class="sev ${escape(f.severity)}">${escape(f.severity)}</span>
        <span class="rule">${escape(f.plugin)} · ${escape(f.rule_id || "")}</span>
        <div style="margin-top:4px;">${escape(f.finding)}</div>
        ${f.remediation ? `<div style="margin-top:3px;font-size:10.5px;color:var(--ink-soft);">→ ${escape(f.remediation)}</div>` : ""}
      </div>
    `).join("") + `</div>`;
  } catch (e) {
    showErr(out, e.message);
  }
});


// ── Tab: QUICK EDIT ───────────────────────────────────────────────────
async function runQuickEdit(action, label) {
  const out = $("edit-out");
  const sel = await readSelection();
  if (!sel || sel.trim().length < 20) {
    showErr(out, "Select at least 20 characters of text in the document first.");
    return;
  }
  showSpin(out, `${label} ${sel.length.toLocaleString()} chars via Sanhita AI scalpel…`);
  try {
    const r = await api("/api/contract/quick-edit", {
      method: "POST",
      body: JSON.stringify({ action, text: sel }),
    });
    if (r.unchanged) { showErr(out, `Unchanged — ${r.reason || "see logs"}`); return; }
    await replaceSelection(r.edited);
    const cites = r.citations_used?.length || 0;
    showOk(out, `${label} done · ${r.model} · ${r.latency_ms}ms${cites ? ` · ${cites} citation${cites>1?"s":""} attached` : ""}.`);
  } catch (e) {
    showErr(out, e.message);
  }
}

$("btn-polish") .addEventListener("click", () => runQuickEdit("polish",  "Polishing"));
$("btn-shorten").addEventListener("click", () => runQuickEdit("shorten", "Shortening"));
$("btn-cite")   .addEventListener("click", () => runQuickEdit("cite",    "Adding citations to"));
