// Sanhita workspace tabs — Vault / Draft / Review / Translate / Citator.
// Research tab is owned by brief.js. This file handles the other five.

(function () {
  "use strict";

  // ── Tab switching ───────────────────────────────────────────────────────
  const tabs = document.querySelectorAll(".ws-tab");
  const panes = document.querySelectorAll(".ws-pane");

  function show(mode) {
    tabs.forEach(t => t.classList.toggle("active", t.dataset.mode === mode));
    panes.forEach(p => {
      const match = p.dataset.pane === mode;
      p.hidden = !match;
    });
    if (mode === "vault") loadVaultDocs();
    if (mode === "draft") loadDraftTemplates();
  }
  tabs.forEach(t => t.addEventListener("click", () => show(t.dataset.mode)));

  // ── Helpers ─────────────────────────────────────────────────────────────
  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, c => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
  }
  function mdToHtml(md) {
    // Tiny markdown: headings, bold, italic, lists, paragraphs, line breaks.
    if (!md) return "";
    let h = escapeHtml(md);
    h = h.replace(/^### (.*)$/gm, "<h3>$1</h3>");
    h = h.replace(/^## (.*)$/gm, "<h2>$1</h2>");
    h = h.replace(/^# (.*)$/gm, "<h1>$1</h1>");
    h = h.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    h = h.replace(/\*(.+?)\*/g, "<em>$1</em>");
    h = h.replace(/^- (.*)$/gm, "<li>$1</li>");
    h = h.replace(/(<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`);
    h = h.replace(/\n{2,}/g, "</p><p>");
    h = h.replace(/\n/g, "<br/>");
    return `<p>${h}</p>`;
  }
  async function api(path, opts = {}) {
    const r = await fetch(path, {
      credentials: "same-origin",
      headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {},
      ...opts,
    });
    if (!r.ok) {
      const txt = await r.text().catch(() => "");
      throw new Error(`${r.status}: ${txt.slice(0, 200)}`);
    }
    return r.json();
  }
  function renderMeta(d) {
    const llm = d.llm || {};
    const v = d.validation || {};
    const conf = typeof v.confidence === "number" ? `${Math.round(v.confidence * 100)}%` : "—";
    const cls = v.confidence >= 0.85 ? "hi" : v.confidence >= 0.6 ? "mid" : "lo";
    return `<div class="ws-meta">
      <span class="ai-chip">✦ ${escapeHtml(llm.provider || "—")} · ${escapeHtml(llm.model || "")}</span>
      ${typeof v.confidence === "number" ? `<span class="conf-pill ${cls}">grounding ${conf}</span>` : ""}
      ${llm.latency_ms ? `<span class="ws-latency">${llm.latency_ms} ms</span>` : ""}
      ${d.refused ? `<span class="refused-badge">refused</span>` : ""}
    </div>`;
  }

  // ══════════════════════════ VAULT ══════════════════════════
  const vaultDocList = document.getElementById("vault-doc-list");
  const vaultUploadForm = document.getElementById("vault-upload-form");
  const vaultFile = document.getElementById("vault-file");
  const vaultStatus = document.getElementById("vault-upload-status");
  const vaultChatLog = document.getElementById("vault-chat-log");
  const vaultChatForm = document.getElementById("vault-chat-form");
  const vaultChatInput = document.getElementById("vault-chat-input");

  async function loadVaultDocs() {
    if (!vaultDocList) return;
    try {
      const { docs } = await api("/api/vault/docs");
      vaultDocList.innerHTML = docs.length
        ? docs.map(d => `
          <li class="vault-doc">
            <span class="vault-doc-name">${escapeHtml(d.filename)}</span>
            <span class="vault-doc-meta">${d.n_chunks} ¶ · ${(d.size_bytes/1024).toFixed(0)} KB</span>
            <button class="vault-doc-del" data-id="${d.id}" title="Delete">×</button>
          </li>`).join("")
        : `<li class="vault-empty">No documents uploaded yet.</li>`;
      vaultDocList.querySelectorAll(".vault-doc-del").forEach(b => {
        b.addEventListener("click", async () => {
          if (!confirm("Delete this document and all its chunks?")) return;
          await api(`/api/vault/docs/${b.dataset.id}`, { method: "DELETE" });
          loadVaultDocs();
        });
      });
    } catch (e) {
      vaultDocList.innerHTML = `<li class="vault-empty">Failed to load: ${escapeHtml(e.message)}</li>`;
    }
  }

  if (vaultUploadForm) {
    vaultUploadForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      if (!vaultFile.files[0]) return;
      vaultStatus.textContent = "Uploading & extracting…";
      const fd = new FormData();
      fd.append("file", vaultFile.files[0]);
      try {
        const r = await api("/api/vault/upload", { method: "POST", body: fd });
        vaultStatus.textContent = `Uploaded — ${r.n_chunks} chunks indexed.`;
        vaultFile.value = "";
        loadVaultDocs();
      } catch (e) {
        vaultStatus.textContent = `Upload failed: ${e.message}`;
      }
    });
  }

  const vaultHistory = [];
  if (vaultChatForm) {
    vaultChatForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const q = vaultChatInput.value.trim();
      if (!q) return;
      vaultChatInput.value = "";
      const empty = vaultChatLog.querySelector(".chat-empty"); if (empty) empty.remove();
      vaultChatLog.insertAdjacentHTML("beforeend",
        `<div class="chat-msg user"><div class="chat-bubble">${escapeHtml(q)}</div></div>`);
      const thinking = document.createElement("div");
      thinking.className = "chat-msg assistant";
      thinking.innerHTML = `<div class="chat-bubble"><em>Searching your vault…</em></div>`;
      vaultChatLog.appendChild(thinking);
      vaultChatLog.scrollTop = vaultChatLog.scrollHeight;
      try {
        const r = await api("/api/vault/chat", {
          method: "POST",
          body: JSON.stringify({ question: q, history: vaultHistory.slice(-6) }),
        });
        vaultHistory.push({ role: "user", content: q });
        vaultHistory.push({ role: "assistant", content: r.answer_markdown });
        const cites = (r.citations || []).map(c =>
          `<div class="ws-cite"><b>[${c.n}]</b> ${escapeHtml(c.doc_title)} ¶ ${escapeHtml(c.para_label||"")}<br/><small>${escapeHtml(c.excerpt||"")}</small></div>`
        ).join("");
        thinking.innerHTML = `<div class="chat-bubble">${mdToHtml(r.answer_markdown)}${renderMeta(r)}${cites ? `<div class="ws-cites">${cites}</div>`:""}</div>`;
      } catch (e) {
        thinking.innerHTML = `<div class="chat-bubble"><em>Error: ${escapeHtml(e.message)}</em></div>`;
      }
      vaultChatLog.scrollTop = vaultChatLog.scrollHeight;
    });
  }

  // ══════════════════════════ DRAFT ══════════════════════════
  const draftTemplate = document.getElementById("draft-template");
  const draftFacts = document.getElementById("draft-facts");
  const draftAdd = document.getElementById("draft-add-row");
  const draftGenerate = document.getElementById("draft-generate");
  const draftOutput = document.getElementById("draft-output");

  async function loadDraftTemplates() {
    if (!draftTemplate || draftTemplate.options.length) return;
    try {
      const { templates } = await api("/api/draft/templates");
      draftTemplate.innerHTML = templates.map(t =>
        `<option value="${t.key}">${escapeHtml(t.title)}</option>`).join("");
    } catch (e) {
      draftTemplate.innerHTML = `<option value="">Failed to load templates</option>`;
    }
  }
  if (draftAdd) {
    draftAdd.addEventListener("click", () => {
      const k = document.createElement("input");
      k.type = "text"; k.className = "draft-fact-key"; k.placeholder = "Field name";
      const v = document.createElement("input");
      v.type = "text"; v.className = "draft-fact-val"; v.placeholder = "Value";
      draftFacts.insertBefore(k, draftAdd);
      draftFacts.insertBefore(v, draftAdd);
    });
  }
  if (draftGenerate) {
    draftGenerate.addEventListener("click", async () => {
      const keys = draftFacts.querySelectorAll(".draft-fact-key");
      const vals = draftFacts.querySelectorAll(".draft-fact-val");
      const facts = {};
      for (let i = 0; i < keys.length; i++) {
        const k = keys[i].value.trim();
        const v = vals[i] ? vals[i].value.trim() : "";
        if (k && v) facts[k] = v;
      }
      draftOutput.innerHTML = `<p class="ws-placeholder">Generating draft…</p>`;
      try {
        const r = await api("/api/draft", {
          method: "POST",
          body: JSON.stringify({ template: draftTemplate.value, facts }),
        });
        draftOutput.innerHTML = `<h2 class="ws-output-title">${escapeHtml(r.title || "Draft")}</h2>${mdToHtml(r.draft_markdown)}${renderMeta(r)}`;
      } catch (e) {
        draftOutput.innerHTML = `<p class="ws-placeholder">Failed: ${escapeHtml(e.message)}</p>`;
      }
    });
  }

  // ══════════════════════════ REVIEW ══════════════════════════
  const reviewClauses = document.getElementById("review-clauses");
  const reviewAdd = document.getElementById("review-add");
  const reviewGo = document.getElementById("review-go");
  const reviewOutput = document.getElementById("review-output");

  if (reviewAdd) {
    reviewAdd.addEventListener("click", () => {
      const ta = document.createElement("textarea");
      ta.className = "review-clause";
      ta.placeholder = `Clause ${reviewClauses.querySelectorAll(".review-clause").length + 1}…`;
      reviewClauses.appendChild(ta);
    });
  }
  if (reviewGo) {
    reviewGo.addEventListener("click", async () => {
      const clauses = [...reviewClauses.querySelectorAll(".review-clause")]
        .map(t => t.value.trim()).filter(Boolean);
      if (!clauses.length) { reviewOutput.innerHTML = `<p class="ws-placeholder">Add at least one clause.</p>`; return; }
      reviewOutput.innerHTML = `<p class="ws-placeholder">Reviewing…</p>`;
      try {
        const r = await api("/api/review", { method: "POST", body: JSON.stringify({ clauses }) });
        reviewOutput.innerHTML = `${mdToHtml(r.review_markdown)}${renderMeta(r)}`;
      } catch (e) {
        reviewOutput.innerHTML = `<p class="ws-placeholder">Failed: ${escapeHtml(e.message)}</p>`;
      }
    });
  }

  // ══════════════════════════ TRANSLATE ══════════════════════════
  const txDir = document.getElementById("tx-direction");
  const txInput = document.getElementById("tx-input");
  const txGo = document.getElementById("tx-go");
  const txOutput = document.getElementById("tx-output");

  if (txGo) {
    txGo.addEventListener("click", async () => {
      const text = txInput.value.trim();
      if (!text) return;
      txOutput.innerHTML = `<p class="ws-placeholder">Translating…</p>`;
      try {
        const r = await api("/api/translate", {
          method: "POST",
          body: JSON.stringify({ text, direction: txDir.value }),
        });
        txOutput.innerHTML = `<div class="tx-result">${mdToHtml(r.translation)}</div>${renderMeta(r)}`;
      } catch (e) {
        txOutput.innerHTML = `<p class="ws-placeholder">Failed: ${escapeHtml(e.message)}</p>`;
      }
    });
  }

  // ══════════════════════════ CITATOR ══════════════════════════
  const citTitle = document.getElementById("cit-title");
  const citExcerpt = document.getElementById("cit-excerpt");
  const citGo = document.getElementById("cit-go");
  const citOutput = document.getElementById("cit-output");

  if (citGo) {
    citGo.addEventListener("click", async () => {
      const case_title = citTitle.value.trim();
      const excerpt = citExcerpt.value.trim();
      if (!case_title || !excerpt) {
        citOutput.innerHTML = `<p class="ws-placeholder">Provide both case title and excerpt.</p>`;
        return;
      }
      citOutput.innerHTML = `<p class="ws-placeholder">Running citator…</p>`;
      try {
        const r = await api("/api/citator", {
          method: "POST", body: JSON.stringify({ case_title, excerpt }),
        });
        citOutput.innerHTML = `<h2 class="ws-output-title">${escapeHtml(case_title)}</h2>${mdToHtml(r.summary_markdown)}${renderMeta(r)}`;
      } catch (e) {
        citOutput.innerHTML = `<p class="ws-placeholder">Failed: ${escapeHtml(e.message)}</p>`;
      }
    });
  }
})();

// ══════════════════════════ REDLINE ══════════════════════════
(function () {
  "use strict";
  const goBtn = document.getElementById("rl-go");
  const inputEl = document.getElementById("rl-input");
  const summaryEl = document.getElementById("rl-summary");
  const countEl = document.getElementById("rl-count");
  const listEl = document.getElementById("rl-list");
  if (!goBtn) return;

  function escapeHtml(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c])); }

  function renderSuggestion(s, idx) {
    const typeColor = s.type === "remove" ? "rl-remove" : s.type === "add" ? "rl-add" : "rl-replace";
    const preview = s.type === "remove"
      ? `<div class="rl-text rl-strike">${escapeHtml(s.original || "")}</div>`
      : s.type === "add"
        ? `<div class="rl-text rl-ins">${escapeHtml(s.replacement || "")}</div>`
        : `<div class="rl-text rl-strike">${escapeHtml(s.original || "")}</div>
           <div class="rl-text rl-ins">${escapeHtml(s.replacement || "")}</div>`;
    return `<div class="rl-card ${typeColor}" data-id="${escapeHtml(s.id||idx)}">
      <div class="rl-card-title">${escapeHtml(s.title || "Suggestion " + (idx+1))}</div>
      ${preview}
      <div class="rl-reason"><em>${escapeHtml(s.reason || "")}</em></div>
      <div class="rl-actions">
        <button class="rl-dismiss">Dismiss</button>
        <button class="rl-apply">Apply</button>
      </div>
    </div>`;
  }

  goBtn.addEventListener("click", async () => {
    const text = (inputEl.value || "").trim();
    if (text.length < 50) { summaryEl.innerHTML = '<p class="ws-placeholder">Paste at least 50 characters.</p>'; return; }
    summaryEl.innerHTML = '<p class="ws-placeholder">Running redline…</p>';
    countEl.textContent = ""; listEl.innerHTML = "";
    try {
      const r = await fetch("/api/redline", {
        method: "POST", credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || r.status);
      summaryEl.innerHTML = (data.summary || []).length
        ? `<ul>${data.summary.map(s => `<li>${escapeHtml(s)}</li>`).join("")}</ul>`
        : '<p class="ws-placeholder">(no summary)</p>';
      const suggestions = data.suggestions || [];
      countEl.textContent = `${suggestions.length} suggestions`;
      listEl.innerHTML = suggestions.map((s, i) => renderSuggestion(s, i)).join("");
      // Apply/Dismiss handlers
      listEl.querySelectorAll(".rl-card").forEach(card => {
        card.querySelector(".rl-dismiss")?.addEventListener("click", () => card.remove());
        card.querySelector(".rl-apply")?.addEventListener("click", () => {
          const id = card.dataset.id;
          const s = suggestions.find(x => String(x.id) === id);
          if (!s) return;
          if (s.type === "remove" && s.original) inputEl.value = inputEl.value.replace(s.original, "");
          else if (s.type === "replace" && s.original) inputEl.value = inputEl.value.replace(s.original, s.replacement || "");
          else if (s.type === "add" && s.replacement) inputEl.value = inputEl.value + "\n\n" + s.replacement;
          card.classList.add("rl-applied");
          card.querySelectorAll("button").forEach(b => b.disabled = true);
        });
      });
    } catch (e) {
      summaryEl.innerHTML = `<p class="ws-placeholder">Failed: ${escapeHtml(e.message)}</p>`;
    }
  });
})();

// ══════════════════════════ GENERIC WORKFLOWS ══════════════════════════
(function () {
  "use strict";
  const WF = {
    reps_warranties: { title: "Reps & Warranties", sub: "Paste an SPA/APA or the representations section." },
    chronology: { title: "Generate Chronology", sub: "Paste briefs, notices, emails, or orders." },
    risks: { title: "Assess Risks", sub: "Paste a fact pattern or transaction summary." },
    interview: { title: "Summarize Interview", sub: "Paste an interview transcript." },
    extract_terms: { title: "Extract Terms", sub: "Paste a contract to extract defined terms, parties, dates." },
    support_argument: { title: "Support Argument", sub: "State the proposition you need to support." },
    client_alert: { title: "Draft Client Alert", sub: "Paste the judgment, notification, or circular." },
    closing_checklist: { title: "Closing Checklist", sub: "Paste a term sheet or SPA." },
    interim_memo: { title: "Interim Memo", sub: "State a discrete point of Indian law." },
  };
  function escapeHtml(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c])); }
  function mdToHtml(md) {
    if (!md) return "";
    let h = escapeHtml(md);
    h = h.replace(/^### (.*)$/gm, "<h3>$1</h3>").replace(/^## (.*)$/gm, "<h2>$1</h2>");
    h = h.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\*(.+?)\*/g, "<em>$1</em>");
    h = h.replace(/^- (.*)$/gm, "<li>$1</li>").replace(/(<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`);
    h = h.replace(/\n{2,}/g, "</p><p>").replace(/\n/g, "<br/>");
    return `<p>${h}</p>`;
  }

  // Re-bind workflow card clicks — intercept keys that belong to generic/redline flow
  document.querySelectorAll(".wf-card[data-wf]").forEach(card => {
    const key = card.dataset.wf;
    if (key === "redline") {
      card.addEventListener("click", (e) => {
        e.stopPropagation();
        document.querySelectorAll(".wf-pane").forEach(p => p.hidden = true);
        document.getElementById("wf-pane-redline").hidden = false;
        document.querySelector('[data-pane="workflows"] .wf-grid').hidden = true;
        document.querySelector('[data-pane="workflows"] .wf-topbar').hidden = true;
      }, true);
    } else if (WF[key]) {
      card.addEventListener("click", (e) => {
        e.stopPropagation();
        document.querySelectorAll(".wf-pane").forEach(p => p.hidden = true);
        const pane = document.getElementById("wf-pane-generic");
        pane.hidden = false;
        document.getElementById("gen-title").textContent = WF[key].title;
        document.getElementById("gen-sub").textContent = WF[key].sub;
        document.getElementById("gen-output").innerHTML = '<p class="ws-placeholder">Paste input and run.</p>';
        document.getElementById("gen-input").value = "";
        document.getElementById("gen-go").dataset.wfKey = key;
        document.querySelector('[data-pane="workflows"] .wf-grid').hidden = true;
        document.querySelector('[data-pane="workflows"] .wf-topbar').hidden = true;
      }, true);
    }
  });

  document.getElementById("gen-go")?.addEventListener("click", async (ev) => {
    const btn = ev.currentTarget;
    const key = btn.dataset.wfKey;
    const text = document.getElementById("gen-input").value.trim();
    const out = document.getElementById("gen-output");
    if (!key || text.length < 10) { out.innerHTML = '<p class="ws-placeholder">Paste at least 10 characters.</p>'; return; }
    out.innerHTML = '<p class="ws-placeholder">Running…</p>';
    try {
      const r = await fetch("/api/workflows/run", {
        method: "POST", credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key, text }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || r.status);
      const llm = data.llm || {};
      out.innerHTML = `<h2 class="ws-output-title">${escapeHtml(data.title || "Output")}</h2>${mdToHtml(data.output_markdown)}
        <div class="ws-meta"><span class="ai-chip">✦ ${escapeHtml(llm.provider||"")} · ${escapeHtml(llm.model||"")}</span>
        ${llm.latency_ms ? `<span class="ws-latency">${llm.latency_ms} ms</span>`:""}</div>`;
    } catch (e) {
      out.innerHTML = `<p class="ws-placeholder">Failed: ${escapeHtml(e.message)}</p>`;
    }
  });
})();
