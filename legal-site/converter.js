/* LexSearch — IPC/BNS Converter */

let direction = "ipc_to_bns";

document.addEventListener("DOMContentLoaded", () => {
  // Direction toggle
  document.querySelectorAll(".dir-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".dir-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      direction = btn.dataset.dir;
      const input = document.getElementById("converter-input");
      if (direction === "bns_to_ipc") {
        input.placeholder = "BNS section number or keyword (e.g. 103, murder)";
      } else {
        input.placeholder = "IPC section number or keyword (e.g. 302, murder, theft)";
      }
    });
  });

  // Search
  document.getElementById("converter-search-btn").addEventListener("click", doSearch);
  document.getElementById("converter-input").addEventListener("keydown", e => {
    if (e.key === "Enter") doSearch();
  });

  // Quick links
  document.querySelectorAll(".quick-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.getElementById("converter-input").value = btn.dataset.q;
      direction = "ipc_to_bns";
      document.querySelectorAll(".dir-btn").forEach(b => b.classList.remove("active"));
      document.querySelector('[data-dir="ipc_to_bns"]').classList.add("active");
      doSearch();
    });
  });
});

async function doSearch() {
  const q = document.getElementById("converter-input").value.trim();
  if (!q) return;

  showState("loading");

  try {
    const res = await fetch(`/converter/search?q=${encodeURIComponent(q)}&direction=${direction}`);
    if (!res.ok) throw new Error("Search failed");
    const data = await res.json();

    if (!data.length) {
      showState("empty");
      return;
    }

    renderResults(data);
    showState("results");
  } catch (e) {
    showError(e.message);
    showState("welcome");
  }
}

function renderResults(items) {
  const container = document.getElementById("converter-results");
  container.innerHTML = items.map((item, i) => `
    <div class="section-card">
      <div class="section-header">
        <div class="section-badges">
          ${item.ipc_section && item.ipc_section !== "—"
            ? `<span class="section-badge old">IPC §${esc(item.ipc_section)}</span>`
            : `<span class="section-badge new-only">NEW</span>`}
          <span class="section-arrow">→</span>
          ${item.bns_section && item.bns_section !== "—"
            ? `<span class="section-badge new">BNS §${esc(item.bns_section)}</span>`
            : `<span class="section-badge removed">Removed</span>`}
        </div>
        <span class="section-category">${esc(item.category)}</span>
      </div>
      <h3 class="section-title">${esc(item.title)}</h3>
      <p class="section-desc">${esc(item.description)}</p>
      <div class="section-tags">
        <span class="tag tag-punishment">${esc(item.punishment)}</span>
        <span class="tag ${item.bailable ? 'tag-bailable' : 'tag-nonbailable'}">${item.bailable ? 'Bailable' : 'Non-Bailable'}</span>
        <span class="tag ${item.cognizable ? 'tag-cognizable' : 'tag-noncognizable'}">${item.cognizable ? 'Cognizable' : 'Non-Cognizable'}</span>
      </div>
      <button class="explain-btn" onclick="explainSection(${i})">Explain in Plain Language</button>
      <div class="explain-panel" id="explain-${i}" style="display:none">
        <div class="explain-loading"><div class="spinner"></div> Getting AI explanation...</div>
        <div class="explain-content"></div>
      </div>
    </div>
  `).join("");

  // Store data for explain calls
  window._converterResults = items;
}

async function explainSection(idx) {
  const panel = document.getElementById(`explain-${idx}`);
  const content = panel.querySelector(".explain-content");
  const loading = panel.querySelector(".explain-loading");

  if (panel.style.display !== "none" && content.innerHTML) {
    panel.style.display = "none";
    return;
  }

  panel.style.display = "";
  loading.style.display = "";
  content.innerHTML = "";

  const item = window._converterResults[idx];

  try {
    const res = await fetch("/converter/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        section: item.ipc_section !== "—" ? item.ipc_section : item.bns_section,
        act: item.ipc_section !== "—" ? "IPC" : "BNS",
        title: item.title,
        bns_section: item.bns_section,
      }),
    });
    if (!res.ok) throw new Error("Explanation failed");
    const data = await res.json();
    content.innerHTML = data.explanation.replace(/\n/g, "<br>");
  } catch (e) {
    content.innerHTML = `<span class="error-text">Failed to get explanation: ${esc(e.message)}</span>`;
  }

  loading.style.display = "none";
}

function showState(state) {
  ["welcome", "loading", "empty", "results"].forEach(s => {
    const el = document.getElementById(`converter-${s}`);
    if (el) el.style.display = s === state ? "" : "none";
  });
}

function showError(msg) {
  const el = document.getElementById("converter-error");
  el.textContent = msg;
  el.style.display = msg ? "" : "none";
}

function esc(s) {
  return String(s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
