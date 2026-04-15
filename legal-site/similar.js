/* LexSearch — Case Similarity Engine */

document.addEventListener("DOMContentLoaded", async () => {
  await loadCourts();
  document.getElementById("similar-search-btn").addEventListener("click", findSimilar);
  document.getElementById("similar-facts").addEventListener("keydown", e => {
    if (e.key === "Enter" && e.ctrlKey) findSimilar();
  });
});

async function loadCourts() {
  try {
    const res = await fetch("/courts");
    const courts = await res.json();
    const sel = document.getElementById("similar-court");
    courts.forEach(c => {
      const opt = document.createElement("option");
      opt.value = c.s3_code;
      opt.textContent = c.name;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.error("Failed to load courts", e);
  }
}

async function findSimilar() {
  const facts = document.getElementById("similar-facts").value.trim();
  if (!facts) return;

  const court = document.getElementById("similar-court").value;

  showState("loading");

  try {
    const res = await fetch("/similar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ facts, court }),
    });

    if (res.status === 503) {
      showState("notready");
      return;
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Search failed.");
    }

    const data = await res.json();

    if (!data.results || !data.results.length) {
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

function renderResults(data) {
  // Stats
  const statsBar = document.getElementById("similar-stats");
  statsBar.style.display = "";
  document.getElementById("similar-stats-text").textContent =
    `Found ${data.results.length} similar cases (searched ${data.corpus_size?.toLocaleString() || "—"} cases)`;

  // Results
  const container = document.getElementById("similar-results");
  container.innerHTML = data.results.map(r => {
    const score = Math.round(r.score * 100);
    const scoreClass = score >= 60 ? "high" : score >= 30 ? "medium" : "low";

    // Build viewer link
    let viewerLink = "";
    if (r.pdf_link) {
      viewerLink = `/viewer.html?pdf=${encodeURIComponent(r.pdf_link)}&court=${encodeURIComponent(r.court || "")}&year=${r.year || ""}`;
    }

    return `
      <div class="similarity-card">
        <div class="similarity-header">
          <span class="similarity-score score-${scoreClass}">${score}%</span>
          <div class="similarity-meta">
            <span class="sim-court">${esc(r.court_name || r.court || "")}</span>
            <span class="sim-year">${r.year || ""}</span>
          </div>
        </div>
        <h3 class="sim-title">${esc(r.title)}</h3>
        ${r.judge ? `<p class="sim-judge">Judge: ${esc(r.judge)}</p>` : ""}
        ${r.disposal ? `<p class="sim-disposal">Disposal: ${esc(r.disposal)}</p>` : ""}
        <div class="sim-actions">
          ${viewerLink ? `<a href="${viewerLink}" class="btn btn-download" target="_blank">Read Judgment</a>` : ""}
        </div>
      </div>
    `;
  }).join("");
}

function showState(state) {
  ["welcome", "loading", "empty", "notready"].forEach(s => {
    const el = document.getElementById(`similar-${s}`);
    if (el) el.style.display = s === state ? "" : "none";
  });
  document.getElementById("similar-results").style.display = state === "results" ? "" : "none";
  document.getElementById("similar-stats").style.display = state === "results" ? "" : "none";
}

function showError(msg) {
  const el = document.getElementById("similar-error");
  el.textContent = msg;
  el.style.display = msg ? "" : "none";
}

function esc(s) {
  return String(s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
