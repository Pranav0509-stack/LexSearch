/**
 * LexSearch – Frontend logic
 */
(function () {
  "use strict";

  let currentPage = 1;
  const PAGE_SIZE = 50;

  // DOM
  const courtSelect  = document.getElementById("court-select");
  const benchSelect  = document.getElementById("bench-select");
  const yearInput    = document.getElementById("year-input");
  const keywordInput = document.getElementById("keyword-input");
  const cnrInput     = document.getElementById("cnr-input");
  const judgeInput   = document.getElementById("judge-input");
  const caseTypeInput = document.getElementById("case-type-input");
  const disposalInput = document.getElementById("disposal-input");
  const searchBtn    = document.getElementById("search-btn");
  const clearBtn     = document.getElementById("clear-btn");
  const resultsGrid  = document.getElementById("results-grid");
  const pagination   = document.getElementById("pagination");
  const statsBar     = document.getElementById("stats-bar");
  const statsText    = document.getElementById("stats-text");
  const statsTime    = document.getElementById("stats-time");
  const errorBox     = document.getElementById("error-container");
  const welcomeEl    = document.getElementById("welcome-screen");
  const loadingEl    = document.getElementById("loading-screen");
  const emptyEl      = document.getElementById("empty-screen");

  let courtsData = [];

  // Load courts
  async function loadCourts() {
    try {
      const res = await fetch("/courts");
      courtsData = await res.json();
      courtSelect.innerHTML = '<option value="">All Courts</option>';
      courtsData.forEach(c => {
        const opt = document.createElement("option");
        opt.value = c.s3_code;
        opt.textContent = c.name;
        courtSelect.appendChild(opt);
      });
    } catch (e) {
      showError("Could not load courts. Is the server running?");
    }
  }

  courtSelect.addEventListener("change", () => {
    const code = courtSelect.value;
    benchSelect.innerHTML = '<option value="">All Benches</option>';
    benchSelect.disabled = !code;
    if (!code) return;
    const court = courtsData.find(c => c.s3_code === code);
    if (!court) return;
    court.benches.forEach(b => {
      const opt = document.createElement("option");
      opt.value = b.code;
      opt.textContent = b.name;
      benchSelect.appendChild(opt);
    });
  });

  // Search
  searchBtn.addEventListener("click", () => runSearch(1));
  clearBtn.addEventListener("click", clearFilters);

  // Enter key on any input triggers search
  document.querySelectorAll(".sidebar input, .sidebar select").forEach(el => {
    el.addEventListener("keydown", e => { if (e.key === "Enter") runSearch(1); });
  });

  function buildParams(page) {
    const p = new URLSearchParams();
    if (courtSelect.value)  p.set("court", courtSelect.value);
    if (benchSelect.value)  p.set("bench", benchSelect.value);
    if (yearInput.value)    p.set("year",  yearInput.value);
    if (keywordInput.value.trim()) p.set("q", keywordInput.value.trim());
    if (cnrInput.value.trim())     p.set("cnr", cnrInput.value.trim());
    if (judgeInput.value.trim())   p.set("judge", judgeInput.value.trim());
    if (caseTypeInput.value)       p.set("case_type", caseTypeInput.value);
    if (disposalInput.value)       p.set("disposal", disposalInput.value);
    p.set("page", page);
    return p;
  }

  async function runSearch(page) {
    currentPage = page;
    const params = buildParams(page);

    showLoading();
    clearError();

    const t0 = Date.now();
    try {
      const res = await fetch(`/search?${params}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      renderResults(data.results, data.total, elapsed, page);
    } catch (e) {
      showError(e.message);
      hideLoading();
    }
  }

  // Render
  function renderResults(results, total, elapsed, page) {
    hideLoading();
    resultsGrid.innerHTML = "";
    pagination.style.display = "none";

    if (!results || results.length === 0) {
      emptyEl.style.display = "block";
      statsBar.style.display = "none";
      return;
    }

    emptyEl.style.display = "none";
    statsBar.style.display = "flex";
    statsText.innerHTML = `<strong>${total.toLocaleString()}</strong> results`;
    statsTime.textContent = `${elapsed}s`;

    results.forEach(r => resultsGrid.appendChild(buildCard(r)));

    if (total > PAGE_SIZE) renderPagination(total, page);
  }

  function buildCard(r) {
    const card = document.createElement("div");
    card.className = "judgment-card";

    const courtName = r.court_name || getCourtName(r.court);
    const benchName = getBenchName(r.court, r.bench);
    const viewerUrl = `viewer.html?key=${encodeURIComponent(r.s3_key)}&title=${encodeURIComponent(r.title || r.case_number)}&court=${encodeURIComponent(courtName)}`;
    const downloadUrl = `/pdf/${r.s3_key}?download=true`;

    card.innerHTML = `
      <div class="card-body">
        <div class="card-title" title="${esc(r.title)}">${esc(r.title || r.case_number || "Untitled")}</div>
        <div class="card-meta">
          <span><span class="label">CNR:</span> ${esc(r.case_number)}</span>
          ${r.judge ? `<span><span class="label">Judge:</span> ${esc(r.judge)}</span>` : ""}
          ${r.date && r.date !== String(r.year) ? `<span><span class="label">Date:</span> ${esc(r.date.split(' ')[0])}</span>` : ""}
        </div>
        <div class="card-tags">
          <span class="tag court">${esc(courtName)}</span>
          ${benchName !== courtName ? `<span class="tag">${esc(benchName)}</span>` : ""}
          <span class="tag year">${r.year}</span>
          ${r.disposal ? `<span class="tag disposal">${esc(r.disposal)}</span>` : ""}
        </div>
      </div>
      <div class="card-actions">
        <a href="${viewerUrl}" target="_blank" class="btn btn-read">Read</a>
        <a href="${downloadUrl}" class="btn btn-download" download>Download</a>
      </div>
    `;
    return card;
  }

  function renderPagination(total, pg) {
    const pages = Math.ceil(total / PAGE_SIZE);
    pagination.style.display = "flex";
    pagination.innerHTML = "";

    addPageBtn("Prev", pg > 1, () => runSearch(pg - 1));

    const start = Math.max(1, pg - 2);
    const end = Math.min(pages, pg + 2);
    for (let i = start; i <= end; i++) {
      const btn = document.createElement("button");
      btn.className = `page-btn${i === pg ? " active" : ""}`;
      btn.textContent = i;
      btn.addEventListener("click", () => runSearch(i));
      pagination.appendChild(btn);
    }

    addPageBtn("Next", pg < pages, () => runSearch(pg + 1));
  }

  function addPageBtn(text, enabled, fn) {
    const btn = document.createElement("button");
    btn.className = "page-btn";
    btn.textContent = text;
    btn.disabled = !enabled;
    btn.addEventListener("click", fn);
    pagination.appendChild(btn);
  }

  // Helpers
  function getCourtName(code) {
    const c = courtsData.find(x => x.s3_code === code);
    return c ? c.name : code;
  }

  function getBenchName(courtCode, benchCode) {
    const c = courtsData.find(x => x.s3_code === courtCode);
    if (!c) return benchCode;
    const b = c.benches.find(x => x.code === benchCode);
    return b ? b.name : benchCode;
  }

  function esc(s) {
    return String(s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function showLoading() {
    welcomeEl.style.display = "none";
    emptyEl.style.display = "none";
    loadingEl.style.display = "block";
    resultsGrid.innerHTML = "";
    pagination.style.display = "none";
    statsBar.style.display = "none";
    searchBtn.disabled = true;
    searchBtn.textContent = "Searching...";
  }

  function hideLoading() {
    loadingEl.style.display = "none";
    searchBtn.disabled = false;
    searchBtn.textContent = "Search";
  }

  function showError(msg) {
    errorBox.innerHTML = `<div class="error-toast">${esc(msg)}</div>`;
  }

  function clearError() { errorBox.innerHTML = ""; }

  function clearFilters() {
    courtSelect.value = "";
    benchSelect.value = "";
    benchSelect.disabled = true;
    benchSelect.innerHTML = '<option value="">All Benches</option>';
    yearInput.value = "";
    keywordInput.value = "";
    cnrInput.value = "";
    judgeInput.value = "";
    caseTypeInput.value = "";
    disposalInput.value = "";
    resultsGrid.innerHTML = "";
    pagination.style.display = "none";
    statsBar.style.display = "none";
    welcomeEl.style.display = "block";
    emptyEl.style.display = "none";
    clearError();
  }

  loadCourts();
})();
