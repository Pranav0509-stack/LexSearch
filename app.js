/**
 * Sanhita – Frontend (HC + SC)
 */
(function () {
  "use strict";

  let currentPage = 1;
  let currentMode = "hc"; // "hc" or "sc"
  let inFlight = false;   // prevents double-submit while a search is running
  const PAGE_SIZE = 50;
  const REQUEST_TIMEOUT_MS = 15000;

  // ── fetch with timeout ───────────────────────────────────────────────
  async function fetchWithTimeout(url, opts = {}, ms = REQUEST_TIMEOUT_MS) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), ms);
    try {
      return await fetch(url, { ...opts, signal: ctrl.signal });
    } finally {
      clearTimeout(t);
    }
  }

  // DOM
  const courtSelect   = document.getElementById("court-select");
  const benchSelect   = document.getElementById("bench-select");
  const yearInput     = document.getElementById("year-input");
  const keywordInput  = document.getElementById("keyword-input");
  const cnrInput      = document.getElementById("cnr-input");
  const judgeInput    = document.getElementById("judge-input");
  const disposalInput = document.getElementById("disposal-input");
  const petitionerInput = document.getElementById("petitioner-input");
  const respondentInput = document.getElementById("respondent-input");
  const citationInput   = document.getElementById("citation-input");
  const searchBtn     = document.getElementById("search-btn");
  const clearBtn      = document.getElementById("clear-btn");
  const resultsGrid   = document.getElementById("results-grid");
  const pagination    = document.getElementById("pagination");
  const statsBar      = document.getElementById("stats-bar");
  const statsText     = document.getElementById("stats-text");
  const statsTime     = document.getElementById("stats-time");
  const errorBox      = document.getElementById("error-container");
  const welcomeEl     = document.getElementById("welcome-screen");
  const loadingEl     = document.getElementById("loading-screen");
  const emptyEl       = document.getElementById("empty-screen");
  const hcFilters     = document.getElementById("hc-filters");
  const scFilters     = document.getElementById("sc-filters");

  let courtsData = [];

  // Toggle HC / SC
  document.querySelectorAll(".toggle-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentMode = btn.dataset.mode;
      hcFilters.style.display = currentMode === "hc" ? "block" : "none";
      scFilters.style.display = currentMode === "sc" ? "block" : "none";
    });
  });

  // Load courts
  async function loadCourts() {
    try {
      const res = await fetchWithTimeout("/courts", {}, 8000);
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

  document.querySelectorAll(".sidebar input, .sidebar select").forEach(el => {
    el.addEventListener("keydown", e => { if (e.key === "Enter") runSearch(1); });
  });

  function buildParams(page) {
    const p = new URLSearchParams();
    p.set("mode", currentMode);
    if (currentMode === "hc") {
      if (courtSelect.value)  p.set("court", courtSelect.value);
      if (benchSelect.value)  p.set("bench", benchSelect.value);
    } else {
      if (petitionerInput.value.trim()) p.set("petitioner", petitionerInput.value.trim());
      if (respondentInput.value.trim()) p.set("respondent", respondentInput.value.trim());
      if (citationInput.value.trim())   p.set("citation", citationInput.value.trim());
    }
    if (yearInput.value)           p.set("year", yearInput.value);
    if (keywordInput.value.trim()) p.set("q", keywordInput.value.trim());
    if (cnrInput.value.trim())     p.set("cnr", cnrInput.value.trim());
    if (judgeInput.value.trim())   p.set("judge", judgeInput.value.trim());
    if (disposalInput.value)       p.set("disposal", disposalInput.value);
    p.set("page", page);
    return p;
  }

  // Client-side filter validation. Returns a human-readable error string
  // if the request would be rejected by the server, else null.
  function validateFilters() {
    if (currentMode === "sc") {
      const any = [
        petitionerInput.value, respondentInput.value, citationInput.value,
        keywordInput.value, cnrInput.value, judgeInput.value, yearInput.value,
      ].some(v => String(v || "").trim());
      if (!any) {
        return "Enter a petitioner, respondent, citation, or year to search the Supreme Court.";
      }
    } else {
      const any = [
        courtSelect.value, keywordInput.value, cnrInput.value,
        judgeInput.value, yearInput.value,
      ].some(v => String(v || "").trim());
      if (!any) {
        return "Pick a High Court, or enter a party name, CNR, judge, or year.";
      }
    }
    return null;
  }

  function shakeButton() {
    searchBtn.classList.remove("shake");
    // force reflow so the animation restarts
    // eslint-disable-next-line no-unused-expressions
    void searchBtn.offsetWidth;
    searchBtn.classList.add("shake");
  }

  async function runSearch(page) {
    if (inFlight) return; // ignore double-fire

    const validationError = validateFilters();
    if (validationError) {
      showError(validationError);
      shakeButton();
      return;
    }

    inFlight = true;
    currentPage = page;
    showLoading();
    clearError();
    const t0 = Date.now();
    try {
      const res = await fetchWithTimeout(`/search?${buildParams(page)}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      renderResults(data.results, data.total, ((Date.now() - t0) / 1000).toFixed(1), page);
    } catch (e) {
      const msg = e && e.name === "AbortError"
        ? "The court archive is slow right now. Try again, or narrow your filters."
        : (e && e.message) || "Something went wrong. Try again.";
      showError(msg);
      hideLoading();
    } finally {
      inFlight = false;
    }
  }

  // Render
  function renderResults(results, total, elapsed, page) {
    hideLoading();
    resultsGrid.innerHTML = "";
    pagination.style.display = "none";

    if (!results || !results.length) {
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

    const isSC = r.type === "sc";
    const courtName = r.court_name || getCourtName(r.court);
    const benchName = !isSC ? getBenchName(r.court, r.bench) : "";

    let pdfUrl, downloadUrl;
    if (isSC) {
      // SC: use /sc-pdf/{year}/{filename}
      const pdfName = r.s3_key.split("/").pop();
      const decoded = decodeURIComponent(pdfName);
      pdfUrl = `/sc-pdf/${r.year}/${decoded}`;
      downloadUrl = `/sc-pdf/${r.year}/${decoded}?download=true`;
    } else {
      pdfUrl = `/pdf/${r.s3_key}`;
      downloadUrl = `/pdf/${r.s3_key}?download=true`;
    }

    const viewerUrl = `viewer.html?${isSC ? 'sc=1&' : ''}key=${encodeURIComponent(r.s3_key)}&year=${r.year}&title=${encodeURIComponent(r.title || r.case_number)}&court=${encodeURIComponent(courtName)}`;

    let metaHtml = "";
    if (r.case_number) metaHtml += `<span><span class="label">${isSC ? 'Case:' : 'CNR:'}</span> ${esc(r.case_number)}</span>`;
    if (r.judge) metaHtml += `<span><span class="label">Judge:</span> ${esc(r.judge)}</span>`;
    if (r.date && r.date !== String(r.year)) metaHtml += `<span><span class="label">Date:</span> ${esc(r.date.split(' ')[0])}</span>`;
    if (isSC && r.citation) metaHtml += `<span><span class="label">Cite:</span> ${esc(r.citation)}</span>`;

    let tagsHtml = `<span class="tag ${isSC ? 'sc' : 'court'}">${esc(courtName)}</span>`;
    if (benchName && benchName !== courtName) tagsHtml += `<span class="tag">${esc(benchName)}</span>`;
    tagsHtml += `<span class="tag year">${r.year}</span>`;
    if (r.disposal) tagsHtml += `<span class="tag disposal">${esc(r.disposal)}</span>`;

    card.innerHTML = `
      <div class="card-body">
        <div class="card-title" title="${esc(r.title)}">${esc(r.title || r.case_number || "Untitled")}</div>
        <div class="card-meta">${metaHtml}</div>
        <div class="card-tags">${tagsHtml}</div>
      </div>
      <div class="card-actions">
        <a href="${viewerUrl}" target="_blank" class="btn btn-read">Read</a>
        <a href="${downloadUrl}" class="btn btn-download" download>Download</a>
      </div>
    `;

    // Probe the PDF with a HEAD request. If it 404s, mark the card
    // unavailable instead of silently opening a broken viewer tab.
    const probeUrl = isSC
      ? `/sc-pdf/${r.year}/${encodeURIComponent(decodeURIComponent(r.s3_key.split("/").pop()))}`
      : `/pdf/${r.s3_key}`;
    fetchWithTimeout(probeUrl, { method: "HEAD" }, 8000)
      .then(res => {
        if (!res.ok) markCardUnavailable(card);
      })
      .catch(() => { /* network blip — let the user try anyway */ });

    return card;
  }

  function markCardUnavailable(card) {
    card.classList.add("unavailable");
    const actions = card.querySelector(".card-actions");
    if (actions) {
      actions.innerHTML = '<span class="unavailable-tag" title="This PDF is not in the public archive">PDF unavailable</span>';
    }
  }

  function renderPagination(total, pg) {
    const pages = Math.ceil(total / PAGE_SIZE);
    pagination.style.display = "flex";
    pagination.innerHTML = "";
    addPageBtn("Prev", pg > 1, () => runSearch(pg - 1));
    const start = Math.max(1, pg - 2), end = Math.min(pages, pg + 2);
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
    if (!c) return benchCode || "";
    const b = c.benches.find(x => x.code === benchCode);
    return b ? b.name : benchCode || "";
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

  function showError(msg) { errorBox.innerHTML = `<div class="error-toast">${esc(msg)}</div>`; }
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
    disposalInput.value = "";
    petitionerInput.value = "";
    respondentInput.value = "";
    citationInput.value = "";
    resultsGrid.innerHTML = "";
    pagination.style.display = "none";
    statsBar.style.display = "none";
    welcomeEl.style.display = "block";
    emptyEl.style.display = "none";
    clearError();
  }

  loadCourts();
})();
