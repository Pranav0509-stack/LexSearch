/**
 * LexSearch – Frontend (HC + SC)
 */
(function () {
  "use strict";

  let currentPage = 1;
  let currentMode = "hc"; // "hc" or "sc"
  const PAGE_SIZE = 50;

  // DOM
  const courtSelect   = document.getElementById("court-select");
  const benchSelect   = document.getElementById("bench-select");
  const yearInput     = document.getElementById("year-input");
  const keywordInput  = document.getElementById("keyword-input");
  const cnrInput      = document.getElementById("cnr-input");
  const judgeInput    = document.getElementById("judge-input");
  const casetypeInput = document.getElementById("casetype-input");
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
    if (casetypeInput.value)        p.set("case_type", casetypeInput.value);
    if (disposalInput.value)       p.set("disposal", disposalInput.value);
    p.set("page", page);
    return p;
  }

  async function runSearch(page) {
    currentPage = page;
    showLoading();
    clearError();
    const t0 = Date.now();
    try {
      const res = await fetch(`/search?${buildParams(page)}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      renderResults(data.results, data.total, ((Date.now() - t0) / 1000).toFixed(1), page);
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
        <button class="btn btn-summarize" data-s3key="${esc(r.s3_key)}" data-sc="${isSC}" data-year="${r.year}" data-pdfname="${isSC ? esc(decodeURIComponent(r.s3_key.split('/').pop())) : ''}">Summarize</button>
      </div>
    `;
    // Summary container (hidden by default)
    const summaryDiv = document.createElement("div");
    summaryDiv.className = "card-summary";
    summaryDiv.style.display = "none";
    card.appendChild(summaryDiv);

    // Summarize button handler
    const sumBtn = card.querySelector(".btn-summarize");
    sumBtn.addEventListener("click", () => handleSummarize(sumBtn, summaryDiv, r));

    return card;
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
    casetypeInput.value = "";
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

  // AI Summarize handler
  async function handleSummarize(btn, container, r) {
    if (container.style.display !== "none") {
      container.style.display = "none";
      btn.textContent = "Summarize";
      return;
    }

    btn.textContent = "Loading...";
    btn.disabled = true;
    container.style.display = "block";
    container.innerHTML = '<div class="summary-loading">Analyzing judgment with AI...</div>';

    const isSC = r.type === "sc";
    const body = {};
    if (isSC) {
      body.sc = true;
      body.year = r.year;
      body.pdf_name = decodeURIComponent(r.s3_key.split("/").pop());
    } else {
      body.s3_key = r.s3_key;
    }

    try {
      const res = await fetch("/ai/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Summarization failed.");
      }
      const data = await res.json();
      renderSummary(container, data, body);
      btn.textContent = "Hide Summary";
    } catch (e) {
      container.innerHTML = `<div class="summary-error">${esc(e.message)}</div>`;
      btn.textContent = "Retry";
    }
    btn.disabled = false;
  }

  function renderSummary(container, data, reqBody) {
    if (data.raw_summary) {
      container.innerHTML = `<div class="summary-content"><p>${esc(data.raw_summary)}</p></div>`;
      return;
    }

    let html = '<div class="summary-content">';
    if (data.facts) html += `<div class="summary-section"><h4>Facts</h4><p>${esc(data.facts)}</p></div>`;
    if (data.issues && data.issues.length) html += `<div class="summary-section"><h4>Issues</h4><ul>${data.issues.map(i => `<li>${esc(i)}</li>`).join("")}</ul></div>`;
    if (data.arguments) html += `<div class="summary-section"><h4>Arguments</h4><p>${esc(data.arguments)}</p></div>`;
    if (data.held) html += `<div class="summary-section"><h4>Held</h4><p>${esc(data.held)}</p></div>`;
    if (data.ratio) html += `<div class="summary-section"><h4>Ratio Decidendi</h4><p>${esc(data.ratio)}</p></div>`;
    if (data.statutes && data.statutes.length) html += `<div class="summary-section"><h4>Statutes Cited</h4><div class="summary-tags">${data.statutes.map(s => `<span class="tag">${esc(s)}</span>`).join("")}</div></div>`;
    if (data.result) html += `<div class="summary-section"><h4>Result</h4><span class="tag disposal">${esc(data.result)}</span></div>`;

    // Translate button
    html += `<button class="btn btn-translate" onclick="this.disabled=true;this.textContent='Translating...';translateSummary(this, ${JSON.stringify(JSON.stringify(data))})">Translate to Hindi</button>`;
    html += '<div class="translation-output" style="display:none"></div>';

    // Q&A section
    html += `<div class="summary-qa">
      <h4>Ask about this case</h4>
      <div class="qa-input-row">
        <input type="text" class="qa-input" placeholder="e.g. What statute was applied?" />
        <button class="btn btn-read qa-btn">Ask</button>
      </div>
      <div class="qa-answer" style="display:none"></div>
    </div>`;
    html += '</div>';

    container.innerHTML = html;

    // Q&A handler
    const qaBtn = container.querySelector(".qa-btn");
    const qaInput = container.querySelector(".qa-input");
    const qaAnswer = container.querySelector(".qa-answer");
    qaBtn.addEventListener("click", () => handleQA(qaInput, qaAnswer, qaBtn, reqBody));
    qaInput.addEventListener("keydown", e => { if (e.key === "Enter") handleQA(qaInput, qaAnswer, qaBtn, reqBody); });
  }

  async function handleQA(input, answerDiv, btn, reqBody) {
    const question = input.value.trim();
    if (!question) return;
    btn.disabled = true;
    btn.textContent = "Thinking...";
    answerDiv.style.display = "block";
    answerDiv.innerHTML = '<em>Analyzing...</em>';

    try {
      const res = await fetch("/ai/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...reqBody, question }),
      });
      if (!res.ok) throw new Error("Failed.");
      const data = await res.json();
      answerDiv.innerHTML = `<p>${esc(data.answer)}</p>`;
    } catch (e) {
      answerDiv.innerHTML = `<p class="summary-error">${esc(e.message)}</p>`;
    }
    btn.disabled = false;
    btn.textContent = "Ask";
  }

  // Global translate function (called from onclick)
  window.translateSummary = async function(btn, summaryJson) {
    const data = JSON.parse(summaryJson);
    const text = [data.facts, data.held, data.ratio].filter(Boolean).join("\n\n");
    const outputDiv = btn.parentElement.querySelector(".translation-output");

    try {
      const res = await fetch("/ai/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error("Translation failed.");
      const result = await res.json();
      outputDiv.style.display = "block";
      outputDiv.innerHTML = `<h4>Hindi Translation</h4><p>${esc(result.translation)}</p>`;
      btn.textContent = "Translated";
    } catch (e) {
      outputDiv.style.display = "block";
      outputDiv.innerHTML = `<p class="summary-error">${esc(e.message)}</p>`;
      btn.textContent = "Retry Translation";
      btn.disabled = false;
    }
  };

  loadCourts();
})();
