/* LexSearch — FIR Analyzer */

document.addEventListener("DOMContentLoaded", () => {
  // Tab switching
  document.querySelectorAll(".fir-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".fir-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById("fir-text-tab").style.display = tab.dataset.tab === "text" ? "" : "none";
      document.getElementById("fir-pdf-tab").style.display = tab.dataset.tab === "pdf" ? "" : "none";
    });
  });

  document.getElementById("analyze-btn").addEventListener("click", analyzeFIR);
  document.getElementById("fir-back").addEventListener("click", () => {
    document.getElementById("fir-results").style.display = "none";
    document.getElementById("fir-input").style.display = "";
  });
});

async function analyzeFIR() {
  const activeTab = document.querySelector(".fir-tab.active").dataset.tab;
  let body;

  if (activeTab === "text") {
    const text = document.getElementById("fir-text").value.trim();
    if (!text) { showError("Please paste the FIR text."); return; }
    body = JSON.stringify({ text });
  } else {
    const file = document.getElementById("fir-file").files[0];
    if (!file) { showError("Please upload a PDF file."); return; }
    const formData = new FormData();
    formData.append("file", file);
    // Use FormData for file upload
    return analyzeFIRWithFile(formData);
  }

  showError("");
  document.getElementById("fir-input").style.display = "none";
  document.getElementById("fir-loading").style.display = "";

  try {
    const res = await fetch("/fir/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Analysis failed.");
    }
    const data = await res.json();
    renderResults(data);
  } catch (e) {
    showError(e.message);
    document.getElementById("fir-input").style.display = "";
  }
  document.getElementById("fir-loading").style.display = "none";
}

async function analyzeFIRWithFile(formData) {
  showError("");
  document.getElementById("fir-input").style.display = "none";
  document.getElementById("fir-loading").style.display = "";

  try {
    const res = await fetch("/fir/analyze-pdf", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Analysis failed.");
    }
    const data = await res.json();
    renderResults(data);
  } catch (e) {
    showError(e.message);
    document.getElementById("fir-input").style.display = "";
  }
  document.getElementById("fir-loading").style.display = "none";
}

function renderResults(data) {
  document.getElementById("fir-results").style.display = "";

  // Severity
  const sevBanner = document.getElementById("severity-banner");
  const severity = (data.severity || "medium").toLowerCase();
  sevBanner.className = `severity-banner severity-${severity}`;
  document.getElementById("severity-value").textContent = severity.charAt(0).toUpperCase() + severity.slice(1);

  // Sections table
  const tbody = document.getElementById("sections-tbody");
  const sections = data.sections_charged || [];
  tbody.innerHTML = sections.map(s => `
    <tr>
      <td><a href="/converter.html?q=${encodeURIComponent(s.section)}" class="section-link">${esc(s.section)}</a></td>
      <td>${esc(s.act || "IPC")}</td>
      <td>${esc(s.title || "—")}</td>
      <td class="${s.bailable ? 'text-green' : 'text-red'}">${s.bailable ? 'Yes' : 'No'}</td>
      <td>${s.cognizable ? 'Yes' : 'No'}</td>
      <td>${esc(s.max_punishment || "—")}</td>
    </tr>
  `).join("");

  // Summary
  document.getElementById("offense-summary").textContent = data.offense_summary || "—";

  // Bail
  document.getElementById("bail-likelihood").textContent = data.bail_analysis?.likelihood || "—";
  document.getElementById("bail-reasoning").textContent = data.bail_analysis?.reasoning || "—";

  // Timeline
  const timeline = document.getElementById("timeline");
  const steps = data.timeline || [];
  timeline.innerHTML = steps.map(step => `
    <div class="timeline-step">
      <div class="timeline-dot"></div>
      <div class="timeline-content">
        <strong>${esc(step.stage)}</strong>
        <span class="timeline-time">${esc(step.timeframe)}</span>
        <p>${esc(step.description)}</p>
      </div>
    </div>
  `).join("");

  // Next steps
  const nextSteps = document.getElementById("next-steps");
  const ns = data.next_steps || [];
  nextSteps.innerHTML = ns.map(s => `<li>${esc(s)}</li>`).join("");
}

function showError(msg) {
  const el = document.getElementById("fir-error");
  el.textContent = msg;
  el.style.display = msg ? "" : "none";
}

function esc(s) {
  return String(s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
