/* LexSearch — Case Predictor */

let outcomeChart = null;

document.addEventListener("DOMContentLoaded", async () => {
  await loadCourts();
  document.getElementById("predict-btn").addEventListener("click", runPrediction);
});

async function loadCourts() {
  try {
    const res = await fetch("/courts");
    const courts = await res.json();
    const sel = document.getElementById("pred-court");
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

async function runPrediction() {
  const court = document.getElementById("pred-court").value;
  const caseType = document.getElementById("pred-casetype").value;

  if (!court || !caseType) {
    showError("Please select both a court and case type.");
    return;
  }

  showError("");
  document.getElementById("predict-welcome").style.display = "none";
  document.getElementById("predict-results").style.display = "none";
  document.getElementById("predict-loading").style.display = "";

  try {
    const res = await fetch(`/predict?court=${court}&case_type=${caseType}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Prediction failed.");
    }
    const data = await res.json();
    renderPrediction(data);
  } catch (e) {
    showError(e.message);
  }

  document.getElementById("predict-loading").style.display = "none";
}

function renderPrediction(data) {
  const results = document.getElementById("predict-results");
  results.style.display = "";

  // Outcome
  const probs = data.outcome_probability || {};
  const topOutcome = Object.entries(probs).sort((a, b) => b[1] - a[1])[0];
  document.getElementById("pred-outcome").textContent = topOutcome ? topOutcome[0] : "Unknown";
  document.getElementById("pred-outcome-pct").textContent = topOutcome ? `${topOutcome[1]}% probability` : "";

  // Color the outcome card
  const outcomeCard = document.querySelector(".outcome-card");
  if (topOutcome) {
    outcomeCard.className = "predict-stat-card outcome-card";
    if (topOutcome[0] === "Allowed") outcomeCard.classList.add("outcome-allowed");
    else if (topOutcome[0] === "Dismissed") outcomeCard.classList.add("outcome-dismissed");
    else outcomeCard.classList.add("outcome-disposed");
  }

  // Duration
  if (data.duration) {
    document.getElementById("pred-duration").textContent = `~${data.duration.median_months} months`;
    document.getElementById("pred-duration-sub").textContent = `Based on ${data.duration.based_on.toLocaleString()} similar cases`;
  } else {
    document.getElementById("pred-duration").textContent = "No data";
    document.getElementById("pred-duration-sub").textContent = "Insufficient historical data";
  }

  // Cost
  if (data.cost_estimate) {
    const min = formatINR(data.cost_estimate.min_inr);
    const max = formatINR(data.cost_estimate.max_inr);
    document.getElementById("pred-cost").textContent = `${min} - ${max}`;
    document.getElementById("pred-cost-sub").textContent = data.cost_estimate.note;
  }

  // Outcome chart
  if (outcomeChart) outcomeChart.destroy();
  const labels = Object.keys(probs);
  const values = Object.values(probs);
  const colors = labels.map(l => {
    if (l === "Allowed") return "#059669";
    if (l === "Dismissed") return "#dc2626";
    return "#c9a84c";
  });

  outcomeChart = new Chart(document.getElementById("outcome-chart"), {
    type: "doughnut",
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderWidth: 2,
        borderColor: "#fff",
      }],
    },
    options: {
      responsive: true,
      cutout: "60%",
      plugins: {
        legend: { position: "bottom", labels: { font: { size: 13 }, padding: 15 } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.label}: ${ctx.parsed}%`
          }
        }
      },
    },
  });

  // Explanation
  const expDiv = document.getElementById("prediction-explanation");
  let expHtml = `<div class="exp-item"><strong>Court:</strong> ${esc(data.court)}</div>`;
  expHtml += `<div class="exp-item"><strong>Case Type:</strong> ${esc(data.case_type)}</div>`;

  if (topOutcome) {
    expHtml += `<div class="exp-item"><strong>Prediction:</strong> Based on historical patterns, a <em>${esc(data.cost_estimate?.label || data.case_type)}</em> in <em>${esc(data.court)}</em> has a <strong>${topOutcome[1]}%</strong> chance of being <strong>${topOutcome[0]}</strong>.</div>`;
  }

  if (data.duration) {
    expHtml += `<div class="exp-item"><strong>Timeline:</strong> The median time from filing to judgment is <strong>${data.duration.median_months} months</strong> (${data.duration.median_days} days). The average is ${data.duration.mean_months} months.</div>`;
  }

  if (data.cost_estimate) {
    expHtml += `<div class="exp-item"><strong>Cost Breakdown:</strong> Estimated legal fees range from <strong>${formatINR(data.cost_estimate.min_inr)}</strong> to <strong>${formatINR(data.cost_estimate.max_inr)}</strong>. This includes lawyer fees and court costs. Actual costs depend on case complexity, number of hearings, and lawyer seniority.</div>`;
  }

  // Outcome breakdown
  expHtml += '<div class="exp-item"><strong>Outcome Breakdown:</strong><ul>';
  Object.entries(probs).sort((a, b) => b[1] - a[1]).forEach(([k, v]) => {
    expHtml += `<li>${esc(k)}: ${v}%</li>`;
  });
  expHtml += '</ul></div>';

  expDiv.innerHTML = expHtml;
}

function formatINR(amount) {
  if (amount >= 100000) return `₹${(amount / 100000).toFixed(1)}L`;
  if (amount >= 1000) return `₹${(amount / 1000).toFixed(0)}K`;
  return `₹${amount}`;
}

function esc(s) {
  return String(s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function showError(msg) {
  const el = document.getElementById("predict-error");
  el.textContent = msg;
  el.style.display = msg ? "" : "none";
}
