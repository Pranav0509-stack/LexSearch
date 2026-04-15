/* LexSearch — Legal Document Drafter */

let selectedDocType = "";

document.addEventListener("DOMContentLoaded", () => {
  // Doc type selection
  document.querySelectorAll(".doc-type-card").forEach(card => {
    card.addEventListener("click", () => {
      document.querySelectorAll(".doc-type-card").forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      selectedDocType = card.dataset.type;
      showStep(2);
      configureForm(selectedDocType);
    });
  });

  document.getElementById("back-to-step1").addEventListener("click", () => showStep(1));
  document.getElementById("back-to-step2").addEventListener("click", () => showStep(2));
  document.getElementById("generate-btn").addEventListener("click", generateDraft);
  document.getElementById("copy-btn").addEventListener("click", copyDraft);
  document.getElementById("download-btn").addEventListener("click", downloadDraft);
});

function configureForm(docType) {
  const extraGroup = document.getElementById("d-extra-group");
  const extraLabel = document.getElementById("d-extra-label");
  const extraInput = document.getElementById("d-extra");

  // Show/hide extra field and customize labels based on doc type
  switch (docType) {
    case "bail_application":
      extraGroup.style.display = "";
      extraLabel.textContent = "Grounds for Bail";
      extraInput.placeholder = "e.g. No flight risk, willing to cooperate, clean record, medical condition...";
      break;
    case "writ_petition":
      extraGroup.style.display = "";
      extraLabel.textContent = "Constitutional Provisions / Fundamental Rights Violated";
      extraInput.placeholder = "e.g. Article 14 (Equality), Article 21 (Life & Liberty)...";
      break;
    case "rti_application":
      extraGroup.style.display = "";
      extraLabel.textContent = "Specific Information Sought";
      extraInput.placeholder = "List the specific details/documents you want under RTI...";
      document.getElementById("d-court").placeholder = "e.g. PIO, Ministry of Education";
      break;
    case "legal_notice":
      extraGroup.style.display = "";
      extraLabel.textContent = "Legal Basis / Relevant Sections";
      extraInput.placeholder = "e.g. Section 138 NI Act, Consumer Protection Act...";
      break;
    case "affidavit":
      extraGroup.style.display = "";
      extraLabel.textContent = "Key Statements to Include";
      extraInput.placeholder = "List the main points you want to state on oath...";
      break;
    case "complaint":
      extraGroup.style.display = "";
      extraLabel.textContent = "IPC/BNS Sections Applicable";
      extraInput.placeholder = "e.g. IPC 420 (Cheating), IPC 406 (Breach of Trust)...";
      break;
    default:
      extraGroup.style.display = "none";
  }
}

async function generateDraft() {
  const party = document.getElementById("d-party").value.trim();
  const opposite = document.getElementById("d-opposite").value.trim();
  const court = document.getElementById("d-court").value.trim();
  const facts = document.getElementById("d-facts").value.trim();

  if (!party || !facts) {
    showError("Please fill at least your name and the facts.");
    return;
  }

  showError("");
  document.getElementById("step-2").style.display = "none";
  document.getElementById("drafter-loading").style.display = "";

  try {
    const body = {
      doc_type: selectedDocType,
      party_name: party,
      opposite_party: opposite,
      court: court,
      case_number: document.getElementById("d-caseno").value.trim(),
      facts: facts,
      relief: document.getElementById("d-relief").value.trim(),
      additional: document.getElementById("d-extra").value.trim(),
    };

    const res = await fetch("/draft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Draft generation failed.");
    }

    const data = await res.json();
    document.getElementById("draft-text").textContent = data.draft;
    document.getElementById("drafter-loading").style.display = "none";
    showStep(3);
  } catch (e) {
    showError(e.message);
    document.getElementById("drafter-loading").style.display = "none";
    document.getElementById("step-2").style.display = "";
  }
}

function copyDraft() {
  const text = document.getElementById("draft-text").textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById("copy-btn");
    btn.textContent = "Copied!";
    setTimeout(() => btn.textContent = "Copy to Clipboard", 2000);
  });
}

function downloadDraft() {
  const text = document.getElementById("draft-text").textContent;
  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${selectedDocType}_draft.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

function showStep(n) {
  [1, 2, 3].forEach(s => {
    const el = document.getElementById(`step-${s}`);
    if (el) el.style.display = s === n ? "" : "none";
  });
  document.getElementById("drafter-loading").style.display = "none";
}

function showError(msg) {
  const el = document.getElementById("drafter-error");
  el.textContent = msg;
  el.style.display = msg ? "" : "none";
}
