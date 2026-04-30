// Sanhita app shell — sidebar navigation, Sources menu, workflow switcher.
// Works on top of brief.js (Assistant chat) and workspaces.js (Vault/Draft/etc).

(function () {
  "use strict";

  // ── Sidebar mode switching ──────────────────────────────────────────────
  const items = document.querySelectorAll(".side-item");
  const panes = document.querySelectorAll(".ws-pane");
  function showMode(mode) {
    items.forEach(i => i.classList.toggle("active", i.dataset.mode === mode));
    panes.forEach(p => { p.hidden = p.dataset.pane !== mode; });
    // Inside Workflows pane, always reset to the grid (hide detail panes)
    if (mode === "workflows") {
      document.querySelectorAll(".wf-pane").forEach(p => p.hidden = true);
      const grid = document.querySelector('[data-pane="workflows"] .wf-grid');
      if (grid) grid.hidden = false;
      const top = document.querySelector('[data-pane="workflows"] .wf-topbar');
      if (top) top.hidden = false;
    }
  }
  items.forEach(b => b.addEventListener("click", () => showMode(b.dataset.mode)));

  // ── Sources menu toggle ─────────────────────────────────────────────────
  const sourcesChip = document.getElementById("sources-chip");
  const sourcesMenu = document.getElementById("sources-menu");
  if (sourcesChip && sourcesMenu) {
    sourcesChip.addEventListener("click", (e) => {
      if (e.target.closest(".tool-menu")) return;
      sourcesMenu.hidden = !sourcesMenu.hidden;
    });
    document.addEventListener("click", (e) => {
      if (!sourcesChip.contains(e.target)) sourcesMenu.hidden = true;
    });
    // Reflect selected jurisdiction on the chip label
    sourcesMenu.querySelectorAll('input[name="juris"]').forEach(r => {
      r.addEventListener("change", () => {
        const label = sourcesMenu.querySelector(`label.menu-row input[value="${r.value}"]`).parentElement.textContent.trim();
        sourcesChip.querySelector(".tool-ic").nextSibling.textContent = " " + label.split(" ").slice(0, 2).join(" ");
        // stash on body so brief.js can pick it up
        document.body.dataset.jurisdiction = r.value;
      });
    });
    // Menu action buttons
    sourcesMenu.querySelectorAll(".menu-btn").forEach(b => {
      b.addEventListener("click", () => {
        const act = b.dataset.action;
        if (act === "upload") {
          showMode("vault");
          sourcesMenu.hidden = true;
        } else if (act === "vault") {
          showMode("vault");
          sourcesMenu.hidden = true;
        } else if (act === "kb") {
          showMode("library");
          sourcesMenu.hidden = true;
        } else if (act === "websearch") {
          document.body.dataset.websearch = "1";
          sourcesMenu.hidden = true;
        }
      });
    });
  }

  // ── Suggest chips → composer ────────────────────────────────────────────
  document.querySelectorAll(".suggest-chip").forEach(c => {
    c.addEventListener("click", () => {
      const t = document.getElementById("chat-input");
      if (t) { t.value = c.dataset.q; t.focus(); t.dispatchEvent(new Event("input", { bubbles: true })); }
    });
  });

  // ── Workflow cards → open sub-pane ──────────────────────────────────────
  const wfGrid = document.querySelector('[data-pane="workflows"] .wf-grid');
  const wfTop  = document.querySelector('[data-pane="workflows"] .wf-topbar');
  document.querySelectorAll(".wf-card[data-wf]").forEach(card => {
    card.addEventListener("click", () => {
      const key = card.dataset.wf;
      if (wfGrid) wfGrid.hidden = true;
      if (wfTop) wfTop.hidden = true;
      document.querySelectorAll(".wf-pane").forEach(p => p.hidden = true);
      const target = document.getElementById(`wf-pane-${key}`);
      if (target) target.hidden = false;
    });
  });
  document.querySelectorAll(".wf-back").forEach(b => {
    b.addEventListener("click", () => {
      document.querySelectorAll(".wf-pane").forEach(p => p.hidden = true);
      if (wfGrid) wfGrid.hidden = false;
      if (wfTop) wfTop.hidden = false;
    });
  });

  // ── File attach chip ────────────────────────────────────────────────────
  const compFiles = document.getElementById("comp-files");
  const compAttached = document.getElementById("comp-attached");
  if (compFiles && compAttached) {
    compFiles.addEventListener("change", () => {
      compAttached.innerHTML = [...compFiles.files].map(f =>
        `<span class="attached-pill">⎙ ${f.name}</span>`).join("");
    });
  }

  // ── /api/me → show user ─────────────────────────────────────────────────
  fetch("/api/me", { credentials: "same-origin" })
    .then(r => r.ok ? r.json() : null)
    .then(d => {
      const u = document.getElementById("who-am-i");
      if (u && d && d.email) u.textContent = d.email;
    }).catch(() => {});

  // ── Improve prompt (local sugar: expands a terse query) ─────────────────
  document.getElementById("improve-btn")?.addEventListener("click", () => {
    const t = document.getElementById("chat-input");
    if (!t || !t.value.trim()) return;
    const juris = document.body.dataset.jurisdiction || "IN";
    t.value = `${t.value.trim()}\n\nScope: ${juris}. Cite controlling authority. If no on-point authority exists, say so plainly.`;
    t.focus();
  });

  // ── New chat: clear log ─────────────────────────────────────────────────
  document.getElementById("new-chat-btn")?.addEventListener("click", () => {
    const log = document.getElementById("chat-log");
    if (log) log.innerHTML = "";
    showMode("research");
  });
})();
