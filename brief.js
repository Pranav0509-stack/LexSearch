/**
 * Sanhita Brief — chat UI.
 *
 * Minimal single-page client:
 *   - loads the user's recent threads
 *   - opens a thread, renders its message history
 *   - sends new questions to /api/brief/chat and renders the reply + citations
 *
 * No framework. Everything is DOM-native so the bundle is ~0 KB.
 */

(function () {
  'use strict';

  const threadList = document.getElementById('thread-list');
  const newThreadBtn = document.getElementById('new-thread-btn');
  const chatLog = document.getElementById('chat-log');
  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input');
  const chatSend = document.getElementById('chat-send');
  const sourcesList = document.getElementById('sources-list');
  const whoAmI = document.getElementById('who-am-i');

  let state = {
    user: null,
    threads: [],
    activeThreadId: null,
    sending: false,
  };

  // ── helpers ────────────────────────────────────────────────────────────

  const esc = (s) =>
    String(s ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

  // Very small markdown: **bold**, *italic*, `code`, and [n] → superscript chip.
  function renderMarkdown(md) {
    let html = esc(md);
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\[(\d+)\]/g, '<sup class="cite-chip" data-n="$1">[$1]</sup>');
    html = html.replace(/\n{2,}/g, '</p><p>');
    html = html.replace(/\n/g, '<br/>');
    return `<p>${html}</p>`;
  }

  async function api(path, opts = {}) {
    const resp = await fetch(path, {
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    });
    if (resp.status === 401) {
      window.location.href = '/login';
      throw new Error('unauthenticated');
    }
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    return data;
  }

  // ── threads sidebar ────────────────────────────────────────────────────

  function renderThreads() {
    threadList.innerHTML = '';
    if (!state.threads.length) {
      threadList.innerHTML = '<li class="thread-empty">No matters yet.</li>';
      return;
    }
    for (const t of state.threads) {
      const li = document.createElement('li');
      li.className = 'thread-item' + (t.id === state.activeThreadId ? ' active' : '');
      li.dataset.id = t.id;
      const title = t.title || 'Untitled matter';
      const when = new Date(t.updated_at * 1000).toLocaleDateString('en-IN', {
        day: '2-digit', month: 'short',
      });
      li.innerHTML = `
        <div class="thread-title">${esc(title)}</div>
        <div class="thread-date">${when}</div>
      `;
      li.addEventListener('click', () => openThread(t.id));
      threadList.appendChild(li);
    }
  }

  async function loadThreads() {
    const data = await api('/api/brief/threads');
    state.threads = data.threads || [];
    state.user = data.user;
    if (whoAmI && state.user) {
      whoAmI.textContent = state.user.name || state.user.email || '';
    }
    renderThreads();
  }

  async function openThread(id) {
    state.activeThreadId = id;
    renderThreads();
    const data = await api(`/api/brief/threads/${id}`);
    chatLog.innerHTML = '';
    sourcesList.innerHTML = '<p class="sources-empty">Citations from your latest answer will appear here.</p>';
    for (const m of data.messages || []) {
      appendMessage(m.role, m.content, m.citations ? safeParse(m.citations) : null, { animate: false });
    }
    if (data.messages && data.messages.length) {
      // show sources of the most recent assistant message
      const lastA = [...data.messages].reverse().find((m) => m.role === 'assistant');
      if (lastA && lastA.citations) renderSources(safeParse(lastA.citations));
    }
  }

  function safeParse(s) {
    try { return JSON.parse(s); } catch { return null; }
  }

  async function newThread() {
    const data = await api('/api/brief/threads', { method: 'POST', body: '{}' });
    state.threads.unshift(data.thread);
    state.activeThreadId = data.thread.id;
    chatLog.innerHTML = '';
    sourcesList.innerHTML = '<p class="sources-empty">Citations from your latest answer will appear here.</p>';
    renderThreads();
  }

  // ── chat log ───────────────────────────────────────────────────────────

  function appendMessage(role, content, citations, opts = {}) {
    // Remove the empty-state block if present
    const empty = chatLog.querySelector('.chat-empty');
    if (empty) empty.remove();

    const wrap = document.createElement('div');
    const refused = !!opts.refused;
    wrap.className = `msg msg-${role}` + (opts.animate === false ? '' : ' msg-fade') + (refused ? ' msg-refused' : '');
    const body =
      role === 'assistant'
        ? renderMarkdown(content)
        : `<p>${esc(content).replace(/\n/g, '<br/>')}</p>`;

    // AI-disclosure chip + confidence pill + refusal badge for assistant turns
    let meta = '';
    if (role === 'assistant') {
      const llm = opts.llm || {};
      const v = opts.validation || {};
      const provider = llm.provider && llm.provider !== 'none' && llm.provider !== 'error'
        ? `<span class="ai-chip" title="Generated by ${esc(llm.model || llm.provider)} via ${esc(llm.provider)} · ${llm.latency_ms || 0}ms">AI · ${esc(llm.provider)}</span>`
        : '';
      const conf = typeof v.confidence === 'number'
        ? `<span class="conf-pill conf-${v.confidence >= 0.85 ? 'hi' : v.confidence >= 0.6 ? 'mid' : 'lo'}" title="${esc((v.reasons || []).join(' · ') || 'all six gates passed')}">${Math.round(v.confidence * 100)}% grounded</span>`
        : '';
      const refusedBadge = refused
        ? `<span class="refused-badge" title="Answer refused — fabrication risk too high">refused · cases only</span>`
        : '';
      meta = `<div class="msg-meta">${provider}${conf}${refusedBadge}</div>`;
    }

    wrap.innerHTML = `<div class="msg-role">${role === 'user' ? 'You' : 'Sanhita Brief'}</div>
                      <div class="msg-body">${body}</div>${meta}`;
    chatLog.appendChild(wrap);
    chatLog.scrollTop = chatLog.scrollHeight;

    // Hover-highlight citation chip → scroll source pane
    if (role === 'assistant' && citations) {
      wrap.querySelectorAll('.cite-chip').forEach((chip) => {
        chip.addEventListener('click', () => {
          const n = chip.dataset.n;
          const el = document.querySelector(`.source-card[data-n="${n}"]`);
          if (el) {
            el.scrollIntoView({ behavior: 'smooth', block: 'center' });
            el.classList.add('flash');
            setTimeout(() => el.classList.remove('flash'), 1200);
          }
        });
      });
    }
    return wrap;
  }

  function appendThinking() {
    const empty = chatLog.querySelector('.chat-empty');
    if (empty) empty.remove();
    const wrap = document.createElement('div');
    wrap.className = 'msg msg-assistant msg-thinking';
    wrap.innerHTML = `<div class="msg-role">Sanhita Brief</div>
                      <div class="msg-body"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>`;
    chatLog.appendChild(wrap);
    chatLog.scrollTop = chatLog.scrollHeight;
    return wrap;
  }

  // ── sources pane ───────────────────────────────────────────────────────

  function renderSources(citations) {
    if (!citations || !citations.length) {
      sourcesList.innerHTML = '<p class="sources-empty">No citations for this answer.</p>';
      return;
    }
    sourcesList.innerHTML = '';
    for (const c of citations) {
      const card = document.createElement('div');
      card.className = 'source-card';
      card.dataset.n = c.n;
      const linkStart = c.pdf_url ? `<a href="${esc(c.pdf_url)}" target="_blank" rel="noopener">` : '<div>';
      const linkEnd = c.pdf_url ? '</a>' : '</div>';
      card.innerHTML = `
        <div class="source-n">[${c.n}]</div>
        ${linkStart}
          <div class="source-title">${esc(c.title)}</div>
          <div class="source-meta">${esc(c.court || '')}${c.year ? ' · ' + esc(c.year) : ''}${c.citation ? ' · ' + esc(c.citation) : ''}</div>
          <div class="source-excerpt">${esc(c.excerpt || '')}</div>
        ${linkEnd}
      `;
      sourcesList.appendChild(card);
    }
  }

  // ── send question ──────────────────────────────────────────────────────

  async function sendQuestion(text) {
    if (state.sending) return;
    if (!text.trim()) return;
    state.sending = true;
    chatSend.disabled = true;
    chatInput.disabled = true;

    // Ensure there's a thread to post into
    if (!state.activeThreadId) {
      try {
        await newThread();
      } catch (e) {
        state.sending = false;
        chatSend.disabled = false;
        chatInput.disabled = false;
        return;
      }
    }

    appendMessage('user', text, null);
    chatInput.value = '';
    const thinking = appendThinking();

    try {
      const jurisSel = document.getElementById('jurisdiction-select');
      const srcSel = document.getElementById('sources-select');
      const jurisdiction = jurisSel?.value || 'IN';
      let rawSrc = srcSel?.value || '';
      // If the Search segmented toggle is on, force web retrieval regardless of
      // the dropdown choice. Matches ai-prompt-box `showSearch` semantics.
      const seg = window.__sanhitaSeg || {};
      if (seg.search && !rawSrc.includes('web')) rawSrc = rawSrc ? rawSrc + ',web' : 'web,seed';
      const sources = rawSrc ? rawSrc.split(',').map((s) => s.trim()).filter(Boolean) : null;
      const data = await api('/api/brief/chat', {
        method: 'POST',
        body: JSON.stringify({
          thread_id: state.activeThreadId,
          question: text,
          jurisdiction,
          sources,
        }),
      });
      thinking.remove();
      appendMessage('assistant', data.answer_markdown, data.citations, {
        llm: data.llm,
        validation: data.validation,
        refused: !!data.refused,
      });
      renderSources(data.citations);
      // Update thread title on first exchange
      const t = state.threads.find((x) => x.id === state.activeThreadId);
      if (t && (!t.title || t.title === 'New chat')) {
        t.title = text.slice(0, 48);
        renderThreads();
      }
    } catch (e) {
      thinking.remove();
      appendMessage('assistant', `**Error:** ${e.message || 'could not get an answer'}`, null);
    } finally {
      state.sending = false;
      chatSend.disabled = false;
      chatInput.disabled = false;
      chatInput.focus();
    }
  }

  // ── wire-up ────────────────────────────────────────────────────────────

  newThreadBtn.addEventListener('click', () => newThread().catch(console.error));

  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    sendQuestion(chatInput.value);
  });

  // Cmd/Ctrl+Enter submits
  chatInput.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      sendQuestion(chatInput.value);
    }
  });

  document.querySelectorAll('.prompt-chip').forEach((b) => {
    b.addEventListener('click', () => {
      chatInput.value = b.dataset.q;
      sendQuestion(b.dataset.q);
    });
  });

  loadThreads().catch((e) => console.error('loadThreads failed:', e));
})();
