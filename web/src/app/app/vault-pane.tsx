"use client";

// Vault — per-user document upload + RAG chat.
// Backend:
//   GET    /api/vault/docs
//   POST   /api/vault/upload   (multipart)
//   DELETE /api/vault/docs/:id
//   POST   /api/vault/chat     {question, doc_ids?}
//
// Ports the structure from `workspaces.js` (vanilla JS prototype) into a
// proper React pane. Uses the same parchment palette + cite-chip rendering.

import { useCallback, useEffect, useRef, useState } from "react";
import { Trash2, Upload, FileText, Sparkles, Loader2 } from "lucide-react";
import { renderMarkdown } from "./markdown";

interface VaultDoc {
  id: number;
  filename: string;
  mime: string;
  size_bytes: number;
  n_chunks: number;
  created_at: number;
}

interface VaultMessage {
  role: "user" | "assistant";
  content: string;
  citations?: Array<{
    chunk_id?: string;
    para_label?: string;
    filename?: string;
    excerpt?: string;
    score?: number;
  }>;
  refused?: boolean;
}

export default function VaultPane() {
  const [docs, setDocs] = useState<VaultDoc[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadErr, setUploadErr] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<VaultMessage[]>([]);
  const [thinking, setThinking] = useState(false);
  // Adalat-style "Analyse" button: one-click structured doc review
  // (parties / dates / obligations / risks / governing-law / next-steps).
  // Result lands as an assistant message in the chat log so the user can
  // follow up ("redline clause 7.2", "translate to Hindi") in the same flow.
  const [analysingId, setAnalysingId] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const logRef = useRef<HTMLDivElement>(null);

  const refreshDocs = useCallback(async () => {
    try {
      const r = await fetch("/api/vault/docs", { credentials: "same-origin" });
      const data = await r.json();
      setDocs(data.docs || []);
    } catch {
      /* silent */
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshDocs();
  }, [refreshDocs]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, thinking]);

  const onUpload = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (!f) return;
      setUploading(true);
      setUploadErr(null);
      try {
        const fd = new FormData();
        fd.append("file", f);
        const r = await fetch("/api/vault/upload", {
          method: "POST",
          credentials: "same-origin",
          body: fd,
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
        await refreshDocs();
      } catch (err) {
        setUploadErr((err as Error).message || "Upload failed.");
      } finally {
        setUploading(false);
        if (fileRef.current) fileRef.current.value = "";
      }
    },
    [refreshDocs]
  );

  const onDelete = useCallback(
    async (id: number) => {
      if (!confirm("Remove this document and all of its chunks from storage?")) return;
      try {
        await fetch(`/api/vault/docs/${id}`, {
          method: "DELETE",
          credentials: "same-origin",
        });
        setSelected((prev) => {
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
        await refreshDocs();
      } catch {
        /* silent */
      }
    },
    [refreshDocs]
  );

  const toggleSelect = useCallback((id: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const onAnalyse = useCallback(async (docId: number) => {
    setAnalysingId(docId);
    const filename = docs.find((d) => d.id === docId)?.filename || "document";
    setMessages((m) => [
      ...m,
      { role: "user", content: `Analyse **${filename}** — parties, key dates, obligations, risks, governing law, next steps.` },
    ]);
    try {
      // Pick up the user's preferred language so the analysis comes back
      // in Hindi/Tamil/Japanese/etc. when they've set the picker upstream.
      const lang =
        typeof window !== "undefined"
          ? window.localStorage.getItem("sanhita.lang") || undefined
          : undefined;
      const r = await fetch("/api/vault/analyse", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: docId, language: lang }),
      });
      const data = await r.json();
      if (!r.ok) {
        setMessages((m) => [
          ...m,
          { role: "assistant", content: `**Analysis failed.** ${data.detail || `HTTP ${r.status}`}`, refused: true },
        ]);
      } else {
        setMessages((m) => [
          ...m,
          { role: "assistant", content: data.analysis_markdown || "(empty analysis)" },
        ]);
      }
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `**Network error.** ${(e as Error).message}`, refused: true },
      ]);
    } finally {
      setAnalysingId(null);
    }
  }, [docs]);

  const onAsk = useCallback(async () => {
    const text = question.trim();
    if (!text) return;
    setMessages((m) => [...m, { role: "user", content: text }]);
    setQuestion("");
    setThinking(true);
    try {
      const r = await fetch("/api/vault/chat", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          doc_ids: selected.size > 0 ? [...selected] : null,
        }),
      });
      const data = await r.json();
      setThinking(false);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: data.answer_markdown || "(no response)",
          citations: data.citations || [],
          refused: !!data.refused,
        },
      ]);
    } catch (e) {
      setThinking(false);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `**Error.** ${(e as Error).message}` },
      ]);
    }
  }, [question, selected]);

  return (
    <div className="grid grid-cols-[320px_1fr] flex-1 min-h-0">
      {/* Doc list */}
      <aside className="border-r border-[var(--line)] bg-[var(--bg-elev)] flex flex-col min-h-0">
        <div className="px-5 py-4 border-b border-[var(--line)]">
          <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)]">
            Documents
          </div>
          <div className="font-display text-lg tracking-tight">
            {docs.length} in storage
          </div>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.docx,.txt,.md"
            onChange={onUpload}
            className="hidden"
            id="vault-upload-input"
          />
          <label
            htmlFor="vault-upload-input"
            className={`mt-3 flex items-center gap-2 justify-center bg-[var(--ink)] text-[var(--bg)] py-2 px-3 rounded-lg text-sm font-medium cursor-pointer hover:bg-[var(--accent)] transition-colors ${
              uploading ? "opacity-60" : ""
            }`}
          >
            <Upload size={14} /> {uploading ? "Uploading…" : "Upload document"}
          </label>
          {uploadErr && (
            <div className="mt-2 text-xs text-[var(--danger)]">{uploadErr}</div>
          )}
          <p className="mt-2 text-[11px] text-[var(--ink-soft)] italic">
            PDF · DOCX · TXT · MD — up to ~25 MB.
          </p>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-3 flex flex-col gap-2">
          {loading && <div className="text-xs italic text-[var(--ink-soft)] px-2">Loading…</div>}
          {!loading && docs.length === 0 && (
            <div className="text-xs italic text-[var(--ink-soft)] px-2">
              Upload a brief, contract or pleading to ask questions across it.
            </div>
          )}
          {docs.map((d) => (
            <div
              key={d.id}
              className={`group rounded-lg border p-3 transition-colors cursor-pointer ${
                selected.has(d.id)
                  ? "border-[var(--accent)] bg-[var(--highlight)]"
                  : "border-[var(--line)] bg-[var(--bg)] hover:border-[var(--accent-soft)]"
              }`}
              onClick={() => toggleSelect(d.id)}
            >
              <div className="flex items-start gap-2">
                <FileText size={14} className="mt-0.5 text-[var(--accent)] shrink-0" />
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium truncate" title={d.filename}>
                    {d.filename}
                  </div>
                  <div className="text-[10px] text-[var(--ink-soft)] mt-0.5">
                    {d.n_chunks} chunks · {(d.size_bytes / 1024).toFixed(0)} KB
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onAnalyse(d.id);
                  }}
                  disabled={analysingId === d.id}
                  className="opacity-0 group-hover:opacity-100 text-[var(--ink-soft)] hover:text-[var(--accent)] transition-opacity disabled:opacity-60"
                  title="Analyse document — parties, dates, obligations, risks"
                >
                  {analysingId === d.id ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <Sparkles size={14} />
                  )}
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(d.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 text-[var(--ink-soft)] hover:text-[var(--danger)] transition-opacity"
                  title="Delete document"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>

        {selected.size > 0 && (
          <div className="px-4 py-3 border-t border-[var(--line)] bg-[var(--bg)] text-xs text-[var(--ink-soft)]">
            Asking across <strong>{selected.size}</strong> selected document
            {selected.size > 1 ? "s" : ""}.{" "}
            <button
              onClick={() => setSelected(new Set())}
              className="text-[var(--accent)] hover:underline"
            >
              clear
            </button>
          </div>
        )}
      </aside>

      {/* Chat column */}
      <section className="flex flex-col min-w-0">
        <div ref={logRef} className="flex-1 overflow-y-auto px-12 py-10">
          {messages.length === 0 ? (
            <div className="max-w-2xl mx-auto pt-16 text-center">
              <div className="font-display italic text-3xl tracking-tight text-[var(--ink)]">
                Ask across your storage
              </div>
              <p className="mt-4 text-[var(--ink-soft)]">
                Sanhita BM25-ranks every paragraph in the documents you upload
                and answers questions strictly from those passages — every claim
                cited with a paragraph reference. Click documents on the left to
                narrow the search; leave nothing selected to ask across all.
              </p>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto flex flex-col gap-6">
              {messages.map((m, i) => (
                <VaultBubble key={i} m={m} />
              ))}
              {thinking && (
                <div className="flex items-center gap-2 text-[var(--ink-soft)]">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Composer */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            onAsk();
          }}
          className="px-12 pb-8 pt-4 border-t border-[var(--line)] bg-[var(--bg)]"
        >
          <div className="max-w-3xl mx-auto flex items-center gap-3 bg-[var(--bg-elev)] border border-[var(--line)] rounded-2xl px-5 py-3 focus-within:border-[var(--accent-soft)] transition-colors">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={
                selected.size > 0
                  ? "Ask a question across the selected document(s)…"
                  : "Ask a question across all uploaded documents…"
              }
              className="flex-1 bg-transparent outline-none text-[15px]"
              disabled={thinking || docs.length === 0}
            />
            <button
              type="submit"
              disabled={thinking || !question.trim() || docs.length === 0}
              className="bg-[var(--ink)] text-[var(--bg)] px-4 py-1.5 rounded-lg text-sm font-medium hover:bg-[var(--accent)] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Ask
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}

function VaultBubble({ m }: { m: VaultMessage }) {
  if (m.role === "user") {
    return (
      <div className="self-end max-w-[80%]">
        <div className="bg-[var(--ink)] text-[var(--bg)] rounded-2xl rounded-br-sm px-5 py-3 leading-relaxed whitespace-pre-wrap">
          {m.content}
        </div>
      </div>
    );
  }
  return (
    <div className="self-start max-w-[92%] flex flex-col gap-2">
      <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] flex items-center gap-2">
        Sanhita Storage
        {m.refused && (
          <span className="bg-[var(--danger)] text-white px-1.5 py-0.5 rounded text-[9px] tracking-wider">
            refused
          </span>
        )}
      </div>
      <div
        className="bg-[var(--bg-elev)] border border-[var(--line)] rounded-2xl rounded-bl-sm px-6 py-4 leading-relaxed text-[15px] prose-style"
        dangerouslySetInnerHTML={{ __html: renderMarkdown(m.content) }}
      />
      {m.citations && m.citations.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-1">
          {m.citations.map((c, i) => (
            <div
              key={i}
              className="bg-[var(--bg)] border border-[var(--line)] rounded-md px-2.5 py-1 text-[11px] text-[var(--ink-soft)]"
              title={c.excerpt || ""}
            >
              <span className="font-mono text-[var(--accent)]">[{i + 1}]</span>{" "}
              {c.filename || "doc"} · {c.para_label || c.chunk_id || ""}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
