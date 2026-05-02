"use client";

// Library — curated statutes, contract templates and pleadings.
// Backend:
//   GET /api/library?jurisdiction=IN&kind=statute
//   GET /api/library/:id
//
// Filter bar (jurisdiction × kind) → list → detail view rendered through
// `renderMarkdown`. "Use this in chat" injects the document as context into
// the Assistant pane via the `onUseInChat` prop — handled in page.tsx.

import { useCallback, useEffect, useMemo, useState } from "react";
import { BookOpen, ExternalLink, ArrowLeftRight } from "lucide-react";
import { renderMarkdown } from "./markdown";

interface LibraryListItem {
  id: number;
  jurisdiction: string;
  kind: string;
  title: string;
  source_url?: string;
  added_at: number;
}

export interface LibraryDoc {
  id: number;
  jurisdiction: string;
  kind: string;
  title: string;
  body_md: string;
  source_url?: string;
  added_at: number;
}

const JURISDICTIONS: { code: string; flag: string; name: string }[] = [
  { code: "IN", flag: "🇮🇳", name: "India" },
];

const KINDS: { value: string; label: string }[] = [
  { value: "", label: "All kinds" },
  { value: "statute", label: "Statutes" },
  { value: "contract", label: "Contract templates" },
  { value: "pleading", label: "Pleading skeletons" },
];

export default function LibraryPane({
  onUseInChat,
}: {
  onUseInChat: (doc: LibraryDoc) => void;
}) {
  const [jurisdiction, setJurisdiction] = useState("");
  const [kind, setKind] = useState("");
  const [items, setItems] = useState<LibraryListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [active, setActive] = useState<LibraryDoc | null>(null);
  const [activeLoading, setActiveLoading] = useState(false);

  // Refetch the library whenever the filter pickers change. We inline the
  // fetch (rather than calling a useCallback) so React 19's purity check
  // sees the effect body itself doing work — no synchronous setState in
  // the effect frame, and the await yields before any setState is reached.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const params = new URLSearchParams();
        if (jurisdiction) params.set("jurisdiction", jurisdiction);
        if (kind) params.set("kind", kind);
        const r = await fetch(
          `/api/library${params.toString() ? "?" + params.toString() : ""}`,
          { credentials: "same-origin" }
        );
        const data = await r.json();
        if (cancelled) return;
        setItems(data.items || []);
      } catch {
        if (!cancelled) setItems([]);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [jurisdiction, kind]);

  const openDoc = useCallback(async (id: number) => {
    setActiveLoading(true);
    try {
      const r = await fetch(`/api/library/${id}`, { credentials: "same-origin" });
      if (!r.ok) return;
      const data: LibraryDoc = await r.json();
      setActive(data);
    } catch {
      /* silent */
    } finally {
      setActiveLoading(false);
    }
  }, []);

  // Group by (jurisdiction, kind) for a clean stacked list.
  const grouped = useMemo(() => {
    const m = new Map<string, LibraryListItem[]>();
    for (const it of items) {
      const key = `${it.jurisdiction}::${it.kind}`;
      const arr = m.get(key) || [];
      arr.push(it);
      m.set(key, arr);
    }
    return [...m.entries()].map(([k, arr]) => {
      const [j, kn] = k.split("::");
      return { jurisdiction: j, kind: kn, items: arr };
    });
  }, [items]);

  return (
    <div className="grid grid-cols-[360px_1fr] flex-1 min-h-0">
      {/* List column */}
      <aside className="border-r border-[var(--line)] bg-[var(--bg-elev)] flex flex-col min-h-0">
        <div className="px-5 py-4 border-b border-[var(--line)] flex flex-col gap-3">
          <div>
            <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)]">
              Library
            </div>
            <div className="font-display text-lg tracking-tight">
              {items.length} document{items.length === 1 ? "" : "s"}
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <select
              value={jurisdiction}
              onChange={(e) => setJurisdiction(e.target.value)}
              className="bg-[var(--bg)] border border-[var(--line)] rounded-lg px-3 py-2 text-sm outline-none hover:border-[var(--accent-soft)]"
            >
              {JURISDICTIONS.map((j) => (
                <option key={j.code} value={j.code}>
                  {j.flag} {j.name}
                </option>
              ))}
            </select>
            <select
              value={kind}
              onChange={(e) => setKind(e.target.value)}
              className="bg-[var(--bg)] border border-[var(--line)] rounded-lg px-3 py-2 text-sm outline-none hover:border-[var(--accent-soft)]"
            >
              {KINDS.map((k) => (
                <option key={k.value} value={k.value}>
                  {k.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-3 flex flex-col gap-3">
          {loading && (
            <div className="text-xs italic text-[var(--ink-soft)] px-2">
              Loading library…
            </div>
          )}
          {!loading && items.length === 0 && (
            <div className="text-xs italic text-[var(--ink-soft)] px-2">
              No documents match this filter.
            </div>
          )}
          {grouped.map((g) => (
            <div key={`${g.jurisdiction}-${g.kind}`} className="flex flex-col gap-1">
              <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] px-2 mt-2">
                {flagFor(g.jurisdiction)} {nameFor(g.jurisdiction)} ·{" "}
                {labelForKind(g.kind)}
              </div>
              {g.items.map((it) => (
                <button
                  key={it.id}
                  onClick={() => openDoc(it.id)}
                  className={`text-left rounded-lg border p-3 transition-colors ${
                    active?.id === it.id
                      ? "border-[var(--accent)] bg-[var(--highlight)]"
                      : "border-[var(--line)] bg-[var(--bg)] hover:border-[var(--accent-soft)]"
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <BookOpen
                      size={14}
                      className="mt-0.5 text-[var(--accent)] shrink-0"
                    />
                    <div className="min-w-0 flex-1">
                      <div
                        className="text-sm font-medium leading-snug"
                        title={it.title}
                      >
                        {it.title}
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          ))}
        </div>
      </aside>

      {/* Detail column */}
      <section className="flex flex-col min-w-0">
        {!active && !activeLoading && (
          <div className="flex-1 flex items-center justify-center p-12 text-center">
            <div className="max-w-md">
              <div className="font-display italic text-3xl tracking-tight text-[var(--ink)]">
                Pick a document
              </div>
              <p className="mt-4 text-[var(--ink-soft)]">
                Sanhita keeps a small curated library of pan-Asian statutes,
                contract templates and pleading skeletons. Open one, or send it
                straight into the Assistant as drafting context.
              </p>
            </div>
          </div>
        )}

        {activeLoading && (
          <div className="flex-1 flex items-center justify-center text-[var(--ink-soft)]">
            Loading…
          </div>
        )}

        {active && !activeLoading && (
          <>
            <div className="flex items-start justify-between border-b border-[var(--line)] px-12 py-6 gap-4">
              <div className="min-w-0">
                <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] mb-1">
                  {flagFor(active.jurisdiction)} {nameFor(active.jurisdiction)} ·{" "}
                  {labelForKind(active.kind)}
                </div>
                <h1 className="font-display text-2xl tracking-tight">
                  {active.title}
                </h1>
                {active.source_url && (
                  <a
                    href={active.source_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-2 inline-flex items-center gap-1 text-xs text-[var(--accent)] hover:underline"
                  >
                    <ExternalLink size={12} /> Source
                  </a>
                )}
              </div>
              <button
                onClick={() => onUseInChat(active)}
                className="shrink-0 flex items-center gap-2 bg-[var(--ink)] text-[var(--bg)] py-2 px-4 rounded-lg text-sm font-medium hover:bg-[var(--accent)] transition-colors"
              >
                <ArrowLeftRight size={14} /> Use in chat
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-12 py-8">
              <div
                className="max-w-3xl mx-auto prose-style text-[15px] leading-relaxed"
                dangerouslySetInnerHTML={{ __html: renderMarkdown(active.body_md) }}
              />
            </div>
          </>
        )}
      </section>
    </div>
  );
}

function flagFor(j: string): string {
  return JURISDICTIONS.find((x) => x.code === j)?.flag || "🌐";
}

function nameFor(j: string): string {
  return JURISDICTIONS.find((x) => x.code === j)?.name || j || "Unknown";
}

function labelForKind(k: string): string {
  return KINDS.find((x) => x.value === k)?.label.replace(/s$/, "") || k;
}
