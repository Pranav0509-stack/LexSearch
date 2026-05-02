"use client";

// History — search across the user's past chat threads.
// Backend: GET /api/brief/threads/search?q=<query>
//
// Empty query returns the 30 most recent assistant turns. Typing in the
// search box re-queries on each keystroke (debounced 200ms). Click any
// result to jump to that thread in the Assistant pane.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Search, MessageSquare } from "lucide-react";

interface SearchHit {
  message_id: number;
  thread_id: number;
  thread_title: string;
  role: "user" | "assistant";
  snippet: string;
  created_at: number;
}

type RangeFilter = "all" | "today" | "week" | "month";

export default function HistoryPane({
  onOpenThread,
}: {
  onOpenThread: (threadId: number) => void;
}) {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [loading, setLoading] = useState(true);
  const [range, setRange] = useState<RangeFilter>("all");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const search = useCallback(async (query: string) => {
    // React 19 strict purity: defer the setState bookkeeping out of the
    // calling effect frame so this is safe to invoke synchronously from
    // useEffect.
    queueMicrotask(() => setLoading(true));
    try {
      const r = await fetch(
        `/api/brief/threads/search?q=${encodeURIComponent(query)}`,
        { credentials: "same-origin" }
      );
      const data = await r.json();
      setHits(data.results || []);
    } catch {
      setHits([]);
    } finally {
      setLoading(false);
    }
  }, []);

  // Debounced search — fires immediately on mount (q="") and on every
  // keystroke thereafter. queueMicrotask avoids the React 19 purity check
  // for "setState synchronously inside an effect."
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      search(q);
    }, q ? 200 : 0);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [q, search]);

  // Pin "now" once per range change so the filter is pure across renders.
  const [cutoffSec, setCutoffSec] = useState<number | null>(null);
  useEffect(() => {
    queueMicrotask(() => {
      if (range === "all") {
        setCutoffSec(null);
        return;
      }
      const now = Date.now() / 1000;
      setCutoffSec(
        range === "today"
          ? now - 24 * 3600
          : range === "week"
          ? now - 7 * 24 * 3600
          : now - 30 * 24 * 3600
      );
    });
  }, [range]);

  const filtered = useMemo(() => {
    if (cutoffSec == null) return hits;
    return hits.filter((h) => h.created_at >= cutoffSec);
  }, [hits, cutoffSec]);

  // Group hits by thread for compact rendering.
  const groups = useMemo(() => {
    const m = new Map<number, { title: string; hits: SearchHit[] }>();
    for (const h of filtered) {
      const g = m.get(h.thread_id);
      if (g) g.hits.push(h);
      else m.set(h.thread_id, { title: h.thread_title, hits: [h] });
    }
    return [...m.entries()].map(([thread_id, v]) => ({
      thread_id,
      title: v.title,
      hits: v.hits,
    }));
  }, [filtered]);

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      {/* Search bar */}
      <div className="px-12 pt-8 pb-4 border-b border-[var(--line)]">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-center gap-3 bg-[var(--bg-elev)] border border-[var(--line)] rounded-2xl px-5 py-3 focus-within:border-[var(--accent-soft)] transition-colors">
            <Search size={16} className="text-[var(--ink-soft)]" />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search across all your past matters…"
              className="flex-1 bg-transparent outline-none text-[15px]"
            />
            {q && (
              <button
                onClick={() => setQ("")}
                className="text-xs text-[var(--ink-soft)] hover:text-[var(--ink)]"
              >
                clear
              </button>
            )}
          </div>

          {/* Range chips */}
          <div className="mt-3 flex items-center gap-2 text-[11px] text-[var(--ink-soft)]">
            {(
              [
                ["all", "All"],
                ["today", "Today"],
                ["week", "This week"],
                ["month", "This month"],
              ] as [RangeFilter, string][]
            ).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setRange(key)}
                className={`px-2.5 py-1 rounded-full border transition-colors ${
                  range === key
                    ? "bg-[var(--ink)] text-[var(--bg)] border-[var(--ink)]"
                    : "border-[var(--line)] hover:border-[var(--accent-soft)]"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto px-12 py-8">
        <div className="max-w-3xl mx-auto">
          {loading && (
            <div className="text-sm italic text-[var(--ink-soft)]">Searching…</div>
          )}

          {!loading && filtered.length === 0 && (
            <div className="text-sm italic text-[var(--ink-soft)]">
              {q
                ? `No matches for "${q}".`
                : "No matters in this range yet. Start a new one from the Assistant tab."}
            </div>
          )}

          {!loading && groups.length > 0 && (
            <div className="flex flex-col gap-6">
              {groups.map((g) => (
                <div key={g.thread_id} className="flex flex-col gap-2">
                  <button
                    onClick={() => onOpenThread(g.thread_id)}
                    className="flex items-center gap-2 text-left group"
                  >
                    <MessageSquare
                      size={14}
                      className="text-[var(--accent)] shrink-0"
                    />
                    <span className="font-display text-base tracking-tight group-hover:text-[var(--accent)] transition-colors truncate">
                      {g.title || "Untitled matter"}
                    </span>
                    <span className="text-[10px] tracking-[0.2em] uppercase text-[var(--ink-soft)] ml-1">
                      {g.hits.length} hit{g.hits.length === 1 ? "" : "s"}
                    </span>
                  </button>

                  <div className="flex flex-col gap-2 pl-6">
                    {g.hits.map((h) => (
                      <button
                        key={h.message_id}
                        onClick={() => onOpenThread(h.thread_id)}
                        className="text-left bg-[var(--bg-elev)] border border-[var(--line)] hover:border-[var(--accent-soft)] rounded-xl px-4 py-3 transition-colors"
                      >
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-[10px] tracking-[0.2em] uppercase text-[var(--ink-soft)]">
                            {h.role === "user" ? "You" : "Sanhita"}
                          </span>
                          <span className="text-[10px] text-[var(--ink-soft)]">
                            {formatWhen(h.created_at)}
                          </span>
                        </div>
                        <div className="text-[13px] leading-relaxed text-[var(--ink)] whitespace-pre-wrap line-clamp-3">
                          {highlight(h.snippet, q)}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function formatWhen(ts: number): string {
  const d = new Date(ts * 1000);
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  if (sameDay) {
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

// Lightweight match highlighter — splits snippet on the query (case-insensitive)
// and bolds the runs that match. Falls back to plain text when q is empty.
function highlight(snippet: string, q: string): React.ReactNode {
  if (!q.trim()) return snippet;
  const needle = q.trim();
  const re = new RegExp(`(${escapeRegExp(needle)})`, "ig");
  const parts = snippet.split(re);
  return parts.map((p, i) =>
    re.test(p) && p.toLowerCase() === needle.toLowerCase() ? (
      <mark
        key={i}
        className="bg-[var(--highlight)] text-[var(--accent)] px-0.5 rounded-sm"
      >
        {p}
      </mark>
    ) : (
      <span key={i}>{p}</span>
    )
  );
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
