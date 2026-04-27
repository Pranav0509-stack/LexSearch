"use client";

// Court Search — browse the BM25 case-law corpus (cuthchow HK CSV +
// upcoming SG/IN ingestors). Two tabs:
//   • Search  — full-text BM25 query, optional juris/tier filters
//   • Latest  — newest cases ingested, per jurisdiction
//
// Each result card has "Use in Assistant" so the lawyer can pull a
// case into a chat thread as context (mirrors the LibraryPane pattern).

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Search,
  Clock,
  ArrowUpRight,
  ExternalLink,
  Scale,
} from "lucide-react";

interface CaseHit {
  case_id: string;
  title?: string;
  court?: string;
  year?: number | string;
  citation?: string;
  jurisdiction?: string;
  tier?: string;
  excerpt?: string;
  score?: number;
  source?: string;
  url?: string;
}

interface CaseDoc extends CaseHit {
  text?: string;
  added_at?: number;
  extra?: Record<string, unknown>;
}

interface IndexStats {
  total: number;
  by_jurisdiction: Record<string, number>;
  by_source: Record<string, number>;
}

interface UseInChatPayload {
  case_id: string;
  title: string;
  body_md: string;
  jurisdiction: string;
}

const JURIS: { code: string; flag: string; label: string }[] = [
  { code: "",   flag: "🌏", label: "All" },
  { code: "IN", flag: "🇮🇳", label: "India" },
  { code: "SG", flag: "🇸🇬", label: "Singapore" },
  { code: "HK", flag: "🇭🇰", label: "Hong Kong" },
];

const TIERS: { code: string; label: string }[] = [
  { code: "",    label: "All courts" },
  { code: "SC",  label: "Supreme Court (IN)" },
  { code: "HC",  label: "High Court" },
  { code: "CFA", label: "Court of Final Appeal (HK)" },
  { code: "CA",  label: "Court of Appeal" },
  { code: "CFI", label: "Court of First Instance (HK)" },
];

type Tab = "search" | "latest";

export default function CourtSearchPane({
  onUseInChat,
}: {
  onUseInChat?: (p: UseInChatPayload) => void;
}) {
  const [tab, setTab] = useState<Tab>("search");
  const [q, setQ] = useState("");
  const [jurisdiction, setJurisdiction] = useState("");
  const [tier, setTier] = useState("");
  const [hits, setHits] = useState<CaseHit[]>([]);
  const [stats, setStats] = useState<IndexStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openCase, setOpenCase] = useState<CaseDoc | null>(null);
  // Compare mode — when on, each card shows a checkbox and the user can
  // pick 2-4 cases. The action bar at the bottom of the pane fetches
  // each selected case's full text and pushes a "compare these N cases"
  // prompt into the Assistant.
  const [compareMode, setCompareMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [comparing, setComparing] = useState(false);

  const toggleSelect = useCallback((case_id: string) => {
    setSelectedIds((prev) => {
      if (prev.includes(case_id)) return prev.filter((x) => x !== case_id);
      if (prev.length >= 4) return prev;       // hard cap — keeps the prompt sane
      return [...prev, case_id];
    });
  }, []);

  // Pull each selected case's full body, stitch them into one rich
  // context block, and hand off to the Assistant pane for the actual
  // analysis. We do the fetch here (not in the parent) so the caller
  // only ever sees a finished payload.
  const compareInAssistant = useCallback(async () => {
    if (!onUseInChat || selectedIds.length < 2) return;
    setComparing(true);
    try {
      const docs = await Promise.all(
        selectedIds.map(async (id) => {
          const r = await fetch(`/api/cases/${encodeURIComponent(id)}`, {
            credentials: "same-origin",
          });
          if (!r.ok) return null;
          return (await r.json()) as CaseDoc;
        })
      );
      const valid = docs.filter((d): d is CaseDoc => !!d);
      if (valid.length < 2) {
        setError("Could not load all selected cases. Try again.");
        return;
      }
      const body_md = valid
        .map((d, i) => {
          const meta: string[] = [];
          if (d.court) meta.push(d.court);
          if (d.year) meta.push(String(d.year));
          if (d.citation) meta.push(d.citation);
          const head = `### Case ${i + 1}: ${d.title || d.case_id}`;
          const sub = meta.length ? `_${meta.join(" · ")}_` : "";
          const body = (d.text || d.excerpt || "").slice(0, 1800);
          return [head, sub, "", body].filter(Boolean).join("\n");
        })
        .join("\n\n---\n\n");
      onUseInChat({
        case_id: `compare-${selectedIds.join("+")}`,
        title: `Compare ${valid.length} cases`,
        body_md: `Compare and analyze these ${valid.length} cases. Identify the legal questions each addresses, the holdings, points of agreement, and points of divergence.\n\n${body_md}`,
        jurisdiction: valid[0]?.jurisdiction || "",
      });
      setSelectedIds([]);
      setCompareMode(false);
    } finally {
      setComparing(false);
    }
  }, [onUseInChat, selectedIds]);

  const runSearch = useCallback(async () => {
    if (tab === "search" && !q.trim()) {
      setHits([]);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (tab === "search") params.set("q", q.trim());
      if (jurisdiction) params.set("jurisdiction", jurisdiction);
      if (tab === "search" && tier) params.set("tier", tier);
      params.set("k", "20");
      const endpoint = tab === "search" ? "/api/cases/search" : "/api/cases/latest";
      const r = await fetch(`${endpoint}?${params.toString()}`, {
        credentials: "same-origin",
      });
      const data = await r.json();
      if (!r.ok) {
        setError(data.detail || `HTTP ${r.status}`);
        setHits([]);
        return;
      }
      setHits(data.hits || []);
      setStats(data.stats || null);
    } catch (e) {
      setError((e as Error).message);
      setHits([]);
    } finally {
      setLoading(false);
    }
  }, [tab, q, jurisdiction, tier]);

  // Latest tab: refetch whenever filters change. Search tab: only on submit.
  useEffect(() => {
    if (tab === "latest") void runSearch();
  }, [tab, jurisdiction, runSearch]);

  // Initial load: pull stats so the header chip is populated even before
  // the user runs their first search.
  useEffect(() => {
    void (async () => {
      try {
        const r = await fetch("/api/cases/latest?k=1", { credentials: "same-origin" });
        if (r.ok) {
          const d = await r.json();
          setStats(d.stats || null);
        }
      } catch {
        /* stats are decorative */
      }
    })();
  }, []);

  const openDetail = useCallback(async (case_id: string) => {
    try {
      const r = await fetch(`/api/cases/${encodeURIComponent(case_id)}`, {
        credentials: "same-origin",
      });
      if (r.ok) setOpenCase(await r.json());
    } catch {
      /* drawer fails silently */
    }
  }, []);

  const totalChip = useMemo(() => {
    if (!stats) return null;
    const parts: string[] = [];
    parts.push(`${stats.total.toLocaleString()} cases indexed`);
    const byJ = stats.by_jurisdiction || {};
    const bits = ["IN", "SG", "HK"]
      .map((j) => (byJ[j] ? `${byJ[j].toLocaleString()} ${j}` : null))
      .filter(Boolean);
    if (bits.length) parts.push(bits.join(" · "));
    return parts.join(" — ");
  }, [stats]);

  return (
    <div className="flex flex-1 min-h-0 min-w-0">
      <section className="flex flex-col flex-1 min-w-0 min-h-0">
        {/* Header strip */}
        <div className="px-4 sm:px-6 lg:px-12 pt-5 pb-3 border-b border-[var(--line)]">
          <div className="flex items-center gap-2 text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)]">
            <Scale size={11} className="text-[var(--accent)]" />
            <span>Court Search</span>
            {totalChip && (
              <span className="ml-auto text-[10px] normal-case tracking-normal text-[var(--ink-soft)]">
                {totalChip}
              </span>
            )}
          </div>
          <h2 className="mt-1.5 font-display text-2xl tracking-tight">
            Browse case law across India, Singapore, Hong Kong
          </h2>

          {/* Tab strip */}
          <div className="mt-4 flex items-center gap-1 text-sm">
            <TabButton active={tab === "search"} onClick={() => setTab("search")} icon={<Search size={13} />} label="Search" />
            <TabButton active={tab === "latest"} onClick={() => setTab("latest")} icon={<Clock size={13} />} label="Latest" />
          </div>

          {/* Filters + search box */}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <select
              value={jurisdiction}
              onChange={(e) => setJurisdiction(e.target.value)}
              className="bg-[var(--bg-elev)] border border-[var(--line)] rounded-full px-3 py-1.5 text-xs cursor-pointer hover:border-[var(--accent-soft)]"
            >
              {JURIS.map((j) => (
                <option key={j.code || "all"} value={j.code}>
                  {j.flag} {j.label}
                </option>
              ))}
            </select>
            {tab === "search" && (
              <select
                value={tier}
                onChange={(e) => setTier(e.target.value)}
                className="bg-[var(--bg-elev)] border border-[var(--line)] rounded-full px-3 py-1.5 text-xs cursor-pointer hover:border-[var(--accent-soft)]"
              >
                {TIERS.map((t) => (
                  <option key={t.code || "all"} value={t.code}>{t.label}</option>
                ))}
              </select>
            )}
            {tab === "search" && (
              <form
                className="flex items-center gap-2 flex-1 min-w-0"
                onSubmit={(e) => {
                  e.preventDefault();
                  void runSearch();
                }}
              >
                <div className="flex items-center gap-2 flex-1 min-w-0 bg-[var(--bg-elev)] border border-[var(--line)] rounded-full px-3 py-1.5 hover:border-[var(--accent-soft)]">
                  <Search size={13} className="text-[var(--ink-soft)] shrink-0" />
                  <input
                    type="search"
                    value={q}
                    onChange={(e) => setQ(e.target.value)}
                    placeholder="Search 'defamation', 'section 138', 'judicial review'…"
                    className="bg-transparent outline-none text-sm flex-1 min-w-0"
                    autoFocus
                  />
                </div>
                <button
                  type="submit"
                  className="text-xs px-3 py-1.5 rounded-full bg-[var(--ink)] text-[var(--bg)] hover:opacity-90"
                >
                  Search
                </button>
              </form>
            )}
          </div>
        </div>

        {/* Results list */}
        <div className="flex-1 overflow-y-auto px-4 sm:px-6 lg:px-12 py-5 min-w-0">
          {error && (
            <div className="text-sm text-[var(--danger)] bg-[var(--bg-elev)] border border-[var(--line)] rounded-xl p-3 mb-3">
              {error}
            </div>
          )}
          {loading && (
            <div className="text-xs uppercase tracking-[0.22em] text-[var(--ink-soft)] py-6 text-center">
              Searching…
            </div>
          )}
          {!loading && hits.length === 0 && (
            <div className="text-sm text-[var(--ink-soft)] py-10 text-center">
              {tab === "search"
                ? q.trim()
                  ? "No cases matched. Try fewer keywords or a broader jurisdiction."
                  : "Type a query above to search the case-law index."
                : "Latest cases will appear here once the index has data."}
            </div>
          )}
          <ul className="flex flex-col gap-3 max-w-3xl mx-auto">
            {hits.map((h) => (
              <CaseCard
                key={h.case_id}
                hit={h}
                onOpen={() => void openDetail(h.case_id)}
                onUseInChat={
                  onUseInChat
                    ? () =>
                        onUseInChat({
                          case_id: h.case_id,
                          title: h.title || h.case_id,
                          body_md: caseToMarkdown(h),
                          jurisdiction: h.jurisdiction || "",
                        })
                    : undefined
                }
              />
            ))}
          </ul>
        </div>
      </section>

      {/* Detail drawer */}
      {openCase && (
        <div
          className="fixed inset-0 bg-black/40 z-30"
          onClick={() => setOpenCase(null)}
          aria-hidden
        />
      )}
      <aside
        className={`fixed inset-y-0 right-0 w-[92vw] sm:w-[480px] z-40 transition-transform duration-200 shadow-xl bg-[var(--bg-elev)] border-l border-[var(--line)] overflow-y-auto ${
          openCase ? "translate-x-0" : "translate-x-full"
        }`}
      >
        {openCase && (
          <div className="p-5 flex flex-col gap-3 min-w-0">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)]">
                  {openCase.jurisdiction || "—"} {openCase.tier ? `· ${openCase.tier}` : ""}
                </div>
                <h3 className="font-display text-lg leading-snug mt-1">
                  {openCase.title}
                </h3>
                <div className="text-xs text-[var(--ink-soft)] mt-1">
                  {openCase.court || "—"}
                  {openCase.year ? ` · ${openCase.year}` : ""}
                </div>
                {openCase.citation && (
                  <div className="text-xs text-[var(--accent)] mt-0.5">{openCase.citation}</div>
                )}
              </div>
              <button
                className="text-xs text-[var(--ink-soft)] hover:text-[var(--ink)]"
                onClick={() => setOpenCase(null)}
              >
                Close
              </button>
            </div>
            {openCase.url && (
              <a
                href={openCase.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-xs text-[var(--accent)] hover:underline w-fit"
              >
                <ExternalLink size={11} /> Open original
              </a>
            )}
            <div className="text-sm leading-relaxed whitespace-pre-wrap text-[var(--ink)] mt-2">
              {openCase.text || openCase.excerpt || "No body text on record."}
            </div>
            {onUseInChat && (
              <button
                className="mt-3 text-xs px-3 py-2 rounded-full bg-[var(--ink)] text-[var(--bg)] hover:opacity-90 w-fit"
                onClick={() => {
                  onUseInChat({
                    case_id: openCase.case_id,
                    title: openCase.title || openCase.case_id,
                    body_md: caseToMarkdown(openCase),
                    jurisdiction: openCase.jurisdiction || "",
                  });
                  setOpenCase(null);
                }}
              >
                Use in Assistant →
              </button>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full transition-colors text-xs ${
        active
          ? "bg-[var(--ink)] text-[var(--bg)]"
          : "bg-[var(--bg-elev)] border border-[var(--line)] text-[var(--ink)] hover:border-[var(--accent-soft)]"
      }`}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

function CaseCard({
  hit,
  onOpen,
  onUseInChat,
}: {
  hit: CaseHit;
  onOpen: () => void;
  onUseInChat?: () => void;
}) {
  return (
    <li className="bg-[var(--bg-elev)] border border-[var(--line)] hover:border-[var(--accent-soft)] rounded-2xl p-4 transition-colors">
      <div className="flex items-start gap-2 text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] mb-1.5">
        <span>{hit.jurisdiction || "—"}</span>
        {hit.tier && <span>· {hit.tier}</span>}
        {hit.year && <span>· {hit.year}</span>}
        {typeof hit.score === "number" && hit.score > 0 && (
          <span className="ml-auto text-[var(--accent)]">{hit.score.toFixed(2)}</span>
        )}
      </div>
      <button
        onClick={onOpen}
        className="font-display text-base leading-snug text-left text-[var(--ink)] hover:text-[var(--accent)] transition-colors"
      >
        {hit.title || hit.case_id}
      </button>
      <div className="text-xs text-[var(--ink-soft)] mt-1">
        {hit.court || "—"}
        {hit.citation && <span className="text-[var(--accent)]"> · {hit.citation}</span>}
      </div>
      {hit.excerpt && (
        <p className="text-sm text-[var(--ink-soft)] mt-2 line-clamp-3">
          {hit.excerpt}
        </p>
      )}
      <div className="flex items-center gap-3 mt-3 text-[11px] text-[var(--ink-soft)]">
        <button
          onClick={onOpen}
          className="hover:text-[var(--ink)] transition-colors"
        >
          Open
        </button>
        {onUseInChat && (
          <button
            onClick={onUseInChat}
            className="flex items-center gap-1 hover:text-[var(--accent)] transition-colors"
          >
            Use in Assistant <ArrowUpRight size={11} />
          </button>
        )}
        {hit.url && (
          <a
            href={hit.url}
            target="_blank"
            rel="noreferrer"
            className="ml-auto flex items-center gap-1 hover:text-[var(--ink)] transition-colors"
          >
            Source <ExternalLink size={11} />
          </a>
        )}
      </div>
    </li>
  );
}

function caseToMarkdown(h: CaseHit | CaseDoc): string {
  const lines: string[] = [];
  lines.push(`**${h.title || h.case_id}**`);
  const meta: string[] = [];
  if (h.court) meta.push(h.court);
  if (h.year) meta.push(String(h.year));
  if (h.citation) meta.push(h.citation);
  if (meta.length) lines.push(meta.join(" · "));
  const body =
    ("text" in h && h.text) ||
    h.excerpt ||
    "";
  if (body) {
    lines.push("");
    lines.push(body.slice(0, 2000));
  }
  return lines.join("\n");
}
