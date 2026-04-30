"use client";

// Court Search — Advanced search dashboard for 31M+ Indian court records.
// India-only. Features:
//   - Full-text BM25 search across judgments, legal docs, statutes, QA
//   - Advanced filters: Court, Year range, Verdict type
//   - Corpus stats banner showing total records
//   - Deduplication by case_id
//   - Compare mode: select 2-4 cases to compare in Assistant
//   - Search + Latest tabs

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Search,
  Clock,
  ArrowUpRight,
  ExternalLink,
  Scale,
  Filter,
  X,
  ChevronDown,
  Database,
  Gavel,
  Calendar,
  Building2,
  FileText,
  BarChart3,
  Sparkles,
} from "lucide-react";

/* ── Types ─────────────────────────────────────────────────── */

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
  verdict?: string;
  judge?: string;
}

interface CaseDoc extends CaseHit {
  text?: string;
  added_at?: number;
  extra?: Record<string, unknown>;
}

interface CourtInfo {
  court: string;
  court_code: string;
  case_count: number;
  year_min?: number;
  year_max?: number;
}

interface VerdictInfo {
  verdict_clean: string;
  case_count: number;
}

interface UseInChatPayload {
  case_id: string;
  title: string;
  body_md: string;
  jurisdiction: string;
}

type Tab = "search" | "latest";

/* ── Constants ─────────────────────────────────────────────── */

const CORPUS_STATS = {
  judgments: 16_901_394,
  legal_docs: 13_654_226,
  legal_qa: 1_364_000,
  statutes: 2_383,
  total: 31_922_003,
  courts: 25,
  yearMin: 1950,
  yearMax: 2025,
};

/* ── Helpers ───────────────────────────────────────────────── */

function formatNum(n: number): string {
  if (n >= 10_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return n.toLocaleString();
}

function deduplicateHits(hits: CaseHit[]): CaseHit[] {
  const seen = new Set<string>();
  return hits.filter((h) => {
    const key = h.case_id || `${h.title}-${h.court}-${h.year}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function caseToMarkdown(h: CaseHit | CaseDoc): string {
  const lines: string[] = [];
  lines.push(`**${h.title || h.case_id}**`);
  const meta: string[] = [];
  if (h.court) meta.push(h.court);
  if (h.year) meta.push(String(h.year));
  if (h.citation) meta.push(h.citation);
  if (h.verdict) meta.push(`Verdict: ${h.verdict}`);
  if (h.judge) meta.push(`Judge: ${h.judge}`);
  if (meta.length) lines.push(meta.join(" · "));
  const body = ("text" in h && h.text) || h.excerpt || "";
  if (body) {
    lines.push("");
    lines.push(body.slice(0, 2000));
  }
  return lines.join("\n");
}

/* ── Main Component ────────────────────────────────────────── */

export default function CourtSearchPane({
  onUseInChat,
}: {
  onUseInChat?: (p: UseInChatPayload) => void;
}) {
  const [tab, setTab] = useState<Tab>("search");
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<CaseHit[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openCase, setOpenCase] = useState<CaseDoc | null>(null);
  const [searchTime, setSearchTime] = useState<number | null>(null);
  const [totalResults, setTotalResults] = useState<number>(0);

  // Advanced filters
  const [showFilters, setShowFilters] = useState(false);
  const [courtCode, setCourtCode] = useState("");
  const [yearFrom, setYearFrom] = useState("");
  const [yearTo, setYearTo] = useState("");
  const [verdict, setVerdict] = useState("");

  // Filter options (loaded from API)
  const [courts, setCourts] = useState<CourtInfo[]>([]);
  const [verdicts, setVerdicts] = useState<VerdictInfo[]>([]);

  // Compare mode
  const [compareMode, setCompareMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [comparing, setComparing] = useState(false);

  const searchInputRef = useRef<HTMLInputElement>(null);

  const activeFilterCount = [courtCode, yearFrom, yearTo, verdict].filter(Boolean).length;

  // Load filter options on mount
  useEffect(() => {
    void (async () => {
      try {
        const [courtsRes, verdictsRes] = await Promise.all([
          fetch("/api/cases/courts", { credentials: "same-origin" }),
          fetch("/api/cases/verdicts", { credentials: "same-origin" }),
        ]);
        if (courtsRes.ok) {
          const d = await courtsRes.json();
          setCourts(d.courts || []);
        }
        if (verdictsRes.ok) {
          const d = await verdictsRes.json();
          setVerdicts(d.verdicts || []);
        }
      } catch {
        /* filter options are optional */
      }
    })();
  }, []);

  const toggleSelect = useCallback((case_id: string) => {
    setSelectedIds((prev) => {
      if (prev.includes(case_id)) return prev.filter((x) => x !== case_id);
      if (prev.length >= 4) return prev;
      return [...prev, case_id];
    });
  }, []);

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
        setError("Could not load all selected cases.");
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
        jurisdiction: "IN",
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
    setSearchTime(null);
    const t0 = performance.now();
    try {
      const params = new URLSearchParams();
      if (tab === "search") params.set("q", q.trim());
      if (courtCode) params.set("court_code", courtCode);
      if (yearFrom) params.set("year_from", yearFrom);
      if (yearTo) params.set("year_to", yearTo);
      if (verdict) params.set("verdict", verdict);
      params.set("k", "25");
      const endpoint =
        tab === "search" ? "/api/cases/search" : "/api/cases/latest";
      const r = await fetch(`${endpoint}?${params.toString()}`, {
        credentials: "same-origin",
      });
      const data = await r.json();
      if (!r.ok) {
        setError(data.detail || `HTTP ${r.status}`);
        setHits([]);
        return;
      }
      const rawHits = data.hits || data.results || [];
      setHits(deduplicateHits(rawHits));
      setTotalResults(data.total || rawHits.length);
      setSearchTime(Math.round(performance.now() - t0));
    } catch (e) {
      setError((e as Error).message);
      setHits([]);
    } finally {
      setLoading(false);
    }
  }, [tab, q, courtCode, yearFrom, yearTo, verdict]);

  useEffect(() => {
    if (tab === "latest") void runSearch();
  }, [tab, runSearch]);

  const clearFilters = useCallback(() => {
    setCourtCode("");
    setYearFrom("");
    setYearTo("");
    setVerdict("");
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

  // Generate year options
  const yearOptions = useMemo(() => {
    const years: number[] = [];
    for (let y = CORPUS_STATS.yearMax; y >= CORPUS_STATS.yearMin; y--) {
      years.push(y);
    }
    return years;
  }, []);

  return (
    <div className="flex flex-1 min-h-0 min-w-0">
      <section className="flex flex-col flex-1 min-w-0 min-h-0">
        {/* ── Header ──────────────────────────────────────── */}
        <div className="px-4 sm:px-6 lg:px-12 pt-5 pb-3 border-b border-[var(--line)]">
          <div className="flex items-center gap-2 text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)]">
            <Scale size={11} className="text-[var(--accent)]" />
            <span>Court Search</span>
            <span className="ml-auto flex items-center gap-1.5">
              <Database size={10} />
              <span className="normal-case tracking-normal font-medium">
                {formatNum(CORPUS_STATS.total)} records
              </span>
            </span>
          </div>
          <h2 className="mt-1.5 font-display text-2xl tracking-tight">
            Search Indian Case Law
          </h2>
          <p className="text-xs text-[var(--ink-soft)] mt-0.5">
            {formatNum(CORPUS_STATS.judgments)} judgments · {formatNum(CORPUS_STATS.legal_docs)} legal docs · {formatNum(CORPUS_STATS.legal_qa)} QA pairs · {formatNum(CORPUS_STATS.statutes)} statutes · 25 High Courts · 1950–2025
          </p>

          {/* ── Corpus stats mini cards ─────────────────── */}
          <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2">
            <StatCard icon={<Gavel size={14} />} label="HC Judgments" value={formatNum(CORPUS_STATS.judgments)} />
            <StatCard icon={<FileText size={14} />} label="Legal Docs" value={formatNum(CORPUS_STATS.legal_docs)} />
            <StatCard icon={<BarChart3 size={14} />} label="Legal QA" value={formatNum(CORPUS_STATS.legal_qa)} />
            <StatCard icon={<Building2 size={14} />} label="High Courts" value="25" />
          </div>

          {/* ── Tab strip ──────────────────────────────── */}
          <div className="mt-4 flex items-center gap-1 text-sm">
            <TabButton
              active={tab === "search"}
              onClick={() => setTab("search")}
              icon={<Search size={13} />}
              label="Search"
            />
            <TabButton
              active={tab === "latest"}
              onClick={() => setTab("latest")}
              icon={<Clock size={13} />}
              label="Latest"
            />
            <div className="flex-1" />
            {tab === "search" && (
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs transition-colors ${
                  showFilters || activeFilterCount > 0
                    ? "bg-[var(--accent)] text-white"
                    : "bg-[var(--bg-elev)] border border-[var(--line)] text-[var(--ink)] hover:border-[var(--accent-soft)]"
                }`}
              >
                <Filter size={12} />
                <span>Filters</span>
                {activeFilterCount > 0 && (
                  <span className="bg-white/30 text-[10px] px-1.5 py-0.5 rounded-full font-bold">
                    {activeFilterCount}
                  </span>
                )}
              </button>
            )}
            {hits.length > 1 && (
              <button
                onClick={() => {
                  setCompareMode(!compareMode);
                  if (compareMode) setSelectedIds([]);
                }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs transition-colors ${
                  compareMode
                    ? "bg-[var(--accent)] text-white"
                    : "bg-[var(--bg-elev)] border border-[var(--line)] text-[var(--ink)] hover:border-[var(--accent-soft)]"
                }`}
              >
                <Sparkles size={12} />
                <span>Compare</span>
              </button>
            )}
          </div>

          {/* ── Search bar ─────────────────────────────── */}
          {tab === "search" && (
            <form
              className="mt-3 flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                void runSearch();
              }}
            >
              <div className="flex items-center gap-2 flex-1 min-w-0 bg-[var(--bg-elev)] border border-[var(--line)] rounded-xl px-3 py-2 hover:border-[var(--accent-soft)] focus-within:border-[var(--accent)] transition-colors">
                <Search size={15} className="text-[var(--ink-soft)] shrink-0" />
                <input
                  ref={searchInputRef}
                  type="search"
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  placeholder="Search 'bail NDPS', 'section 138 NI Act', 'writ petition article 226'..."
                  className="bg-transparent outline-none text-sm flex-1 min-w-0"
                  autoFocus
                />
                {q && (
                  <button
                    type="button"
                    onClick={() => { setQ(""); setHits([]); searchInputRef.current?.focus(); }}
                    className="text-[var(--ink-soft)] hover:text-[var(--ink)]"
                  >
                    <X size={14} />
                  </button>
                )}
              </div>
              <button
                type="submit"
                disabled={loading}
                className="text-xs px-4 py-2.5 rounded-xl bg-[var(--ink)] text-[var(--bg)] hover:opacity-90 disabled:opacity-50 font-medium"
              >
                {loading ? "Searching..." : "Search"}
              </button>
            </form>
          )}

          {/* ── Advanced Filters Panel ─────────────────── */}
          {showFilters && tab === "search" && (
            <div className="mt-3 p-3 bg-[var(--bg-elev)] border border-[var(--line)] rounded-xl">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-[var(--ink)] flex items-center gap-1.5">
                  <Filter size={12} /> Advanced Filters
                </span>
                {activeFilterCount > 0 && (
                  <button
                    onClick={clearFilters}
                    className="text-[10px] text-[var(--accent)] hover:underline"
                  >
                    Clear all
                  </button>
                )}
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
                {/* Court */}
                <div>
                  <label className="text-[10px] uppercase tracking-wider text-[var(--ink-soft)] mb-1 block">
                    Court
                  </label>
                  <select
                    value={courtCode}
                    onChange={(e) => setCourtCode(e.target.value)}
                    className="w-full bg-[var(--bg)] border border-[var(--line)] rounded-lg px-2.5 py-1.5 text-xs cursor-pointer hover:border-[var(--accent-soft)]"
                  >
                    <option value="">All 25 High Courts</option>
                    {courts.map((c) => (
                      <option key={c.court_code} value={c.court_code}>
                        {c.court} ({formatNum(c.case_count)})
                      </option>
                    ))}
                  </select>
                </div>

                {/* Year from */}
                <div>
                  <label className="text-[10px] uppercase tracking-wider text-[var(--ink-soft)] mb-1 block">
                    Year from
                  </label>
                  <select
                    value={yearFrom}
                    onChange={(e) => setYearFrom(e.target.value)}
                    className="w-full bg-[var(--bg)] border border-[var(--line)] rounded-lg px-2.5 py-1.5 text-xs cursor-pointer hover:border-[var(--accent-soft)]"
                  >
                    <option value="">Any</option>
                    {yearOptions.map((y) => (
                      <option key={y} value={y}>{y}</option>
                    ))}
                  </select>
                </div>

                {/* Year to */}
                <div>
                  <label className="text-[10px] uppercase tracking-wider text-[var(--ink-soft)] mb-1 block">
                    Year to
                  </label>
                  <select
                    value={yearTo}
                    onChange={(e) => setYearTo(e.target.value)}
                    className="w-full bg-[var(--bg)] border border-[var(--line)] rounded-lg px-2.5 py-1.5 text-xs cursor-pointer hover:border-[var(--accent-soft)]"
                  >
                    <option value="">Any</option>
                    {yearOptions.map((y) => (
                      <option key={y} value={y}>{y}</option>
                    ))}
                  </select>
                </div>

                {/* Verdict */}
                <div>
                  <label className="text-[10px] uppercase tracking-wider text-[var(--ink-soft)] mb-1 block">
                    Verdict
                  </label>
                  <select
                    value={verdict}
                    onChange={(e) => setVerdict(e.target.value)}
                    className="w-full bg-[var(--bg)] border border-[var(--line)] rounded-lg px-2.5 py-1.5 text-xs cursor-pointer hover:border-[var(--accent-soft)]"
                  >
                    <option value="">All verdicts</option>
                    {verdicts.map((v) => (
                      <option key={v.verdict_clean} value={v.verdict_clean}>
                        {v.verdict_clean} ({formatNum(v.case_count)})
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Active filter chips */}
              {activeFilterCount > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {courtCode && (
                    <FilterChip
                      label={courts.find((c) => c.court_code === courtCode)?.court || courtCode}
                      onRemove={() => setCourtCode("")}
                    />
                  )}
                  {yearFrom && (
                    <FilterChip label={`From ${yearFrom}`} onRemove={() => setYearFrom("")} />
                  )}
                  {yearTo && (
                    <FilterChip label={`To ${yearTo}`} onRemove={() => setYearTo("")} />
                  )}
                  {verdict && (
                    <FilterChip label={verdict} onRemove={() => setVerdict("")} />
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Results ─────────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto px-4 sm:px-6 lg:px-12 py-4 min-w-0">
          {/* Search results meta bar */}
          {!loading && hits.length > 0 && (
            <div className="flex items-center gap-3 mb-3 text-xs text-[var(--ink-soft)]">
              <span className="font-medium">
                {hits.length} results
                {totalResults > hits.length && ` of ${formatNum(totalResults)}`}
              </span>
              {searchTime !== null && (
                <span>in {searchTime < 1000 ? `${searchTime}ms` : `${(searchTime / 1000).toFixed(1)}s`}</span>
              )}
              {activeFilterCount > 0 && (
                <span className="text-[var(--accent)]">{activeFilterCount} filter{activeFilterCount > 1 ? "s" : ""} active</span>
              )}
            </div>
          )}

          {error && (
            <div className="text-sm text-[var(--danger)] bg-[var(--bg-elev)] border border-[var(--line)] rounded-xl p-3 mb-3">
              {error}
            </div>
          )}
          {loading && (
            <div className="flex flex-col items-center gap-2 py-12">
              <div className="w-6 h-6 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
              <span className="text-xs uppercase tracking-[0.22em] text-[var(--ink-soft)]">
                Searching {formatNum(CORPUS_STATS.total)} records...
              </span>
            </div>
          )}
          {!loading && hits.length === 0 && (
            <div className="text-center py-12">
              <Scale size={32} className="mx-auto text-[var(--ink-soft)] opacity-30 mb-3" />
              <p className="text-sm text-[var(--ink-soft)]">
                {tab === "search"
                  ? q.trim()
                    ? "No cases matched. Try different keywords or adjust filters."
                    : "Search across 31.9M Indian court records — judgments, statutes, legal QA."
                  : "Latest cases will appear here."}
              </p>
              {tab === "search" && !q.trim() && (
                <div className="mt-4 flex flex-wrap justify-center gap-2">
                  {["Bail under NDPS Act", "Section 138 NI Act", "Writ petition Article 226", "Motor accident compensation", "Domestic violence protection", "Cheque bounce case"].map((s) => (
                    <button
                      key={s}
                      onClick={() => {
                        setQ(s);
                        setTimeout(() => {
                          void runSearch();
                        }, 0);
                      }}
                      className="text-xs px-3 py-1.5 rounded-full bg-[var(--bg-elev)] border border-[var(--line)] text-[var(--ink-soft)] hover:border-[var(--accent-soft)] hover:text-[var(--ink)] transition-colors"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Compare action bar */}
          {compareMode && selectedIds.length > 0 && (
            <div className="sticky top-0 z-10 mb-3 p-3 bg-[var(--accent)]/10 border border-[var(--accent)]/30 rounded-xl flex items-center gap-3">
              <span className="text-xs font-medium">
                {selectedIds.length} case{selectedIds.length > 1 ? "s" : ""} selected
              </span>
              <div className="flex-1" />
              {selectedIds.length >= 2 && (
                <button
                  onClick={() => void compareInAssistant()}
                  disabled={comparing}
                  className="text-xs px-3 py-1.5 rounded-full bg-[var(--accent)] text-white hover:opacity-90 disabled:opacity-50"
                >
                  {comparing ? "Loading..." : `Compare ${selectedIds.length} in Assistant`}
                </button>
              )}
              <button
                onClick={() => { setSelectedIds([]); setCompareMode(false); }}
                className="text-xs text-[var(--ink-soft)] hover:text-[var(--ink)]"
              >
                Cancel
              </button>
            </div>
          )}

          <ul className="flex flex-col gap-3 max-w-3xl mx-auto">
            {hits.map((h) => (
              <CaseCard
                key={h.case_id}
                hit={h}
                compareMode={compareMode}
                selected={selectedIds.includes(h.case_id)}
                onToggleSelect={() => toggleSelect(h.case_id)}
                onOpen={() => void openDetail(h.case_id)}
                onUseInChat={
                  onUseInChat
                    ? () =>
                        onUseInChat({
                          case_id: h.case_id,
                          title: h.title || h.case_id,
                          body_md: caseToMarkdown(h),
                          jurisdiction: "IN",
                        })
                    : undefined
                }
              />
            ))}
          </ul>
        </div>
      </section>

      {/* ── Detail drawer ─────────────────────────────────── */}
      {openCase && (
        <div
          className="fixed inset-0 bg-black/40 z-30"
          onClick={() => setOpenCase(null)}
          aria-hidden
        />
      )}
      <aside
        className={`fixed inset-y-0 right-0 w-[92vw] sm:w-[520px] z-40 transition-transform duration-200 shadow-xl bg-[var(--bg-elev)] border-l border-[var(--line)] overflow-y-auto ${
          openCase ? "translate-x-0" : "translate-x-full"
        }`}
      >
        {openCase && (
          <div className="p-5 flex flex-col gap-3 min-w-0">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] flex items-center gap-1.5">
                  <span>IN</span>
                  {openCase.tier && <span>· {openCase.tier}</span>}
                  {openCase.year && <span>· {openCase.year}</span>}
                  {openCase.verdict && (
                    <VerdictBadge verdict={openCase.verdict} />
                  )}
                </div>
                <h3 className="font-display text-lg leading-snug mt-1">
                  {openCase.title}
                </h3>
                <div className="text-xs text-[var(--ink-soft)] mt-1">
                  {openCase.court || "—"}
                </div>
                {openCase.judge && (
                  <div className="text-xs text-[var(--ink-soft)] mt-0.5">
                    Judge: {openCase.judge}
                  </div>
                )}
                {openCase.citation && (
                  <div className="text-xs text-[var(--accent)] mt-0.5 font-mono">
                    {openCase.citation}
                  </div>
                )}
              </div>
              <button
                className="text-xs text-[var(--ink-soft)] hover:text-[var(--ink)] p-1"
                onClick={() => setOpenCase(null)}
              >
                <X size={16} />
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
            <div className="text-sm leading-relaxed whitespace-pre-wrap text-[var(--ink)] mt-2 bg-[var(--bg)] p-3 rounded-lg border border-[var(--line)]">
              {openCase.text || openCase.excerpt || "No body text on record."}
            </div>
            {onUseInChat && (
              <button
                className="mt-3 text-xs px-3 py-2 rounded-full bg-[var(--ink)] text-[var(--bg)] hover:opacity-90 w-fit flex items-center gap-1.5"
                onClick={() => {
                  onUseInChat({
                    case_id: openCase.case_id,
                    title: openCase.title || openCase.case_id,
                    body_md: caseToMarkdown(openCase),
                    jurisdiction: "IN",
                  });
                  setOpenCase(null);
                }}
              >
                Use in Assistant <ArrowUpRight size={11} />
              </button>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}

/* ── Sub-components ────────────────────────────────────────── */

function StatCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-2 p-2 bg-[var(--bg-elev)] border border-[var(--line)] rounded-lg">
      <div className="text-[var(--accent)]">{icon}</div>
      <div className="min-w-0">
        <div className="text-sm font-semibold text-[var(--ink)] leading-tight">{value}</div>
        <div className="text-[10px] text-[var(--ink-soft)] leading-tight">{label}</div>
      </div>
    </div>
  );
}

function FilterChip({
  label,
  onRemove,
}: {
  label: string;
  onRemove: () => void;
}) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[var(--accent)]/10 text-[var(--accent)] text-[10px] font-medium">
      {label}
      <button onClick={onRemove} className="hover:opacity-70">
        <X size={10} />
      </button>
    </span>
  );
}

function VerdictBadge({ verdict }: { verdict: string }) {
  const v = verdict.toLowerCase();
  let color = "text-[var(--ink-soft)]";
  if (v.includes("allowed") || v.includes("granted")) color = "text-green-600";
  else if (v.includes("dismissed") || v.includes("rejected")) color = "text-red-500";
  else if (v.includes("disposed")) color = "text-amber-600";
  return (
    <span className={`ml-1 text-[9px] font-medium uppercase ${color}`}>
      {verdict}
    </span>
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
  compareMode,
  selected,
  onToggleSelect,
  onOpen,
  onUseInChat,
}: {
  hit: CaseHit;
  compareMode: boolean;
  selected: boolean;
  onToggleSelect: () => void;
  onOpen: () => void;
  onUseInChat?: () => void;
}) {
  const tierLabel = hit.tier === "STATUTE" ? "Statute" : hit.tier === "QA" ? "Legal QA" : hit.tier || "HC";
  const sourceLabel = hit.source?.includes("legal_docs") ? "Full text" : hit.source?.includes("statutes") ? "Statute" : hit.source?.includes("legal_qa") ? "QA" : "Judgment";

  return (
    <li
      className={`bg-[var(--bg-elev)] border rounded-2xl p-4 transition-colors ${
        selected
          ? "border-[var(--accent)] ring-1 ring-[var(--accent)]/30"
          : "border-[var(--line)] hover:border-[var(--accent-soft)]"
      }`}
    >
      <div className="flex items-start gap-3">
        {compareMode && (
          <input
            type="checkbox"
            checked={selected}
            onChange={onToggleSelect}
            className="mt-1 accent-[var(--accent)]"
          />
        )}
        <div className="flex-1 min-w-0">
          {/* Meta line */}
          <div className="flex items-center gap-2 text-[10px] tracking-[0.18em] uppercase text-[var(--ink-soft)] mb-1">
            <span className="flex items-center gap-1">
              <span>IN</span>
              <span>·</span>
              <span>{tierLabel}</span>
            </span>
            {hit.year && <span>· {hit.year}</span>}
            {hit.verdict && <VerdictBadge verdict={hit.verdict} />}
            {typeof hit.score === "number" && hit.score > 0 && (
              <span className="ml-auto text-[var(--accent)] font-mono">
                {hit.score.toFixed(2)}
              </span>
            )}
          </div>

          {/* Title */}
          <button
            onClick={onOpen}
            className="font-display text-base leading-snug text-left text-[var(--ink)] hover:text-[var(--accent)] transition-colors"
          >
            {hit.title || hit.case_id}
          </button>

          {/* Court + citation */}
          <div className="text-xs text-[var(--ink-soft)] mt-1 flex items-center gap-1.5 flex-wrap">
            {hit.court && <span>{hit.court}</span>}
            {hit.citation && (
              <span className="text-[var(--accent)] font-mono text-[10px]">
                {hit.citation}
              </span>
            )}
            {hit.judge && (
              <span className="text-[var(--ink-soft)]">· {hit.judge}</span>
            )}
          </div>

          {/* Excerpt */}
          {hit.excerpt && (
            <p className="text-xs text-[var(--ink-soft)] mt-2 line-clamp-2 leading-relaxed">
              {hit.excerpt}
            </p>
          )}

          {/* Source tag + actions */}
          <div className="flex items-center gap-3 mt-2.5 text-[11px] text-[var(--ink-soft)]">
            <span className="px-1.5 py-0.5 rounded bg-[var(--bg)] border border-[var(--line)] text-[9px] uppercase tracking-wider">
              {sourceLabel}
            </span>
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
        </div>
      </div>
    </li>
  );
}
