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
  UserSearch,
  TrendingUp,
  Award,
  Network,
} from "lucide-react";

/* ── Types ─────────────────────────────────────────────────── */

type DocType = "JUDGMENT" | "LEGAL_DOC" | "STATUTE" | "LEGAL_QA";

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
  bench?: string;
  date_decided?: string;
  explanation?: string;
  description?: string;
  has_full_text?: boolean;
  doc_type?: DocType;
}

interface CaseDoc extends CaseHit {
  text?: string;
  full_text?: string;
  added_at?: number;
  extra?: Record<string, unknown>;
  petitioner?: string;
  respondent?: string;
  verdict_raw?: string;
  pdf_link?: string;
  pdf_available?: boolean;
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

type Tab = "search" | "latest" | "judge";

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

function classifyDocType(h: CaseHit | CaseDoc): DocType {
  const source = (h.source || "").toLowerCase();
  const tier = (h.tier || "").toUpperCase();
  if (source.includes("statutes") || tier === "STATUTE") return "STATUTE";
  if (source.includes("legal_qa") || tier === "QA") return "LEGAL_QA";
  if (source.includes("legal_docs")) return "LEGAL_DOC";
  return "JUDGMENT";
}

const DOC_TYPE_LABELS: Record<DocType, string> = {
  JUDGMENT: "Judgment",
  LEGAL_DOC: "Legal Document",
  STATUTE: "Statute",
  LEGAL_QA: "Legal Q&A",
};

const DOC_TYPE_COLORS: Record<DocType, { bg: string; text: string; border: string }> = {
  JUDGMENT: { bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-200" },
  LEGAL_DOC: { bg: "bg-blue-50", text: "text-blue-700", border: "border-blue-200" },
  STATUTE: { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200" },
  LEGAL_QA: { bg: "bg-purple-50", text: "text-purple-700", border: "border-purple-200" },
};

function caseToMarkdown(h: CaseHit | CaseDoc): string {
  const docType = h.doc_type || classifyDocType(h);
  const lines: string[] = [];

  // Header with document type
  lines.push(`## ${h.title || h.case_id}`);
  lines.push(`**Document Type:** ${DOC_TYPE_LABELS[docType]}`);

  // Metadata
  const meta: string[] = [];
  if (h.court) meta.push(`**Court:** ${h.court}`);
  if (h.year) meta.push(`**Year:** ${h.year}`);
  if (h.citation) meta.push(`**Citation:** ${h.citation}`);
  if (h.judge) meta.push(`**Judge:** ${h.judge}`);
  if ("bench" in h && h.bench) meta.push(`**Bench:** ${h.bench}`);
  if ("date_decided" in h && h.date_decided) meta.push(`**Date Decided:** ${h.date_decided}`);
  if (meta.length) {
    lines.push("");
    lines.push(meta.join(" · "));
  }

  // Verdict as a separate section
  if (h.verdict) {
    lines.push("");
    lines.push("### Verdict");
    lines.push(h.verdict);
  }

  // Full document text — use the richest source available
  const fullText = ("full_text" in h && h.full_text) || "";
  const text = ("text" in h && h.text) || "";
  const explanation = ("explanation" in h && h.explanation) || "";
  const excerpt = h.excerpt || "";

  const body = fullText || text || explanation || excerpt;
  if (body) {
    lines.push("");
    lines.push("### Document Content");
    lines.push(body.slice(0, 8000));
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
          const docType = d.doc_type || classifyDocType(d);
          const meta: string[] = [];
          if (d.court) meta.push(d.court);
          if (d.year) meta.push(String(d.year));
          if (d.citation) meta.push(d.citation);
          const head = `### Case ${i + 1}: ${d.title || d.case_id}`;
          const typeLine = `**Document Type:** ${DOC_TYPE_LABELS[docType]}`;
          const sub = meta.length ? `_${meta.join(" · ")}_` : "";
          const verdictLine = d.verdict ? `**Verdict:** ${d.verdict}` : "";
          const body = (d.full_text || d.text || d.excerpt || "").slice(0, 4000);
          return [head, typeLine, sub, verdictLine, "", body].filter(Boolean).join("\n");
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
            <TabButton
              active={tab === "judge"}
              onClick={() => setTab("judge")}
              icon={<UserSearch size={13} />}
              label="Judge"
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

        {/* ── Judge Analytics tab ─────────────────────────── */}
        {tab === "judge" && (
          <div className="flex-1 overflow-y-auto px-4 sm:px-6 lg:px-12 py-4">
            <JudgeAnalyticsPanel />
          </div>
        )}

        {/* ── Results ─────────────────────────────────────── */}
        {tab !== "judge" && (
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
                    ? async () => {
                        // Fetch full case detail (includes full_text) before sending to assistant
                        try {
                          const r = await fetch(`/api/cases/${encodeURIComponent(h.case_id)}`, { credentials: "same-origin" });
                          if (r.ok) {
                            const full = await r.json() as CaseDoc;
                            onUseInChat({
                              case_id: full.case_id,
                              title: full.title || full.case_id,
                              body_md: caseToMarkdown(full),
                              jurisdiction: "IN",
                            });
                            return;
                          }
                        } catch { /* fall through to search hit */ }
                        // Fallback: use search hit data
                        onUseInChat({
                          case_id: h.case_id,
                          title: h.title || h.case_id,
                          body_md: caseToMarkdown(h),
                          jurisdiction: "IN",
                        });
                      }
                    : undefined
                }
              />
            ))}
          </ul>
        </div>
        )} {/* end tab !== "judge" */}
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
        {openCase && (() => {
          const docType = openCase.doc_type || classifyDocType(openCase);
          const dtColors = DOC_TYPE_COLORS[docType];
          const dtLabel = DOC_TYPE_LABELS[docType];
          const bodyText = openCase.full_text || openCase.text || openCase.excerpt || "";

          return (
          <div className="p-5 flex flex-col gap-3 min-w-0">
            {/* Header */}
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                {/* Doc type badge + tier + year */}
                <div className="flex items-center gap-2 mb-2 flex-wrap">
                  <span className={`px-2.5 py-1 rounded-full text-[10px] font-semibold uppercase tracking-wider ${dtColors.bg} ${dtColors.text} border ${dtColors.border}`}>
                    {dtLabel}
                  </span>
                  {openCase.tier && (
                    <span className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)]">
                      {openCase.tier}
                    </span>
                  )}
                  {openCase.year && (
                    <span className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)]">
                      {openCase.year}
                    </span>
                  )}
                </div>
                <h3 className="font-display text-lg leading-snug">
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
                {openCase.bench && (
                  <div className="text-xs text-[var(--ink-soft)] mt-0.5">
                    Bench: {openCase.bench}
                  </div>
                )}
                {openCase.date_decided && (
                  <div className="text-xs text-[var(--ink-soft)] mt-0.5">
                    Date Decided: {openCase.date_decided}
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

            {/* PDF actions */}
            {openCase.url && (
              <div className="flex items-center gap-3 flex-wrap">
                <a
                  href={openCase.url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 text-xs text-[var(--accent)] hover:underline"
                >
                  <ExternalLink size={11} /> View PDF
                </a>
                <a
                  href={`${openCase.url}${openCase.url.includes("?") ? "&" : "?"}download=true`}
                  className="inline-flex items-center gap-1 text-xs text-[var(--ink-soft)] hover:text-[var(--ink)] border border-[var(--line)] rounded px-2 py-0.5 bg-[var(--bg)] transition-colors"
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                  Download PDF
                </a>
              </div>
            )}

            {/* ── Verdict Section ── */}
            {openCase.verdict && (
              <div className="bg-[var(--bg)] border border-[var(--line)] rounded-xl p-3">
                <div className="text-[10px] uppercase tracking-[0.22em] text-[var(--ink-soft)] font-medium mb-1.5">
                  Verdict
                </div>
                <div className={`text-sm font-semibold ${
                  openCase.verdict.toLowerCase().includes("allowed") || openCase.verdict.toLowerCase().includes("granted")
                    ? "text-green-600"
                    : openCase.verdict.toLowerCase().includes("dismissed") || openCase.verdict.toLowerCase().includes("rejected")
                      ? "text-red-500"
                      : "text-amber-600"
                }`}>
                  {openCase.verdict}
                </div>
                {openCase.verdict_raw && openCase.verdict_raw !== openCase.verdict && (
                  <div className="text-[10px] text-[var(--ink-soft)] mt-1">
                    Raw: {openCase.verdict_raw}
                  </div>
                )}
              </div>
            )}

            {/* ── Explanation / Case Analysis Section ── */}
            {openCase.explanation && (
              <div className="bg-[var(--bg)] border border-[var(--line)] rounded-xl p-3">
                <div className="text-[10px] uppercase tracking-[0.22em] text-[var(--ink-soft)] font-medium mb-1.5">
                  Case Analysis
                </div>
                <div className="text-sm leading-relaxed text-[var(--ink)] whitespace-pre-wrap">
                  {openCase.explanation}
                </div>
              </div>
            )}

            {/* ── Document Content ── */}
            <div className="bg-[var(--bg)] border border-[var(--line)] rounded-xl p-3">
              <div className="text-[10px] uppercase tracking-[0.22em] text-[var(--ink-soft)] font-medium mb-1.5">
                {docType === "STATUTE" ? "Statute Text" : docType === "LEGAL_QA" ? "Q&A Content" : "Document Content"}
              </div>
              <div className="text-sm leading-relaxed whitespace-pre-wrap text-[var(--ink)] max-h-[60vh] overflow-y-auto">
                {bodyText || "No body text on record."}
              </div>
            </div>

            {/* Use in Assistant */}
            {onUseInChat && (
              <button
                className="mt-1 text-xs px-3 py-2 rounded-full bg-[var(--ink)] text-[var(--bg)] hover:opacity-90 w-fit flex items-center gap-1.5"
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

            {/* Related Cases via Citation Graph */}
            <RelatedCases caseId={openCase.case_id} onOpen={(id) => openDetail(id)} />
          </div>
          );
        })()}
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
  const docType = hit.doc_type || classifyDocType(hit);
  const dtColors = DOC_TYPE_COLORS[docType];
  const dtLabel = DOC_TYPE_LABELS[docType];

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
          {/* Meta line with doc type badge */}
          <div className="flex items-center gap-2 text-[10px] mb-1.5 flex-wrap">
            <span className={`px-2 py-0.5 rounded-full text-[9px] font-semibold uppercase tracking-wider ${dtColors.bg} ${dtColors.text} border ${dtColors.border}`}>
              {dtLabel}
            </span>
            <span className="tracking-[0.18em] uppercase text-[var(--ink-soft)]">
              {hit.year && <span>{hit.year}</span>}
            </span>
            {hit.verdict && <VerdictBadge verdict={hit.verdict} />}
            {typeof hit.score === "number" && hit.score > 0 && (
              <span className="ml-auto text-[var(--accent)] font-mono text-[10px]">
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

          {/* Court + citation + judge */}
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
            {hit.date_decided && (
              <span className="text-[var(--ink-soft)]">· {hit.date_decided}</span>
            )}
          </div>

          {/* Verdict — distinct section */}
          {hit.verdict && (
            <div className="mt-2 flex items-center gap-2">
              <span className="text-[10px] uppercase tracking-wider text-[var(--ink-soft)] font-medium">Verdict:</span>
              <span className={`text-xs font-semibold ${
                hit.verdict.toLowerCase().includes("allowed") || hit.verdict.toLowerCase().includes("granted")
                  ? "text-green-600"
                  : hit.verdict.toLowerCase().includes("dismissed") || hit.verdict.toLowerCase().includes("rejected")
                    ? "text-red-500"
                    : "text-amber-600"
              }`}>
                {hit.verdict}
              </span>
            </div>
          )}

          {/* Excerpt */}
          {hit.excerpt && (
            <p className="text-xs text-[var(--ink-soft)] mt-2 line-clamp-3 leading-relaxed">
              {hit.excerpt}
            </p>
          )}

          {/* Source tag + actions */}
          <div className="flex items-center gap-3 mt-2.5 text-[11px] text-[var(--ink-soft)] flex-wrap">
            {hit.has_full_text && (
              <span className="px-1.5 py-0.5 rounded bg-green-50 border border-green-200 text-[9px] text-green-700 uppercase tracking-wider shrink-0">
                Full text
              </span>
            )}
            <button
              onClick={onOpen}
              className="hover:text-[var(--ink)] transition-colors shrink-0"
            >
              Open
            </button>
            {onUseInChat && (
              <button
                onClick={onUseInChat}
                className="flex items-center gap-1 hover:text-[var(--accent)] transition-colors shrink-0"
              >
                Use in Assistant <ArrowUpRight size={11} />
              </button>
            )}
            {hit.url && (
              <div className="ml-auto flex items-center gap-2 shrink-0">
                <a
                  href={hit.url}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-1 hover:text-[var(--accent)] transition-colors"
                  title="View judgment PDF"
                >
                  <ExternalLink size={11} /> View PDF
                </a>
                <a
                  href={`${hit.url}${hit.url.includes("?") ? "&" : "?"}download=true`}
                  className="flex items-center gap-1 hover:text-[var(--ink)] transition-colors px-1.5 py-0.5 rounded bg-[var(--bg)] border border-[var(--line)] text-[10px]"
                  title="Download judgment as PDF"
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                  Download
                </a>
              </div>
            )}
          </div>
        </div>
      </div>
    </li>
  );
}


/* ── Citation Network Graph ───────────────────────────────────── */

interface RelatedCase {
  case_id: string;
  pagerank: number;
  relationship: string;
}

function CitationGraph({ caseId, nodes, onOpen }: {
  caseId: string;
  nodes: RelatedCase[];
  onOpen: (id: string) => void;
}) {
  const W = 300, H = 220, cx = W / 2, cy = H / 2, r = 85;
  const angles = nodes.map((_, i) => (2 * Math.PI * i) / nodes.length - Math.PI / 2);
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="w-full max-w-[300px] mx-auto">
      {/* Edges */}
      {nodes.map((n, i) => {
        const nx = cx + r * Math.cos(angles[i]);
        const ny = cy + r * Math.sin(angles[i]);
        const isCitedBy = n.relationship === "cited_by";
        return (
          <line key={n.case_id}
            x1={cx} y1={cy} x2={nx} y2={ny}
            stroke={isCitedBy ? "#1a73e8" : "#1e8e3e"}
            strokeWidth={1.5} strokeOpacity={0.5}
            strokeDasharray={isCitedBy ? "none" : "4 2"}
          />
        );
      })}
      {/* Central node */}
      <circle cx={cx} cy={cy} r={18} fill="var(--accent)" />
      <text x={cx} y={cy + 1} textAnchor="middle" dominantBaseline="middle"
        fontSize={7} fill="white" fontWeight="600">
        THIS
      </text>
      {/* Satellite nodes */}
      {nodes.map((n, i) => {
        const nx = cx + r * Math.cos(angles[i]);
        const ny = cy + r * Math.sin(angles[i]);
        const isCitedBy = n.relationship === "cited_by";
        const label = n.case_id.length > 12 ? n.case_id.slice(0, 12) + "..." : n.case_id;
        return (
          <g key={n.case_id} className="cursor-pointer" onClick={() => onOpen(n.case_id)}>
            <circle cx={nx} cy={ny} r={14}
              fill={isCitedBy ? "#e8f0fe" : "#e6f4ea"}
              stroke={isCitedBy ? "#1a73e8" : "#1e8e3e"}
              strokeWidth={1.5}
              className="hover:opacity-80 transition-opacity"
            />
            <text x={nx} y={ny + 1} textAnchor="middle" dominantBaseline="middle"
              fontSize={5.5} fill={isCitedBy ? "#1a73e8" : "#1e8e3e"}>
              {label}
            </text>
            {/* Arrowhead */}
            <text x={(cx + nx) / 2} y={(cy + ny) / 2 - 6}
              textAnchor="middle" fontSize={8}
              fill={isCitedBy ? "#1a73e8" : "#1e8e3e"} opacity={0.7}>
              {isCitedBy ? "→" : "←"}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function RelatedCases({ caseId, onOpen }: { caseId: string; onOpen: (id: string) => void }) {
  const [related, setRelated] = useState<RelatedCase[]>([]);
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState<"graph" | "list">("graph");

  useEffect(() => {
    if (!caseId) return;
    let cancelled = false;
    setLoading(true);
    fetch(`/api/cases/related/${encodeURIComponent(caseId)}?limit=8`, { credentials: "same-origin" })
      .then((r) => r.ok ? r.json() : { related: [] })
      .then((d) => { if (!cancelled) setRelated(d.related || []); })
      .catch(() => {})
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [caseId]);

  if (loading) {
    return (
      <div className="mt-4 pt-4 border-t border-[var(--line)]">
        <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] mb-2">Citation Graph</div>
        <div className="text-xs text-[var(--ink-soft)] animate-pulse">Mapping citations...</div>
      </div>
    );
  }
  if (related.length === 0) return null;

  const citedBy = related.filter(r => r.relationship === "cited_by");
  const cites = related.filter(r => r.relationship !== "cited_by");

  return (
    <div className="mt-4 pt-4 border-t border-[var(--line)]">
      <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] mb-3 flex items-center gap-1.5">
        <Network size={10} />
        <span>Citation Network</span>
        <span className="ml-auto text-[9px] normal-case tracking-normal font-normal">
          {citedBy.length} cited-by · {cites.length} cites
        </span>
        <button
          onClick={() => setView(v => v === "graph" ? "list" : "graph")}
          className="text-[9px] px-1.5 py-0.5 rounded bg-[var(--bg-elev)] border border-[var(--line)] hover:border-[var(--accent-soft)] transition-colors normal-case tracking-normal"
        >
          {view === "graph" ? "List" : "Graph"}
        </button>
      </div>

      {view === "graph" ? (
        <div className="bg-[var(--bg)] rounded-xl border border-[var(--line)] p-3">
          <CitationGraph caseId={caseId} nodes={related.slice(0, 7)} onOpen={onOpen} />
          <div className="flex items-center justify-center gap-4 mt-2 text-[9px] text-[var(--ink-soft)]">
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-blue-500 inline-block rounded" /> cited-by (→)</span>
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-green-500 inline-block rounded border-dashed border-t border-green-500" style={{borderStyle:"dashed"}} /> cites (←)</span>
          </div>
        </div>
      ) : (
        <ul className="space-y-1.5">
          {related.map((r) => (
            <li key={r.case_id}>
              <button
                onClick={() => onOpen(r.case_id)}
                className="w-full text-left px-3 py-2 rounded-lg bg-[var(--bg)] border border-[var(--line)] hover:border-[var(--accent-soft)] transition-colors text-xs group"
              >
                <span className="text-[var(--ink)] group-hover:text-[var(--accent)] transition-colors font-mono truncate block">
                  {r.case_id}
                </span>
                <span className="flex items-center gap-2 mt-0.5 text-[var(--ink-soft)]">
                  <span className={`px-1 py-0.5 rounded text-[8px] uppercase tracking-wider ${
                    r.relationship === "cited_by" ? "bg-blue-100 text-blue-700" : "bg-green-100 text-green-700"
                  }`}>
                    {r.relationship === "cited_by" ? "Cited by this" : "Cites this"}
                  </span>
                  {r.pagerank > 0 && (
                    <span className="text-[9px]">PR: {r.pagerank.toFixed(5)}</span>
                  )}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}


/* ── Judge Analytics Panel ────────────────────────────────────── */

interface JudgeProfile {
  query: string;
  matched_names: string[];
  total: number;
  cases: Array<{
    cnr: string;
    title: string;
    court: string;
    year: number | null;
    verdict: string;
    judge: string;
    citation: string;
  }>;
  verdict_breakdown: Record<string, number>;
  court_breakdown: Record<string, number>;
  yearly_volume: Record<string, number>;
}

function JudgeAnalyticsPanel() {
  const [judgeQuery, setJudgeQuery] = useState("");
  const [profile, setProfile] = useState<JudgeProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = async () => {
    if (!judgeQuery.trim()) return;
    setLoading(true);
    setError(null);
    setProfile(null);
    try {
      const res = await fetch(
        `/api/analytics/judge-profile?judge=${encodeURIComponent(judgeQuery.trim())}&limit=30`,
        { credentials: "same-origin" }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const d: JudgeProfile = await res.json();
      setProfile(d);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const topVerdicts = profile
    ? Object.entries(profile.verdict_breakdown)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 6)
    : [];

  const topCourts = profile
    ? Object.entries(profile.court_breakdown)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
    : [];

  const yearlyEntries = profile
    ? Object.entries(profile.yearly_volume).sort(([a], [b]) => Number(a) - Number(b))
    : [];

  const maxYearly = yearlyEntries.length
    ? Math.max(...yearlyEntries.map(([, v]) => v))
    : 1;

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-6">
        <h3 className="font-display text-lg text-[var(--ink)] mb-1">Judge Intelligence</h3>
        <p className="text-xs text-[var(--ink-soft)]">
          Analyse how a judge rules — verdict breakdown, active courts, yearly volume across 16M+ judgments.
        </p>
      </div>

      {/* Search bar */}
      <div className="flex gap-2 mb-6">
        <div className="relative flex-1">
          <UserSearch size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--ink-soft)]" />
          <input
            type="text"
            value={judgeQuery}
            onChange={e => setJudgeQuery(e.target.value)}
            onKeyDown={e => e.key === "Enter" && search()}
            placeholder='e.g. "Chandrachud" or "Justice R.F. Nariman"'
            className="w-full pl-8 pr-3 py-2.5 text-sm border border-[var(--line)] rounded-xl focus:outline-none focus:border-[var(--accent)] bg-white text-[var(--ink)] placeholder:text-[var(--ink-soft)]"
          />
        </div>
        <button
          onClick={search}
          disabled={loading || !judgeQuery.trim()}
          className="px-4 py-2.5 rounded-xl text-sm font-medium text-white bg-[var(--accent)] hover:opacity-90 disabled:opacity-40 transition-opacity"
        >
          {loading ? "..." : "Search"}
        </button>
      </div>

      {error && (
        <div className="text-sm text-[var(--danger)] bg-red-50 border border-red-100 rounded-xl p-3 mb-4">{error}</div>
      )}

      {loading && (
        <div className="flex items-center gap-2 py-8 justify-center text-[var(--ink-soft)]">
          <div className="w-4 h-4 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
          <span className="text-sm">Querying 16M+ judgments...</span>
        </div>
      )}

      {profile && (
        <div className="space-y-5">
          {/* Header */}
          <div className="bg-white border border-[var(--line)] rounded-2xl p-5">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-xl bg-[var(--accent)]/10 flex items-center justify-center shrink-0">
                <Award size={22} style={{ color: "var(--accent)" }} />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-[var(--ink-soft)] mb-1">Matched judge name(s)</p>
                <p className="text-sm font-semibold text-[var(--ink)] truncate">
                  {profile.matched_names.length > 0 ? profile.matched_names.slice(0, 3).join(", ") : profile.query}
                </p>
                <div className="flex items-center gap-4 mt-2 text-xs text-[var(--ink-soft)]">
                  <span className="font-semibold text-[var(--ink)]">{profile.total.toLocaleString()}</span> total cases
                  <span className="font-semibold text-[var(--ink)]">{Object.keys(profile.court_breakdown).length}</span> courts
                </div>
              </div>
            </div>
          </div>

          {/* Verdict breakdown */}
          {topVerdicts.length > 0 && (
            <div className="bg-white border border-[var(--line)] rounded-2xl p-5">
              <p className="text-[10px] tracking-[0.18em] uppercase text-[var(--ink-soft)] mb-3 flex items-center gap-1.5">
                <Gavel size={10} /> Verdict Breakdown
              </p>
              <div className="space-y-2">
                {topVerdicts.map(([verdict, count]) => {
                  const pct = Math.round((count / Math.max(...topVerdicts.map(([, v]) => v))) * 100);
                  return (
                    <div key={verdict} className="flex items-center gap-3 text-xs">
                      <span className="text-[var(--ink-soft)] w-32 shrink-0 truncate capitalize">{verdict || "Unspecified"}</span>
                      <div className="flex-1 h-2 bg-[var(--bg)] rounded-full overflow-hidden">
                        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: "var(--accent)" }} />
                      </div>
                      <span className="text-[var(--ink)] font-medium w-10 text-right">{count.toLocaleString()}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Courts + Yearly volume side-by-side */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {topCourts.length > 0 && (
              <div className="bg-white border border-[var(--line)] rounded-2xl p-4">
                <p className="text-[10px] tracking-[0.18em] uppercase text-[var(--ink-soft)] mb-3 flex items-center gap-1.5">
                  <Building2 size={10} /> Courts
                </p>
                <div className="space-y-1.5">
                  {topCourts.map(([court, count]) => (
                    <div key={court} className="flex items-center justify-between gap-2 text-xs">
                      <span className="text-[var(--ink-soft)] truncate text-[11px]">{court}</span>
                      <span className="text-[var(--ink)] font-medium shrink-0">{count.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {yearlyEntries.length > 0 && (
              <div className="bg-white border border-[var(--line)] rounded-2xl p-4">
                <p className="text-[10px] tracking-[0.18em] uppercase text-[var(--ink-soft)] mb-3 flex items-center gap-1.5">
                  <TrendingUp size={10} /> Yearly Volume
                </p>
                <div className="flex items-end gap-1 h-20">
                  {yearlyEntries.slice(-15).map(([year, count]) => (
                    <div key={year} className="flex-1 flex flex-col items-center gap-1 min-w-0">
                      <div
                        className="w-full rounded-sm transition-all"
                        style={{
                          height: `${Math.max(4, (count / maxYearly) * 64)}px`,
                          background: "var(--accent)",
                          opacity: 0.75,
                        }}
                        title={`${year}: ${count}`}
                      />
                      {yearlyEntries.length <= 8 && (
                        <span className="text-[8px] text-[var(--ink-soft)] truncate">{year}</span>
                      )}
                    </div>
                  ))}
                </div>
                {yearlyEntries.length > 8 && (
                  <p className="text-[9px] text-[var(--ink-soft)] mt-1 text-center">
                    {yearlyEntries[0][0]} – {yearlyEntries[yearlyEntries.length - 1][0]}
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Recent cases sample */}
          {profile.cases.length > 0 && (
            <div className="bg-white border border-[var(--line)] rounded-2xl p-5">
              <p className="text-[10px] tracking-[0.18em] uppercase text-[var(--ink-soft)] mb-3 flex items-center gap-1.5">
                <FileText size={10} /> Recent Cases ({Math.min(profile.cases.length, 10)} of {profile.total.toLocaleString()})
              </p>
              <ul className="divide-y divide-[var(--line)]">
                {profile.cases.slice(0, 10).map((c, i) => (
                  <li key={i} className="py-2.5 flex items-start gap-3 text-xs">
                    <span className="text-[var(--ink-soft)] shrink-0 w-8">{c.year || "—"}</span>
                    <div className="flex-1 min-w-0">
                      <p className="text-[var(--ink)] truncate font-medium">{c.title || c.citation || c.cnr}</p>
                      <p className="text-[var(--ink-soft)] truncate text-[11px]">{c.court}</p>
                    </div>
                    {c.verdict && (
                      <span className="shrink-0 text-[9px] px-1.5 py-0.5 rounded bg-[var(--highlight)] text-[var(--accent)] border border-[var(--line-strong)] capitalize">
                        {c.verdict.slice(0, 15)}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {!profile && !loading && (
        <div className="text-center py-12">
          <UserSearch size={36} className="mx-auto text-[var(--ink-soft)] opacity-20 mb-3" />
          <p className="text-sm text-[var(--ink-soft)]">Search for a judge to see analytics</p>
          <p className="text-xs text-[var(--ink-soft)] mt-1 opacity-70">
            Try "Chandrachud", "Nariman", or any High Court judge name
          </p>
        </div>
      )}
    </div>
  );
}
