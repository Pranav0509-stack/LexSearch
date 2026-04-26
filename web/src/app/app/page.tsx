"use client";

import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  Plus,
  MessageSquare,
  FolderClosed,
  Workflow,
  History as HistoryIcon,
  Library as LibraryIcon,
  Settings as SettingsIcon,
  LogOut,
  Globe,
  Database,
  Languages,
  Inbox,
  Menu,
  X,
  PanelRightClose,
  PanelRightOpen,
  Copy,
  Check,
  ArrowUpRight,
  Cpu,
  Mail,
  FileDown,
  Sparkles,
  Scale,
} from "lucide-react";
import { PromptInputBox } from "@/components/ui/ai-prompt-box";
import { renderMarkdown } from "./markdown";
import VaultPane from "./vault-pane";
import WorkflowsPane from "./workflows-pane";
import HistoryPane from "./history-pane";
import LibraryPane from "./library-pane";
import CourtSearchPane from "./court-search-pane";
import SettingsPane from "./settings-pane";
import ClientsPane from "./clients-pane";

// Sanhita serves three jurisdictions: India, Singapore, Hong Kong.
// Each has a live BM25 corpus (ingested from open GitHub datasets) + a
// dedicated default-source stack in `connectors._default_sources_for`.
const JURISDICTIONS: { code: string; flag: string; name: string }[] = [
  { code: "IN", flag: "🇮🇳", name: "India" },
  { code: "SG", flag: "🇸🇬", name: "Singapore" },
  { code: "HK", flag: "🇭🇰", name: "Hong Kong SAR" },
];

// Source databases — `available_connectors()` from the FastAPI backend tells
// us at runtime which ones have API keys wired so we can grey out the rest.
const SOURCES: { value: string; icon: string; label: string }[] = [
  { value: "", icon: "⚖️", label: "All available" },
  { value: "indian_kanoon", icon: "🇮🇳", label: "Indian Kanoon" },
  { value: "ecourts", icon: "🏛️", label: "eCourts (India)" },
  { value: "egov_japan", icon: "🇯🇵", label: "e-Gov (Japan)" },
  { value: "web", icon: "🌐", label: "Open web" },
  { value: "seed", icon: "📚", label: "Seed corpus" },
  { value: "indian_kanoon,ecourts", icon: "⚖️", label: "Indian sources only" },
  { value: "web,seed", icon: "🌍", label: "Web + seed" },
];

// Reasoning model picker. The `value` maps to llm.router.generate's
// `prefer` kwarg ("gemini" | "anthropic" | "groq" | "cloudflare"). Empty
// value = router default (Gemini Flash). Labels are user-facing — keep
// them short, concrete, and brand-aligned with each provider's own naming.
const MODELS: { value: string; label: string }[] = [
  { value: "",          label: "Auto · Gemini" },
  { value: "gemini",    label: "Gemini 2.5 Flash" },
  { value: "anthropic", label: "Claude Sonnet 4.5" },
  { value: "groq",      label: "Llama 3.3 70B" },
  { value: "cloudflare",label: "Workers AI" },
];

const SUGGESTIONS: { q: string; tag: string }[] = [
  {
    tag: "Anticipatory bail",
    q: "Conditions for anticipatory bail under §482 BNSS — leading authority since 2024.",
  },
  {
    tag: "Section 138 NI Act",
    q: "Standard of proof for cheque dishonour under Section 138 NI Act, recent SC view.",
  },
  {
    tag: "Japan labour law",
    q: "Termination for poor performance in Japan — Article 16 Labour Contracts Act test.",
  },
  {
    tag: "Singapore IP",
    q: "Trademark passing-off test under Singapore law, citing recent High Court decisions.",
  },
];

type Mode =
  | "assistant"
  | "vault"
  | "workflows"
  | "court-search"
  | "history"
  | "library"
  | "clients"
  | "settings";

interface LanguageOpt {
  code: string;
  label: string;
  native: string;
  family: string;
  rtl: boolean;
}

interface Thread {
  id: number;
  title: string;
  updated_at: number;
}

interface Citation {
  n: number;
  title: string;
  court?: string;
  year?: number | string;
  citation?: string;
  excerpt?: string;
  pdf_url?: string;
}

interface TraceStep {
  tool?: string;
  args?: Record<string, unknown>;
  result_preview?: string;
  ms?: number;
  error?: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  llm?: { provider?: string; model?: string; latency_ms?: number };
  validation?: { confidence?: number; reasons?: string[] };
  refused?: boolean;
  trace?: TraceStep[];
  mode?: string;
  followups?: string[];
}

export default function AppPage() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("assistant");
  const [user, setUser] = useState<{ email?: string; name?: string } | null>(null);
  // Threads list is no longer rendered in the sidebar (History pane owns
  // that surface). We still keep `setThreads` to seed the list on boot and
  // bump titles after the first chat — the History pane refetches its own
  // search results on demand. The `threads` array itself is unused so we
  // mute the lint warning by tossing the read with a `void`.
  const [threads, setThreads] = useState<Thread[]>([]);
  void threads;
  const [activeThread, setActiveThread] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [thinking, setThinking] = useState(false);
  // Live "what's happening right now" phases shown under the answer
  // bubble while we wait for the API. Each phase is a short ChatGPT-style
  // status line ("Searching case law…", "Drafting memo…"). The handler
  // advances them on a timer so the user sees motion even though the
  // backend isn't streaming.
  const [thinkingPhases, setThinkingPhases] = useState<string[]>([]);
  const [jurisdiction, setJurisdiction] = useState("IN");
  const [source, setSource] = useState("");
  const [connectors, setConnectors] = useState<Record<string, boolean>>({});
  // Output language for the AI's reply. Persisted in localStorage so a
  // Hindi-speaking user keeps Hindi across sessions. Empty string = English
  // (we never send an "en" code; null/empty round-trips as default).
  const [language, setLanguage] = useState<string>(
    typeof window !== "undefined"
      ? window.localStorage.getItem("sanhita.lang") || ""
      : ""
  );
  const [languages, setLanguages] = useState<LanguageOpt[]>([]);
  // Model preference. Empty = let the router pick (default: Gemini Flash).
  // Persisted in localStorage so the user's pick survives reloads. The
  // backend honours this via the `prefer` arg on llm.router.generate.
  const [model, setModel] = useState<string>(
    typeof window !== "undefined"
      ? window.localStorage.getItem("sanhita.model") || ""
      : ""
  );
  // Count of `new` NyayaSathi leads — drives the sidebar badge.
  const [newClientCount, setNewClientCount] = useState(0);
  // Mobile drawer state. Sidebar is permanent on >= md; on phone it slides
  // in over the chat. Citations rail is permanent on >= lg; on tablet it
  // toggles via a button in the topbar so the chat column gets full width.
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [railOpen, setRailOpen] = useState(false);
  const chatLogRef = useRef<HTMLDivElement>(null);

  // ── boot: load threads + connectors. 401 ⇒ /login.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch("/api/brief/threads", { credentials: "same-origin" });
        if (r.status === 401) {
          router.replace("/login");
          return;
        }
        const data = await r.json();
        if (cancelled) return;
        setThreads(data.threads || []);
        setUser(data.user || null);
      } catch {
        router.replace("/login");
      }
      try {
        const r = await fetch("/api/connectors");
        const data = await r.json();
        if (!cancelled) setConnectors(data.connectors || {});
      } catch {
        /* connector status is decorative */
      }
      // Load language catalog from the backend (single source of truth —
      // adding a language only requires extending brief_service.LANGUAGES).
      try {
        const r = await fetch("/api/languages");
        const data = await r.json();
        if (!cancelled) setLanguages(data.languages || []);
      } catch {
        /* fall back to English-only display */
      }
      // Initial NyayaSathi inbox count (drives sidebar badge).
      try {
        const r = await fetch("/api/clients?status=new", {
          credentials: "same-origin",
        });
        if (r.ok) {
          const data = await r.json();
          if (!cancelled) setNewClientCount(data.counts?.new ?? 0);
        }
      } catch {
        /* badge is decorative */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  // Auto-scroll the chat log on new messages
  useEffect(() => {
    chatLogRef.current?.scrollTo({
      top: chatLogRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, thinking]);

  const newThread = useCallback(async (): Promise<number | null> => {
    try {
      const r = await fetch("/api/brief/threads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: "{}",
      });
      if (!r.ok) return null;
      const data = await r.json();
      const t: Thread = data.thread;
      setThreads((prev) => [t, ...prev]);
      setActiveThread(t.id);
      setMessages([]);
      return t.id;
    } catch {
      return null;
    }
  }, []);

  const openThread = useCallback(async (id: number) => {
    setActiveThread(id);
    setMessages([]);
    try {
      const r = await fetch(`/api/brief/threads/${id}`, { credentials: "same-origin" });
      if (!r.ok) return;
      const data = await r.json();
      const msgs: Message[] = (data.messages || []).map((m: { role: string; content: string; citations?: string }) => ({
        role: m.role as "user" | "assistant",
        content: m.content,
        citations: m.citations ? safeParse(m.citations) : undefined,
      }));
      setMessages(msgs);
    } catch {
      /* silent */
    }
  }, []);

  const handleSend = useCallback(
    async (
      message: string,
      _files?: File[],
      opts?: { search?: boolean; think?: boolean; canvas?: boolean; agent?: boolean }
    ) => {
      const text = message.trim();
      if (!text) return;

      let tid = activeThread;
      if (!tid) {
        tid = await newThread();
        if (!tid) return;
      }

      setMessages((prev) => [...prev, { role: "user", content: text }]);
      setThinking(true);

      // Mode routing — Harvey-style action toggles.
      //   Agent   → /api/brief/agent (Gemini chains tools across turns)
      //   Canvas  → /api/brief/draft  (open drafting, no retrieval)
      //   Search  → /api/brief/web    (real web search + grounded answer)
      //   else    → /api/brief/chat   (research mode: BM25 + connectors)
      const endpoint = opts?.agent
        ? "/api/brief/agent"
        : opts?.canvas
        ? "/api/brief/draft"
        : opts?.search
        ? "/api/brief/web"
        : "/api/brief/chat";

      // Live "thinking" phases — ChatGPT-style status under the bubble.
      // The backend isn't streaming, so we advance through plausible
      // phases on a timer. The phases mirror what the server is actually
      // doing in that mode (retrieve → draft → validate, or just draft
      // for canvas, or web fetch for search). Stops when the response
      // lands.
      const phasesByMode: Record<string, { ms: number; label: string }[]> =
        opts?.agent
          ? {
              agent: [
                { ms: 0, label: "Planning the agent loop…" },
                { ms: 1200, label: "Searching case law…" },
                { ms: 4000, label: "Pulling statutes…" },
                { ms: 7000, label: "Composing answer with citations…" },
                { ms: 11000, label: "Validating sources…" },
              ],
            }
          : opts?.canvas
          ? {
              draft: [
                { ms: 0, label: "Reading your request…" },
                { ms: 800, label: "Drafting with Gemini Flash…" },
                { ms: 4500, label: "Polishing structure…" },
              ],
            }
          : opts?.search
          ? {
              web: [
                { ms: 0, label: "Searching the open web…" },
                { ms: 1200, label: "Reading top results…" },
                { ms: 3000, label: "Composing grounded answer…" },
                { ms: 7000, label: "Mapping citations to URLs…" },
              ],
            }
          : {
              chat: [
                { ms: 0, label: "Searching the case-law index…" },
                { ms: 900, label: "Ranking the strongest authorities…" },
                { ms: 2200, label: "Drafting the memo…" },
                { ms: 6500, label: "Checking citations resolve…" },
                { ms: 10000, label: "Final pass on grounding…" },
              ],
            };
      const phases = Object.values(phasesByMode)[0]!;
      setThinkingPhases([phases[0].label]);
      const phaseTimers = phases.slice(1).map((p) =>
        setTimeout(() => {
          setThinkingPhases((prev) => [...prev, p.label]);
        }, p.ms)
      );

      // Source allowlist only applies to research mode.
      let srcs: string[] | null = null;
      if (!opts?.canvas && !opts?.search && !opts?.agent && source) {
        srcs = source.split(",").map((s) => s.trim()).filter(Boolean);
      }

      try {
        const r = await fetch(endpoint, {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            thread_id: tid,
            question: text,
            jurisdiction,
            sources: srcs,
            language: language || undefined,
            // `prefer` matches llm.router.generate's prefer kwarg —
            // "gemini" | "anthropic" | "groq" | "cloudflare" reorder the
            // chain to put that provider first. Empty string = router
            // default (Gemini Flash).
            prefer: model || undefined,
          }),
        });
        const data = await r.json();
        phaseTimers.forEach(clearTimeout);
        setThinkingPhases([]);
        setThinking(false);
        if (!r.ok) {
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: `**Error:** ${data.detail || `HTTP ${r.status}`}`,
            },
          ]);
          return;
        }
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.answer_markdown,
            citations: data.citations,
            llm: data.llm,
            validation: data.validation,
            refused: !!data.refused,
            trace: data.trace,
            mode: data.mode,
            followups: data.followups,
          },
        ]);
        // Bump the thread title if it was a fresh "New matter".
        setThreads((prev) =>
          prev.map((t) =>
            t.id === tid && (!t.title || t.title === "New matter")
              ? { ...t, title: text.slice(0, 48) }
              : t
          )
        );
      } catch (e) {
        phaseTimers.forEach(clearTimeout);
        setThinkingPhases([]);
        setThinking(false);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `**Network error.** ${(e as Error).message}`,
          },
        ]);
      }
    },
    [activeThread, jurisdiction, source, language, model, newThread]
  );

  // Persist the language picker value across sessions. Keeping it on the
  // window/localStorage layer (not just React state) means a returning
  // Tamil-speaking user lands on Tamil without having to re-pick.
  useEffect(() => {
    if (typeof window === "undefined") return;
    if (language) window.localStorage.setItem("sanhita.lang", language);
    else window.localStorage.removeItem("sanhita.lang");
  }, [language]);

  // Persist model preference across reloads.
  useEffect(() => {
    if (typeof window === "undefined") return;
    if (model) window.localStorage.setItem("sanhita.model", model);
    else window.localStorage.removeItem("sanhita.model");
  }, [model]);

  const lastCitations = useMemo<Citation[]>(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant" && messages[i].citations?.length) {
        return messages[i].citations!;
      }
    }
    return [];
  }, [messages]);

  return (
    <div className="flex h-[100dvh] w-screen bg-[var(--bg)] overflow-hidden">
      {/* Mobile backdrop — only renders when the drawer is open on phones. */}
      {sidebarOpen && (
        <div
          className="md:hidden fixed inset-0 bg-black/40 z-30"
          onClick={() => setSidebarOpen(false)}
          aria-hidden
        />
      )}

      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <aside
        className={`flex flex-col border-r border-[var(--line)] bg-[var(--bg)] min-w-0 z-40 transition-transform duration-200
          fixed md:static inset-y-0 left-0 w-[260px] md:w-[230px] shrink-0
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0`}
      >
        <div className="px-6 pt-7 pb-6 flex items-start justify-between">
          <div>
            <div className="font-display text-3xl tracking-tight">Sanhita</div>
            <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] mt-1">
              Research counsel
            </div>
          </div>
          {/* Close button — visible only on phone where the sidebar is a
              drawer over the chat column. */}
          <button
            className="md:hidden p-1.5 rounded-lg text-[var(--ink-soft)] hover:bg-[var(--bg-elev)]"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close menu"
          >
            <X size={18} />
          </button>
        </div>

        <button
          onClick={() => {
            setMode("assistant");
            newThread();
            setSidebarOpen(false);
          }}
          className="mx-4 mb-4 flex items-center gap-2 rounded-xl bg-[var(--ink)] text-[var(--bg)] py-2.5 px-4 text-sm font-medium hover:bg-[var(--accent)] transition-colors"
        >
          <Plus size={16} strokeWidth={2.4} />
          New matter
        </button>

        <nav className="px-2 flex flex-col gap-0.5">
          <SideItem icon={<MessageSquare size={16} />} label="Assistant" active={mode === "assistant"} onClick={() => { setMode("assistant"); setSidebarOpen(false); }} />
          <SideItem icon={<FolderClosed size={16} />} label="Storage" active={mode === "vault"} onClick={() => { setMode("vault"); setSidebarOpen(false); }} />
          <SideItem icon={<Workflow size={16} />} label="Workflows" active={mode === "workflows"} onClick={() => { setMode("workflows"); setSidebarOpen(false); }} />
          <SideItem icon={<Scale size={16} />} label="Court Search" active={mode === "court-search"} onClick={() => { setMode("court-search"); setSidebarOpen(false); }} />
          <SideItem icon={<Inbox size={16} />} label="Clients" active={mode === "clients"} onClick={() => { setMode("clients"); setSidebarOpen(false); }} badge={newClientCount > 0 ? newClientCount : undefined} />
          <SideItem icon={<HistoryIcon size={16} />} label="History" active={mode === "history"} onClick={() => { setMode("history"); setSidebarOpen(false); }} />
          <SideItem icon={<LibraryIcon size={16} />} label="Library" active={mode === "library"} onClick={() => { setMode("library"); setSidebarOpen(false); }} />
          <SideItem icon={<SettingsIcon size={16} />} label="Settings" active={mode === "settings"} onClick={() => { setMode("settings"); setSidebarOpen(false); }} />
        </nav>

        {/* Spacer pushes footer to bottom — past threads now live in History pane */}
        <div className="flex-1" />

        {/* Footer — who am I + sign out */}
        <div className="mt-auto p-4 border-t border-[var(--line)]">
          <div className="text-xs text-[var(--ink-soft)] truncate" title={user?.email}>
            {user?.name || user?.email || "—"}
          </div>
          <a
            href="/api/logout"
            className="mt-2 flex items-center gap-2 text-xs text-[var(--ink-soft)] hover:text-[var(--danger)] transition-colors"
          >
            <LogOut size={13} /> Sign out
          </a>
        </div>
      </aside>

      {/* ── Main column ─────────────────────────────────────────────────── */}
      <main className="flex flex-col flex-1 min-w-0 paper-grain overflow-hidden">
        {/* Topbar with jurisdiction + source pickers — flex-wrap so the
            selects fall to a second row on narrow viewports rather than
            pushing the whole layout horizontally. */}
        <div className="flex items-center justify-between gap-2 flex-wrap border-b border-[var(--line)] px-3 sm:px-6 py-3 min-w-0">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0">
            {/* Hamburger — only on phone (sidebar is a drawer there). */}
            <button
              className="md:hidden p-1.5 rounded-lg text-[var(--ink-soft)] hover:bg-[var(--bg-elev)] shrink-0"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open menu"
            >
              <Menu size={20} />
            </button>
            <h1 className="font-display text-lg tracking-tight capitalize truncate">
              {modeTitle(mode)}
            </h1>
          </div>

          {mode === "assistant" && (
            // Compact pill row — single line. Each chip is icon + value
            // with no trailing chevron padding; the native <select> arrow
            // is hidden on the closed state and only revealed on hover.
            // Tighter horizontal rhythm (gap-1.5) so all 4 selectors +
            // citations toggle fit on a 14" laptop without wrapping.
            <div className="flex items-center gap-1.5 flex-wrap min-w-0 toolbar-pills">
              <label className="pill" title="Jurisdiction">
                <Globe size={12} className="text-[var(--ink-soft)] shrink-0" />
                <span className="shrink-0">{JURISDICTIONS.find((j) => j.code === jurisdiction)?.flag}</span>
                <select
                  value={jurisdiction}
                  onChange={(e) => setJurisdiction(e.target.value)}
                  className="pill-select"
                >
                  {JURISDICTIONS.map((j) => (
                    <option key={j.code} value={j.code}>
                      {j.name}
                    </option>
                  ))}
                </select>
              </label>

              <label className="pill" title="Source databases">
                <LibraryIcon size={12} className="text-[var(--accent)] shrink-0" />
                <select
                  value={source}
                  onChange={(e) => setSource(e.target.value)}
                  className="pill-select"
                >
                  {SOURCES.map((s) => {
                    const single = !s.value.includes(",") && s.value;
                    const disabled = single ? connectors[single] === false : false;
                    return (
                      <option key={s.value} value={s.value} disabled={disabled}>
                        {s.label}
                        {disabled ? " (no key)" : ""}
                      </option>
                    );
                  })}
                </select>
              </label>

              <label className="pill" title="Reasoning model">
                <Cpu size={12} className="text-[var(--ink-soft)] shrink-0" />
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="pill-select"
                >
                  {MODELS.map((m) => (
                    <option key={m.value} value={m.value}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </label>

              {languages.length > 0 && (
                <label className="pill" title="Reply language">
                  <Languages size={12} className="text-[var(--ink-soft)] shrink-0" />
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    className="pill-select"
                  >
                    {languages.map((l) => (
                      <option key={l.code} value={l.code === "en" ? "" : l.code}>
                        {l.native} {l.code !== "en" ? `(${l.label})` : ""}
                      </option>
                    ))}
                  </select>
                </label>
              )}

              <button
                className="pill pill-button"
                onClick={() => setRailOpen((v) => !v)}
                aria-label="Toggle citations"
                title="Show sources / citations"
              >
                {railOpen ? <PanelRightClose size={12} /> : <PanelRightOpen size={12} />}
                <span className="hidden sm:inline">
                  {lastCitations.length > 0
                    ? `${lastCitations.length}`
                    : "Sources"}
                </span>
              </button>
            </div>
          )}
        </div>

        {/* Body — switches by mode */}
        {mode === "assistant" && (
          <AssistantPane
            messages={messages}
            thinking={thinking}
            thinkingPhases={thinkingPhases}
            onSend={handleSend}
            chatLogRef={chatLogRef}
            citations={lastCitations}
            suggestions={SUGGESTIONS}
            railOpen={railOpen}
            onCloseRail={() => setRailOpen(false)}
          />
        )}
        {mode === "vault" && <VaultPane />}
        {mode === "workflows" && <WorkflowsPane />}
        {mode === "court-search" && (
          <CourtSearchPane
            onUseInChat={async (c) => {
              const tid = activeThread || (await newThread());
              if (!tid) return;
              setMode("assistant");
              const seed = `Use this case as context:\n\n${c.body_md}\n\nNow help me with: `;
              setMessages((prev) => [
                ...prev,
                { role: "assistant", content: seed },
              ]);
            }}
          />
        )}
        {mode === "clients" && (
          <ClientsPane
            onOpenThread={async (id) => {
              setMode("assistant");
              await openThread(id);
              // Refresh badge — the client we just opened has flipped to
              // in_progress, so the "new" count drops by one.
              try {
                const r = await fetch("/api/clients?status=new", {
                  credentials: "same-origin",
                });
                if (r.ok) {
                  const data = await r.json();
                  setNewClientCount(data.counts?.new ?? 0);
                }
              } catch {
                /* badge is decorative */
              }
            }}
          />
        )}
        {mode === "history" && (
          <HistoryPane
            onOpenThread={(id) => {
              setMode("assistant");
              openThread(id);
            }}
          />
        )}
        {mode === "library" && (
          <LibraryPane
            onUseInChat={async (doc) => {
              const tid = activeThread || (await newThread());
              if (!tid) return;
              setMode("assistant");
              const seed = `Use this ${doc.kind} as context:\n\n**${doc.title}**\n\n${doc.body_md.slice(0, 1200)}…\n\nNow help me with: `;
              setMessages((prev) => [
                ...prev,
                { role: "assistant", content: seed },
              ]);
            }}
          />
        )}
        {mode === "settings" && (
          <SettingsPane
            onChange={async () => {
              try {
                const r = await fetch("/api/connectors");
                const data = await r.json();
                setConnectors(data.connectors || {});
              } catch {
                /* connector status is decorative */
              }
            }}
          />
        )}
      </main>
    </div>
  );
}

// ───────────────────────────────────────────────────────────────────────────

function SideItem({
  icon,
  label,
  active,
  onClick,
  badge,
}: {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  onClick: () => void;
  badge?: number;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-3 px-4 py-2 rounded-lg text-sm font-medium transition-colors text-left ${
        active
          ? "bg-[var(--highlight)] text-[var(--ink)]"
          : "text-[var(--ink-soft)] hover:bg-[var(--bg-elev)] hover:text-[var(--ink)]"
      }`}
    >
      <span className="text-[var(--accent)]">{icon}</span>
      <span className="flex-1">{label}</span>
      {badge !== undefined && badge > 0 && (
        <span className="text-[10px] tracking-wide bg-[var(--accent)] text-[var(--bg)] rounded-full px-1.5 py-0.5 min-w-[18px] text-center font-mono">
          {badge > 99 ? "99+" : badge}
        </span>
      )}
    </button>
  );
}

function modeTitle(mode: Mode): string {
  switch (mode) {
    case "assistant":
      return "Assistant";
    case "vault":
      return "Storage";
    case "workflows":
      return "Workflows";
    case "clients":
      return "Clients";
    case "history":
      return "History";
    case "library":
      return "Library";
    case "settings":
      return "Settings";
  }
}

// ── Assistant pane: chat log + citations rail + composer ───────────────────
function AssistantPane({
  messages,
  thinking,
  thinkingPhases,
  onSend,
  chatLogRef,
  citations,
  suggestions,
  railOpen,
  onCloseRail,
}: {
  messages: Message[];
  thinking: boolean;
  thinkingPhases: string[];
  onSend: (m: string, files?: File[], opts?: { search?: boolean; think?: boolean; canvas?: boolean }) => void;
  chatLogRef: React.RefObject<HTMLDivElement | null>;
  citations: Citation[];
  suggestions: { q: string; tag: string }[];
  railOpen: boolean;
  onCloseRail: () => void;
}) {
  const empty = messages.length === 0;

  return (
    // Single-column on phone/tablet (< lg), two-column with permanent rail
    // on lg+. The rail becomes a slide-over drawer on smaller viewports —
    // toggled from the topbar button.
    <div className="flex flex-1 min-h-0 min-w-0 relative">
      {/* Chat column */}
      <section className="flex flex-col flex-1 min-w-0 min-h-0">
        <div ref={chatLogRef} className="flex-1 overflow-y-auto px-4 sm:px-6 lg:px-12 py-6 sm:py-8 min-w-0">
          {empty ? (
            <EmptyState onPick={(q) => onSend(q)} suggestions={suggestions} />
          ) : (
            <div className="max-w-3xl mx-auto flex flex-col gap-6 min-w-0">
              {messages.map((m, i) => (
                <ChatBubble key={i} m={m} onPickFollowup={(q) => onSend(q)} />
              ))}
              {thinking && <ThinkingPanel phases={thinkingPhases} />}
            </div>
          )}
        </div>

        {/* Composer — compact on phone, generous on desktop. The
            safe-area-inset keeps the send button above the iPhone home bar. */}
        <div
          className="px-3 sm:px-6 lg:px-12 pt-3 sm:pt-4 border-t border-[var(--line)] bg-[var(--bg)] min-w-0"
          style={{ paddingBottom: "max(0.75rem, env(safe-area-inset-bottom))" }}
        >
          <div className="max-w-3xl mx-auto min-w-0">
            <PromptInputBox
              onSend={onSend}
              isLoading={thinking}
              placeholder="Ask Sanhita — cite-grounded answers across Asia"
            />
          </div>
        </div>
      </section>

      {/* Backdrop — only renders while the rail drawer is open. Click
          outside to dismiss. */}
      {railOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-30"
          onClick={onCloseRail}
          aria-hidden
        />
      )}

      {/* Citations rail — always a slide-over drawer. Hidden by default on
          every viewport; opens when the user clicks the Sources button in
          the topbar. overflow-x-hidden so long titles wrap instead of
          pushing the column wider than its track. */}
      <aside
        className={`border-l border-[var(--line)] bg-[var(--bg-elev)] overflow-y-auto overflow-x-hidden min-w-0
          fixed inset-y-0 right-0 w-[88vw] sm:w-[340px] z-40 transition-transform duration-200 shadow-xl
          ${railOpen ? "translate-x-0" : "translate-x-full"}`}
      >
        <div className="px-5 py-4 border-b border-[var(--line)] min-w-0 flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)]">
              Sources
            </div>
            <div className="font-display text-lg tracking-tight">
              {citations.length ? `${citations.length} citation${citations.length === 1 ? "" : "s"}` : "Citations"}
            </div>
          </div>
          <button
            className="p-1 rounded-md text-[var(--ink-soft)] hover:bg-[var(--bg)] shrink-0"
            onClick={onCloseRail}
            aria-label="Close citations"
          >
            <X size={16} />
          </button>
        </div>

        <div className="px-3 py-3 flex flex-col gap-2 min-w-0">
          {citations.length === 0 && (
            <p className="text-sm italic text-[var(--ink-soft)] px-2">
              Citations from your latest answer will appear here.
            </p>
          )}
          {citations.map((c) => (
            <SourceCard key={c.n} c={c} />
          ))}
        </div>
      </aside>
    </div>
  );
}

// Live "what's happening right now" panel — ChatGPT-style. Each entry
// in `phases` was appended on a timer by the handler, so the list grows
// as the request progresses. The most recent phase has a pulsing dot
// (in-flight); earlier phases get a static check (done).
function ThinkingPanel({ phases }: { phases: string[] }) {
  return (
    <div className="self-start w-full sm:max-w-[92%] flex flex-col gap-2 min-w-0">
      <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] flex items-center gap-2">
        <Sparkles size={11} className="text-[var(--accent)]" />
        Thinking
      </div>
      <div className="bg-[var(--bg-elev)] border border-[var(--line)] rounded-2xl rounded-bl-sm px-5 py-3 min-w-0">
        <ul className="flex flex-col gap-1.5 text-[13px]">
          {phases.length === 0 && (
            <li className="flex items-center gap-2 text-[var(--ink-soft)]">
              <span className="thinking-pulse" />
              <span>Working…</span>
            </li>
          )}
          {phases.map((p, i) => {
            const isLast = i === phases.length - 1;
            return (
              <li
                key={`${i}-${p}`}
                className={`flex items-center gap-2 ${
                  isLast ? "text-[var(--ink)]" : "text-[var(--ink-soft)]"
                }`}
              >
                {isLast ? (
                  <span className="thinking-pulse" />
                ) : (
                  <Check size={12} className="text-[var(--accent)] shrink-0" />
                )}
                <span>{p}</span>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}

function EmptyState({
  onPick: _onPick,
  suggestions: _suggestions,
}: {
  onPick: (q: string) => void;
  suggestions: { q: string; tag: string }[];
}) {
  // Suggestion cards intentionally removed — the empty state now shows only
  // the headline + tagline so the screen feels uncluttered and focuses the
  // user on the composer below.
  void _onPick;
  void _suggestions;
  return (
    // Vertically centered greeting — flex column so the heading sits
    // optically at the visual centre of the chat column rather than
    // hugging the topbar. No description; "Ask Sanhita" prompt copy
    // lives in the composer below and is the call to action.
    <div className="h-full min-h-[60vh] flex items-center justify-center px-4">
      <h2 className="font-display text-3xl sm:text-4xl lg:text-5xl tracking-[-0.025em] text-[var(--ink)] leading-[1.1] text-center max-w-2xl">
        Where would you like to begin, Counsel?
      </h2>
    </div>
  );
}

function ChatBubble({ m, onPickFollowup }: { m: Message; onPickFollowup?: (q: string) => void }) {
  const [copied, setCopied] = useState(false);
  const [savingDoc, setSavingDoc] = useState(false);
  const [savedDoc, setSavedDoc] = useState<string | null>(null);

  if (m.role === "user") {
    return (
      <div className="self-end max-w-[88%] sm:max-w-[80%]">
        <div className="bg-[var(--ink)] text-[var(--bg)] rounded-2xl rounded-br-sm px-4 sm:px-5 py-2.5 sm:py-3 leading-relaxed whitespace-pre-wrap break-words">
          {m.content}
        </div>
      </div>
    );
  }

  // Strip markdown for paste-friendly outputs (clipboard, email body).
  const toPlain = (md: string) =>
    md
      .replace(/\*\*(.*?)\*\*/g, "$1")
      .replace(/[*_`]/g, "")
      .replace(/^#+\s*/gm, "");

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(toPlain(m.content));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard API unavailable — silently no-op */
    }
  };

  // Email: open the user's mail client with the answer pre-filled.
  // Lightweight, no OAuth needed — works everywhere.
  const onEmail = () => {
    const subject = "Sanhita research note";
    const body = toPlain(m.content);
    const url = `mailto:?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
    window.open(url, "_blank");
  };

  // Save-as-Doc: tries the Google Docs endpoint if Google is connected;
  // otherwise falls back to a local .md download so the user always gets
  // a file out of the click.
  const onSaveDoc = async () => {
    if (savingDoc) return;
    setSavingDoc(true);
    try {
      const res = await fetch("/api/google/docs/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          title: "Sanhita research note",
          body_markdown: m.content,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data?.url) {
          window.open(data.url, "_blank");
          setSavedDoc("Opened in Google Docs");
          setTimeout(() => setSavedDoc(null), 1800);
          return;
        }
      }
      // Fallback — download as .md
      const blob = new Blob([m.content], { type: "text/markdown;charset=utf-8" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `sanhita-${Date.now()}.md`;
      a.click();
      URL.revokeObjectURL(a.href);
      setSavedDoc("Downloaded");
      setTimeout(() => setSavedDoc(null), 1800);
    } catch {
      // Same fallback path on network error.
      const blob = new Blob([m.content], { type: "text/markdown;charset=utf-8" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `sanhita-${Date.now()}.md`;
      a.click();
      URL.revokeObjectURL(a.href);
      setSavedDoc("Downloaded");
      setTimeout(() => setSavedDoc(null), 1800);
    } finally {
      setSavingDoc(false);
    }
  };

  // assistant
  return (
    <div className="self-start w-full sm:max-w-[92%] flex flex-col gap-2 min-w-0">
      <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] flex items-center gap-2 flex-wrap">
        Sanhita
        {m.llm?.provider && m.llm.provider !== "guard" && (
          <span className="bg-[var(--bg-elev)] border border-[var(--line)] px-1.5 py-0.5 rounded text-[9px] tracking-wider">
            via {m.llm.provider}
          </span>
        )}
        {typeof m.validation?.confidence === "number" && (
          <span
            className={`px-1.5 py-0.5 rounded text-[9px] tracking-wider ${
              m.validation.confidence >= 0.85
                ? "bg-[var(--highlight)] text-[var(--accent)]"
                : "bg-[var(--bg-elev)] text-[var(--ink-soft)]"
            }`}
            title={(m.validation.reasons || []).join(" · ")}
          >
            {Math.round(m.validation.confidence * 100)}% grounded
          </span>
        )}
        {m.refused && (
          <span className="bg-[var(--danger)] text-white px-1.5 py-0.5 rounded text-[9px] tracking-wider">
            refused
          </span>
        )}
        {m.mode === "agent" && (
          <span className="bg-[#22C55E]/15 border border-[#22C55E]/40 text-[#22C55E] px-1.5 py-0.5 rounded text-[9px] tracking-wider">
            agent
          </span>
        )}
      </div>
      {m.trace && m.trace.length > 0 && <TraceBreadcrumbs trace={m.trace} />}
      <div
        className="bg-[var(--bg-elev)] border border-[var(--line)] rounded-2xl rounded-bl-sm px-6 py-4 leading-relaxed text-[15px] prose-style"
        dangerouslySetInnerHTML={{ __html: renderMarkdown(m.content) }}
      />

      {/* Action row — Copy / Email / Save-as-Doc. Lawyers want the
          answer OUT of the chat: into a draft email, into a shared Doc,
          or simply paste into Word. Hidden for refusals. */}
      {!m.refused && m.content && m.content.length > 40 && (
        <div className="flex items-center gap-4 text-[11px] text-[var(--ink-soft)] pl-1">
          <button
            onClick={onCopy}
            className="flex items-center gap-1.5 hover:text-[var(--ink)] transition-colors"
            title="Copy answer (plain text)"
          >
            {copied ? <Check size={12} /> : <Copy size={12} />}
            <span>{copied ? "Copied" : "Copy"}</span>
          </button>
          <button
            onClick={onEmail}
            className="flex items-center gap-1.5 hover:text-[var(--ink)] transition-colors"
            title="Send as email — opens your mail client with the answer pre-filled"
          >
            <Mail size={12} />
            <span>Email</span>
          </button>
          <button
            onClick={onSaveDoc}
            disabled={savingDoc}
            className="flex items-center gap-1.5 hover:text-[var(--ink)] transition-colors disabled:opacity-60"
            title="Save to Google Docs (or download as Markdown)"
          >
            <FileDown size={12} />
            <span>
              {savingDoc ? "Saving…" : savedDoc ?? "Save as Doc"}
            </span>
          </button>
        </div>
      )}

      {/* Follow-up suggestion cards — Harvey-style "what to ask next".
          Card grid (2-up on md+) so all three suggestions sit in the
          user's eye-line at once. Click sends the question back through
          the composer's onSend handler. */}
      {m.followups && m.followups.length > 0 && onPickFollowup && (
        <div className="mt-4 flex flex-col gap-2.5">
          <div className="flex items-center gap-1.5 text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] pl-1">
            <Sparkles size={11} className="text-[var(--accent)]" />
            <span>Suggested next steps</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {m.followups.map((q, i) => (
              <button
                key={i}
                onClick={() => onPickFollowup(q)}
                className="group relative flex flex-col gap-2 text-left text-[13.5px] leading-snug bg-gradient-to-br from-[var(--bg-elev)] to-[var(--bg)] border border-[var(--line)] hover:border-[var(--accent)] hover:shadow-[0_4px_14px_rgba(120,80,40,0.08)] rounded-2xl px-4 py-3 transition-all duration-150 min-w-0"
              >
                <span className="text-[var(--ink)] min-w-0 break-words pr-5">
                  {q}
                </span>
                <span className="flex items-center justify-between text-[10px] tracking-[0.18em] uppercase text-[var(--ink-soft)] group-hover:text-[var(--accent)] transition-colors">
                  <span>Ask Sanhita</span>
                  <ArrowUpRight
                    size={13}
                    className="shrink-0 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
                  />
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Harvey-style friendly progress labels. The agent-trace endpoint returns
// raw tool names; users want plain English for what's happening.
const TOOL_ICONS: Record<string, string> = {
  retrieve_cases: "⚖️",
  retrieve_statutes: "📖",
  web_search: "🌐",
  redline_contract: "✂️",
  translate: "🌐",
  vault_search: "📂",
  semantic_search: "🔎",
};

const TOOL_LABELS: Record<string, string> = {
  retrieve_cases: "Searching case law",
  retrieve_statutes: "Pulling statutes",
  web_search: "Researching the open web",
  redline_contract: "Redlining the contract",
  translate: "Translating",
  vault_search: "Searching uploaded documents",
  semantic_search: "Semantic vault retrieval",
};

function _toolLabel(tool?: string): string {
  if (!tool) return "Working";
  return TOOL_LABELS[tool] || tool.replace(/_/g, " ");
}

function TraceBreadcrumbs({ trace }: { trace: TraceStep[] }) {
  return (
    <details className="text-[11px] text-[var(--ink-soft)] bg-[var(--bg-elev)] border border-[var(--line)] rounded-lg px-3 py-2 group">
      <summary className="cursor-pointer flex items-center gap-2 select-none">
        <span className="text-[var(--accent)] font-mono">▸</span>
        <span className="tracking-wider uppercase text-[10px]">
          {trace.length} step{trace.length === 1 ? "" : "s"}
        </span>
        <span className="flex items-center gap-1.5 ml-1 truncate">
          {trace.map((s, i) => (
            <span key={i} className="inline-flex items-center gap-1">
              <span>{TOOL_ICONS[s.tool || ""] || "🔧"}</span>
              <span>{_toolLabel(s.tool)}</span>
              {i < trace.length - 1 && <span className="opacity-40">·</span>}
            </span>
          ))}
        </span>
      </summary>
      <div className="mt-2 space-y-1.5 pl-4 border-l border-[var(--line)]">
        {trace.map((s, i) => (
          <div key={i} className="leading-snug">
            <div className="flex items-center gap-2">
              <span className="text-[var(--accent)]">✓</span>
              <span>{TOOL_ICONS[s.tool || ""] || "🔧"}</span>
              <span className="text-[var(--ink)]">{_toolLabel(s.tool)}</span>
              {typeof s.ms === "number" && (
                <span className="text-[10px] opacity-60">{s.ms}ms</span>
              )}
              {s.error && (
                <span className="text-[10px] text-[var(--danger)]">⚠ {s.error}</span>
              )}
            </div>
            {s.args && Object.keys(s.args).length > 0 && (
              <div className="font-mono text-[10px] opacity-70 break-all pl-6">
                {JSON.stringify(s.args)}
              </div>
            )}
            {s.result_preview && (
              <div className="text-[11px] opacity-80 pl-6 break-words">
                → {s.result_preview}
              </div>
            )}
          </div>
        ))}
      </div>
    </details>
  );
}

function SourceCard({ c }: { c: Citation }) {
  const body = (
    <>
      <div className="flex items-baseline gap-2 mb-1 min-w-0">
        <span className="font-mono text-[var(--accent)] text-xs shrink-0">[{c.n}]</span>
        <span className="font-display text-sm leading-tight text-[var(--ink)] line-clamp-2 break-words min-w-0">
          {c.title}
        </span>
      </div>
      <div className="text-[11px] text-[var(--ink-soft)] mb-2 break-words">
        {[c.court, c.year, c.citation].filter(Boolean).join(" · ")}
      </div>
      {c.excerpt && (
        <div className="text-xs text-[var(--ink-soft)] italic line-clamp-4 break-words">
          “{c.excerpt}”
        </div>
      )}
    </>
  );
  return (
    <div
      data-n={c.n}
      className="source-card bg-[var(--bg)] border border-[var(--line)] hover:border-[var(--accent-soft)] rounded-lg p-3 transition-colors min-w-0 overflow-hidden"
    >
      {c.pdf_url ? (
        <a href={c.pdf_url} target="_blank" rel="noopener noreferrer" className="block min-w-0">
          {body}
        </a>
      ) : (
        body
      )}
    </div>
  );
}

function safeParse(s: string): Citation[] | undefined {
  try {
    const parsed = JSON.parse(s);
    return Array.isArray(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
}
