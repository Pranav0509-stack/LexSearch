"use client";

// Dashboard — in-house admin pane for Sanhita. Four widgets:
//   • Stats          — corpus + DB counts (poll on mount + every 30s)
//   • Users          — access-code holders, with revoke action
//   • Activity feed  — live audit log (Socket.io stream)
//   • System health  — DB / BM25 / LLM / web-search availability
//
// "Ask Sanhita" CTA in the corner hands the assistant the current
// dashboard snapshot so the lawyer can ask "who logged in this week?"
// or "delete the seed corpus rows" and get an actionable answer.

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  LayoutDashboard,
  Users as UsersIcon,
  Activity as ActivityIcon,
  Cpu,
  Database,
  Globe,
  Sparkles,
  RefreshCw,
  ShieldOff,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import { on as subscribe, getSocket } from "@/lib/realtime";

// ─── Types ──────────────────────────────────────────────────────────

interface StatsPayload {
  users: number;
  threads: number;
  messages: number;
  library: number;
  bm25: {
    total: number;
    by_jurisdiction: Record<string, number>;
    by_source: Record<string, number>;
  };
}

interface UserRow {
  id: number;
  email: string | null;
  name: string | null;
  created_at: number;
  revoked_at: number | null;
  last_seen: number | null;
  thread_count: number;
}

interface ActivityRow {
  id?: number;
  actor: string;
  action: string;
  target: string | null;
  payload: string | null;
  created_at: number;
}

interface SystemPayload {
  db: { mode: string; url_host: string; ok: boolean; error?: string };
  bm25: {
    available: boolean;
    enabled: boolean;
    loaded: boolean;
    loading: boolean;
    doc_count: number;
    load_error: string | null;
  };
  llm: Record<string, boolean>;
  web: Record<string, boolean>;
}

// ─── Component ──────────────────────────────────────────────────────

export default function DashboardPane({
  onAskSanhita,
}: {
  onAskSanhita?: (seedPrompt: string) => void;
}) {
  const [stats, setStats] = useState<StatsPayload | null>(null);
  const [users, setUsers] = useState<UserRow[]>([]);
  const [activity, setActivity] = useState<ActivityRow[]>([]);
  const [system, setSystem] = useState<SystemPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [presence, setPresence] = useState(1);

  const refreshAll = useCallback(async () => {
    setRefreshing(true);
    setError(null);
    try {
      const [s, u, a, sys] = await Promise.all([
        fetch("/api/dashboard/stats", { credentials: "same-origin" }).then((r) => r.json()),
        fetch("/api/dashboard/users", { credentials: "same-origin" }).then((r) => r.json()),
        fetch("/api/dashboard/activity?limit=30", { credentials: "same-origin" }).then((r) => r.json()),
        fetch("/api/dashboard/system", { credentials: "same-origin" }).then((r) => r.json()),
      ]);
      setStats(s);
      setUsers(u.users || []);
      setActivity(a.activity || []);
      setSystem(sys);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refreshAll();
    const t = setInterval(() => void refreshAll(), 30000);
    return () => clearInterval(t);
  }, [refreshAll]);

  // Live: append new activity rows from the socket stream so a second
  // admin's edits appear here without a manual refresh.
  useEffect(() => {
    const sock = getSocket();
    const offAppend = subscribe<ActivityRow>("activity:append", (row) => {
      setActivity((prev) => [row, ...prev].slice(0, 60));
    });
    const offJoin = subscribe<{ sid: string }>("presence:join", () => {
      setPresence((n) => n + 1);
    });
    const offLeave = subscribe<{ sid: string }>("presence:leave", () => {
      setPresence((n) => Math.max(1, n - 1));
    });
    const onConnect = () => setPresence(1);
    sock.on("connect", onConnect);
    return () => {
      offAppend();
      offJoin();
      offLeave();
      sock.off("connect", onConnect);
    };
  }, []);

  const revokeUser = useCallback(
    async (uid: number) => {
      if (!confirm("Revoke this user's access? They'll be signed out immediately.")) return;
      const r = await fetch(`/api/dashboard/users/${uid}/revoke`, {
        method: "POST",
        credentials: "same-origin",
      });
      if (r.ok) await refreshAll();
    },
    [refreshAll]
  );

  const askSanhitaSnapshot = useCallback(() => {
    if (!onAskSanhita) return;
    const summary = stats
      ? `I'm the Sanhita admin looking at the dashboard. Right now the system shows:\n` +
        `- ${stats.users} active users · ${stats.threads} threads · ${stats.messages} messages\n` +
        `- ${stats.library} library docs\n` +
        `- BM25 corpus: ${stats.bm25.total} cases (${formatJurisBreakdown(stats.bm25.by_jurisdiction)})\n\n` +
        `What should I look at first?`
      : "I'm the Sanhita admin opening the dashboard. What should I look at first?";
    onAskSanhita(summary);
  }, [onAskSanhita, stats]);

  return (
    <div className="flex flex-col flex-1 min-h-0 min-w-0 overflow-y-auto px-4 sm:px-6 lg:px-12 py-5">
      {/* Header */}
      <div className="flex items-center gap-2 text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)]">
        <LayoutDashboard size={11} className="text-[var(--accent)]" />
        <span>Dashboard</span>
        <span className="ml-2 inline-flex items-center gap-1 rounded-full bg-[var(--bg-elev)] border border-[var(--line)] px-2 py-0.5 text-[10px] normal-case tracking-normal">
          <span className="size-1.5 rounded-full bg-emerald-500 animate-pulse" />
          {presence} live
        </span>
        <button
          onClick={() => void refreshAll()}
          disabled={refreshing}
          className="ml-auto inline-flex items-center gap-1 text-[10px] normal-case tracking-normal text-[var(--ink-soft)] hover:text-[var(--ink)] transition-colors"
        >
          <RefreshCw size={11} className={refreshing ? "animate-spin" : ""} />
          {refreshing ? "Refreshing…" : "Refresh"}
        </button>
        {onAskSanhita && (
          <button
            onClick={askSanhitaSnapshot}
            className="inline-flex items-center gap-1 rounded-full bg-[var(--ink)] text-[var(--bg)] px-3 py-1.5 text-[10px] normal-case tracking-normal hover:opacity-90"
          >
            <Sparkles size={11} />
            Ask Sanhita
          </button>
        )}
      </div>
      <h2 className="mt-1.5 font-display text-2xl tracking-tight">
        Inhouse control panel
      </h2>
      <p className="mt-1 text-sm text-[var(--ink-soft)] max-w-2xl">
        Stats, audit log, and system health for everything Sanhita serves —
        with live updates so two admins watching the same screen stay in sync.
      </p>

      {error && (
        <div className="mt-4 text-sm text-[var(--danger)] bg-[var(--bg-elev)] border border-[var(--line)] rounded-xl p-3">
          {error}
        </div>
      )}

      {/* Widget grid */}
      <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-4 max-w-6xl">
        <StatsWidget stats={stats} />
        <SystemWidget system={system} />
        <UsersWidget users={users} onRevoke={(id) => void revokeUser(id)} />
        <ActivityWidget rows={activity} />
      </div>
    </div>
  );
}

// ─── Widgets ────────────────────────────────────────────────────────

function StatsWidget({ stats }: { stats: StatsPayload | null }) {
  return (
    <Card title="At a glance" icon={<Database size={11} className="text-[var(--accent)]" />}>
      {!stats ? (
        <Skeleton lines={4} />
      ) : (
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Active users" value={stats.users} />
          <Stat label="Threads" value={stats.threads} />
          <Stat label="Messages" value={stats.messages} />
          <Stat label="Library docs" value={stats.library} />
          <Stat
            label="BM25 corpus"
            value={stats.bm25.total}
            sub={formatJurisBreakdown(stats.bm25.by_jurisdiction)}
            wide
          />
        </div>
      )}
    </Card>
  );
}

function SystemWidget({ system }: { system: SystemPayload | null }) {
  return (
    <Card title="System health" icon={<Cpu size={11} className="text-[var(--accent)]" />}>
      {!system ? (
        <Skeleton lines={4} />
      ) : (
        <ul className="text-sm flex flex-col gap-2">
          <HealthRow
            ok={system.db.ok}
            label={`DB (${system.db.mode})`}
            detail={system.db.url_host}
          />
          <HealthRow
            ok={system.bm25.loaded && !system.bm25.load_error}
            label="BM25 index"
            detail={`${system.bm25.doc_count.toLocaleString()} docs · ${
              system.bm25.loading ? "loading" : "ready"
            }`}
          />
          <HealthRow
            ok={Object.values(system.llm).some(Boolean)}
            label="LLM router"
            detail={Object.entries(system.llm)
              .filter(([, on]) => on)
              .map(([k]) => k)
              .join(" · ") || "no key set"}
            icon={<Cpu size={11} />}
          />
          <HealthRow
            ok={Object.values(system.web).some(Boolean)}
            label="Web search"
            detail={Object.entries(system.web)
              .filter(([, on]) => on)
              .map(([k]) => k)
              .join(" · ") || "ddg only"}
            icon={<Globe size={11} />}
          />
        </ul>
      )}
    </Card>
  );
}

function UsersWidget({
  users,
  onRevoke,
}: {
  users: UserRow[];
  onRevoke: (id: number) => void;
}) {
  return (
    <Card title={`Users (${users.length})`} icon={<UsersIcon size={11} className="text-[var(--accent)]" />}>
      {users.length === 0 ? (
        <Skeleton lines={3} />
      ) : (
        <ul className="text-sm flex flex-col gap-2 max-h-72 overflow-y-auto pr-1">
          {users.slice(0, 25).map((u) => {
            const revoked = !!u.revoked_at;
            return (
              <li
                key={u.id}
                className={`flex items-center gap-3 py-1.5 ${revoked ? "opacity-60" : ""}`}
              >
                <div className="min-w-0 flex-1">
                  <div className="font-medium truncate">
                    {u.name || u.email || `user #${u.id}`}
                  </div>
                  <div className="text-[11px] text-[var(--ink-soft)] truncate">
                    {u.email || "—"} · {u.thread_count} thread
                    {u.thread_count === 1 ? "" : "s"}
                    {u.last_seen ? ` · last seen ${formatRel(u.last_seen)}` : ""}
                  </div>
                </div>
                {revoked ? (
                  <span className="text-[10px] uppercase tracking-wider text-[var(--ink-soft)]">
                    revoked
                  </span>
                ) : (
                  <button
                    onClick={() => onRevoke(u.id)}
                    title="Revoke access"
                    className="text-[var(--ink-soft)] hover:text-[var(--danger)] transition-colors"
                  >
                    <ShieldOff size={14} />
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </Card>
  );
}

function ActivityWidget({ rows }: { rows: ActivityRow[] }) {
  return (
    <Card
      title="Activity"
      icon={<ActivityIcon size={11} className="text-[var(--accent)]" />}
      sub="Live · last 30"
    >
      {rows.length === 0 ? (
        <p className="text-sm text-[var(--ink-soft)]">
          No admin actions yet. Revoke a user or update a key to see entries here in real time.
        </p>
      ) : (
        <ul className="text-sm flex flex-col gap-2 max-h-72 overflow-y-auto pr-1">
          {rows.map((r, i) => (
            <li key={r.id ?? i} className="flex items-start gap-3 py-1">
              <div className="size-2 rounded-full bg-[var(--accent)] mt-1.5 shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="text-[13px]">
                  <span className="font-medium">{r.actor || "system"}</span>{" "}
                  <span className="text-[var(--ink-soft)]">{humanAction(r.action)}</span>{" "}
                  {r.target && (
                    <span className="text-[var(--ink)]">{r.target}</span>
                  )}
                </div>
                <div className="text-[11px] text-[var(--ink-soft)]">
                  {formatRel(r.created_at)}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

// ─── Reusable bits ──────────────────────────────────────────────────

function Card({
  title,
  icon,
  sub,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  sub?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="bg-[var(--bg-elev)] border border-[var(--line)] rounded-2xl p-4 min-w-0">
      <header className="flex items-center gap-2 text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] mb-3">
        {icon}
        <span>{title}</span>
        {sub && (
          <span className="ml-auto normal-case tracking-normal text-[10px] text-[var(--ink-soft)]">
            {sub}
          </span>
        )}
      </header>
      {children}
    </section>
  );
}

function Stat({
  label,
  value,
  sub,
  wide,
}: {
  label: string;
  value: number | string;
  sub?: string;
  wide?: boolean;
}) {
  return (
    <div
      className={`bg-[var(--bg)] border border-[var(--line)] rounded-xl p-3 ${
        wide ? "col-span-2" : ""
      }`}
    >
      <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)]">
        {label}
      </div>
      <div className="mt-1 font-display text-2xl tracking-tight">
        {typeof value === "number" ? value.toLocaleString() : value}
      </div>
      {sub && <div className="text-[11px] text-[var(--ink-soft)] mt-0.5">{sub}</div>}
    </div>
  );
}

function HealthRow({
  ok,
  label,
  detail,
  icon,
}: {
  ok: boolean;
  label: string;
  detail?: string;
  icon?: React.ReactNode;
}) {
  return (
    <li className="flex items-center gap-3 py-1">
      {ok ? (
        <CheckCircle2 size={14} className="text-emerald-600 shrink-0" />
      ) : (
        <XCircle size={14} className="text-[var(--danger)] shrink-0" />
      )}
      <div className="min-w-0 flex-1">
        <div className="font-medium flex items-center gap-1.5">
          {icon}
          <span>{label}</span>
        </div>
        {detail && (
          <div className="text-[11px] text-[var(--ink-soft)] truncate">{detail}</div>
        )}
      </div>
    </li>
  );
}

function Skeleton({ lines = 3 }: { lines?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="h-8 rounded bg-[var(--bg)] border border-[var(--line)] animate-pulse"
        />
      ))}
    </div>
  );
}

// ─── helpers ────────────────────────────────────────────────────────

function formatJurisBreakdown(by: Record<string, number>): string {
  const ordered = ["IN", "SG", "HK"];
  const bits: string[] = [];
  for (const k of ordered) {
    if (by[k]) bits.push(`${by[k].toLocaleString()} ${k}`);
  }
  for (const [k, v] of Object.entries(by)) {
    if (!ordered.includes(k) && v) bits.push(`${v.toLocaleString()} ${k}`);
  }
  return bits.join(" · ") || "empty";
}

function formatRel(unix: number): string {
  if (!unix) return "—";
  const now = Date.now() / 1000;
  const d = Math.max(0, now - unix);
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}

function humanAction(a: string): string {
  switch (a) {
    case "revoke_access":
      return "revoked access for";
    case "set_key":
      return "updated key";
    case "delete_key":
      return "deleted key";
    case "library_add":
      return "added library doc";
    default:
      return a;
  }
}
