"use client";

// Clients — inbound NyayaSathi leads.
//
// Three tabs (New / In progress / Closed) over GET /api/clients?status=…
// Click a row to open a detail drawer; "Open as thread" hits
// POST /api/clients/:id/open which spins up a chat thread pre-loaded with
// the intake summary, then hands the thread_id back so the parent can
// switch to the Assistant pane.

import { useCallback, useEffect, useState } from "react";
import {
  Search,
  MessageSquarePlus,
  Phone,
  Mail,
  Globe,
  Loader2,
  CheckCircle2,
} from "lucide-react";

interface ClientRow {
  id: number;
  source: "whatsapp" | "voice" | "web" | "manual";
  name: string | null;
  phone: string | null;
  email: string | null;
  language: string | null;
  jurisdiction: string | null;
  intake_summary: string | null;
  status: "new" | "in_progress" | "closed";
  assigned_user_id: number | null;
  thread_id: number | null;
  arrived_at: number;
  updated_at: number;
}

interface ClientDetail extends ClientRow {
  intake_transcript: string | null;
  notes: string | null;
}

interface CountsBlock {
  new: number;
  in_progress: number;
  closed: number;
}

const FLAG: Record<string, string> = {
  IN: "🇮🇳", JP: "🇯🇵", SG: "🇸🇬", HK: "🇭🇰", AE: "🇦🇪", KR: "🇰🇷",
  MY: "🇲🇾", ID: "🇮🇩", TH: "🇹🇭", VN: "🇻🇳", PH: "🇵🇭", BD: "🇧🇩",
  PK: "🇵🇰", LK: "🇱🇰", NP: "🇳🇵", TW: "🇹🇼",
};

const SOURCE_BADGE: Record<ClientRow["source"], string> = {
  whatsapp: "WhatsApp",
  voice: "Voice",
  web: "Web",
  manual: "Manual",
};

function timeAgo(unix: number): string {
  const s = Math.floor(Date.now() / 1000) - unix;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export default function ClientsPane({
  onOpenThread,
}: {
  onOpenThread: (threadId: number) => void;
}) {
  const [tab, setTab] = useState<"new" | "in_progress" | "closed">("new");
  const [rows, setRows] = useState<ClientRow[]>([]);
  const [counts, setCounts] = useState<CountsBlock>({
    new: 0,
    in_progress: 0,
    closed: 0,
  });
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(true);
  const [openId, setOpenId] = useState<number | null>(null);
  const [detail, setDetail] = useState<ClientDetail | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const u = new URL("/api/clients", window.location.origin);
      u.searchParams.set("status", tab);
      if (q.trim()) u.searchParams.set("q", q.trim());
      const r = await fetch(u.toString(), { credentials: "same-origin" });
      const data = await r.json();
      setRows(data.clients || []);
      setCounts(data.counts || { new: 0, in_progress: 0, closed: 0 });
    } catch {
      /* silent */
    } finally {
      setLoading(false);
    }
  }, [tab, q]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    (async () => {
      try {
        const u = new URL("/api/clients", window.location.origin);
        u.searchParams.set("status", tab);
        if (q.trim()) u.searchParams.set("q", q.trim());
        const r = await fetch(u.toString(), { credentials: "same-origin" });
        const data = await r.json();
        if (cancelled) return;
        setRows(data.clients || []);
        setCounts(data.counts || { new: 0, in_progress: 0, closed: 0 });
      } catch {
        /* silent */
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [tab, q]);

  const openDetail = useCallback(async (id: number) => {
    setOpenId(id);
    setDetail(null);
    try {
      const r = await fetch(`/api/clients/${id}`, {
        credentials: "same-origin",
      });
      if (r.ok) {
        const data = (await r.json()) as ClientDetail;
        setDetail(data);
      }
    } catch {
      /* silent */
    }
  }, []);

  const openAsThread = useCallback(
    async (id: number) => {
      setBusy(true);
      try {
        const r = await fetch(`/api/clients/${id}/open`, {
          method: "POST",
          credentials: "same-origin",
        });
        const data = await r.json();
        if (data?.thread_id) {
          // Refresh the list (status flips to in_progress) and bounce the
          // user into the Assistant pane on the new thread.
          await refresh();
          onOpenThread(data.thread_id);
        }
      } catch {
        /* silent */
      } finally {
        setBusy(false);
        setOpenId(null);
      }
    },
    [onOpenThread, refresh]
  );

  const updateStatus = useCallback(
    async (id: number, status: "new" | "in_progress" | "closed") => {
      setBusy(true);
      try {
        await fetch(`/api/clients/${id}`, {
          method: "PATCH",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ status, assign_to_me: status !== "closed" }),
        });
        await refresh();
      } catch {
        /* silent */
      } finally {
        setBusy(false);
      }
    },
    [refresh]
  );

  return (
    <div className="flex-1 min-h-0 overflow-y-auto px-12 py-10">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] mb-1">
            Clients
          </div>
          <h1 className="font-display text-3xl tracking-tight">
            NyayaSathi inbox
          </h1>
          <p className="mt-3 text-[var(--ink-soft)] max-w-xl">
            Inbound consumer leads from the NyayaSathi WhatsApp / voice /
            web surface. Open one as a thread and Sanhita drops you into
            context with the intake summary already at the top.
          </p>
        </div>

        {/* Tabs with live counts */}
        <div className="flex items-center gap-1 mb-4 border-b border-[var(--line)]">
          {(["new", "in_progress", "closed"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
                tab === t
                  ? "border-[var(--accent)] text-[var(--ink)]"
                  : "border-transparent text-[var(--ink-soft)] hover:text-[var(--ink)]"
              }`}
            >
              {t === "new"
                ? "New"
                : t === "in_progress"
                  ? "In progress"
                  : "Closed"}
              <span className="ml-2 text-xs text-[var(--ink-soft)] font-mono">
                {counts[t]}
              </span>
            </button>
          ))}
          <div className="flex-1" />
          <div className="flex items-center gap-2 border border-[var(--line)] rounded-lg px-3 py-1.5 text-sm bg-[var(--bg-elev)]">
            <Search size={13} className="text-[var(--ink-soft)]" />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search name, phone, summary…"
              className="bg-transparent outline-none w-56 text-sm"
            />
          </div>
        </div>

        {loading && (
          <div className="text-sm italic text-[var(--ink-soft)] py-8">
            Loading…
          </div>
        )}
        {!loading && rows.length === 0 && (
          <div className="text-sm italic text-[var(--ink-soft)] py-8">
            {tab === "new"
              ? "Inbox zero — no new leads waiting."
              : tab === "in_progress"
                ? "Nothing in progress."
                : "No closed matters yet."}
          </div>
        )}

        <div className="flex flex-col gap-2">
          {rows.map((c) => {
            const flag = FLAG[c.jurisdiction || ""] || "🌐";
            return (
              <button
                key={c.id}
                onClick={() => openDetail(c.id)}
                className="text-left rounded-xl border border-[var(--line)] bg-[var(--bg-elev)] p-4 hover:border-[var(--accent-soft)] transition-colors"
              >
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="text-base">{flag}</span>
                  <span className="font-display text-base tracking-tight">
                    {c.name || "Unknown caller"}
                  </span>
                  <span className="text-[10px] tracking-[0.2em] uppercase text-[var(--ink-soft)] border border-[var(--line)] rounded-full px-2 py-0.5">
                    {SOURCE_BADGE[c.source]}
                  </span>
                  {c.language && (
                    <span className="text-[10px] tracking-[0.2em] uppercase text-[var(--ink-soft)] border border-[var(--line)] rounded-full px-2 py-0.5 font-mono">
                      {c.language}
                    </span>
                  )}
                  <div className="flex-1" />
                  <span className="text-xs text-[var(--ink-soft)]">
                    {timeAgo(c.arrived_at)}
                  </span>
                </div>
                {c.intake_summary && (
                  <div className="mt-2 text-sm text-[var(--ink)] line-clamp-2">
                    {c.intake_summary}
                  </div>
                )}
                <div className="mt-2 flex items-center gap-3 text-xs text-[var(--ink-soft)]">
                  {c.phone && (
                    <span className="flex items-center gap-1">
                      <Phone size={11} />
                      {c.phone}
                    </span>
                  )}
                  {c.email && (
                    <span className="flex items-center gap-1">
                      <Mail size={11} />
                      {c.email}
                    </span>
                  )}
                  {c.jurisdiction && (
                    <span className="flex items-center gap-1">
                      <Globe size={11} />
                      {c.jurisdiction}
                    </span>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Detail drawer */}
      {openId && (
        <div
          className="fixed inset-0 bg-black/30 z-30 flex justify-end"
          onClick={() => setOpenId(null)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-xl h-full bg-[var(--bg)] border-l border-[var(--line)] overflow-y-auto"
          >
            {!detail ? (
              <div className="p-10 text-sm italic text-[var(--ink-soft)]">
                Loading…
              </div>
            ) : (
              <div className="p-8">
                <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] mb-1">
                  Client #{detail.id} · {SOURCE_BADGE[detail.source]}
                </div>
                <h2 className="font-display text-2xl tracking-tight">
                  {detail.name || "Unknown caller"}
                </h2>

                <div className="mt-3 flex flex-wrap gap-2 text-sm text-[var(--ink-soft)]">
                  {detail.phone && (
                    <span className="flex items-center gap-1">
                      <Phone size={13} />
                      {detail.phone}
                    </span>
                  )}
                  {detail.email && (
                    <span className="flex items-center gap-1">
                      <Mail size={13} />
                      {detail.email}
                    </span>
                  )}
                  {detail.jurisdiction && (
                    <span className="flex items-center gap-1">
                      <Globe size={13} />
                      {detail.jurisdiction}
                    </span>
                  )}
                  {detail.language && (
                    <span className="font-mono">lang={detail.language}</span>
                  )}
                </div>

                {detail.intake_summary && (
                  <>
                    <h3 className="mt-6 text-xs tracking-[0.18em] uppercase text-[var(--ink-soft)]">
                      Summary
                    </h3>
                    <div className="mt-2 text-sm whitespace-pre-wrap">
                      {detail.intake_summary}
                    </div>
                  </>
                )}
                {detail.intake_transcript && (
                  <>
                    <h3 className="mt-6 text-xs tracking-[0.18em] uppercase text-[var(--ink-soft)]">
                      Transcript
                    </h3>
                    <div className="mt-2 text-xs whitespace-pre-wrap text-[var(--ink-soft)] max-h-72 overflow-y-auto border border-[var(--line)] rounded-lg p-3 bg-[var(--bg-elev)]">
                      {detail.intake_transcript}
                    </div>
                  </>
                )}

                <div className="mt-8 flex items-center gap-2">
                  <button
                    onClick={() => openAsThread(detail.id)}
                    disabled={busy}
                    className="bg-[var(--ink)] text-[var(--bg)] px-4 py-2 rounded-lg text-sm font-medium hover:bg-[var(--accent)] disabled:opacity-40 transition-colors flex items-center gap-2"
                  >
                    {busy ? (
                      <Loader2 size={13} className="animate-spin" />
                    ) : (
                      <MessageSquarePlus size={13} />
                    )}
                    {detail.thread_id ? "Open thread" : "Open as thread"}
                  </button>
                  {detail.status !== "closed" && (
                    <button
                      onClick={() => updateStatus(detail.id, "closed")}
                      disabled={busy}
                      className="px-4 py-2 rounded-lg text-sm border border-[var(--line)] hover:border-[var(--ink-soft)] transition-colors flex items-center gap-2 text-[var(--ink-soft)]"
                    >
                      <CheckCircle2 size={13} />
                      Mark closed
                    </button>
                  )}
                  {detail.status === "new" && (
                    <button
                      onClick={() => updateStatus(detail.id, "in_progress")}
                      disabled={busy}
                      className="px-4 py-2 rounded-lg text-sm border border-[var(--line)] hover:border-[var(--ink-soft)] transition-colors text-[var(--ink-soft)]"
                    >
                      Take it
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
