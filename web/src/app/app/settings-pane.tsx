"use client";

// Settings — per-connector API keys, plug-and-play.
// Backend:
//   GET    /api/settings/keys              — catalog + state (never plaintext)
//   POST   /api/settings/keys              — {name, key, note?}
//   DELETE /api/settings/keys/:name        — revoke
//
// Keys land in the SQLite `connector_keys` table; `connectors._key()` reads
// DB-first, env-fallback. So saving a key here takes effect on the next
// request without a restart.

import { useCallback, useEffect, useState } from "react";
import { Key, Check, Trash2, Loader2, Link2, Unlink } from "lucide-react";

interface SettingsKey {
  name: string;
  label: string;
  country: string;
  country_label: string;
  kind: "case" | "statute" | "web";
  free: boolean;
  has_key: boolean;
  masked_tail: string;
  set_at: number | null;
}

interface GoogleStatus {
  configured: boolean;
  connected: boolean;
  email?: string | null;
  scopes?: string[];
  connected_at?: number | null;
  tracker_sheet?: string | null;
}

const FLAG_BY_COUNTRY: Record<string, string> = {
  IN: "🇮🇳",
  JP: "🇯🇵",
  SG: "🇸🇬",
  HK: "🇭🇰",
  AE: "🇦🇪",
  KR: "🇰🇷",
  MY: "🇲🇾",
  ID: "🇮🇩",
  "*": "🌐",
};

export default function SettingsPane({
  onChange,
}: {
  onChange?: () => void | Promise<void>;
}) {
  const [keys, setKeys] = useState<SettingsKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  // Names whose "saved" pill is currently visible. We toggle this from a
  // setTimeout (impure clocks belong outside render).
  const [recentlySaved, setRecentlySaved] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  // Google Workspace connection state. Driven by GET /api/google/status, which
  // returns {configured: bool, connected: bool, email?, scopes?}. `configured`
  // is false when the deploy has no GOOGLE_CLIENT_ID/SECRET — we still render
  // the card but disable the button and explain why.
  const [google, setGoogle] = useState<GoogleStatus | null>(null);
  const [googleBusy, setGoogleBusy] = useState(false);
  const [googleBanner, setGoogleBanner] = useState<{
    kind: "ok" | "warn" | "err";
    text: string;
  } | null>(null);

  const refreshGoogle = useCallback(async () => {
    try {
      const r = await fetch("/api/google/status", {
        credentials: "same-origin",
      });
      if (!r.ok) return;
      const data = (await r.json()) as GoogleStatus;
      setGoogle(data);
    } catch {
      /* silent */
    }
  }, []);

  useEffect(() => {
    refreshGoogle();
  }, [refreshGoogle]);

  // Surface the OAuth callback result. Our backend redirects to
  // /app?google=connected | google=denied | google=error so the user lands
  // back here with a banner explaining what happened. We strip the param
  // from the URL after reading it so a refresh doesn't re-show the banner.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    const flag = url.searchParams.get("google");
    if (!flag) return;
    if (flag === "connected") {
      setGoogleBanner({
        kind: "ok",
        text: "Google Workspace connected. Sanhita can now save Docs, draft emails, and log matters to your Sheet.",
      });
    } else if (flag === "denied") {
      setGoogleBanner({
        kind: "warn",
        text: "Google connection cancelled. You can connect any time.",
      });
    } else {
      setGoogleBanner({
        kind: "err",
        text: "Google connection failed. Try again or check the deploy's GOOGLE_CLIENT_ID/SECRET.",
      });
    }
    url.searchParams.delete("google");
    window.history.replaceState({}, "", url.toString());
    // After a callback, status is now different — refetch.
    refreshGoogle();
  }, [refreshGoogle]);

  const connectGoogle = useCallback(() => {
    if (typeof window === "undefined") return;
    // Server-side redirect carries the OAuth state nonce + scope list. We
    // don't want a fetch here — we want a real top-level navigation so the
    // browser follows the 302 to accounts.google.com.
    window.location.href = "/api/google/oauth/start";
  }, []);

  const disconnectGoogle = useCallback(async () => {
    if (
      !confirm(
        "Disconnect Google Workspace? Sanhita will stop being able to save Docs, draft emails, or update your matter sheet."
      )
    )
      return;
    setGoogleBusy(true);
    try {
      await fetch("/api/google/disconnect", {
        method: "POST",
        credentials: "same-origin",
      });
      await refreshGoogle();
      setGoogleBanner({ kind: "ok", text: "Google Workspace disconnected." });
    } catch {
      setGoogleBanner({
        kind: "err",
        text: "Could not disconnect. Try again.",
      });
    } finally {
      setGoogleBusy(false);
    }
  }, [refreshGoogle]);

  // Refetch the catalog. Used by the initial-load effect, by saveKey, and
  // by revokeKey. Initial `loading: true` covers the first paint; afterwards
  // we just swap items in.
  const refresh = useCallback(async () => {
    try {
      const r = await fetch("/api/settings/keys", { credentials: "same-origin" });
      const data = await r.json();
      setKeys(data.keys || []);
    } catch {
      /* silent */
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load — inlined to satisfy React 19's purity rule about setState
  // synchronously inside an effect (the `await fetch` here yields before any
  // setState reaches React).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch("/api/settings/keys", { credentials: "same-origin" });
        const data = await r.json();
        if (cancelled) return;
        setKeys(data.keys || []);
      } catch {
        /* silent */
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const saveKey = useCallback(
    async (name: string) => {
      const key = (drafts[name] || "").trim();
      if (!key) return;
      setBusy(name);
      setError(null);
      try {
        const r = await fetch("/api/settings/keys", {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, key }),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
        setDrafts((d) => ({ ...d, [name]: "" }));
        setRecentlySaved((s) => {
          const next = new Set(s);
          next.add(name);
          return next;
        });
        // Auto-clear the "saved" pill after 4s, off the render path.
        setTimeout(() => {
          setRecentlySaved((s) => {
            const next = new Set(s);
            next.delete(name);
            return next;
          });
        }, 4000);
        await refresh();
        if (onChange) await onChange();
      } catch (e) {
        setError((e as Error).message || "Failed to save key.");
      } finally {
        setBusy(null);
      }
    },
    [drafts, refresh, onChange]
  );

  const revokeKey = useCallback(
    async (name: string) => {
      if (!confirm(`Revoke the API key for ${name}?`)) return;
      setBusy(name);
      setError(null);
      try {
        await fetch(`/api/settings/keys/${name}`, {
          method: "DELETE",
          credentials: "same-origin",
        });
        await refresh();
        if (onChange) await onChange();
      } catch (e) {
        setError((e as Error).message || "Failed to revoke key.");
      } finally {
        setBusy(null);
      }
    },
    [refresh, onChange]
  );

  return (
    <div className="flex-1 min-h-0 overflow-y-auto px-12 py-10">
      <div className="max-w-3xl mx-auto">
        <div className="mb-8">
          <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] mb-1">
            Settings
          </div>
          <h1 className="font-display text-3xl tracking-tight">
            Connector API keys
          </h1>
          <p className="mt-3 text-[var(--ink-soft)] max-w-xl">
            Plug in country-specific case-law and web-search providers. Keys are
            stored encrypted-at-rest in the local SQLite, never logged, and
            never returned in plaintext after save. Saving a key takes effect
            on the next request — no restart required.
          </p>
        </div>

        {/* Google Workspace card. Lives at the top of Settings because it's
            the highest-leverage integration: once connected, the agent can
            save drafts to Drive, queue Gmail drafts, and log matters to a
            tracker Sheet without leaving Sanhita. */}
        <div className="mb-8 rounded-xl border border-[var(--line)] bg-[var(--bg-elev)] p-5">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-base">🔗</span>
                <span className="font-display text-lg tracking-tight">
                  Google Workspace
                </span>
                <span className="text-[10px] tracking-[0.2em] uppercase text-[var(--ink-soft)] border border-[var(--line)] rounded-full px-2 py-0.5">
                  agent tools
                </span>
                {google?.connected && (
                  <span className="text-[10px] tracking-[0.2em] uppercase bg-[var(--highlight)] text-[var(--accent)] rounded-full px-2 py-0.5 flex items-center gap-1">
                    <Check size={10} /> connected
                  </span>
                )}
                {google && !google.configured && (
                  <span className="text-[10px] tracking-[0.2em] uppercase text-[var(--ink-soft)] border border-[var(--line)] rounded-full px-2 py-0.5">
                    not configured
                  </span>
                )}
                {google && google.configured && !google.connected && (
                  <span className="text-[10px] tracking-[0.2em] uppercase text-[var(--ink-soft)] border border-[var(--line)] rounded-full px-2 py-0.5">
                    not connected
                  </span>
                )}
              </div>
              <div className="mt-1 text-xs text-[var(--ink-soft)]">
                {google?.connected ? (
                  <>
                    Signed in as{" "}
                    <span className="font-mono">{google.email || "—"}</span>
                    {google.connected_at && (
                      <>
                        {" · since "}
                        {new Date(
                          google.connected_at * 1000
                        ).toLocaleDateString()}
                      </>
                    )}
                  </>
                ) : (
                  <>
                    Lets the agent save Docs, queue Gmail drafts (compose
                    only — never auto-sends), append rows to your matter
                    Sheet, and search your Drive. Scopes: docs, gmail.compose,
                    sheets, drive.file, userinfo.
                  </>
                )}
              </div>
            </div>

            <div className="shrink-0">
              {google?.connected ? (
                <button
                  onClick={disconnectGoogle}
                  disabled={googleBusy}
                  className="text-[var(--ink-soft)] hover:text-[var(--danger)] transition-colors disabled:opacity-40 flex items-center gap-1.5 text-sm border border-[var(--line)] rounded-lg px-3 py-2"
                  title="Disconnect Google Workspace"
                >
                  {googleBusy ? (
                    <Loader2 size={13} className="animate-spin" />
                  ) : (
                    <Unlink size={13} />
                  )}
                  Disconnect
                </button>
              ) : (
                <button
                  onClick={connectGoogle}
                  disabled={!google?.configured}
                  className="bg-[var(--ink)] text-[var(--bg)] px-4 py-2 rounded-lg text-sm font-medium hover:bg-[var(--accent)] disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                  title={
                    google?.configured
                      ? "Connect Google Workspace"
                      : "Deploy missing GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET"
                  }
                >
                  <Link2 size={13} />
                  Connect Google
                </button>
              )}
            </div>
          </div>

          {googleBanner && (
            <div
              className={`mt-3 rounded-lg border px-3 py-2 text-xs ${
                googleBanner.kind === "ok"
                  ? "border-[var(--accent-soft)] text-[var(--accent)]"
                  : googleBanner.kind === "warn"
                    ? "border-[var(--line)] text-[var(--ink-soft)]"
                    : "border-[var(--danger)] text-[var(--danger)]"
              }`}
            >
              {googleBanner.text}
            </div>
          )}

          {google && !google.configured && (
            <div className="mt-3 text-xs text-[var(--ink-soft)] italic">
              The deploy is missing{" "}
              <span className="font-mono">GOOGLE_CLIENT_ID</span> and/or{" "}
              <span className="font-mono">GOOGLE_CLIENT_SECRET</span>. Set
              them in <span className="font-mono">.claude/launch.json</span>{" "}
              and restart the FastAPI Backend to enable this integration.
            </div>
          )}
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-[var(--danger)] bg-[var(--bg-elev)] px-4 py-2 text-sm text-[var(--danger)]">
            {error}
          </div>
        )}

        {loading && (
          <div className="text-sm italic text-[var(--ink-soft)]">
            Loading connectors…
          </div>
        )}

        {!loading && keys.length === 0 && (
          <div className="text-sm italic text-[var(--ink-soft)]">
            No connectors configured for this build.
          </div>
        )}

        <div className="flex flex-col gap-3">
          {keys.map((k) => {
            const flag = FLAG_BY_COUNTRY[k.country] || "🌐";
            const draft = drafts[k.name] ?? "";
            const justSaved = recentlySaved.has(k.name);

            return (
              <div
                key={k.name}
                className="rounded-xl border border-[var(--line)] bg-[var(--bg-elev)] p-5"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-base">{flag}</span>
                      <span className="font-display text-lg tracking-tight">
                        {k.label}
                      </span>
                      <span className="text-[10px] tracking-[0.2em] uppercase text-[var(--ink-soft)] border border-[var(--line)] rounded-full px-2 py-0.5">
                        {k.kind}
                      </span>
                      {k.has_key && (
                        <span className="text-[10px] tracking-[0.2em] uppercase bg-[var(--highlight)] text-[var(--accent)] rounded-full px-2 py-0.5 flex items-center gap-1">
                          <Check size={10} /> live
                        </span>
                      )}
                      {!k.has_key && (
                        <span className="text-[10px] tracking-[0.2em] uppercase text-[var(--ink-soft)] border border-[var(--line)] rounded-full px-2 py-0.5">
                          missing key
                        </span>
                      )}
                      {justSaved && (
                        <span className="text-[10px] tracking-[0.2em] uppercase text-[var(--accent)]">
                          saved
                        </span>
                      )}
                    </div>
                    <div className="mt-1 text-xs text-[var(--ink-soft)]">
                      {k.country_label}
                      {k.has_key && (
                        <>
                          {" · key ending "}
                          <span className="font-mono">{k.masked_tail}</span>
                          {k.set_at && (
                            <>
                              {" · added "}
                              {new Date(k.set_at * 1000).toLocaleDateString()}
                            </>
                          )}
                        </>
                      )}
                    </div>
                  </div>

                  {k.has_key && (
                    <button
                      onClick={() => revokeKey(k.name)}
                      disabled={busy === k.name}
                      className="text-[var(--ink-soft)] hover:text-[var(--danger)] transition-colors disabled:opacity-40"
                      title="Revoke key"
                    >
                      <Trash2 size={15} />
                    </button>
                  )}
                </div>

                <div className="mt-4 flex items-center gap-2">
                  <div className="flex-1 flex items-center gap-2 bg-[var(--bg)] border border-[var(--line)] rounded-lg px-3 py-2 focus-within:border-[var(--accent-soft)] transition-colors">
                    <Key size={13} className="text-[var(--ink-soft)] shrink-0" />
                    <input
                      type="password"
                      autoComplete="off"
                      value={draft}
                      onChange={(e) =>
                        setDrafts((d) => ({ ...d, [k.name]: e.target.value }))
                      }
                      placeholder={
                        k.has_key
                          ? "Replace key… (paste new value)"
                          : "Paste API key…"
                      }
                      className="flex-1 bg-transparent outline-none text-sm font-mono"
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          saveKey(k.name);
                        }
                      }}
                    />
                  </div>
                  <button
                    onClick={() => saveKey(k.name)}
                    disabled={!draft.trim() || busy === k.name}
                    className="bg-[var(--ink)] text-[var(--bg)] px-4 py-2 rounded-lg text-sm font-medium hover:bg-[var(--accent)] disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                  >
                    {busy === k.name ? (
                      <Loader2 size={13} className="animate-spin" />
                    ) : null}
                    Save
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        <p className="mt-8 text-xs text-[var(--ink-soft)] italic max-w-xl">
          Don&apos;t have a key? Sanhita falls back automatically — Indian Kanoon
          and eCourts work without keys for low-volume use; Serper / Tavily
          paid tiers give the cleanest web results, otherwise we fall back to
          DuckDuckGo HTML scraping.
        </p>
      </div>
    </div>
  );
}
