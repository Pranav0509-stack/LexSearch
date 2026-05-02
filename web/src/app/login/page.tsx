"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

// Single-input access-code login. The backend (`/api/login`) sets the
// httpOnly `ls_session` cookie on success — we just navigate to /app.
// Preview demo code: SNHT-DEMO-2026.
export default function LoginPage() {
  const router = useRouter();
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!code.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ code: code.trim() }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      router.replace("/app");
    } catch (e) {
      setErr((e as Error).message || "Could not sign in.");
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen flex items-center justify-center px-6 paper-grain">
      <div className="w-full max-w-md">
        <div className="text-center mb-12">
          <div className="font-display text-5xl tracking-tight text-[var(--ink)]">
            Sanhita
          </div>
          <div className="mt-2 text-sm tracking-[0.18em] uppercase text-[var(--ink-soft)]">
            Research counsel · Asia
          </div>
        </div>

        <form
          onSubmit={submit}
          className="bg-[var(--bg-elev)] border border-[var(--line)] rounded-2xl p-8 shadow-[0_1px_0_rgba(0,0,0,0.02),0_24px_60px_-30px_rgba(107,79,29,0.18)]"
        >
          <label className="block">
            <div className="text-xs tracking-[0.14em] uppercase text-[var(--ink-soft)] mb-2">
              Access code
            </div>
            <input
              type="text"
              value={code}
              onChange={(e) => setCode(e.target.value.toUpperCase())}
              placeholder="SNHT-DEMO-2026"
              className="w-full bg-transparent border-b border-[var(--line-strong)] focus:border-[var(--accent)] py-3 text-lg tracking-wider outline-none transition-colors font-mono"
              autoFocus
              autoComplete="off"
              spellCheck={false}
            />
          </label>

          {err && (
            <div className="mt-4 text-sm text-[var(--danger)]">{err}</div>
          )}

          <button
            type="submit"
            disabled={busy || !code.trim()}
            className="mt-8 w-full py-3 rounded-xl bg-[var(--ink)] text-[var(--bg)] font-medium tracking-wide hover:bg-[var(--accent)] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {busy ? "Signing in…" : "Enter chambers"}
          </button>

          <p className="mt-6 text-xs text-center text-[var(--ink-soft)]">
            Preview access:{" "}
            <code className="bg-[var(--highlight)] px-1.5 py-0.5 rounded">
              SNHT-DEMO-2026
            </code>
          </p>
        </form>

        <p className="mt-8 text-center text-xs text-[var(--ink-soft)] italic">
          &ldquo;Where the law speaks, we listen.&rdquo;
        </p>
      </div>
    </main>
  );
}
