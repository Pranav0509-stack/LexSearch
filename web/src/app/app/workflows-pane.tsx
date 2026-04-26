"use client";

// Workflows — Draft / Review / Translate / Citator / Redline + 9 generic.
// Backed by:
//   GET  /api/draft/templates
//   POST /api/draft        {template, facts}
//   POST /api/review       {clauses}
//   POST /api/translate    {text, direction}
//   POST /api/citator      {case_title, excerpt, holdings}
//   POST /api/redline      {text}
//   GET  /api/workflows
//   POST /api/workflows/run {key, text}

import { useEffect, useState } from "react";
import {
  ArrowLeft,
  ArrowRightLeft,
  ClipboardCheck,
  FileSignature,
  Languages,
  PenLine,
  ScrollText,
  Wand2,
} from "lucide-react";
import { renderMarkdown } from "./markdown";

type WorkflowKey =
  | "draft"
  | "review"
  | "translate"
  | "citator"
  | "redline"
  | { generic: string; title: string; sub: string };

interface DraftTemplate {
  key: string;
  title: string;
}

interface GenericWorkflow {
  key: string;
  title: string;
  sub: string;
}

interface RedlineSuggestion {
  type: "remove" | "replace" | "add";
  before?: string;
  after?: string;
  text?: string;
  reason?: string;
  position?: string;
}

interface RunResult {
  markdown?: string;
  draft_markdown?: string;
  review_markdown?: string;
  translation?: string;
  citator_markdown?: string;
  redline_markdown?: string;
  suggestions?: RedlineSuggestion[];
  llm?: { provider?: string; model?: string; latency_ms?: number };
  refused?: boolean;
  [k: string]: unknown;
}

const BUILTINS = [
  {
    key: "draft" as const,
    title: "Draft",
    sub: "Court-ready Indian-law drafts (4 templates).",
    icon: <FileSignature size={20} />,
  },
  {
    key: "review" as const,
    title: "Review",
    sub: "Clause-by-clause contract review with risk flags.",
    icon: <ClipboardCheck size={20} />,
  },
  {
    key: "translate" as const,
    title: "Translate",
    sub: "EN ↔ HI with legal-force preservation.",
    icon: <Languages size={20} />,
  },
  {
    key: "citator" as const,
    title: "Citator",
    sub: "Judicial history + related cases for a judgment.",
    icon: <ScrollText size={20} />,
  },
  {
    key: "redline" as const,
    title: "Redline",
    sub: "Suggest remove/replace/add edits across a contract.",
    icon: <PenLine size={20} />,
  },
];

export default function WorkflowsPane() {
  const [generics, setGenerics] = useState<GenericWorkflow[]>([]);
  const [active, setActive] = useState<WorkflowKey | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch("/api/workflows");
        const data = await r.json();
        setGenerics(data.workflows || []);
      } catch {
        /* silent */
      }
    })();
  }, []);

  if (active) {
    return (
      <div className="flex-1 overflow-y-auto px-12 py-10">
        <div className="max-w-3xl mx-auto">
          <button
            onClick={() => setActive(null)}
            className="flex items-center gap-2 text-sm text-[var(--ink-soft)] hover:text-[var(--ink)] mb-6 transition-colors"
          >
            <ArrowLeft size={14} /> All workflows
          </button>
          <WorkflowDetail wf={active} />
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-12 py-10">
      <div className="max-w-5xl mx-auto">
        <div className="font-display italic text-3xl tracking-tight text-[var(--ink)] mb-2">
          Workflows
        </div>
        <p className="text-[var(--ink-soft)] max-w-2xl mb-10">
          Production-grade pipelines: drafting court applications, clause-level
          review, redlining, citator and translation. Five built-ins, plus the
          generic chain — pick one to start.
        </p>

        <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] mb-3">
          Built-ins
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mb-10">
          {BUILTINS.map((b) => (
            <button
              key={b.key}
              onClick={() => setActive(b.key)}
              className="text-left bg-[var(--bg-elev)] border border-[var(--line)] hover:border-[var(--accent-soft)] hover:bg-[var(--highlight)] rounded-xl p-5 transition-colors group"
            >
              <div className="text-[var(--accent)] mb-3">{b.icon}</div>
              <div className="font-display text-lg tracking-tight">{b.title}</div>
              <div className="text-xs text-[var(--ink-soft)] mt-1 leading-snug">
                {b.sub}
              </div>
            </button>
          ))}
        </div>

        <div className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)] mb-3">
          Generic ({generics.length})
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {generics.map((g) => (
            <button
              key={g.key}
              onClick={() =>
                setActive({ generic: g.key, title: g.title, sub: g.sub })
              }
              className="text-left bg-[var(--bg-elev)] border border-[var(--line)] hover:border-[var(--accent-soft)] hover:bg-[var(--highlight)] rounded-xl p-5 transition-colors"
            >
              <div className="text-[var(--accent)] mb-3">
                <Wand2 size={18} />
              </div>
              <div className="font-display text-base tracking-tight">{g.title}</div>
              <div className="text-xs text-[var(--ink-soft)] mt-1 leading-snug">
                {g.sub}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function WorkflowDetail({ wf }: { wf: WorkflowKey }) {
  if (typeof wf === "object" && "generic" in wf) {
    return <GenericForm wf={wf} />;
  }
  switch (wf) {
    case "draft":
      return <DraftForm />;
    case "review":
      return <ReviewForm />;
    case "translate":
      return <TranslateForm />;
    case "citator":
      return <CitatorForm />;
    case "redline":
      return <RedlineForm />;
  }
}

// ── Draft ───────────────────────────────────────────────────────────────────
function DraftForm() {
  const [templates, setTemplates] = useState<DraftTemplate[]>([]);
  const [template, setTemplate] = useState<string>("");
  const [factsRaw, setFactsRaw] = useState(
    '{\n  "applicant": "",\n  "respondent": "",\n  "facts": ""\n}'
  );
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<RunResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch("/api/draft/templates", { credentials: "same-origin" });
        const data = await r.json();
        setTemplates(data.templates || []);
        if ((data.templates || []).length > 0) setTemplate(data.templates[0].key);
      } catch {
        /* silent */
      }
    })();
  }, []);

  async function run() {
    setBusy(true);
    setErr(null);
    setResult(null);
    let facts: Record<string, unknown> = {};
    try {
      facts = factsRaw.trim() ? JSON.parse(factsRaw) : {};
    } catch (e) {
      setBusy(false);
      setErr("Facts must be valid JSON. " + (e as Error).message);
      return;
    }
    try {
      const r = await fetch("/api/draft", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ template, facts }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      setResult(data);
    } catch (e) {
      setErr((e as Error).message || "Draft failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <FormShell title="Draft" sub="Court-ready Indian-law drafts (BNSS bail, §138 NI notice, Article 226 writ, Commercial Plaint).">
      <Field label="Template">
        <select
          value={template}
          onChange={(e) => setTemplate(e.target.value)}
          className="w-full bg-[var(--bg-elev)] border border-[var(--line)] rounded-lg px-3 py-2 text-sm outline-none focus:border-[var(--accent-soft)]"
        >
          {templates.map((t) => (
            <option key={t.key} value={t.key}>
              {t.title}
            </option>
          ))}
        </select>
      </Field>
      <Field
        label="Facts (JSON)"
        hint="Free-form key/value pairs. Sanhita inserts <BRACKETED PLACEHOLDER> for anything you leave blank."
      >
        <textarea
          value={factsRaw}
          onChange={(e) => setFactsRaw(e.target.value)}
          rows={10}
          className="w-full bg-[var(--bg-elev)] border border-[var(--line)] rounded-lg px-3 py-2 font-mono text-sm outline-none focus:border-[var(--accent-soft)] resize-vertical"
          spellCheck={false}
        />
      </Field>
      <RunButton busy={busy} disabled={!template} onClick={run}>
        Generate draft
      </RunButton>
      {err && <ErrorBlock msg={err} />}
      {result && (
        <ResultBlock
          markdown={result.draft_markdown || ""}
          meta={result.llm}
          refused={result.refused}
        />
      )}
    </FormShell>
  );
}

// ── Review ──────────────────────────────────────────────────────────────────
function ReviewForm() {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<RunResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setErr(null);
    setResult(null);
    const clauses = text
      .split(/\n\s*\n/)
      .map((s) => s.trim())
      .filter(Boolean)
      .slice(0, 40);
    if (clauses.length === 0) {
      setBusy(false);
      setErr("Paste at least one clause (separated by a blank line).");
      return;
    }
    try {
      const r = await fetch("/api/review", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clauses }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      setResult(data);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <FormShell
      title="Review"
      sub="Paste clauses (separated by a blank line). Sanhita returns risk levels, issues and suggested rewrites."
    >
      <Field label="Clauses (one per blank-line-separated block)">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={14}
          className="w-full bg-[var(--bg-elev)] border border-[var(--line)] rounded-lg px-3 py-2 text-sm outline-none focus:border-[var(--accent-soft)] resize-vertical"
        />
      </Field>
      <RunButton busy={busy} onClick={run}>
        Run review
      </RunButton>
      {err && <ErrorBlock msg={err} />}
      {result && (
        <ResultBlock
          markdown={result.review_markdown || result.markdown || ""}
          meta={result.llm}
        />
      )}
    </FormShell>
  );
}

// ── Translate ───────────────────────────────────────────────────────────────
function TranslateForm() {
  const [text, setText] = useState("");
  const [direction, setDirection] = useState<"en->hi" | "hi->en">("en->hi");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<RunResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      const r = await fetch("/api/translate", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, direction }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      setResult(data);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <FormShell
      title="Translate"
      sub="Legal-grade EN↔HI. Section numbers and clause IDs are preserved verbatim."
    >
      <Field label="Direction">
        <button
          onClick={() =>
            setDirection((d) => (d === "en->hi" ? "hi->en" : "en->hi"))
          }
          className="flex items-center gap-2 bg-[var(--bg-elev)] border border-[var(--line)] rounded-lg px-3 py-2 text-sm hover:border-[var(--accent-soft)] transition-colors"
        >
          {direction === "en->hi" ? "English → हिन्दी" : "हिन्दी → English"}
          <ArrowRightLeft size={14} className="text-[var(--accent)]" />
        </button>
      </Field>
      <Field label="Text">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={10}
          className="w-full bg-[var(--bg-elev)] border border-[var(--line)] rounded-lg px-3 py-2 text-sm outline-none focus:border-[var(--accent-soft)] resize-vertical"
        />
      </Field>
      <RunButton busy={busy} disabled={text.trim().length < 1} onClick={run}>
        Translate
      </RunButton>
      {err && <ErrorBlock msg={err} />}
      {result && (
        <ResultBlock
          markdown={result.translation || result.markdown || ""}
          meta={result.llm}
        />
      )}
    </FormShell>
  );
}

// ── Citator ─────────────────────────────────────────────────────────────────
function CitatorForm() {
  const [caseTitle, setCaseTitle] = useState("");
  const [excerpt, setExcerpt] = useState("");
  const [holdings, setHoldings] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<RunResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      const r = await fetch("/api/citator", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case_title: caseTitle, excerpt, holdings }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      setResult(data);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <FormShell
      title="Citator"
      sub="Paste a judgment's title, key excerpt, and holdings — get a structured judicial-history summary."
    >
      <Field label="Case title">
        <input
          value={caseTitle}
          onChange={(e) => setCaseTitle(e.target.value)}
          placeholder="e.g. Sushila Aggarwal v. State (NCT of Delhi)"
          className="w-full bg-[var(--bg-elev)] border border-[var(--line)] rounded-lg px-3 py-2 text-sm outline-none focus:border-[var(--accent-soft)]"
        />
      </Field>
      <Field label="Key excerpt (paste from judgment)">
        <textarea
          value={excerpt}
          onChange={(e) => setExcerpt(e.target.value)}
          rows={8}
          className="w-full bg-[var(--bg-elev)] border border-[var(--line)] rounded-lg px-3 py-2 text-sm outline-none focus:border-[var(--accent-soft)] resize-vertical"
        />
      </Field>
      <Field label="Holdings (optional)">
        <textarea
          value={holdings}
          onChange={(e) => setHoldings(e.target.value)}
          rows={4}
          className="w-full bg-[var(--bg-elev)] border border-[var(--line)] rounded-lg px-3 py-2 text-sm outline-none focus:border-[var(--accent-soft)] resize-vertical"
        />
      </Field>
      <RunButton
        busy={busy}
        disabled={caseTitle.trim().length < 2 || excerpt.trim().length < 10}
        onClick={run}
      >
        Run citator
      </RunButton>
      {err && <ErrorBlock msg={err} />}
      {result && (
        <ResultBlock
          markdown={result.citator_markdown || result.markdown || ""}
          meta={result.llm}
        />
      )}
    </FormShell>
  );
}

// ── Redline ─────────────────────────────────────────────────────────────────
function RedlineForm() {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<RunResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      const r = await fetch("/api/redline", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      setResult(data);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <FormShell
      title="Redline"
      sub="Paste a contract (50–40,000 chars). Sanhita returns structured remove/replace/add edits with reasons."
    >
      <Field label="Contract text">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={16}
          className="w-full bg-[var(--bg-elev)] border border-[var(--line)] rounded-lg px-3 py-2 text-sm outline-none focus:border-[var(--accent-soft)] resize-vertical"
          placeholder="Paste the agreement here…"
        />
      </Field>
      <RunButton busy={busy} disabled={text.length < 50} onClick={run}>
        Suggest redlines
      </RunButton>
      {err && <ErrorBlock msg={err} />}
      {result && (
        <div className="mt-6 flex flex-col gap-3">
          <RunMeta meta={result.llm} />
          {(result.suggestions || []).map((s, i) => (
            <SuggestionCard key={i} s={s} idx={i + 1} />
          ))}
          {result.suggestions && result.suggestions.length === 0 && (
            <div className="text-sm italic text-[var(--ink-soft)]">
              Sanhita didn&apos;t find anything to flag in this draft.
            </div>
          )}
          {result.markdown && (
            <ResultBlock markdown={result.markdown} meta={undefined} />
          )}
        </div>
      )}
    </FormShell>
  );
}

function SuggestionCard({ s, idx }: { s: RedlineSuggestion; idx: number }) {
  const colour =
    s.type === "remove"
      ? "border-[var(--danger)]"
      : s.type === "add"
      ? "border-[var(--accent)]"
      : "border-[var(--line-strong)]";
  return (
    <div className={`bg-[var(--bg-elev)] border-l-4 ${colour} rounded-lg p-4`}>
      <div className="flex items-baseline gap-2 mb-2">
        <span className="text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)]">
          [{idx}] {s.type}
        </span>
        {s.position && (
          <span className="text-[11px] text-[var(--ink-soft)] italic">
            {s.position}
          </span>
        )}
      </div>
      {s.before && (
        <div className="text-sm mb-1">
          <span className="text-[var(--ink-soft)] text-xs">before:</span>{" "}
          <span className="line-through text-[var(--danger)]">{s.before}</span>
        </div>
      )}
      {s.after && (
        <div className="text-sm mb-1">
          <span className="text-[var(--ink-soft)] text-xs">after:</span>{" "}
          <span className="text-[var(--accent)]">{s.after}</span>
        </div>
      )}
      {s.text && !s.before && !s.after && (
        <div className="text-sm mb-1">{s.text}</div>
      )}
      {s.reason && (
        <div className="text-xs text-[var(--ink-soft)] italic mt-2">
          {s.reason}
        </div>
      )}
    </div>
  );
}

// ── Generic ─────────────────────────────────────────────────────────────────
function GenericForm({
  wf,
}: {
  wf: { generic: string; title: string; sub: string };
}) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<RunResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      const r = await fetch("/api/workflows/run", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: wf.generic, text }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      setResult(data);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <FormShell title={wf.title} sub={wf.sub}>
      <Field label="Input">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={12}
          className="w-full bg-[var(--bg-elev)] border border-[var(--line)] rounded-lg px-3 py-2 text-sm outline-none focus:border-[var(--accent-soft)] resize-vertical"
        />
      </Field>
      <RunButton busy={busy} disabled={text.length < 10} onClick={run}>
        Run
      </RunButton>
      {err && <ErrorBlock msg={err} />}
      {result && (
        <ResultBlock
          markdown={result.markdown || result.draft_markdown || ""}
          meta={result.llm}
        />
      )}
    </FormShell>
  );
}

// ── shared form chrome ──────────────────────────────────────────────────────
function FormShell({
  title,
  sub,
  children,
}: {
  title: string;
  sub: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="font-display italic text-3xl tracking-tight text-[var(--ink)]">
        {title}
      </div>
      <p className="mt-2 text-[var(--ink-soft)] max-w-2xl mb-8">{sub}</p>
      <div className="flex flex-col gap-5">{children}</div>
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <div className="text-xs tracking-[0.14em] uppercase text-[var(--ink-soft)] mb-2">
        {label}
      </div>
      {children}
      {hint && (
        <div className="mt-1 text-[11px] italic text-[var(--ink-soft)]">{hint}</div>
      )}
    </label>
  );
}

function RunButton({
  busy,
  disabled,
  onClick,
  children,
}: {
  busy: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={busy || disabled}
      className="self-start bg-[var(--ink)] text-[var(--bg)] py-2.5 px-6 rounded-xl text-sm font-medium hover:bg-[var(--accent)] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
    >
      {busy ? "Running…" : children}
    </button>
  );
}

function ErrorBlock({ msg }: { msg: string }) {
  return (
    <div className="bg-[var(--bg-elev)] border-l-4 border-[var(--danger)] rounded-lg p-4 text-sm text-[var(--danger)]">
      {msg}
    </div>
  );
}

function RunMeta({ meta }: { meta?: { provider?: string; model?: string; latency_ms?: number } }) {
  if (!meta || !meta.provider) return null;
  return (
    <div className="flex items-center gap-2 text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)]">
      <span className="bg-[var(--bg-elev)] border border-[var(--line)] px-1.5 py-0.5 rounded">
        via {meta.provider}
      </span>
      {typeof meta.latency_ms === "number" && <span>{meta.latency_ms} ms</span>}
    </div>
  );
}

function ResultBlock({
  markdown,
  meta,
  refused,
}: {
  markdown: string;
  meta?: { provider?: string; model?: string; latency_ms?: number };
  refused?: boolean;
}) {
  return (
    <div className="mt-4 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <RunMeta meta={meta} />
        {refused && (
          <span className="bg-[var(--danger)] text-white px-1.5 py-0.5 rounded text-[9px] tracking-wider">
            refused
          </span>
        )}
      </div>
      <div
        className="bg-[var(--bg-elev)] border border-[var(--line)] rounded-2xl px-6 py-5 leading-relaxed text-[15px] prose-style"
        dangerouslySetInnerHTML={{ __html: renderMarkdown(markdown) }}
      />
    </div>
  );
}
