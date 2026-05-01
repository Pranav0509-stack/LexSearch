"use client";

/**
 * Workflows — Real multi-step legal pipeline system.
 *
 * Each workflow is a sequential pipeline of steps:
 *   form   → user inputs facts / paste text
 *   search → auto-runs BM25 case search, returns top cases
 *   draft  → auto-drafts using retrieved context + LLM
 *   review → auto-reviews / risk-flags
 *
 * API endpoints used:
 *   GET  /api/cases/search?q=...&k=10
 *   POST /api/draft          {template, facts}
 *   POST /api/workflows/run  {key, text}
 *   POST /api/review         {clauses}
 *   POST /api/redline        {text}
 *   POST /api/citator        {case_title, excerpt, holdings}
 */

import { useState, useCallback, useRef } from "react";
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  Circle,
  Loader2,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  FileDown,
  RefreshCw,
  Search,
  FileText,
  Gavel,
  Scale,
  ClipboardList,
  FileSignature,
  BookOpen,
  Layers,
  ExternalLink,
  Sparkles,
  PenLine,
} from "lucide-react";
import { renderMarkdown } from "./markdown";

/* ─── Types ──────────────────────────────────────────────────── */

type StepType = "form" | "search" | "analyze" | "draft" | "review";
type StepStatus = "waiting" | "running" | "done" | "error";

interface StepField {
  id: string;
  label: string;
  placeholder: string;
  type?: "text" | "textarea" | "select";
  options?: string[];
  required?: boolean;
}

interface WorkflowStepDef {
  id: string;
  title: string;
  subtitle: string;
  type: StepType;
  fields?: StepField[];
  searchQuery?: (inputs: Record<string, string>) => string;
  draftTemplate?: string;
  workflowKey?: string;
  buildPrompt?: (inputs: Record<string, string>, prevOutputs: string[]) => string;
}

interface WorkflowDef {
  id: string;
  title: string;
  tagline: string;
  description: string;
  icon: React.ReactNode;
  category: "litigation" | "drafting" | "research" | "contract";
  steps: WorkflowStepDef[];
  color: string;
}

interface CaseResult {
  case_id: string;
  title?: string;
  court?: string;
  year?: number | string;
  citation?: string;
  verdict?: string;
  excerpt?: string;
  url?: string;
  score?: number;
  tier?: string;
}

interface StepState {
  status: StepStatus;
  output?: string;
  cases?: CaseResult[];
  elapsed?: number;
  error?: string;
  collapsed?: boolean;
}

/* ─── Workflow Definitions ───────────────────────────────────── */

const WORKFLOWS: WorkflowDef[] = [
  {
    id: "research_memo",
    title: "Legal Research Memo",
    tagline: "Question → Cases → Full memo with citations",
    description: "Searches 31M judgments, clusters by legal principle, drafts a structured memo with verified citations.",
    icon: <BookOpen size={22} />,
    category: "research",
    color: "#1a73e8",
    steps: [
      {
        id: "question",
        title: "Research Question",
        subtitle: "Define the legal question and relevant context",
        type: "form",
        fields: [
          { id: "question", label: "Legal question", placeholder: "e.g. What are the grounds for bail under Section 37 NDPS Act?", type: "textarea", required: true },
          { id: "jurisdiction", label: "Jurisdiction / court (optional)", placeholder: "e.g. Bombay High Court, Supreme Court of India", type: "text" },
          { id: "context", label: "Case context (optional)", placeholder: "Brief facts, party names, offences charged…", type: "textarea" },
        ],
      },
      {
        id: "cases",
        title: "Case Retrieval",
        subtitle: "Searching 31M judgments via BM25",
        type: "search",
        searchQuery: (inp) => `${inp.question} ${inp.context || ""}`.slice(0, 300),
      },
      {
        id: "memo",
        title: "Legal Memo",
        subtitle: "AI-drafted memo with verified citations",
        type: "draft",
        draftTemplate: "support_argument",
        buildPrompt: (inp, prevOutputs) =>
          `Research question: ${inp.question}\n\nJurisdiction: ${inp.jurisdiction || "Indian courts"}\n\nContext: ${inp.context || "None"}\n\nRetrieved cases:\n${prevOutputs[0] || ""}`,
      },
    ],
  },

  {
    id: "bail_application",
    title: "Bail Application",
    tagline: "Facts → Precedents → Court-ready bail application",
    description: "Finds the most relevant bail judgments for your offence, then drafts a complete bail application with embedded citations.",
    icon: <Gavel size={22} />,
    category: "litigation",
    color: "#e8710a",
    steps: [
      {
        id: "facts",
        title: "Client & Case Details",
        subtitle: "Enter the accused and offence details",
        type: "form",
        fields: [
          { id: "accused", label: "Name of accused", placeholder: "e.g. Rajesh Kumar", type: "text", required: true },
          { id: "offence", label: "Offence / sections charged", placeholder: "e.g. Section 302 IPC, Section 20 NDPS Act", type: "text", required: true },
          { id: "court", label: "Court where FIR registered / matter pending", placeholder: "e.g. Sessions Court, Pune", type: "text", required: true },
          { id: "arrest_date", label: "Date of arrest / custody since", placeholder: "e.g. 12 April 2025", type: "text" },
          { id: "facts", label: "Brief facts in support of bail", placeholder: "Describe why bail should be granted — prior convictions (if none), family circumstances, flight risk, etc.", type: "textarea", required: true },
        ],
      },
      {
        id: "precedents",
        title: "Bail Precedents",
        subtitle: "Searching for relevant bail judgments",
        type: "search",
        searchQuery: (inp) => `bail application ${inp.offence} grounds custody`,
      },
      {
        id: "draft",
        title: "Draft Application",
        subtitle: "Generating court-ready bail application",
        type: "draft",
        draftTemplate: "anticipatory_bail_482",
        buildPrompt: (inp, prevOutputs) =>
          `BAIL APPLICATION DRAFTING\n\nAccused: ${inp.accused}\nOffence: ${inp.offence}\nCourt: ${inp.court}\nDate of arrest: ${inp.arrest_date || "Not specified"}\nFacts in support: ${inp.facts}\n\nRelevant precedents found:\n${prevOutputs[0] || "None retrieved"}`,
      },
    ],
  },

  {
    id: "section138_notice",
    title: "Section 138 Demand Notice",
    tagline: "Cheque details → Precedents → Statutory demand notice",
    description: "Drafts a legally valid demand notice under Section 138 NI Act with 30-day demand period and latest case law.",
    icon: <FileSignature size={22} />,
    category: "litigation",
    color: "#d93025",
    steps: [
      {
        id: "cheque_details",
        title: "Cheque & Party Details",
        subtitle: "Enter the dishonoured cheque information",
        type: "form",
        fields: [
          { id: "payee", label: "Payee (your client — notice sender)", placeholder: "e.g. Sharma Traders Pvt. Ltd.", type: "text", required: true },
          { id: "drawer", label: "Drawer (accused — notice recipient)", placeholder: "e.g. Ramesh Verma, 123 MG Road, Pune", type: "text", required: true },
          { id: "cheque_no", label: "Cheque number", placeholder: "e.g. 004567", type: "text", required: true },
          { id: "cheque_date", label: "Cheque date", placeholder: "e.g. 01 March 2025", type: "text", required: true },
          { id: "cheque_amount", label: "Cheque amount (Rs.)", placeholder: "e.g. 5,00,000", type: "text", required: true },
          { id: "bank", label: "Drawee bank", placeholder: "e.g. HDFC Bank, Koregaon Park Branch", type: "text", required: true },
          { id: "dishonour_date", label: "Date of dishonour / return memo", placeholder: "e.g. 15 March 2025", type: "text", required: true },
          { id: "reason", label: "Reason for dishonour (from bank memo)", placeholder: "e.g. Insufficient funds / Account closed", type: "text", required: true },
          { id: "underlying_liability", label: "Underlying liability (legally enforceable debt)", placeholder: "e.g. towards repayment of loan of Rs. 5 lakhs advanced on 10 Jan 2025", type: "textarea", required: true },
        ],
      },
      {
        id: "precedents",
        title: "Section 138 Precedents",
        subtitle: "Loading latest NI Act jurisprudence",
        type: "search",
        searchQuery: () => "Section 138 Negotiable Instruments Act cheque dishonour demand notice limitation",
      },
      {
        id: "notice",
        title: "Demand Notice",
        subtitle: "Generating Section 138 demand notice",
        type: "draft",
        draftTemplate: "ni_138_notice",
        buildPrompt: (inp) =>
          `SECTION 138 NI ACT DEMAND NOTICE\n\nPayee / Complainant: ${inp.payee}\nDrawer / Accused: ${inp.drawer}\nCheque No: ${inp.cheque_no}, dated ${inp.cheque_date}, for Rs. ${inp.cheque_amount}\nBank: ${inp.bank}\nDishonour date: ${inp.dishonour_date}\nReason for dishonour: ${inp.reason}\nUnderlying liability: ${inp.underlying_liability}\n\nDraft a formal legal demand notice under Section 138 of the Negotiable Instruments Act, 1881. The notice must: (a) state the facts of dishonour, (b) demand payment of Rs. ${inp.cheque_amount} within 30 days, (c) warn of criminal prosecution under Section 138 NI Act if unpaid. Use formal legal language with proper cause-of-action paragraph.`,
      },
    ],
  },

  {
    id: "contract_review",
    title: "Contract Due Diligence",
    tagline: "Paste agreement → Risk clauses → Case law → Risk report",
    description: "Identifies high-risk clauses, finds supporting Indian case law for each risk, and generates a comprehensive due diligence report.",
    icon: <ClipboardList size={22} />,
    category: "contract",
    color: "#1e8e3e",
    steps: [
      {
        id: "contract",
        title: "Upload Contract",
        subtitle: "Paste the agreement text for analysis",
        type: "form",
        fields: [
          { id: "contract_type", label: "Agreement type", placeholder: "e.g. Employment Agreement, Share Purchase Agreement, Lease Deed", type: "text", required: true },
          { id: "parties", label: "Parties to the agreement", placeholder: "e.g. ABC Pvt. Ltd. (Company) and John Doe (Employee)", type: "text", required: true },
          { id: "text", label: "Agreement text (paste full or key clauses)", placeholder: "Paste the full contract or the clauses you want reviewed…", type: "textarea", required: true },
        ],
      },
      {
        id: "risk_analysis",
        title: "Risk Identification",
        subtitle: "Identifying high-risk and non-standard clauses",
        type: "analyze",
        workflowKey: "risks",
        buildPrompt: (inp) =>
          `CONTRACT TYPE: ${inp.contract_type}\nPARTIES: ${inp.parties}\n\n${inp.text}`,
      },
      {
        id: "case_law",
        title: "Supporting Case Law",
        subtitle: "Searching for Indian judgments on flagged risks",
        type: "search",
        searchQuery: (inp) => `${inp.contract_type} contract clause risk enforceability Indian courts`,
      },
      {
        id: "report",
        title: "Due Diligence Report",
        subtitle: "Generating final risk report with citations",
        type: "analyze",
        workflowKey: "risks",
        buildPrompt: (inp, prevOutputs) =>
          `CONTRACT DUE DILIGENCE REPORT\n\nAgreement type: ${inp.contract_type}\nParties: ${inp.parties}\n\nRisk analysis:\n${prevOutputs[0] || ""}\n\nRelevant case law:\n${prevOutputs[1] || ""}\n\nOriginal text:\n${inp.text.slice(0, 1500)}`,
      },
    ],
  },

  {
    id: "writ_petition",
    title: "Writ Petition Drafter",
    tagline: "Grounds → Article 226 precedents → Full petition",
    description: "Researches relevant Article 226/32 precedents for each ground of challenge and drafts a complete writ petition with prayer clause.",
    icon: <Scale size={22} />,
    category: "litigation",
    color: "#8430ce",
    steps: [
      {
        id: "petition_facts",
        title: "Petition Details",
        subtitle: "Enter the petitioner, respondent and grounds",
        type: "form",
        fields: [
          { id: "petitioner", label: "Petitioner(s)", placeholder: "e.g. XYZ Pvt. Ltd., a company incorporated under the Companies Act", type: "text", required: true },
          { id: "respondent", label: "Respondent(s)", placeholder: "e.g. State of Maharashtra through its Principal Secretary, Ministry of Finance", type: "text", required: true },
          { id: "high_court", label: "High Court", placeholder: "e.g. High Court of Bombay", type: "text", required: true },
          { id: "impugned_order", label: "Impugned order / action", placeholder: "e.g. Assessment order dated 12 March 2025 issued by the Deputy Commissioner of Income Tax", type: "textarea", required: true },
          { id: "grounds", label: "Grounds of challenge", placeholder: "List the grounds — e.g. (1) Violation of natural justice — no show-cause notice issued; (2) Without jurisdiction — barred by limitation; (3) Arbitrary and unreasonable — violates Article 14", type: "textarea", required: true },
          { id: "relief", label: "Relief sought", placeholder: "e.g. Quash and set aside the impugned order; Issue mandamus directing re-assessment with notice; Stay operation of the order pending disposal", type: "textarea", required: true },
        ],
      },
      {
        id: "precedents",
        title: "Writ Precedents",
        subtitle: "Searching Article 226 / natural justice case law",
        type: "search",
        searchQuery: (inp) => `writ petition Article 226 ${inp.grounds.slice(0, 150)} High Court`,
      },
      {
        id: "petition",
        title: "Draft Petition",
        subtitle: "Drafting complete writ petition",
        type: "draft",
        draftTemplate: "writ_226",
        buildPrompt: (inp, prevOutputs) =>
          `WRIT PETITION UNDER ARTICLE 226\n\nHigh Court: ${inp.high_court}\nPetitioner: ${inp.petitioner}\nRespondent: ${inp.respondent}\nImpugned order/action: ${inp.impugned_order}\nGrounds of challenge:\n${inp.grounds}\nRelief sought:\n${inp.relief}\n\nRelevant precedents:\n${prevOutputs[0] || ""}`,
      },
    ],
  },

  {
    id: "case_timeline",
    title: "Case Chronology Builder",
    tagline: "Paste materials → Extract events → Formatted timeline",
    description: "Extracts all dates, events and orders from case materials and builds a structured chronology for use in arguments or briefs.",
    icon: <Layers size={22} />,
    category: "research",
    color: "#b5770d",
    steps: [
      {
        id: "materials",
        title: "Case Materials",
        subtitle: "Paste pleadings, orders, correspondence or notes",
        type: "form",
        fields: [
          { id: "matter_name", label: "Matter name", placeholder: "e.g. ABC Pvt. Ltd. vs Income Tax Department", type: "text", required: true },
          { id: "materials", label: "Case materials (paste all relevant text)", placeholder: "Paste FIRs, orders, notices, emails, letters, HC/SC orders — everything you want events extracted from…", type: "textarea", required: true },
          { id: "focus", label: "Focus area (optional)", placeholder: "e.g. only include events after 01 Jan 2023, or focus on court hearings only", type: "text" },
        ],
      },
      {
        id: "timeline",
        title: "Chronology",
        subtitle: "Extracting and organizing events by date",
        type: "analyze",
        workflowKey: "chronology",
        buildPrompt: (inp) =>
          `MATTER: ${inp.matter_name}\n${inp.focus ? `FOCUS: ${inp.focus}\n` : ""}\n${inp.materials}`,
      },
      {
        id: "summary",
        title: "Brief Summary",
        subtitle: "Generating factual summary from timeline",
        type: "analyze",
        workflowKey: "interim_memo",
        buildPrompt: (inp, prevOutputs) =>
          `Write a concise factual summary of the matter "${inp.matter_name}" based on this chronology:\n\n${prevOutputs[0] || ""}\n\nThe summary should be suitable for use in an opening argument or brief — approximately 200-300 words, organized by phase (pre-litigation, litigation, current status).`,
      },
    ],
  },
];

/* ─── API helpers ─────────────────────────────────────────────── */

const api = async (path: string, opts?: RequestInit) => {
  const r = await fetch(path, { credentials: "include", ...opts });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
    throw new Error(err.detail || `HTTP ${r.status}`);
  }
  return r.json();
};
const post = (path: string, body: unknown) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

async function runSearchStep(query: string): Promise<{ output: string; cases: CaseResult[] }> {
  const d = await api(`/api/cases/search?q=${encodeURIComponent(query)}&k=8`);
  const hits: CaseResult[] = d.hits || d.results || [];
  if (hits.length === 0) return { output: "No relevant cases found in the corpus for this query.", cases: [] };

  const output = hits.slice(0, 8).map((h, i) =>
    `[${i + 1}] ${h.title || h.case_id}\n  Court: ${h.court || "Unknown"} | Year: ${h.year || "?"} | Verdict: ${h.verdict || "?"}\n  ${(h.excerpt || "").slice(0, 200)}`
  ).join("\n\n");

  return { output, cases: hits };
}

async function runDraftStep(template: string, facts: string): Promise<string> {
  try {
    // facts must be a dict per backend DraftBody schema
    const d = await post("/api/draft", { template, facts: { text: facts } });
    return d.draft_markdown || d.markdown || d.text || JSON.stringify(d).slice(0, 2000);
  } catch {
    // Fallback to generic workflow run
    try {
      const d = await post("/api/workflows/run", { key: "risks", text: facts });
      return d.markdown || d.text || "Draft generation requires an LLM API key (Gemini/Anthropic/Groq). Add one in Settings.";
    } catch {
      return "Draft generation requires an LLM API key. Add one in Settings → AI Keys.";
    }
  }
}

async function runAnalyzeStep(key: string, text: string): Promise<string> {
  try {
    const d = await post("/api/workflows/run", { key, text });
    // /api/workflows/run returns output_markdown, not markdown
    return d.output_markdown || d.markdown || d.text || JSON.stringify(d).slice(0, 2000);
  } catch (e) {
    throw new Error(`Analysis step failed: ${(e as Error).message}`);
  }
}

/* ─── Main component ─────────────────────────────────────────── */

interface WorkflowsPaneProps {
  onOpenInEditor?: (content: string, title: string) => void;
}

export default function WorkflowsPane({ onOpenInEditor }: WorkflowsPaneProps = {}) {
  const [selected, setSelected] = useState<WorkflowDef | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [stepStates, setStepStates] = useState<Record<string, StepState>>({});
  const [formInputs, setFormInputs] = useState<Record<string, string>>({});
  const [running, setRunning] = useState(false);
  const [copyState, setCopyState] = useState<Record<string, boolean>>({});
  const outputRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const reset = useCallback(() => {
    setCurrentStep(0);
    setStepStates({});
    setFormInputs({});
    setRunning(false);
  }, []);

  const selectWorkflow = useCallback((wf: WorkflowDef) => {
    setSelected(wf);
    reset();
  }, [reset]);

  const updateStep = useCallback((stepId: string, patch: Partial<StepState>) => {
    setStepStates(prev => ({ ...prev, [stepId]: { ...prev[stepId], ...patch } }));
  }, []);

  const copyText = useCallback((id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopyState(c => ({ ...c, [id]: true }));
    setTimeout(() => setCopyState(c => ({ ...c, [id]: false })), 2000);
  }, []);

  const downloadDoc = useCallback((title: string, text: string) => {
    const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${title.replace(/\s+/g, "-").toLowerCase()}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, []);

  const runStep = useCallback(async (wf: WorkflowDef, stepIdx: number, inputs: Record<string, string>) => {
    const step = wf.steps[stepIdx];
    if (!step) return;

    // Collect previous step outputs for context
    const prevOutputs = wf.steps.slice(0, stepIdx)
      .map(s => stepStates[s.id]?.output || "")
      .filter(Boolean);

    updateStep(step.id, { status: "running", error: undefined });
    const t0 = Date.now();

    try {
      if (step.type === "form") {
        // Form step — just advance
        updateStep(step.id, { status: "done", output: Object.entries(inputs).map(([k, v]) => `**${k}**: ${v}`).join("\n"), elapsed: 0 });
      } else if (step.type === "search") {
        const query = step.searchQuery ? step.searchQuery(inputs) : inputs.question || "";
        const { output, cases } = await runSearchStep(query);
        updateStep(step.id, { status: "done", output, cases, elapsed: Date.now() - t0 });
      } else if (step.type === "analyze") {
        const text = step.buildPrompt ? step.buildPrompt(inputs, prevOutputs) : prevOutputs.join("\n\n");
        const output = await runAnalyzeStep(step.workflowKey || "risks", text);
        updateStep(step.id, { status: "done", output, elapsed: Date.now() - t0 });
      } else if (step.type === "draft") {
        const facts = step.buildPrompt ? step.buildPrompt(inputs, prevOutputs) : prevOutputs.join("\n\n");
        const output = await runDraftStep(step.draftTemplate || "general", facts);
        updateStep(step.id, { status: "done", output, elapsed: Date.now() - t0 });
      }

      // Advance to next step
      if (stepIdx + 1 < wf.steps.length) {
        setCurrentStep(stepIdx + 1);
        // Auto-run non-form steps
        const nextStep = wf.steps[stepIdx + 1];
        if (nextStep.type !== "form") {
          setTimeout(() => {
            setRunning(true);
            runStep(wf, stepIdx + 1, inputs).finally(() => setRunning(false));
          }, 400);
        }
      }
    } catch (e) {
      updateStep(step.id, { status: "error", error: (e as Error).message, elapsed: Date.now() - t0 });
    }
  }, [stepStates, updateStep]);

  const handleFormSubmit = useCallback(async () => {
    if (!selected || running) return;

    // Validate required fields
    const step = selected.steps[currentStep];
    if (step.type === "form") {
      const missing = (step.fields || []).filter(f => f.required && !formInputs[f.id]?.trim());
      if (missing.length > 0) {
        updateStep(step.id, { status: "error", error: `Please fill in: ${missing.map(f => f.label).join(", ")}` });
        return;
      }
    }

    setRunning(true);
    updateStep(step.id, { status: "done", output: "Inputs captured.", elapsed: 0 });

    // Start next step immediately
    const nextIdx = currentStep + 1;
    if (nextIdx < selected.steps.length) {
      setCurrentStep(nextIdx);
      const nextStep = selected.steps[nextIdx];
      if (nextStep.type !== "form") {
        await runStep(selected, nextIdx, formInputs);
      }
    }
    setRunning(false);
  }, [selected, running, currentStep, formInputs, updateStep, runStep]);

  const rerunStep = useCallback(async (stepIdx: number) => {
    if (!selected || running) return;
    setRunning(true);
    await runStep(selected, stepIdx, formInputs);
    setRunning(false);
  }, [selected, running, formInputs, runStep]);

  /* ── Workflow list view ──────────────────────────────────────── */
  if (!selected) {
    return (
      <div className="flex flex-col h-full bg-[#f8f9fa] overflow-hidden">
        <div className="bg-white border-b border-[var(--line)] px-4 sm:px-8 py-5">
          <h2 className="font-display text-xl tracking-tight text-[var(--ink)]">Workflows</h2>
          <p className="text-xs text-[var(--ink-soft)] mt-0.5">Multi-step legal pipelines — from facts to court-ready output</p>
        </div>

        <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-5 sm:py-6">
          {/* Category sections */}
          {(["litigation", "research", "contract", "drafting"] as const).map(cat => {
            const wfs = WORKFLOWS.filter(w => w.category === cat);
            if (wfs.length === 0) return null;
            const catLabel = { litigation: "Litigation", research: "Research", contract: "Contract & Due Diligence", drafting: "Drafting" }[cat];
            return (
              <div key={cat} className="mb-8">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--ink-soft)] mb-3">{catLabel}</div>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  {wfs.map(wf => (
                    <button
                      key={wf.id}
                      onClick={() => selectWorkflow(wf)}
                      className="group text-left bg-white border border-[var(--line)] hover:border-[var(--accent)] hover:shadow-[0_4px_20px_rgba(0,0,0,0.08)] rounded-2xl p-5 transition-all duration-200 flex flex-col gap-3"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0 transition-colors" style={{ background: `${wf.color}18`, color: wf.color }}>
                          {wf.icon}
                        </div>
                        <div className="shrink-0 w-6 h-6 rounded-full bg-[var(--bg-elev)] flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity" style={{ color: wf.color }}>
                          <ArrowRight size={12} />
                        </div>
                      </div>
                      <div>
                        <p className="font-semibold text-sm text-[var(--ink)] leading-snug">{wf.title}</p>
                        <p className="text-xs text-[var(--ink-soft)] mt-0.5 leading-snug">{wf.tagline}</p>
                      </div>
                      <p className="text-[11px] text-[var(--ink-soft)] leading-relaxed line-clamp-2">{wf.description}</p>
                      {/* Step count */}
                      <div className="flex items-center gap-1.5 mt-auto">
                        {wf.steps.map((s, i) => (
                          <div key={i} className="flex items-center gap-1 text-[9px] uppercase tracking-wider text-[var(--ink-soft)]">
                            <div className="w-1.5 h-1.5 rounded-full" style={{ background: `${wf.color}60` }} />
                            {s.title}
                          </div>
                        ))}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  /* ── Active workflow pipeline view ───────────────────────────── */
  const allDone = selected.steps.every(s => stepStates[s.id]?.status === "done");
  const lastOutput = stepStates[selected.steps[selected.steps.length - 1]?.id]?.output || "";

  return (
    <div className="flex flex-col h-full bg-[#f8f9fa] overflow-hidden">

      {/* Header */}
      <div className="bg-white border-b border-[var(--line)] px-3 sm:px-6 py-4 flex items-center gap-3 sm:gap-4 shrink-0">
        <button onClick={() => { setSelected(null); reset(); }} className="p-1.5 hover:bg-[var(--bg-elev)] rounded-lg transition-colors text-[var(--ink-soft)]">
          <ArrowLeft size={16} />
        </button>
        <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: `${selected.color}18`, color: selected.color }}>
          {selected.icon}
        </div>
        <div className="flex-1 min-w-0">
          <h2 className="font-display text-base font-semibold text-[var(--ink)] leading-tight">{selected.title}</h2>
          <p className="text-xs text-[var(--ink-soft)]">{selected.tagline}</p>
        </div>
        {allDone && (
          <div className="flex items-center gap-2">
            <button
              onClick={() => downloadDoc(selected.title, lastOutput)}
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border border-[var(--line)] text-[var(--ink-soft)] hover:border-[var(--accent)] hover:text-[var(--accent)] transition-colors"
            >
              <FileDown size={12} /> Download
            </button>
            <button
              onClick={() => { reset(); }}
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border border-[var(--line)] text-[var(--ink-soft)] hover:bg-[var(--bg-elev)] transition-colors"
            >
              <RefreshCw size={12} /> Start over
            </button>
          </div>
        )}
      </div>

      {/* Step tracker — scrollable on mobile */}
      <div className="bg-white border-b border-[var(--line)] px-3 sm:px-6 py-3 shrink-0 overflow-x-auto">
        <div className="flex items-center gap-0 min-w-max sm:min-w-0">
          {selected.steps.map((step, i) => {
            const state = stepStates[step.id];
            const isCurrent = i === currentStep;
            const isDone = state?.status === "done";
            const isError = state?.status === "error";
            const isRunning = state?.status === "running";

            return (
              <div key={step.id} className="flex items-center flex-1 min-w-0">
                <div className={`flex items-center gap-2 min-w-0 py-1 px-2 rounded-lg transition-all ${isCurrent ? "bg-[var(--bg-elev)]" : ""}`}>
                  <div className="shrink-0">
                    {isDone ? (
                      <CheckCircle2 size={16} style={{ color: selected.color }} />
                    ) : isError ? (
                      <AlertCircle size={16} className="text-[var(--danger)]" />
                    ) : isRunning ? (
                      <Loader2 size={16} className="animate-spin text-[var(--accent)]" />
                    ) : (
                      <Circle size={16} className={isCurrent ? "text-[var(--accent)]" : "text-[var(--ink-soft)]"} />
                    )}
                  </div>
                  <div className="min-w-0">
                    <p className={`text-[11px] font-medium truncate ${isCurrent ? "text-[var(--ink)]" : isDone ? "text-[var(--ink-soft)]" : "text-[var(--ink-soft)]"}`}>
                      {step.title}
                    </p>
                    {state?.elapsed && <p className="text-[9px] text-[var(--ink-soft)]">{(state.elapsed / 1000).toFixed(1)}s</p>}
                  </div>
                </div>
                {i < selected.steps.length - 1 && (
                  <div className="flex-1 h-px bg-[var(--line)] mx-1 min-w-[12px]" />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Step outputs + active form */}
      <div className="flex-1 overflow-y-auto px-3 sm:px-6 py-4 sm:py-5 flex flex-col gap-4">

        {/* Completed step outputs */}
        {selected.steps.map((step, i) => {
          const state = stepStates[step.id];
          if (!state || state.status === "waiting") return null;
          const isLast = i === selected.steps.length - 1;
          const isSearchStep = step.type === "search";

          return (
            <div key={step.id} className="bg-white border border-[var(--line)] rounded-2xl overflow-hidden">
              {/* Step header */}
              <div
                className="flex items-center gap-3 px-5 py-3 cursor-pointer select-none"
                onClick={() => updateStep(step.id, { collapsed: !state.collapsed })}
              >
                <div className="shrink-0">
                  {state.status === "running" ? (
                    <Loader2 size={15} className="animate-spin text-[var(--accent)]" />
                  ) : state.status === "done" ? (
                    <CheckCircle2 size={15} style={{ color: selected.color }} />
                  ) : (
                    <AlertCircle size={15} className="text-[var(--danger)]" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <span className="text-sm font-medium text-[var(--ink)]">{step.title}</span>
                  {state.status === "running" && <span className="ml-2 text-xs text-[var(--accent)] animate-pulse">{step.subtitle}…</span>}
                  {state.status === "done" && isSearchStep && state.cases && (
                    <span className="ml-2 text-xs text-[var(--ink-soft)]">{state.cases.length} cases retrieved</span>
                  )}
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  {state.status === "done" && state.output && (
                    <>
                      <button onClick={(e) => { e.stopPropagation(); copyText(step.id, state.output || ""); }} className="p-1 hover:bg-[var(--bg-elev)] rounded transition-colors text-[var(--ink-soft)]" title="Copy">
                        {copyState[step.id] ? <Check size={13} /> : <Copy size={13} />}
                      </button>
                      <button onClick={(e) => { e.stopPropagation(); rerunStep(i); }} className="p-1 hover:bg-[var(--bg-elev)] rounded transition-colors text-[var(--ink-soft)]" title="Re-run">
                        <RefreshCw size={13} />
                      </button>
                    </>
                  )}
                  <button className="p-1 text-[var(--ink-soft)]">
                    {state.collapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
                  </button>
                </div>
              </div>

              {/* Step content */}
              {!state.collapsed && (
                <div className="border-t border-[var(--line)]">
                  {state.status === "error" && (
                    <div className="px-5 py-3 text-sm text-[var(--danger)] bg-red-50 flex items-start gap-2">
                      <AlertCircle size={14} className="shrink-0 mt-0.5" />
                      <span>{state.error}</span>
                    </div>
                  )}

                  {/* Search results — case cards */}
                  {isSearchStep && state.cases && state.cases.length > 0 && (
                    <div className="px-5 py-4 grid grid-cols-1 md:grid-cols-2 gap-3">
                      {state.cases.map((c, ci) => (
                        <CaseCard key={ci} c={c} color={selected.color} />
                      ))}
                    </div>
                  )}

                  {/* Text / markdown output */}
                  {!isSearchStep && state.output && (
                    <div className="px-5 py-4">
                      {isLast && allDone ? (
                        <div
                          className="prose-style text-sm leading-relaxed max-h-[520px] overflow-y-auto"
                          dangerouslySetInnerHTML={{ __html: renderMarkdown(state.output) }}
                        />
                      ) : (
                        <div className="text-xs text-[var(--ink-soft)] font-mono bg-[var(--bg-elev)] rounded-lg px-3 py-3 max-h-[200px] overflow-y-auto whitespace-pre-wrap leading-relaxed">
                          {state.output.slice(0, 600)}{state.output.length > 600 ? "…" : ""}
                        </div>
                      )}

                      {/* Final output actions */}
                      {isLast && allDone && (
                        <div className="flex items-center gap-3 mt-4 pt-4 border-t border-[var(--line)] flex-wrap">
                          <button
                            onClick={() => copyText(`${step.id}_full`, state.output || "")}
                            className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full bg-[var(--bg-elev)] border border-[var(--line)] hover:border-[var(--accent)] hover:text-[var(--accent)] transition-colors"
                          >
                            {copyState[`${step.id}_full`] ? <Check size={11} /> : <Copy size={11} />}
                            {copyState[`${step.id}_full`] ? "Copied!" : "Copy"}
                          </button>
                          <button
                            onClick={() => downloadDoc(selected.title, state.output || "")}
                            className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full bg-[var(--bg-elev)] border border-[var(--line)] hover:border-[var(--accent)] hover:text-[var(--accent)] transition-colors"
                          >
                            <FileDown size={11} /> Download .md
                          </button>
                          {onOpenInEditor && state.output && (
                            <button
                              onClick={() => onOpenInEditor(state.output!, selected.title)}
                              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full text-white transition-opacity hover:opacity-90"
                              style={{ background: selected.color }}
                            >
                              <PenLine size={11} /> Open in Draft Editor
                            </button>
                          )}
                          <div className="flex items-center gap-1.5 text-[10px] text-[var(--ink-soft)] ml-auto">
                            <Sparkles size={10} style={{ color: selected.color }} />
                            <span className="hidden sm:inline">Powered by Sanhita · 31.9M judgments</span>
                            <span className="sm:hidden">Sanhita</span>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}

        {/* Active form step */}
        {!allDone && (() => {
          const step = selected.steps[currentStep];
          if (!step || step.type !== "form") return null;
          const state = stepStates[step.id];
          if (state?.status === "running" || state?.status === "done") return null;

          return (
            <div className="bg-white border border-[var(--line)] rounded-2xl overflow-hidden">
              <div className="px-5 py-4 border-b border-[var(--line)] flex items-center gap-2">
                <div className="w-6 h-6 rounded-full flex items-center justify-center text-[11px] font-bold text-white" style={{ background: selected.color }}>
                  {currentStep + 1}
                </div>
                <div>
                  <p className="font-medium text-sm text-[var(--ink)]">{step.title}</p>
                  <p className="text-xs text-[var(--ink-soft)]">{step.subtitle}</p>
                </div>
              </div>

              <div className="px-5 py-5 flex flex-col gap-4">
                {(step.fields || []).map(field => (
                  <div key={field.id}>
                    <label className="text-xs font-medium text-[var(--ink-soft)] uppercase tracking-wider flex items-center gap-1">
                      {field.label}
                      {field.required && <span className="text-[var(--danger)] text-[10px]">*</span>}
                    </label>
                    {field.type === "textarea" ? (
                      <textarea
                        value={formInputs[field.id] || ""}
                        onChange={e => setFormInputs(prev => ({ ...prev, [field.id]: e.target.value }))}
                        placeholder={field.placeholder}
                        rows={4}
                        className="mt-1.5 w-full border border-[var(--line)] rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:border-[var(--accent)] resize-none text-[var(--ink)] placeholder:text-[var(--ink-soft)] bg-[var(--bg)]"
                      />
                    ) : field.type === "select" ? (
                      <select
                        value={formInputs[field.id] || ""}
                        onChange={e => setFormInputs(prev => ({ ...prev, [field.id]: e.target.value }))}
                        className="mt-1.5 w-full border border-[var(--line)] rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:border-[var(--accent)] bg-[var(--bg)] text-[var(--ink)]"
                      >
                        <option value="">Select…</option>
                        {(field.options || []).map(o => <option key={o} value={o}>{o}</option>)}
                      </select>
                    ) : (
                      <input
                        type="text"
                        value={formInputs[field.id] || ""}
                        onChange={e => setFormInputs(prev => ({ ...prev, [field.id]: e.target.value }))}
                        onKeyDown={e => e.key === "Enter" && !e.shiftKey && handleFormSubmit()}
                        placeholder={field.placeholder}
                        className="mt-1.5 w-full border border-[var(--line)] rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:border-[var(--accent)] text-[var(--ink)] placeholder:text-[var(--ink-soft)] bg-[var(--bg)]"
                      />
                    )}
                  </div>
                ))}

                {stepStates[step.id]?.error && (
                  <div className="flex items-center gap-2 text-xs text-[var(--danger)] bg-red-50 border border-red-100 rounded-lg px-3 py-2">
                    <AlertCircle size={12} />
                    {stepStates[step.id]?.error}
                  </div>
                )}

                <button
                  onClick={handleFormSubmit}
                  disabled={running}
                  className="w-full py-3 rounded-xl text-sm font-semibold text-white flex items-center justify-center gap-2 transition-opacity disabled:opacity-60 hover:opacity-90"
                  style={{ background: selected.color }}
                >
                  {running ? <Loader2 size={15} className="animate-spin" /> : <ArrowRight size={15} />}
                  {running ? "Running pipeline…" : currentStep === 0 ? `Start: ${selected.title}` : "Continue →"}
                </button>
              </div>
            </div>
          );
        })()}

        {/* Running indicator when auto-steps are executing */}
        {running && (() => {
          const step = selected.steps[currentStep];
          if (!step || step.type === "form") return null;
          const state = stepStates[step.id];
          if (state?.status !== "running" && state?.status !== "waiting") return null;

          return (
            <div className="bg-white border border-[var(--accent)]/30 rounded-2xl px-5 py-4 flex items-center gap-3">
              <Loader2 size={16} className="animate-spin shrink-0" style={{ color: selected.color }} />
              <div>
                <p className="text-sm font-medium text-[var(--ink)]">{step.title}</p>
                <p className="text-xs text-[var(--ink-soft)] animate-pulse">{step.subtitle}…</p>
              </div>
            </div>
          );
        })()}

      </div>
    </div>
  );
}

/* ─── Case card for search results ──────────────────────────── */

function CaseCard({ c, color }: { c: CaseResult; color: string }) {
  const [expanded, setExpanded] = useState(false);
  const pdfUrl = c.url || null;

  return (
    <div className="border border-[var(--line)] rounded-xl p-3 hover:border-[var(--accent)] transition-colors group">
      {/* Tier + verdict row */}
      <div className="flex items-center gap-1.5 mb-1.5 flex-wrap">
        {c.tier && (
          <span className="text-[9px] font-semibold uppercase tracking-wider border rounded px-1.5 py-0.5"
            style={{ background: `${color}12`, color, borderColor: `${color}40` }}>
            {c.tier === "SC" ? "Supreme Court" : c.tier === "HC" ? "High Court" : c.tier}
          </span>
        )}
        {c.verdict && (
          <span className="text-[9px] font-semibold uppercase tracking-wider border rounded px-1.5 py-0.5 bg-[var(--bg-elev)] text-[var(--ink-soft)] border-[var(--line)]">
            {c.verdict.length > 20 ? c.verdict.slice(0, 20) + "…" : c.verdict}
          </span>
        )}
      </div>

      {/* Title */}
      <p className="text-xs font-semibold text-[var(--ink)] leading-snug mb-1 line-clamp-2">{c.title || c.case_id}</p>

      {/* Meta */}
      <div className="flex items-center gap-1.5 text-[10px] text-[var(--ink-soft)] mb-2 flex-wrap">
        {c.court && <span>{c.court}</span>}
        {c.year && <><span>·</span><span>{c.year}</span></>}
        {c.citation && <><span>·</span><span className="font-mono text-[var(--accent)]">{c.citation}</span></>}
      </div>

      {/* Excerpt (expandable) */}
      {c.excerpt && (
        <div>
          <p className={`text-[11px] text-[var(--ink-soft)] italic leading-relaxed ${expanded ? "" : "line-clamp-2"}`}>
            "{c.excerpt}"
          </p>
          {c.excerpt.length > 120 && (
            <button onClick={() => setExpanded(e => !e)} className="text-[10px] text-[var(--accent)] mt-0.5">
              {expanded ? "Show less" : "Show more"}
            </button>
          )}
        </div>
      )}

      {/* Action row */}
      <div className="flex items-center gap-3 mt-2 pt-2 border-t border-[var(--line)]">
        {pdfUrl && (
          <>
            <a
              href={pdfUrl}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 text-[10px] text-[var(--accent)] hover:underline font-medium"
            >
              <ExternalLink size={10} /> View
            </a>
            <a
              href={`${pdfUrl}${pdfUrl.includes("?") ? "&" : "?"}download=true`}
              className="flex items-center gap-1 text-[10px] text-[var(--ink-soft)] hover:text-[var(--ink)] transition-colors"
            >
              <FileDown size={10} /> Download PDF
            </a>
          </>
        )}
        <span className="ml-auto text-[9px] text-[var(--ink-soft)] font-mono">
          {typeof c.score === "number" ? `score: ${c.score.toFixed(2)}` : ""}
        </span>
      </div>
    </div>
  );
}
