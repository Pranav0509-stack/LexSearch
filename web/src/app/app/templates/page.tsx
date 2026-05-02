"use client";

// Legal Document Templates — common Indian legal documents that lawyers
// and law students draft daily. Each template provides structure + guidance.

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  FileText,
  ArrowRight,
  Scale,
  Shield,
  AlertTriangle,
  FileCheck,
  Gavel,
  BookOpen,
  Search,
  Sparkles,
} from "lucide-react";

interface Template {
  id: string;
  title: string;
  category: string;
  description: string;
  icon: React.ReactNode;
  prompt: string;
  fields: string[];
}

const CATEGORIES = [
  { id: "all", label: "All Templates" },
  { id: "criminal", label: "Criminal" },
  { id: "civil", label: "Civil" },
  { id: "constitutional", label: "Constitutional" },
  { id: "corporate", label: "Corporate" },
  { id: "family", label: "Family" },
  { id: "consumer", label: "Consumer" },
  { id: "student", label: "For Students" },
];

const TEMPLATES: Template[] = [
  {
    id: "bail-application",
    title: "Bail Application",
    category: "criminal",
    description: "Application for regular/anticipatory bail under CrPC/BNSS with grounds, prayer, and supporting arguments.",
    icon: <Shield size={20} />,
    prompt: "Draft a bail application for the following case. Include: court header, case details, grounds for bail (citing relevant precedents from our database), undertakings, and prayer clause.",
    fields: ["Court Name", "Case Number", "Accused Name", "Offence Sections", "Brief Facts", "Grounds for Bail"],
  },
  {
    id: "writ-petition",
    title: "Writ Petition (Art. 226/32)",
    category: "constitutional",
    description: "Petition under Article 226 or 32 seeking certiorari, mandamus, habeas corpus, prohibition, or quo warranto.",
    icon: <Scale size={20} />,
    prompt: "Draft a writ petition under Article 226 of the Constitution. Include: cause title, synopsis & list of dates, grounds, prayer, and relevant constitutional provisions.",
    fields: ["High Court", "Petitioner Details", "Respondent (Authority)", "Fundamental Right Violated", "Relief Sought", "Brief Facts"],
  },
  {
    id: "legal-notice",
    title: "Legal Notice (S.80 CPC)",
    category: "civil",
    description: "Pre-litigation legal notice under Section 80 CPC to government bodies, or general legal notice for demand/cease-and-desist.",
    icon: <AlertTriangle size={20} />,
    prompt: "Draft a legal notice. Include: sender/recipient details, subject, factual background, legal grounds, demand/relief sought, and time limit for response.",
    fields: ["Sender Name & Address", "Recipient Name & Address", "Subject", "Facts", "Legal Grounds", "Demand/Relief"],
  },
  {
    id: "cheque-bounce-complaint",
    title: "S.138 NI Act Complaint",
    category: "criminal",
    description: "Complaint for cheque dishonour under Section 138 of Negotiable Instruments Act, 1881.",
    icon: <FileCheck size={20} />,
    prompt: "Draft a complaint under Section 138 of the Negotiable Instruments Act. Include: complainant/accused details, cheque details, dishonour memo, legal notice sent, and prayer.",
    fields: ["Complainant Details", "Accused Details", "Cheque Number & Amount", "Date of Dishonour", "Reason for Dishonour", "Legal Notice Date"],
  },
  {
    id: "consumer-complaint",
    title: "Consumer Complaint",
    category: "consumer",
    description: "Complaint before Consumer Disputes Redressal Commission under Consumer Protection Act, 2019.",
    icon: <Shield size={20} />,
    prompt: "Draft a consumer complaint. Include: complainant details, opposite party details, deficiency in service/defect in goods, relief claimed, and supporting documents.",
    fields: ["Complainant", "Opposite Party", "Product/Service", "Deficiency/Defect", "Amount Claimed", "Date of Purchase"],
  },
  {
    id: "divorce-petition",
    title: "Divorce Petition",
    category: "family",
    description: "Petition for divorce under Hindu Marriage Act, 1955 or Special Marriage Act, 1954.",
    icon: <Gavel size={20} />,
    prompt: "Draft a divorce petition. Include: court header, parties' details, marriage details, grounds for divorce under the applicable Act, prayer for dissolution and ancillary reliefs.",
    fields: ["Court", "Petitioner Details", "Respondent Details", "Date of Marriage", "Grounds for Divorce", "Children (if any)"],
  },
  {
    id: "rti-application",
    title: "RTI Application",
    category: "constitutional",
    description: "Application under Right to Information Act, 2005 seeking information from public authorities.",
    icon: <BookOpen size={20} />,
    prompt: "Draft an RTI application under the Right to Information Act, 2005. Include: public authority name, information sought (specific questions), fee details, and applicant declaration.",
    fields: ["Public Authority", "Subject", "Information Sought (list specific questions)", "Period of Information", "Preferred Format"],
  },
  {
    id: "internship-memo",
    title: "Legal Research Memo",
    category: "student",
    description: "Research memorandum format for law students — issue, rule, application, conclusion (IRAC method).",
    icon: <BookOpen size={20} />,
    prompt: "Draft a legal research memorandum using IRAC method. Include: issue statement, applicable rules/statutes, analysis with case law citations from our database, and conclusion with recommendation.",
    fields: ["Legal Issue/Question", "Relevant Statutes", "Jurisdiction", "Key Facts", "Client's Position"],
  },
  {
    id: "moot-court-memorial",
    title: "Moot Court Memorial",
    category: "student",
    description: "Memorial format for moot court competitions — statement of jurisdiction, facts, issues, arguments, prayer.",
    icon: <Gavel size={20} />,
    prompt: "Draft a moot court memorial. Include: cover page, table of contents, index of authorities, statement of jurisdiction, statement of facts, issues raised, summary of arguments, arguments advanced (with citations), and prayer.",
    fields: ["Competition Name", "Team Number", "Side (Petitioner/Respondent)", "Problem Statement", "Issues Framed"],
  },
  {
    id: "case-brief",
    title: "Case Brief / Analysis",
    category: "student",
    description: "Structured case brief for academic study — facts, issues, held, reasoning, significance.",
    icon: <FileText size={20} />,
    prompt: "Create a comprehensive case brief. Include: case citation, facts, procedural history, issues, rule of law, court's analysis/reasoning, holding, and significance/impact of the decision.",
    fields: ["Case Name", "Court", "Year", "Key Legal Issue"],
  },
];

export default function TemplatesPage() {
  const router = useRouter();
  const [category, setCategory] = useState("all");
  const [search, setSearch] = useState("");

  const filtered = TEMPLATES.filter((t) => {
    if (category !== "all" && t.category !== category) return false;
    if (search && !t.title.toLowerCase().includes(search.toLowerCase()) && !t.description.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  const useTemplate = (template: Template) => {
    // Navigate to assistant with the template prompt pre-loaded
    sessionStorage.setItem(
      "sanhita.caseContext",
      JSON.stringify({
        case_id: `template-${template.id}`,
        title: template.title,
        body_md: `${template.prompt}\n\n**Required fields:**\n${template.fields.map((f) => `- ${f}: [fill in]`).join("\n")}`,
        jurisdiction: "IN",
      })
    );
    router.push("/app");
  };

  return (
    <div className="flex-1 overflow-y-auto">
      {/* Header */}
      <div className="px-4 sm:px-6 lg:px-12 pt-6 pb-4 border-b border-[var(--line)]">
        <div className="flex items-center gap-2 text-[10px] tracking-[0.22em] uppercase text-[var(--ink-soft)]">
          <FileText size={11} className="text-[var(--accent)]" />
          <span>Legal Templates</span>
          <span className="ml-auto text-xs normal-case tracking-normal">
            {TEMPLATES.length} templates
          </span>
        </div>
        <h2 className="mt-1.5 font-display text-2xl tracking-tight">
          Draft Legal Documents
        </h2>
        <p className="text-xs text-[var(--ink-soft)] mt-1 max-w-xl">
          Professional Indian legal document templates. Select a template, fill in your details,
          and Sanhita will draft a complete document citing real case law from 31.9M judgments.
        </p>

        {/* Search */}
        <div className="mt-4 flex items-center gap-2">
          <div className="flex items-center gap-2 flex-1 min-w-0 bg-[var(--bg-elev)] border border-[var(--line)] rounded-xl px-3 py-2 hover:border-[var(--accent-soft)] focus-within:border-[var(--accent)] transition-colors">
            <Search size={15} className="text-[var(--ink-soft)] shrink-0" />
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search templates..."
              className="bg-transparent outline-none text-sm flex-1 min-w-0"
            />
          </div>
        </div>

        {/* Category tabs */}
        <div className="mt-3 flex items-center gap-1.5 flex-wrap">
          {CATEGORIES.map((c) => (
            <button
              key={c.id}
              onClick={() => setCategory(c.id)}
              className={`px-3 py-1.5 rounded-full text-xs transition-colors ${
                category === c.id
                  ? "bg-[var(--ink)] text-[var(--bg)]"
                  : "bg-[var(--bg-elev)] border border-[var(--line)] text-[var(--ink-soft)] hover:border-[var(--accent-soft)]"
              }`}
            >
              {c.label}
            </button>
          ))}
        </div>
      </div>

      {/* Template grid */}
      <div className="px-4 sm:px-6 lg:px-12 py-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 max-w-5xl">
          {filtered.map((t) => (
            <button
              key={t.id}
              onClick={() => useTemplate(t)}
              className="text-left bg-[var(--bg-elev)] border border-[var(--line)] rounded-2xl p-5 hover:border-[var(--accent-soft)] hover:shadow-sm transition-all group"
            >
              <div className="flex items-start gap-3">
                <div className="p-2 rounded-lg bg-[var(--bg)] border border-[var(--line)] text-[var(--accent)] shrink-0">
                  {t.icon}
                </div>
                <div className="min-w-0 flex-1">
                  <h3 className="font-medium text-sm text-[var(--ink)] group-hover:text-[var(--accent)] transition-colors">
                    {t.title}
                  </h3>
                  <span className="text-[9px] uppercase tracking-wider text-[var(--ink-soft)] mt-0.5 block">
                    {t.category}
                  </span>
                </div>
                <ArrowRight size={14} className="text-[var(--ink-soft)] group-hover:text-[var(--accent)] mt-1 shrink-0 transition-colors" />
              </div>
              <p className="text-xs text-[var(--ink-soft)] mt-3 leading-relaxed line-clamp-2">
                {t.description}
              </p>
              <div className="mt-3 flex flex-wrap gap-1">
                {t.fields.slice(0, 3).map((f) => (
                  <span key={f} className="text-[9px] px-1.5 py-0.5 rounded bg-[var(--bg)] border border-[var(--line)] text-[var(--ink-soft)]">
                    {f}
                  </span>
                ))}
                {t.fields.length > 3 && (
                  <span className="text-[9px] px-1.5 py-0.5 text-[var(--ink-soft)]">
                    +{t.fields.length - 3} more
                  </span>
                )}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
