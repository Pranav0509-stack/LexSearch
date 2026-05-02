"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useEditor, EditorContent } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import { Table } from "@tiptap/extension-table";
import TableRow from "@tiptap/extension-table-row";
import TableCell from "@tiptap/extension-table-cell";
import TableHeader from "@tiptap/extension-table-header";
import Image from "@tiptap/extension-image";
import Link from "@tiptap/extension-link";
import TextAlign from "@tiptap/extension-text-align";
import Color from "@tiptap/extension-color";
import { TextStyle } from "@tiptap/extension-text-style";
import Highlight from "@tiptap/extension-highlight";
import Underline from "@tiptap/extension-underline";
import TaskList from "@tiptap/extension-task-list";
import TaskItem from "@tiptap/extension-task-item";
import Subscript from "@tiptap/extension-subscript";
import Superscript from "@tiptap/extension-superscript";
import CharacterCount from "@tiptap/extension-character-count";
import Placeholder from "@tiptap/extension-placeholder";
import Typography from "@tiptap/extension-typography";
import Focus from "@tiptap/extension-focus";
import HorizontalRule from "@tiptap/extension-horizontal-rule";
import { FontFamily } from "@tiptap/extension-font-family";

// ── Types ──────────────────────────────────────────────────────────────────
interface LegalDoc {
  id: number;
  title: string;
  doc_type: string;
  content: string;
  word_count: number;
  updated_at: number;
}

interface DocType {
  id: string;
  label: string;
  description: string;
  icon: string;
}

// ── API helpers ────────────────────────────────────────────────────────────
const api = async (path: string, opts?: RequestInit) => {
  const res = await fetch(path, { credentials: "include", ...opts });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json();
};
const post = (path: string, body: unknown) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const put = (path: string, body: unknown) =>
  api(path, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const del = (path: string) => api(path, { method: "DELETE" });

const fmtDate = (ts: number) =>
  new Intl.DateTimeFormat("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }).format(new Date(ts * 1000));

// ── Font sizes ─────────────────────────────────────────────────────────────
const FONT_SIZES = ["8", "9", "10", "11", "12", "14", "16", "18", "20", "22", "24", "26", "28", "36", "48", "72"];
const FONT_FAMILIES = [
  { label: "Arial", value: "Arial, sans-serif" },
  { label: "Georgia", value: "Georgia, serif" },
  { label: "Times New Roman", value: "'Times New Roman', Times, serif" },
  { label: "Courier New", value: "'Courier New', Courier, monospace" },
  { label: "Verdana", value: "Verdana, sans-serif" },
  { label: "Garamond", value: "Garamond, serif" },
];

// ── FontSize extension ─────────────────────────────────────────────────────
import { Extension } from "@tiptap/core";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const FontSize = Extension.create<any>({
  name: "fontSize",
  addOptions() { return { types: ["textStyle"] }; },
  addGlobalAttributes() {
    return [{
      types: this.options.types,
      attributes: {
        fontSize: {
          default: null,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          parseHTML: (el: HTMLElement) => el.style.fontSize?.replace(/['"]+/g, "") || null,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          renderHTML: (attrs: any) => attrs.fontSize ? { style: `font-size: ${attrs.fontSize}` } : {},
        },
      },
    }];
  },
  addCommands() {
    return {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      setFontSize: (fontSize: string) => ({ chain }: any) =>
        chain().setMark("textStyle", { fontSize: `${fontSize}pt` }).run(),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      unsetFontSize: () => ({ chain }: any) =>
        chain().setMark("textStyle", { fontSize: null }).removeEmptyTextStyle().run(),
    };
  },
});

// ── LineHeight extension ────────────────────────────────────────────────────
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const LineHeight = Extension.create<any>({
  name: "lineHeight",
  addOptions() { return { types: ["paragraph", "heading"], defaultLineHeight: "normal" }; },
  addGlobalAttributes() {
    return [{
      types: this.options.types,
      attributes: {
        lineHeight: {
          default: this.options.defaultLineHeight,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          parseHTML: (el: HTMLElement) => el.style.lineHeight || this.options.defaultLineHeight,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          renderHTML: (attrs: any) =>
            attrs.lineHeight && attrs.lineHeight !== this.options.defaultLineHeight
              ? { style: `line-height: ${attrs.lineHeight}` } : {},
        },
      },
    }];
  },
  addCommands() {
    return {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      setLineHeight: (lineHeight: string) => ({ tr, state, dispatch }: any) => {
        const { selection } = state;
        if (dispatch) {
          selection.ranges.forEach(({ $from, $to }: { $from: { pos: number }; $to: { pos: number } }) => {
            state.doc.nodesBetween($from.pos, $to.pos, (node: { isBlock: boolean; attrs: Record<string, unknown> }, pos: number) => {
              if (node.isBlock) {
                tr.setNodeMarkup(pos, undefined, { ...node.attrs, lineHeight });
              }
            });
          });
          dispatch(tr);
        }
        return true;
      },
    };
  },
});

// ─────────────────────────────────────────────────────────────────────────────
// EditorPane — Google Docs-style full-featured legal document editor
// ─────────────────────────────────────────────────────────────────────────────
export default function EditorPane() {
  const [docs, setDocs] = useState<LegalDoc[]>([]);
  const [activeDoc, setActiveDoc] = useState<LegalDoc | null>(null);
  const [docTypes, setDocTypes] = useState<DocType[]>([]);
  const [showNewModal, setShowNewModal] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [newType, setNewType] = useState("general");
  const [saveStatus, setSaveStatus] = useState<"saved" | "unsaved" | "saving">("saved");
  const [fontSize, setFontSize] = useState("12");
  const [fontFamily, setFontFamily] = useState("Georgia, serif");
  const [showAiPanel, setShowAiPanel] = useState(false);
  const [aiMode, setAiMode] = useState<"complete" | "improve" | "cases" | "write">("complete");
  const [aiLoading, setAiLoading] = useState(false);
  const [aiResult, setAiResult] = useState("");
  const [aiQuery, setAiQuery] = useState("");
  const [aiCases, setAiCases] = useState<Array<{ case_id: string; title: string; court: string; year: number | null; citation: string; excerpt: string }>>([]);
  const [showLinkModal, setShowLinkModal] = useState(false);
  const [linkUrl, setLinkUrl] = useState("");
  const [showImageModal, setShowImageModal] = useState(false);
  const [imageUrl, setImageUrl] = useState("");
  const [showFindReplace, setShowFindReplace] = useState(false);
  const [findText, setFindText] = useState("");
  const [replaceText, setReplaceText] = useState("");
  const [showTableMenu, setShowTableMenu] = useState(false);
  const [zoom, setZoom] = useState(100);
  const [showComments, setShowComments] = useState(false);
  const [showClausePanel, setShowClausePanel] = useState(false);
  const [clauses, setClauses] = useState<Array<{id: string; category: string; label: string}>>([]);
  const [clauseSearch, setClauseSearch] = useState("");
  const [lineHeight, setLineHeight] = useState("1.5");
  const [marginSize, setMarginSize] = useState<"normal" | "narrow" | "wide">("normal");
  const [bubbleMenuPos, setBubbleMenuPos] = useState<{ x: number; y: number } | null>(null);
  const [hasSelection, setHasSelection] = useState(false);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const titleRef = useRef<HTMLInputElement>(null);
  const editorWrapRef = useRef<HTMLDivElement>(null);

  // ── Tiptap editor ────────────────────────────────────────────────────────
  const editor = useEditor({
    immediatelyRender: false,   // required in Next.js to avoid SSR hydration mismatch
    extensions: [
      StarterKit.configure({ codeBlock: false }),
      Underline,
      TextStyle,
      Color,
      FontSize,
      FontFamily,
      LineHeight,
      Highlight.configure({ multicolor: true }),
      TextAlign.configure({ types: ["heading", "paragraph"] }),
      Link.configure({ openOnClick: false, HTMLAttributes: { class: "editor-link" } }),
      Image.configure({ inline: false, allowBase64: true }),
      Table.configure({ resizable: true }),
      TableRow,
      TableHeader,
      TableCell,
      TaskList,
      TaskItem.configure({ nested: true }),
      Subscript,
      Superscript,
      HorizontalRule,
      Typography,
      Focus.configure({ className: "has-focus" }),
      CharacterCount,
      Placeholder.configure({
        placeholder: "Start writing your document…",
        emptyEditorClass: "is-editor-empty",
      }),
    ],
    content: "",
    editorProps: {
      attributes: {
        class: "tiptap-doc",
        spellcheck: "true",
      },
    },
    onUpdate: ({ editor: e }) => {
      if (!activeDoc) return;
      setSaveStatus("unsaved");
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
      saveTimerRef.current = setTimeout(() => {
        saveDoc(e.getHTML());
      }, 2000);
    },
    onSelectionUpdate: ({ editor: e }) => {
      const { from, to } = e.state.selection;
      if (from === to) {
        setHasSelection(false);
        setBubbleMenuPos(null);
        return;
      }
      setHasSelection(true);
      // Position bubble menu above the selection
      const domSel = window.getSelection();
      if (domSel && domSel.rangeCount > 0) {
        const range = domSel.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        const wrap = editorWrapRef.current;
        if (wrap) {
          const wrapRect = wrap.getBoundingClientRect();
          setBubbleMenuPos({
            x: rect.left + rect.width / 2 - wrapRect.left,
            y: rect.top - wrapRect.top - 8,
          });
        }
      }
    },
  });

  // Apply font-family when changed
  useEffect(() => {
    if (!editor) return;
    editor.chain().focus().setFontFamily(fontFamily).run();
  }, [fontFamily, editor]);

  // ── Load docs + doc types ─────────────────────────────────────────────────
  useEffect(() => {
    api("/api/editor/docs").then(d => setDocs(d.documents || [])).catch(() => {});
    api("/api/editor/doc-types").then(d => setDocTypes(d.doc_types || [])).catch(() => {});
  }, []);

  // Load legal clauses when panel opens
  useEffect(() => {
    if (showClausePanel && clauses.length === 0) {
      api("/api/editor/clauses").then(d => setClauses(d.clauses || [])).catch(() => {});
    }
  }, [showClausePanel]); // eslint-disable-line react-hooks/exhaustive-deps

  const insertClause = useCallback(async (clauseId: string) => {
    if (!editor) return;
    try {
      const c = await api(`/api/editor/clauses/${clauseId}`);
      if (c.text) {
        const html = c.text.split("\n").map((l: string) => {
          if (!l.trim()) return "";
          return `<p>${l}</p>`;
        }).filter((l: string) => l).join("");
        editor.chain().focus().insertContent(html).run();
        setSaveStatus("unsaved");
      }
    } catch (e) { console.error(e); }
  }, [editor]);

  // When activeDoc changes (e.g. after create), push its content into the editor.
  // We can't do this inside openDoc because the editor might not be mounted yet.
  useEffect(() => {
    if (!activeDoc || !editor) return;
    editor.commands.setContent(activeDoc.content || "");
  }, [activeDoc?.id, editor]); // eslint-disable-line react-hooks/exhaustive-deps

  const openDoc = useCallback(async (doc: LegalDoc) => {
    // Fetch full content (doc_list only returns metadata)
    try {
      const full = await api(`/api/editor/docs/${doc.id}`);
      setActiveDoc({ ...doc, ...full.doc });
    } catch {
      setActiveDoc(doc);
    }
    setSaveStatus("saved");
  }, []);

  const createDoc = useCallback(async () => {
    try {
      let tmplContent = "";
      if (newType !== "general") {
        try { const t = await api(`/api/editor/template/${newType}`); tmplContent = `<p>${t.template || ""}</p>`; } catch {}
      }
      const d = await post("/api/editor/docs", { title: newTitle || "Untitled Document", doc_type: newType, content: tmplContent });
      const newDoc: LegalDoc = d.doc;
      setDocs(prev => [newDoc, ...prev]);
      setShowNewModal(false);
      setNewTitle("");
      setNewType("general");
      openDoc(newDoc);
    } catch (e) { console.error(e); }
  }, [newTitle, newType, openDoc]);

  // ── Import draft from Workflows pane via sessionStorage ──────────────────
  // When user clicks "Open in Draft Editor" in Workflows, the markdown content
  // is stored in sessionStorage. We pick it up here and create a new document.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const content = window.sessionStorage.getItem("editor_draft_content");
    const title = window.sessionStorage.getItem("editor_draft_title");
    if (!content) return;
    window.sessionStorage.removeItem("editor_draft_content");
    window.sessionStorage.removeItem("editor_draft_title");
    // Convert markdown to HTML for TipTap
    const lines = content.split("\n");
    const htmlLines = lines.map(line => {
      if (line.startsWith("# ")) return `<h1>${line.slice(2)}</h1>`;
      if (line.startsWith("## ")) return `<h2>${line.slice(3)}</h2>`;
      if (line.startsWith("### ")) return `<h3>${line.slice(4)}</h3>`;
      if (line.startsWith("#### ")) return `<h4>${line.slice(5)}</h4>`;
      if (line.trim() === "") return "";
      const boldLine = line.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\*(.+?)\*/g, "<em>$1</em>");
      return `<p>${boldLine}</p>`;
    });
    const htmlContent = htmlLines.filter(l => l !== "").join("\n");
    post("/api/editor/docs", {
      title: title || "Workflow Draft",
      doc_type: "general",
      content: htmlContent,
    }).then(d => {
      const newDoc: LegalDoc = d.doc;
      setDocs(prev => [newDoc, ...prev]);
      openDoc(newDoc);
    }).catch(console.error);
  }, [openDoc]); // eslint-disable-line react-hooks/exhaustive-deps

  const saveDoc = useCallback(async (htmlContent?: string) => {
    if (!activeDoc || !editor) return;
    const content = htmlContent ?? editor.getHTML();
    setSaveStatus("saving");
    try {
      // Send title so backend can keep it in sync
      const d = await put(`/api/editor/docs/${activeDoc.id}`, { title: activeDoc.title, content });
      setActiveDoc(prev => prev ? { ...prev, ...d.doc } : null);
      setDocs(prev => prev.map(doc => doc.id === activeDoc.id ? { ...doc, ...d.doc } : doc));
      setSaveStatus("saved");
    } catch { setSaveStatus("unsaved"); }
  }, [activeDoc, editor]);

  const deleteDoc = useCallback(async (id: number) => {
    if (!confirm("Delete this document? This cannot be undone.")) return;
    await del(`/api/editor/docs/${id}`);
    setDocs(prev => prev.filter(d => d.id !== id));
    if (activeDoc?.id === id) { setActiveDoc(null); editor?.commands.clearContent(); }
  }, [activeDoc, editor]);

  const renameDoc = useCallback(async (title: string) => {
    if (!activeDoc || !title.trim()) return;
    const content = editor?.getHTML() ?? activeDoc.content ?? "";
    await put(`/api/editor/docs/${activeDoc.id}`, { title, content });
    setActiveDoc(prev => prev ? { ...prev, title } : null);
    setDocs(prev => prev.map(doc => doc.id === activeDoc.id ? { ...doc, title } : doc));
  }, [activeDoc, editor]);

  // ── AI actions ───────────────────────────────────────────────────────────
  const runAI = useCallback(async () => {
    if (!editor) return;
    setAiLoading(true);
    setAiResult("");
    setAiCases([]);
    try {
      const selected = editor.state.selection.empty ? "" : editor.state.doc.textBetween(editor.state.selection.from, editor.state.selection.to);
      const fullText = editor.getText().slice(0, 2000);
      if (aiMode === "complete") {
        const d = await post("/api/editor/ai/complete", { content: fullText, doc_type: activeDoc?.doc_type ?? "general", cursor_text: selected });
        setAiResult(d.completion || "");
      } else if (aiMode === "improve") {
        const d = await post("/api/editor/ai/improve", { selected_text: selected || fullText.slice(0, 600), doc_type: activeDoc?.doc_type ?? "general" });
        setAiResult(d.improved || "");
      } else if (aiMode === "write") {
        const d = await post("/api/editor/ai/write-section", { instruction: aiQuery, doc_type: activeDoc?.doc_type ?? "general", context: fullText.slice(0, 800) });
        setAiResult(d.text || "");
      } else if (aiMode === "cases") {
        const d = await post("/api/editor/ai/suggest-cases", { argument: aiQuery || fullText.slice(0, 400), doc_type: activeDoc?.doc_type ?? "general" });
        setAiCases(d.cases || []);
        setAiResult(d.suggested_query || "");
      }
    } catch (e) { setAiResult("Error: " + (e as Error).message); }
    setAiLoading(false);
  }, [editor, aiMode, aiQuery, activeDoc]);

  const insertAiResult = useCallback(() => {
    if (!editor || !aiResult) return;
    editor.chain().focus().insertContent(aiResult).run();
    setAiResult("");
  }, [editor, aiResult]);

  const insertCitation = useCallback((c: { case_id: string; title: string; court: string; year: number | null; citation: string }) => {
    if (!editor) return;
    const cite = `<p><em>[${c.title}${c.citation ? `, ${c.citation}` : ""}${c.court ? `, ${c.court}` : ""}${c.year ? ` (${c.year})` : ""}]</em></p>`;
    editor.chain().focus().insertContent(cite).run();
  }, [editor]);

  // ── Find & Replace ────────────────────────────────────────────────────────
  const findAndReplace = useCallback(() => {
    if (!editor || !findText) return;
    const html = editor.getHTML();
    const escaped = findText.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const replaced = html.replace(new RegExp(escaped, "gi"), replaceText);
    editor.commands.setContent(replaced);
  }, [editor, findText, replaceText]);

  // ── Insert table ──────────────────────────────────────────────────────────
  const insertTable = useCallback((rows: number, cols: number) => {
    editor?.chain().focus().insertTable({ rows, cols, withHeaderRow: true }).run();
    setShowTableMenu(false);
  }, [editor]);

  // ── Export ────────────────────────────────────────────────────────────────
  const exportDoc = useCallback((format: "html" | "txt" | "md") => {
    if (!editor || !activeDoc) return;
    let content = "", mime = "text/plain", ext = "txt";
    if (format === "html") { content = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>${activeDoc.title}</title></head><body>${editor.getHTML()}</body></html>`; mime = "text/html"; ext = "html"; }
    else if (format === "txt") { content = editor.getText(); ext = "txt"; }
    else { content = editor.getHTML().replace(/<[^>]+>/g, ""); ext = "md"; }
    const blob = new Blob([content], { type: `${mime};charset=utf-8` });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `${activeDoc.title}.${ext}`; a.click(); URL.revokeObjectURL(a.href);
  }, [editor, activeDoc]);

  // ── Keyboard shortcuts ─────────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey)) {
        if (e.key === "s") { e.preventDefault(); saveDoc(); }
        if (e.key === "h") { e.preventDefault(); setShowFindReplace(v => !v); }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [saveDoc]);

  // ── Toolbar helpers ───────────────────────────────────────────────────────
  const isActive = (name: string, attrs?: Record<string, unknown>) => editor?.isActive(name, attrs) ?? false;
  const btn = (active: boolean) =>
    `toolbar-btn ${active ? "toolbar-btn-active" : ""}`;

  const marginClass = marginSize === "narrow" ? "mx-auto max-w-[700px] px-[48px]" : marginSize === "wide" ? "mx-auto max-w-[900px] px-[96px]" : "mx-auto max-w-[816px] px-[72px]";

  // ── Table hover grid ──────────────────────────────────────────────────────
  const [tableHover, setTableHover] = useState({ r: 0, c: 0 });
  const TABLE_GRID = { rows: 8, cols: 10 };

  if (!activeDoc) {
    // ── Document list / home ────────────────────────────────────────────────
    return (
      <div className="flex flex-col h-full bg-[#f8f9fa] overflow-hidden">
        {/* Header */}
        <div className="bg-white border-b border-[var(--line)] px-8 py-5 flex items-center justify-between">
          <div>
            <h2 className="font-display text-xl tracking-tight text-[var(--ink)]">Documents</h2>
            <p className="text-xs text-[var(--ink-soft)] mt-0.5">Your legal drafts & filings</p>
          </div>
          <button
            onClick={() => setShowNewModal(true)}
            className="flex items-center gap-2 bg-[#1a73e8] hover:bg-[#1557b0] text-white text-sm font-medium px-4 py-2 rounded-md transition-colors"
          >
            <span className="text-base leading-none">+</span> New document
          </button>
        </div>

        {/* Doc grid */}
        <div className="flex-1 overflow-y-auto px-8 py-6">
          {docs.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
              <div className="text-6xl">📄</div>
              <p className="text-lg font-medium text-[var(--ink)]">No documents yet</p>
              <p className="text-sm text-[var(--ink-soft)]">Create your first legal draft</p>
              <button onClick={() => setShowNewModal(true)} className="mt-2 px-5 py-2 bg-[#1a73e8] text-white rounded-md text-sm font-medium">New document</button>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {docs.map(doc => (
                <div key={doc.id} className="group flex flex-col gap-2 cursor-pointer" onClick={() => openDoc(doc)}>
                  <div className="aspect-[3/4] bg-white border border-[var(--line)] rounded-lg shadow-sm group-hover:shadow-md group-hover:border-[#1a73e8] transition-all overflow-hidden relative">
                    <div className="p-3 text-[9px] leading-relaxed text-[var(--ink-soft)] overflow-hidden h-full" dangerouslySetInnerHTML={{ __html: doc.content?.slice(0, 400) || "<p class='text-gray-400 italic'>Empty document</p>" }} />
                    <div className="absolute inset-0 bg-gradient-to-b from-transparent to-white/80 pointer-events-none" />
                    <button
                      onClick={e => { e.stopPropagation(); deleteDoc(doc.id); }}
                      className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-white border border-[var(--line)] rounded p-1 text-[var(--danger)] hover:bg-red-50 text-xs"
                    >✕</button>
                  </div>
                  <div className="px-1">
                    <p className="text-xs font-medium text-[var(--ink)] truncate">{doc.title}</p>
                    <p className="text-[10px] text-[var(--ink-soft)]">{fmtDate(doc.updated_at)}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* New document modal */}
        {showNewModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={() => setShowNewModal(false)}>
            <div className="bg-white rounded-xl shadow-2xl w-[520px] max-h-[80vh] overflow-hidden flex flex-col" onClick={e => e.stopPropagation()}>
              <div className="px-6 py-5 border-b border-[var(--line)]">
                <h3 className="font-display text-lg">New document</h3>
              </div>
              <div className="px-6 py-4 flex flex-col gap-4 flex-1 overflow-y-auto">
                <div>
                  <label className="text-xs font-medium text-[var(--ink-soft)] uppercase tracking-wider">Title</label>
                  <input
                    autoFocus
                    value={newTitle}
                    onChange={e => setNewTitle(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && createDoc()}
                    placeholder="e.g. Bail Application — State vs. Sharma"
                    className="mt-1.5 w-full border border-[var(--line)] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-[#1a73e8]"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-[var(--ink-soft)] uppercase tracking-wider">Document type</label>
                  <div className="mt-2 grid grid-cols-2 gap-2">
                    {[{ id: "general", label: "General", description: "Blank document", icon: "📄" }, ...docTypes].map(t => (
                      <button
                        key={t.id}
                        onClick={() => setNewType(t.id)}
                        className={`flex items-center gap-3 p-3 border rounded-lg text-left text-sm transition-all ${newType === t.id ? "border-[#1a73e8] bg-[#e8f0fe]" : "border-[var(--line)] hover:border-[#1a73e8]/50"}`}
                      >
                        <span className="text-xl">{t.icon}</span>
                        <div><p className="font-medium text-xs">{t.label}</p><p className="text-[10px] text-[var(--ink-soft)] truncate">{t.description}</p></div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
              <div className="px-6 py-4 border-t border-[var(--line)] flex justify-end gap-3">
                <button onClick={() => setShowNewModal(false)} className="px-4 py-2 text-sm text-[var(--ink-soft)] hover:text-[var(--ink)]">Cancel</button>
                <button onClick={createDoc} className="px-5 py-2 bg-[#1a73e8] hover:bg-[#1557b0] text-white text-sm font-medium rounded-md">Create</button>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // ── Full editor view ──────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-full bg-[#f8f9fa] overflow-hidden">

      {/* ── Menu bar (Google Docs style) ──────────────────────────────────── */}
      <div className="bg-white border-b border-[#e0e0e0] shrink-0">
        {/* Title row */}
        <div className="flex items-center gap-3 px-4 pt-3 pb-1">
          <button
            onClick={() => setActiveDoc(null)}
            className="text-[#4285f4] hover:bg-[#e8f0fe] rounded p-1 transition-colors"
            title="Back to documents"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
          </button>
          <div className="flex-1 flex items-center gap-3 min-w-0">
            <input
              ref={titleRef}
              defaultValue={activeDoc.title}
              onBlur={e => renameDoc(e.target.value)}
              onKeyDown={e => e.key === "Enter" && titleRef.current?.blur()}
              className="text-lg font-medium text-[#202124] bg-transparent border-b-2 border-transparent focus:border-[#1a73e8] focus:outline-none px-1 py-0.5 truncate max-w-[400px]"
            />
            <span className={`text-xs px-2 py-0.5 rounded-full ${saveStatus === "saved" ? "text-[#1e8e3e] bg-[#e6f4ea]" : saveStatus === "saving" ? "text-[#b5770d] bg-[#fef7e0]" : "text-[#d93025] bg-[#fce8e6]"}`}>
              {saveStatus === "saved" ? "✓ Saved" : saveStatus === "saving" ? "Saving…" : "● Unsaved"}
            </span>
          </div>
          <div className="flex items-center gap-1 shrink-0">
            <button onClick={() => setShowAiPanel(v => !v)} className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border transition-all ${showAiPanel ? "bg-[#1a73e8] text-white border-[#1a73e8]" : "border-[#dadce0] text-[#444746] hover:bg-[#f6f8fe]"}`}>
              ✨ AI Assist
            </button>
            <div className="relative">
              <button onClick={() => { }} className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border border-[#dadce0] text-[#444746] hover:bg-[#f8f9fa] transition-all">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                Export
              </button>
              {/* Inline export dropdown on hover */}
            </div>
            <div className="flex items-center gap-1">
              <button onClick={() => exportDoc("html")} title="Export HTML" className="text-[10px] px-2 py-1 border border-[#dadce0] rounded hover:bg-[#f8f9fa] text-[#444746]">.html</button>
              <button onClick={() => exportDoc("txt")} title="Export TXT" className="text-[10px] px-2 py-1 border border-[#dadce0] rounded hover:bg-[#f8f9fa] text-[#444746]">.txt</button>
            </div>
          </div>
        </div>

        {/* Menu items row */}
        <div className="flex items-center gap-0 px-4 text-[13px] text-[#444746]">
          {[
            { label: "File", items: [
              { label: "New document", action: () => setShowNewModal(true) },
              { label: "Rename", action: () => titleRef.current?.focus() },
              { label: "Download (.html)", action: () => exportDoc("html") },
              { label: "Download (.txt)", action: () => exportDoc("txt") },
              { label: "Delete document", action: () => deleteDoc(activeDoc.id) },
            ]},
            { label: "Edit", items: [
              { label: "Undo (⌘Z)", action: () => editor?.chain().focus().undo().run() },
              { label: "Redo (⌘⇧Z)", action: () => editor?.chain().focus().redo().run() },
              { label: "Find & replace (⌘H)", action: () => setShowFindReplace(v => !v) },
              { label: "Select all", action: () => editor?.chain().focus().selectAll().run() },
            ]},
            { label: "View", items: [
              { label: "Zoom in", action: () => setZoom(z => Math.min(z + 10, 200)) },
              { label: "Zoom out", action: () => setZoom(z => Math.max(z - 10, 50)) },
              { label: "100%", action: () => setZoom(100) },
              { label: "Show AI panel", action: () => setShowAiPanel(v => !v) },
              { label: "Find & replace", action: () => setShowFindReplace(v => !v) },
            ]},
            { label: "Insert", items: [
              { label: "⚖️ Legal Clause Library", action: () => setShowClausePanel(v => !v) },
              { label: "Image from URL", action: () => setShowImageModal(true) },
              { label: "Link", action: () => setShowLinkModal(true) },
              { label: "Table", action: () => setShowTableMenu(v => !v) },
              { label: "Horizontal rule", action: () => editor?.chain().focus().setHorizontalRule().run() },
              { label: "Page break", action: () => editor?.chain().focus().insertContent("<hr class='page-break' />").run() },
            ]},
            { label: "Format", items: [
              { label: "Bold", action: () => editor?.chain().focus().toggleBold().run() },
              { label: "Italic", action: () => editor?.chain().focus().toggleItalic().run() },
              { label: "Underline", action: () => editor?.chain().focus().toggleUnderline().run() },
              { label: "Strikethrough", action: () => editor?.chain().focus().toggleStrike().run() },
              { label: "Clear formatting", action: () => editor?.chain().focus().unsetAllMarks().clearNodes().run() },
              { label: "Line spacing: 1", action: () => (editor?.chain().focus() as unknown as { setLineHeight: (v: string) => { run: () => void } }).setLineHeight("1").run() },
              { label: "Line spacing: 1.5", action: () => (editor?.chain().focus() as unknown as { setLineHeight: (v: string) => { run: () => void } }).setLineHeight("1.5").run() },
              { label: "Line spacing: 2", action: () => (editor?.chain().focus() as unknown as { setLineHeight: (v: string) => { run: () => void } }).setLineHeight("2").run() },
            ]},
            { label: "Table", items: [
              { label: "Insert table…", action: () => setShowTableMenu(v => !v) },
              { label: "Insert row above", action: () => editor?.chain().focus().addRowBefore().run() },
              { label: "Insert row below", action: () => editor?.chain().focus().addRowAfter().run() },
              { label: "Insert column left", action: () => editor?.chain().focus().addColumnBefore().run() },
              { label: "Insert column right", action: () => editor?.chain().focus().addColumnAfter().run() },
              { label: "Delete row", action: () => editor?.chain().focus().deleteRow().run() },
              { label: "Delete column", action: () => editor?.chain().focus().deleteColumn().run() },
              { label: "Delete table", action: () => editor?.chain().focus().deleteTable().run() },
              { label: "Merge cells", action: () => editor?.chain().focus().mergeCells().run() },
              { label: "Split cell", action: () => editor?.chain().focus().splitCell().run() },
            ]},
          ].map(menu => (
            <MenuDropdown key={menu.label} label={menu.label} items={menu.items} />
          ))}
        </div>

        {/* Formatting toolbar */}
        <div className="flex items-center gap-0.5 px-3 py-1.5 border-t border-[#e0e0e0] flex-wrap">
          {/* Undo / Redo */}
          <ToolBtn title="Undo (⌘Z)" onClick={() => editor?.chain().focus().undo().run()} disabled={!editor?.can().undo()}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 7v6h6"/><path d="M21 17a9 9 0 00-9-9 9 9 0 00-6 2.3L3 13"/></svg>
          </ToolBtn>
          <ToolBtn title="Redo (⌘⇧Z)" onClick={() => editor?.chain().focus().redo().run()} disabled={!editor?.can().redo()}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 7v6h-6"/><path d="M3 17a9 9 0 019-9 9 9 0 016 2.3l3 2.7"/></svg>
          </ToolBtn>

          <Divider />

          {/* Zoom */}
          <select value={zoom} onChange={e => setZoom(Number(e.target.value))} className="toolbar-select text-xs w-[62px]">
            {[50, 75, 90, 100, 110, 125, 150, 200].map(z => <option key={z} value={z}>{z}%</option>)}
          </select>

          <Divider />

          {/* Heading / paragraph style */}
          <select
            className="toolbar-select text-xs w-[110px]"
            value={
              isActive("heading", { level: 1 }) ? "h1" :
              isActive("heading", { level: 2 }) ? "h2" :
              isActive("heading", { level: 3 }) ? "h3" :
              isActive("heading", { level: 4 }) ? "h4" :
              isActive("heading", { level: 5 }) ? "h5" :
              isActive("heading", { level: 6 }) ? "h6" :
              isActive("blockquote") ? "quote" :
              isActive("codeBlock") ? "code" : "normal"
            }
            onChange={e => {
              const v = e.target.value;
              if (v.startsWith("h")) editor?.chain().focus().toggleHeading({ level: Number(v[1]) as 1|2|3|4|5|6 }).run();
              else if (v === "quote") editor?.chain().focus().toggleBlockquote().run();
              else editor?.chain().focus().setParagraph().run();
            }}
          >
            <option value="normal">Normal text</option>
            <option value="h1">Heading 1</option>
            <option value="h2">Heading 2</option>
            <option value="h3">Heading 3</option>
            <option value="h4">Heading 4</option>
            <option value="h5">Heading 5</option>
            <option value="h6">Heading 6</option>
            <option value="quote">Block quote</option>
          </select>

          <Divider />

          {/* Font family */}
          <select value={fontFamily} onChange={e => setFontFamily(e.target.value)} className="toolbar-select text-xs w-[110px]">
            {FONT_FAMILIES.map(f => <option key={f.value} value={f.value} style={{ fontFamily: f.value }}>{f.label}</option>)}
          </select>

          {/* Font size */}
          <select value={fontSize} onChange={e => { setFontSize(e.target.value); (editor?.chain().focus() as unknown as { setFontSize: (s: string) => { run: () => void } }).setFontSize(e.target.value).run(); }} className="toolbar-select text-xs w-[52px]">
            {FONT_SIZES.map(s => <option key={s} value={s}>{s}</option>)}
          </select>

          <Divider />

          {/* Bold / Italic / Underline / Strike */}
          <ToolBtn title="Bold (⌘B)" active={isActive("bold")} onClick={() => editor?.chain().focus().toggleBold().run()}><b className="text-xs">B</b></ToolBtn>
          <ToolBtn title="Italic (⌘I)" active={isActive("italic")} onClick={() => editor?.chain().focus().toggleItalic().run()}><i className="text-xs">I</i></ToolBtn>
          <ToolBtn title="Underline (⌘U)" active={isActive("underline")} onClick={() => editor?.chain().focus().toggleUnderline().run()}><u className="text-xs">U</u></ToolBtn>
          <ToolBtn title="Strikethrough" active={isActive("strike")} onClick={() => editor?.chain().focus().toggleStrike().run()}><s className="text-xs">S</s></ToolBtn>

          <Divider />

          {/* Text color */}
          <label className="relative" title="Text color">
            <input type="color" defaultValue="#000000" onChange={e => editor?.chain().focus().setColor(e.target.value).run()} className="sr-only" />
            <span className="toolbar-btn flex flex-col items-center gap-0 cursor-pointer">
              <span className="font-bold text-xs" style={{ color: "#000" }}>A</span>
              <span className="w-4 h-[3px] rounded-sm bg-current" style={{ color: editor?.getAttributes("textStyle").color || "#000" }}></span>
            </span>
          </label>

          {/* Highlight */}
          <label className="relative" title="Highlight color">
            <input type="color" defaultValue="#ffff00" onChange={e => editor?.chain().focus().toggleHighlight({ color: e.target.value }).run()} className="sr-only" />
            <span className={`toolbar-btn cursor-pointer flex items-center text-xs ${isActive("highlight") ? "toolbar-btn-active" : ""}`}>
              <span className="text-xs">🖊</span>
            </span>
          </label>

          <Divider />

          {/* Link */}
          <ToolBtn title="Insert link" active={isActive("link")} onClick={() => { setLinkUrl(editor?.getAttributes("link").href || ""); setShowLinkModal(true); }}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"/></svg>
          </ToolBtn>

          {/* Image */}
          <ToolBtn title="Insert image" onClick={() => setShowImageModal(true)}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
          </ToolBtn>

          {/* Table */}
          <div className="relative">
            <ToolBtn title="Insert table" active={showTableMenu} onClick={() => setShowTableMenu(v => !v)}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="1"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>
            </ToolBtn>
            {showTableMenu && (
              <div className="absolute top-full left-0 z-50 bg-white border border-[#dadce0] rounded-lg shadow-xl p-3" onMouseLeave={() => setShowTableMenu(false)}>
                <p className="text-[10px] text-[var(--ink-soft)] mb-2">
                  {tableHover.r > 0 ? `${tableHover.r} × ${tableHover.c} table` : "Select table size"}
                </p>
                <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${TABLE_GRID.cols}, 18px)` }}>
                  {Array.from({ length: TABLE_GRID.rows }, (_, r) =>
                    Array.from({ length: TABLE_GRID.cols }, (_, c) => (
                      <div
                        key={`${r}-${c}`}
                        className={`w-[18px] h-[18px] border rounded-sm cursor-pointer transition-colors ${r < tableHover.r && c < tableHover.c ? "bg-[#1a73e8]/20 border-[#1a73e8]" : "border-[#dadce0] hover:border-[#1a73e8]"}`}
                        onMouseEnter={() => setTableHover({ r: r + 1, c: c + 1 })}
                        onClick={() => insertTable(r + 1, c + 1)}
                      />
                    ))
                  )}
                </div>
              </div>
            )}
          </div>

          <Divider />

          {/* Alignment */}
          <ToolBtn title="Align left" active={editor?.isActive({ textAlign: "left" }) ?? false} onClick={() => editor?.chain().focus().setTextAlign("left").run()}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="15" y2="12"/><line x1="3" y1="18" x2="18" y2="18"/></svg>
          </ToolBtn>
          <ToolBtn title="Align center" active={editor?.isActive({ textAlign: "center" }) ?? false} onClick={() => editor?.chain().focus().setTextAlign("center").run()}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="6" y1="12" x2="18" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/></svg>
          </ToolBtn>
          <ToolBtn title="Align right" active={editor?.isActive({ textAlign: "right" }) ?? false} onClick={() => editor?.chain().focus().setTextAlign("right").run()}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="9" y1="12" x2="21" y2="12"/><line x1="6" y1="18" x2="21" y2="18"/></svg>
          </ToolBtn>
          <ToolBtn title="Justify" active={editor?.isActive({ textAlign: "justify" }) ?? false} onClick={() => editor?.chain().focus().setTextAlign("justify").run()}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
          </ToolBtn>

          <Divider />

          {/* Lists */}
          <ToolBtn title="Bullet list" active={isActive("bulletList")} onClick={() => editor?.chain().focus().toggleBulletList().run()}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="9" y1="6" x2="20" y2="6"/><line x1="9" y1="12" x2="20" y2="12"/><line x1="9" y1="18" x2="20" y2="18"/><circle cx="4" cy="6" r="1" fill="currentColor"/><circle cx="4" cy="12" r="1" fill="currentColor"/><circle cx="4" cy="18" r="1" fill="currentColor"/></svg>
          </ToolBtn>
          <ToolBtn title="Numbered list" active={isActive("orderedList")} onClick={() => editor?.chain().focus().toggleOrderedList().run()}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="10" y1="6" x2="21" y2="6"/><line x1="10" y1="12" x2="21" y2="12"/><line x1="10" y1="18" x2="21" y2="18"/><path d="M4 6h1v4"/><path d="M4 10h2"/><path d="M6 18H4c0-1 2-2 2-3s-1-1.5-2-1"/></svg>
          </ToolBtn>
          <ToolBtn title="Task list / Checklist" active={isActive("taskList")} onClick={() => editor?.chain().focus().toggleTaskList().run()}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11"/></svg>
          </ToolBtn>

          <Divider />

          {/* Indent */}
          <ToolBtn title="Decrease indent" onClick={() => editor?.chain().focus().liftListItem("listItem").run()}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="3" y1="8" x2="21" y2="8"/><line x1="3" y1="16" x2="21" y2="16"/><polyline points="7 12 3 8 7 4"/></svg>
          </ToolBtn>
          <ToolBtn title="Increase indent" onClick={() => editor?.chain().focus().sinkListItem("listItem").run()}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="3" y1="8" x2="21" y2="8"/><line x1="3" y1="16" x2="21" y2="16"/><polyline points="17 4 21 8 17 12"/></svg>
          </ToolBtn>

          <Divider />

          {/* Subscript / Superscript */}
          <ToolBtn title="Subscript" active={isActive("subscript")} onClick={() => editor?.chain().focus().toggleSubscript().run()}>
            <span className="text-xs">x<sub>2</sub></span>
          </ToolBtn>
          <ToolBtn title="Superscript" active={isActive("superscript")} onClick={() => editor?.chain().focus().toggleSuperscript().run()}>
            <span className="text-xs">x<sup>2</sup></span>
          </ToolBtn>

          <Divider />

          {/* Line height */}
          <select value={lineHeight} onChange={e => { setLineHeight(e.target.value); (editor?.chain().focus() as unknown as { setLineHeight: (v: string) => { run: () => void } }).setLineHeight(e.target.value).run(); }} className="toolbar-select text-xs w-[60px]" title="Line spacing">
            {["1", "1.15", "1.5", "2", "2.5", "3"].map(v => <option key={v} value={v}>{v}×</option>)}
          </select>

          <Divider />

          {/* Find & Replace */}
          <ToolBtn title="Find & Replace (⌘H)" active={showFindReplace} onClick={() => setShowFindReplace(v => !v)}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          </ToolBtn>

          {/* Word count */}
          <span className="ml-auto text-[10px] text-[#9aa0a6] pr-2 whitespace-nowrap">
            {editor?.storage.characterCount?.words?.() ?? 0} words · {editor?.storage.characterCount?.characters?.() ?? 0} chars
          </span>
        </div>

        {/* Find & Replace bar */}
        {showFindReplace && (
          <div className="flex items-center gap-2 px-4 py-2 bg-[#f8f9fa] border-t border-[#e0e0e0]">
            <input value={findText} onChange={e => setFindText(e.target.value)} placeholder="Find…" className="text-sm border border-[#dadce0] rounded px-2 py-1 w-40 focus:outline-none focus:border-[#1a73e8]" />
            <input value={replaceText} onChange={e => setReplaceText(e.target.value)} placeholder="Replace with…" className="text-sm border border-[#dadce0] rounded px-2 py-1 w-40 focus:outline-none focus:border-[#1a73e8]" />
            <button onClick={findAndReplace} className="text-xs px-3 py-1.5 bg-[#1a73e8] text-white rounded hover:bg-[#1557b0]">Replace all</button>
            <button onClick={() => setShowFindReplace(false)} className="text-xs px-2 py-1.5 text-[#5f6368] hover:bg-[#e8eaed] rounded">✕</button>
          </div>
        )}
      </div>

      {/* ── Editor body ──────────────────────────────────────────────────────── */}
      <div className="flex flex-1 min-h-0 overflow-hidden">

        {/* Page area */}
        <div ref={editorWrapRef} className="flex-1 overflow-y-auto bg-[#f8f9fa] min-h-0 relative" style={{ padding: "24px 0" }}>
          {/* Floating bubble menu — appears above any text selection */}
          {hasSelection && bubbleMenuPos && editor && (
            <div
              className="fixed z-50 flex items-center gap-0.5 bg-[#202124] rounded-lg px-1.5 py-1.5 shadow-2xl pointer-events-auto"
              style={{
                left: bubbleMenuPos.x + (editorWrapRef.current?.getBoundingClientRect().left ?? 0),
                top: bubbleMenuPos.y + (editorWrapRef.current?.getBoundingClientRect().top ?? 0) - 44,
                transform: "translateX(-50%)",
              }}
              onMouseDown={e => e.preventDefault()}
            >
              <BubbleBtn active={isActive("bold")} onClick={() => editor.chain().focus().toggleBold().run()} title="Bold"><b className="text-[11px] text-white px-0.5">B</b></BubbleBtn>
              <BubbleBtn active={isActive("italic")} onClick={() => editor.chain().focus().toggleItalic().run()} title="Italic"><i className="text-[11px] text-white px-0.5">I</i></BubbleBtn>
              <BubbleBtn active={isActive("underline")} onClick={() => editor.chain().focus().toggleUnderline().run()} title="Underline"><u className="text-[11px] text-white px-0.5">U</u></BubbleBtn>
              <BubbleBtn active={isActive("strike")} onClick={() => editor.chain().focus().toggleStrike().run()} title="Strikethrough"><s className="text-[11px] text-white px-0.5">S</s></BubbleBtn>
              <div className="w-px h-4 bg-white/20 mx-0.5" />
              <BubbleBtn active={isActive("link")} onClick={() => setShowLinkModal(true)} title="Link">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"/></svg>
              </BubbleBtn>
              <BubbleBtn active={false} onClick={() => editor.chain().focus().unsetAllMarks().run()} title="Clear formatting">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
              </BubbleBtn>
              <div className="w-px h-4 bg-white/20 mx-0.5" />
              <BubbleBtn active={false} onClick={() => { setAiMode("improve"); setShowAiPanel(true); runAI(); }} title="✨ AI Improve">
                <span className="text-[10px] text-[#8ab4f8] px-0.5">✨ AI</span>
              </BubbleBtn>
            </div>
          )}
          {/* Ruler */}
          <div className="mx-auto mb-2 bg-[#e8eaed] h-[20px] rounded-sm relative overflow-hidden" style={{ width: zoom < 100 ? `${zoom}%` : "816px", maxWidth: "100%" }}>
            <div className="absolute inset-0 flex items-center px-[72px]">
              {Array.from({ length: 20 }, (_, i) => (
                <div key={i} className="flex-1 border-l border-[#bdc1c6] h-2 first:border-l-0 relative">
                  <span className="absolute -top-0.5 left-1 text-[8px] text-[#9aa0a6]">{i > 0 ? i : ""}</span>
                </div>
              ))}
            </div>
          </div>

          {/* The "page" */}
          <div
            className="bg-white shadow-[0_1px_3px_rgba(0,0,0,0.12),0_2px_8px_rgba(0,0,0,0.08)] mx-auto min-h-[1056px]"
            style={{
              width: zoom < 100 ? `${zoom}%` : "816px",
              maxWidth: "100%",
              transform: zoom > 100 ? `scale(${zoom / 100})` : undefined,
              transformOrigin: "top center",
            }}
          >
            <div className={`${marginClass} py-[72px]`}>
              {editor && <EditorContent editor={editor} />}
            </div>
          </div>
        </div>

        {/* ── AI side panel ─────────────────────────────────────────────────── */}
        {showAiPanel && (
          <div className="w-[300px] shrink-0 bg-white border-l border-[#e0e0e0] flex flex-col overflow-hidden">
            <div className="px-4 py-3 border-b border-[#e0e0e0] flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span>✨</span>
                <span className="font-medium text-sm">AI Assist</span>
              </div>
              <button onClick={() => setShowAiPanel(false)} className="text-[#5f6368] hover:bg-[#f1f3f4] rounded p-1">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
              </button>
            </div>

            {/* Mode tabs */}
            <div className="flex border-b border-[#e0e0e0]">
              {(["complete", "improve", "write", "cases"] as const).map(m => (
                <button key={m} onClick={() => setAiMode(m)} className={`flex-1 py-2 text-[11px] font-medium capitalize transition-colors ${aiMode === m ? "border-b-2 border-[#1a73e8] text-[#1a73e8]" : "text-[#5f6368] hover:bg-[#f8f9fa]"}`}>
                  {m === "cases" ? "Find Cases" : m === "complete" ? "Complete" : m === "improve" ? "Improve" : "Write"}
                </button>
              ))}
            </div>

            <div className="flex flex-col gap-3 p-4 overflow-y-auto flex-1">
              {(aiMode === "write" || aiMode === "cases") && (
                <textarea
                  value={aiQuery}
                  onChange={e => setAiQuery(e.target.value)}
                  placeholder={aiMode === "write" ? "Describe what to write (e.g. 'grounds for bail citing NDPS Section 37')…" : "Describe the legal argument (e.g. 'anticipatory bail for white collar crime')…"}
                  rows={3}
                  className="text-sm border border-[#dadce0] rounded-lg px-3 py-2 resize-none focus:outline-none focus:border-[#1a73e8]"
                />
              )}
              {aiMode === "complete" && (
                <p className="text-xs text-[#5f6368]">AI will continue your document from where your cursor is, or complete the selected text.</p>
              )}
              {aiMode === "improve" && (
                <p className="text-xs text-[#5f6368]">Select text in the document, then click Improve to enhance clarity, legal precision, and style.</p>
              )}

              <button
                onClick={runAI}
                disabled={aiLoading}
                className="w-full py-2 bg-[#1a73e8] hover:bg-[#1557b0] disabled:opacity-60 text-white text-sm font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                {aiLoading ? (
                  <><span className="animate-spin">⟳</span> Working…</>
                ) : (
                  <><span>✨</span> {aiMode === "complete" ? "Complete" : aiMode === "improve" ? "Improve" : aiMode === "write" ? "Write" : "Find Cases"}</>
                )}
              </button>

              {/* Result */}
              {aiResult && aiMode !== "cases" && (
                <div className="flex flex-col gap-2">
                  <div className="text-[11px] font-medium text-[#5f6368] uppercase tracking-wide">Result</div>
                  <div className="text-sm bg-[#f8f9fa] border border-[#e0e0e0] rounded-lg px-3 py-3 max-h-[200px] overflow-y-auto whitespace-pre-wrap leading-relaxed">
                    {aiResult}
                  </div>
                  <button onClick={insertAiResult} className="w-full py-1.5 text-sm text-[#1a73e8] border border-[#1a73e8] rounded-lg hover:bg-[#e8f0fe] transition-colors font-medium">
                    ↩ Insert into document
                  </button>
                </div>
              )}

              {/* Case results */}
              {aiMode === "cases" && aiCases.length > 0 && (
                <div className="flex flex-col gap-2">
                  <div className="text-[11px] font-medium text-[#5f6368] uppercase tracking-wide">Cases found ({aiCases.length})</div>
                  {aiCases.map((c, i) => (
                    <div key={i} className="border border-[#e0e0e0] rounded-lg p-3 text-xs flex flex-col gap-1 hover:border-[#1a73e8] transition-colors">
                      <p className="font-medium text-[#202124] leading-snug line-clamp-2">{c.title}</p>
                      <p className="text-[#5f6368]">{[c.court, c.year].filter(Boolean).join(" · ")}</p>
                      {c.citation && <p className="text-[#1a73e8] font-mono text-[10px]">{c.citation}</p>}
                      {c.excerpt && <p className="text-[#5f6368] italic line-clamp-2">"{c.excerpt}"</p>}
                      <button
                        onClick={() => insertCitation(c)}
                        className="mt-1 text-[11px] text-[#1a73e8] hover:underline text-left font-medium"
                      >
                        ↩ Insert citation
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Legal Clause Library panel ────────────────────────────────────── */}
        {showClausePanel && (
          <div className="w-[300px] shrink-0 bg-white border-l border-[#e0e0e0] flex flex-col overflow-hidden">
            <div className="px-4 py-3 border-b border-[#e0e0e0] flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span>⚖️</span>
                <span className="font-medium text-sm">Legal Clauses</span>
              </div>
              <button onClick={() => setShowClausePanel(false)} className="text-[#5f6368] hover:bg-[#f1f3f4] rounded p-1">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
              </button>
            </div>
            <div className="px-3 py-2 border-b border-[#e0e0e0]">
              <input
                value={clauseSearch}
                onChange={e => setClauseSearch(e.target.value)}
                placeholder="Search clauses..."
                className="w-full text-sm border border-[#dadce0] rounded-lg px-3 py-1.5 focus:outline-none focus:border-[#1a73e8]"
              />
            </div>
            <div className="flex-1 overflow-y-auto px-3 py-2">
              {(() => {
                const filtered = clauses.filter(c =>
                  !clauseSearch || c.label.toLowerCase().includes(clauseSearch.toLowerCase()) || c.category.toLowerCase().includes(clauseSearch.toLowerCase())
                );
                const groups = filtered.reduce<Record<string, typeof filtered>>((acc, c) => {
                  (acc[c.category] = acc[c.category] || []).push(c);
                  return acc;
                }, {});
                return Object.entries(groups).map(([cat, items]) => (
                  <div key={cat} className="mb-3">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[#9aa0a6] px-1 mb-1">{cat}</div>
                    {items.map(c => (
                      <button
                        key={c.id}
                        onClick={() => insertClause(c.id)}
                        className="w-full text-left text-xs px-2 py-2 rounded hover:bg-[#e8f0fe] transition-colors flex items-center gap-2 group"
                      >
                        <span className="text-[#1a73e8] opacity-0 group-hover:opacity-100 transition-opacity text-[10px]">+</span>
                        <span className="text-[#202124]">{c.label}</span>
                      </button>
                    ))}
                  </div>
                ));
              })()}
              {clauses.length === 0 && <p className="text-xs text-[#9aa0a6] text-center py-4">Loading clauses...</p>}
            </div>
            <div className="px-3 py-2 border-t border-[#e0e0e0]">
              <p className="text-[10px] text-[#9aa0a6]">Click a clause to insert it at cursor position. Customize the [PLACEHOLDERS] after insertion.</p>
            </div>
          </div>
        )}
      </div>

      {/* ── Link modal ────────────────────────────────────────────────────────── */}
      {showLinkModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={() => setShowLinkModal(false)}>
          <div className="bg-white rounded-xl shadow-2xl p-6 w-[360px]" onClick={e => e.stopPropagation()}>
            <h3 className="font-medium text-sm mb-3">Insert link</h3>
            <input
              autoFocus
              value={linkUrl}
              onChange={e => setLinkUrl(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter") {
                  editor?.chain().focus().extendMarkRange("link").setLink({ href: linkUrl }).run();
                  setShowLinkModal(false);
                }
              }}
              placeholder="https://…"
              className="w-full border border-[#dadce0] rounded-lg px-3 py-2 text-sm mb-4 focus:outline-none focus:border-[#1a73e8]"
            />
            <div className="flex justify-end gap-3">
              <button onClick={() => { editor?.chain().focus().unsetLink().run(); setShowLinkModal(false); }} className="text-sm text-[#d93025] hover:underline">Remove link</button>
              <button onClick={() => setShowLinkModal(false)} className="text-sm text-[#5f6368] px-3 py-1.5">Cancel</button>
              <button
                onClick={() => { editor?.chain().focus().extendMarkRange("link").setLink({ href: linkUrl }).run(); setShowLinkModal(false); }}
                className="text-sm px-4 py-1.5 bg-[#1a73e8] text-white rounded-md hover:bg-[#1557b0]"
              >Apply</button>
            </div>
          </div>
        </div>
      )}

      {/* ── Image modal ───────────────────────────────────────────────────────── */}
      {showImageModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={() => setShowImageModal(false)}>
          <div className="bg-white rounded-xl shadow-2xl p-6 w-[400px]" onClick={e => e.stopPropagation()}>
            <h3 className="font-medium text-sm mb-3">Insert image</h3>
            <input
              autoFocus
              value={imageUrl}
              onChange={e => setImageUrl(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter") {
                  editor?.chain().focus().setImage({ src: imageUrl }).run();
                  setShowImageModal(false); setImageUrl("");
                }
              }}
              placeholder="Image URL (https://…) or paste base64"
              className="w-full border border-[#dadce0] rounded-lg px-3 py-2 text-sm mb-2 focus:outline-none focus:border-[#1a73e8]"
            />
            <p className="text-[11px] text-[#5f6368] mb-4">Or upload from your computer:</p>
            <input type="file" accept="image/*" className="text-xs mb-4" onChange={e => {
              const file = e.target.files?.[0];
              if (!file) return;
              const reader = new FileReader();
              reader.onload = ev => {
                editor?.chain().focus().setImage({ src: ev.target?.result as string }).run();
                setShowImageModal(false);
              };
              reader.readAsDataURL(file);
            }} />
            <div className="flex justify-end gap-3">
              <button onClick={() => setShowImageModal(false)} className="text-sm text-[#5f6368] px-3 py-1.5">Cancel</button>
              <button
                onClick={() => { editor?.chain().focus().setImage({ src: imageUrl }).run(); setShowImageModal(false); setImageUrl(""); }}
                className="text-sm px-4 py-1.5 bg-[#1a73e8] text-white rounded-md hover:bg-[#1557b0]"
              >Insert</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Toolbar button ─────────────────────────────────────────────────────────
function ToolBtn({ children, onClick, active, disabled, title }: {
  children: React.ReactNode; onClick: () => void; active?: boolean; disabled?: boolean; title?: string;
}) {
  return (
    <button
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={`min-w-[26px] h-[26px] px-1 flex items-center justify-center rounded transition-colors text-[#444746] ${active ? "bg-[#d3e3fd] text-[#1a73e8]" : "hover:bg-[#f1f3f4]"} ${disabled ? "opacity-40 cursor-not-allowed" : ""}`}
    >
      {children}
    </button>
  );
}

function BubbleBtn({ children, onClick, active, title }: {
  children: React.ReactNode; onClick: () => void; active: boolean; title?: string;
}) {
  return (
    <button title={title} onClick={onClick} className={`min-w-[24px] h-[24px] px-1 flex items-center justify-center rounded transition-colors ${active ? "bg-white/20" : "hover:bg-white/10"}`}>
      {children}
    </button>
  );
}

function Divider() {
  return <div className="w-px h-5 bg-[#dadce0] mx-0.5 shrink-0" />;
}

// ── Menu dropdown ──────────────────────────────────────────────────────────
function MenuDropdown({ label, items }: { label: string; items: { label: string; action: () => void }[] }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(v => !v)}
        className={`px-2.5 py-1.5 rounded text-[13px] transition-colors ${open ? "bg-[#e8f0fe] text-[#1a73e8]" : "hover:bg-[#f1f3f4]"}`}
      >
        {label}
      </button>
      {open && (
        <div className="absolute top-full left-0 z-50 bg-white border border-[#e0e0e0] rounded-lg shadow-xl py-1 min-w-[180px]">
          {items.map(item => (
            <button
              key={item.label}
              onClick={() => { item.action(); setOpen(false); }}
              className="w-full text-left px-4 py-2 text-[13px] text-[#202124] hover:bg-[#f8f9fa] transition-colors"
            >
              {item.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
