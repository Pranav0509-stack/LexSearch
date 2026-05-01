// Tiny Markdown → safe HTML renderer used across the /app panes.
//
// We deliberately avoid pulling in a real Markdown library (≈30KB) — the
// LLM only ever emits a small, predictable subset: headings, bullets,
// numbered lists, bold/italic, inline code, and `[n]` citation chips.
// This renderer covers exactly that.
//
// Output is structured (real <h2>/<ul>/<ol>) so the "prose-style" CSS in
// globals.css can space sections properly instead of everything landing
// in one giant <p> blob.

export function esc(s: unknown): string {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Inline pass — bold, italic, code, citation chip. Operates on already-
// escaped text so HTML in the source is rendered as literal characters.
function renderInline(s: string): string {
  let h = s;
  h = h.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // Italic: single * not preceded by another * and not at start of a
  // bullet line (handled separately as a list marker).
  h = h.replace(/(^|[^*\s])\*([^*\n]+?)\*(?!\*)/g, "$1<em>$2</em>");
  h = h.replace(/`([^`]+)`/g, "<code>$1</code>");
  // Citation chip — drop a leading space so chips sit flush after the
  // sentence's last word ("…sparingly[1].") instead of a stranded gap.
  h = h.replace(/\s?\[(\d+)\]/g, '<sup class="cite-chip" data-n="$1">$1</sup>');
  return h;
}

export function renderMarkdown(md: string): string {
  const lines = String(md ?? "").split("\n");
  const out: string[] = [];

  let listType: "ul" | "ol" | null = null;
  let blockquoteBuf: string[] = [];
  let paragraphBuf: string[] = [];

  const flushParagraph = () => {
    if (paragraphBuf.length === 0) return;
    const text = paragraphBuf.join(" ").trim();
    if (text) out.push(`<p>${renderInline(text)}</p>`);
    paragraphBuf = [];
  };
  const closeList = () => {
    if (listType) { out.push(`</${listType}>`); listType = null; }
  };
  const flushBlockquote = () => {
    if (blockquoteBuf.length === 0) return;
    const inner = blockquoteBuf.join(" ").trim();
    if (inner) out.push(`<blockquote>${renderInline(esc(inner))}</blockquote>`);
    blockquoteBuf = [];
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    const escaped = esc(line);

    // Blank line — flush everything
    if (line.trim() === "") {
      flushParagraph();
      closeList();
      flushBlockquote();
      continue;
    }

    // Horizontal rule: --- or ***
    if (/^[-*]{3,}\s*$/.test(line.trim())) {
      flushParagraph();
      closeList();
      flushBlockquote();
      out.push('<hr class="prose-hr" />');
      continue;
    }

    // Headings: ####, ###, ##, #
    const h4m = /^####\s+(.*)$/.exec(line);
    const h3m = /^###\s+(.*)$/.exec(line);
    const h2m = /^##\s+(.*)$/.exec(line);
    const h1m = /^#\s+(.*)$/.exec(line);
    if (h4m || h3m || h2m || h1m) {
      flushParagraph(); closeList(); flushBlockquote();
      const tag = h4m ? "h4" : h3m ? "h3" : h2m ? "h2" : "h1";
      const body = (h4m?.[1] ?? h3m?.[1] ?? h2m?.[1] ?? h1m?.[1] ?? "").trim();
      out.push(`<${tag}>${renderInline(esc(body))}</${tag}>`);
      continue;
    }

    // Blockquote: > text
    const bq = /^>\s*(.*)$/.exec(line);
    if (bq) {
      flushParagraph(); closeList();
      blockquoteBuf.push(bq[1]);
      continue;
    } else {
      flushBlockquote();
    }

    // Bullets: `* foo` or `- foo` (but not `---`)
    const bullet = /^\s*[*-]\s+(.+)$/.exec(line);
    if (bullet) {
      flushParagraph();
      if (listType !== "ul") { closeList(); out.push("<ul>"); listType = "ul"; }
      out.push(`<li>${renderInline(esc(bullet[1]))}</li>`);
      continue;
    }

    // Numbered: `1. foo`
    const num = /^\s*(\d+)\.\s+(.+)$/.exec(line);
    if (num) {
      flushParagraph();
      if (listType !== "ol") { closeList(); out.push("<ol>"); listType = "ol"; }
      out.push(`<li>${renderInline(esc(num[2]))}</li>`);
      continue;
    }

    // Plain text
    closeList();
    paragraphBuf.push(escaped);
  }

  flushParagraph();
  closeList();
  flushBlockquote();
  return out.join("\n");
}
