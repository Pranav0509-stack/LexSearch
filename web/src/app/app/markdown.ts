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

  // Tracks open list state so consecutive bullet lines collapse into a
  // single <ul>/<ol> block.
  let listType: "ul" | "ol" | null = null;
  let paragraphBuf: string[] = [];

  const flushParagraph = () => {
    if (paragraphBuf.length === 0) return;
    const text = paragraphBuf.join(" ").trim();
    if (text) out.push(`<p>${renderInline(text)}</p>`);
    paragraphBuf = [];
  };
  const closeList = () => {
    if (listType) {
      out.push(`</${listType}>`);
      listType = null;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    const escaped = esc(line);

    // Blank line — paragraph/list break.
    if (line.trim() === "") {
      flushParagraph();
      closeList();
      continue;
    }

    // Headings: ## or ### (the LLM uses ## for memo sections).
    const h3 = /^###\s+(.*)$/.exec(line);
    const h2 = /^##\s+(.*)$/.exec(line);
    if (h2 || h3) {
      flushParagraph();
      closeList();
      const tag = h3 ? "h3" : "h2";
      const body = (h3?.[1] ?? h2?.[1] ?? "").trim();
      out.push(`<${tag}>${renderInline(esc(body))}</${tag}>`);
      continue;
    }

    // Bullets: `* foo` or `- foo` at line start.
    const bullet = /^\s*[*-]\s+(.+)$/.exec(line);
    if (bullet) {
      flushParagraph();
      if (listType !== "ul") {
        closeList();
        out.push("<ul>");
        listType = "ul";
      }
      out.push(`<li>${renderInline(esc(bullet[1]))}</li>`);
      continue;
    }

    // Numbered: `1. foo`
    const num = /^\s*(\d+)\.\s+(.+)$/.exec(line);
    if (num) {
      flushParagraph();
      if (listType !== "ol") {
        closeList();
        out.push("<ol>");
        listType = "ol";
      }
      out.push(`<li>${renderInline(esc(num[2]))}</li>`);
      continue;
    }

    // Plain text — accumulate. (Closing any open list because plain
    // prose between bullets is a real paragraph, not a continuation.)
    closeList();
    paragraphBuf.push(escaped);
  }

  flushParagraph();
  closeList();
  return out.join("\n");
}
