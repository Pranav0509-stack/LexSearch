#!/usr/bin/env python3
"""Render the Sanhita master docs to PDF.

Inputs:  SANHITA.md and CALL_SURFACE_PLAN.md (both at the repo root)
Output:  Sanhita.pdf — combined, page-numbered, with a clean serif
         body and a Fraunces-ish display heading.

We deliberately keep this dependency-light: only `markdown` + `reportlab`
(both pure-Python, no native bindings). That means simpler tables and
no syntax-highlighted code blocks, but the doc is for executives /
investors / future Claude sessions — readability matters more than
typography.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    Preformatted,
    KeepTogether,
)

ROOT = Path(__file__).resolve().parent.parent
INPUTS = [
    ROOT / "SANHITA.md",
    ROOT / "CALL_SURFACE_PLAN.md",
]
OUT = ROOT / "Sanhita.pdf"


# ---------- styles ----------------------------------------------------------

def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    serif = "Times-Roman"
    serif_bold = "Times-Bold"
    serif_italic = "Times-Italic"
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"],
            fontName=serif_bold, fontSize=24, leading=28,
            spaceAfter=16, textColor=colors.HexColor("#1a1a1a"),
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"],
            fontName=serif_bold, fontSize=18, leading=22,
            spaceBefore=18, spaceAfter=10, textColor=colors.HexColor("#6b4f1d"),
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"],
            fontName=serif_bold, fontSize=14, leading=18,
            spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#6b4f1d"),
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"],
            fontName=serif_bold, fontSize=12, leading=15,
            spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#1a1a1a"),
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"],
            fontName=serif, fontSize=10, leading=14,
            spaceAfter=4, alignment=TA_LEFT,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["BodyText"],
            fontName=serif, fontSize=10, leading=14,
            leftIndent=18, bulletIndent=6, spaceAfter=2,
        ),
        "blockquote": ParagraphStyle(
            "blockquote", parent=base["BodyText"],
            fontName=serif_italic, fontSize=10, leading=14,
            leftIndent=18, rightIndent=18, spaceAfter=8,
            textColor=colors.HexColor("#555555"),
        ),
        "code": ParagraphStyle(
            "code", parent=base["Code"],
            fontName="Courier", fontSize=8.5, leading=11,
            leftIndent=12, spaceBefore=4, spaceAfter=8,
            textColor=colors.HexColor("#222222"),
        ),
        "tablecell": ParagraphStyle(
            "tablecell", parent=base["BodyText"],
            fontName=serif, fontSize=8.5, leading=11, spaceAfter=0,
        ),
    }


# ---------- markdown → flowables --------------------------------------------

INLINE_BOLD = re.compile(r"\*\*(.+?)\*\*")
INLINE_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
INLINE_CODE = re.compile(r"`([^`]+)`")
INLINE_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _inline(s: str) -> str:
    """Escape XML and render simple markdown inline syntax to ReportLab tags."""
    s = (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = INLINE_BOLD.sub(r"<b>\1</b>", s)
    s = INLINE_ITALIC.sub(r"<i>\1</i>", s)
    s = INLINE_CODE.sub(r"<font face='Courier' size='9'>\1</font>", s)

    def _link_repl(m: re.Match) -> str:
        text, url = m.group(1), m.group(2)
        # In-document anchors don't resolve in a flat PDF. Drop the
        # link wrapper and keep just the visible text in italic so the
        # ToC still reads naturally.
        if url.startswith("#"):
            return f"<i>{text}</i>"
        return f'<link href="{url}" color="#6b4f1d">{text}</link>'
    s = INLINE_LINK.sub(_link_repl, s)
    return s


def _table_flowable(rows: list[list[str]], styles: dict) -> Table:
    body = [[Paragraph(_inline(c), styles["tablecell"]) for c in r] for r in rows]
    t = Table(body, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3eddd")),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfaf6")]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#6b4f1d")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor("#dcd5c0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def md_to_flowables(md: str, styles: dict) -> list:
    """Tiny Markdown parser → ReportLab flowables. Covers headings, bullets,
    numbered lists, blockquotes, fenced code, and pipe-tables — that's the
    full surface of our two source MD files."""
    out: list = []
    lines = md.split("\n")
    i = 0
    in_code = False
    code_buf: list[str] = []

    def flush_code():
        nonlocal code_buf
        if code_buf:
            out.append(Preformatted("\n".join(code_buf), styles["code"]))
            out.append(Spacer(1, 6))
            code_buf = []

    while i < len(lines):
        ln = lines[i]
        stripped = ln.rstrip()

        # fenced code
        if stripped.startswith("```"):
            if in_code:
                in_code = False
                flush_code()
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(ln)
            i += 1
            continue

        # tables — pipe rows separated by alignment row
        if "|" in stripped and i + 1 < len(lines) and re.match(r"^[\s\|:\-]+$", lines[i + 1].strip()):
            header = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2  # skip alignment row
            body_rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                body_rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            out.append(_table_flowable([header, *body_rows], styles))
            out.append(Spacer(1, 8))
            continue

        # headings
        m = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if m:
            depth = len(m.group(1))
            txt = _inline(m.group(2))
            style = styles[{1: "h1", 2: "h2", 3: "h3"}[depth]]
            out.append(Paragraph(txt, style))
            i += 1
            continue

        # horizontal rule
        if re.match(r"^-{3,}$", stripped) or re.match(r"^\*{3,}$", stripped):
            out.append(Spacer(1, 6))
            i += 1
            continue

        # blockquote
        if stripped.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                quote.append(lines[i].lstrip().lstrip(">").lstrip())
                i += 1
            out.append(Paragraph(_inline(" ".join(quote)), styles["blockquote"]))
            continue

        # bullet
        m = re.match(r"^\s*[-*]\s+(.*)$", ln)
        if m:
            out.append(Paragraph(_inline(m.group(1)), styles["bullet"], bulletText="•"))
            i += 1
            continue

        # numbered
        m = re.match(r"^\s*(\d+)\.\s+(.*)$", ln)
        if m:
            out.append(Paragraph(_inline(m.group(2)), styles["bullet"], bulletText=f"{m.group(1)}."))
            i += 1
            continue

        # blank
        if stripped == "":
            out.append(Spacer(1, 4))
            i += 1
            continue

        # paragraph (collapse continuation lines)
        para = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not _is_block_start(lines[i]):
            para.append(lines[i].rstrip())
            i += 1
        out.append(Paragraph(_inline(" ".join(para)), styles["body"]))

    flush_code()
    return out


def _is_block_start(ln: str) -> bool:
    s = ln.lstrip()
    return (
        s.startswith("#")
        or s.startswith(">")
        or s.startswith("|")
        or s.startswith("- ")
        or s.startswith("* ")
        or s.startswith("```")
        or bool(re.match(r"^\d+\. ", s))
    )


# ---------- header / footer -------------------------------------------------

def _on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Italic", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(0.7 * inch, 0.5 * inch, "Sanhita — Pan-Asia Legal Research Counsel")
    canvas.drawRightString(A4[0] - 0.7 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


# ---------- main ------------------------------------------------------------

def main() -> int:
    styles = _styles()
    flowables: list = []
    flowables.append(Paragraph("Sanhita", styles["title"]))
    flowables.append(Paragraph(
        "<i>Pan-Asia legal research counsel — what we built, what we ship next, what it costs.</i>",
        styles["body"],
    ))
    flowables.append(Spacer(1, 12))

    for path in INPUTS:
        if not path.exists():
            print(f"warn: {path} missing", file=sys.stderr)
            continue
        flowables.extend(md_to_flowables(path.read_text(encoding="utf-8"), styles))
        flowables.append(PageBreak())

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.8 * inch,
        title="Sanhita — Pan-Asia Legal Research Counsel",
        author="Sanhita",
    )
    doc.build(flowables, onFirstPage=_on_page, onLaterPages=_on_page)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024}KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
