"""
Generate MASTER-PLAN.pdf — the canonical NyayaSathi × Sanhita launch plan.

Run from the project root:
  python3 docs/generate_master_plan.py

Re-run any time the plan changes — the PDF is the source of truth that lives
alongside the codebase, and CI can regenerate it on every PR that touches docs/.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ── Output ────────────────────────────────────────────────────────────────
OUT = Path(__file__).resolve().parent / "MASTER-PLAN.pdf"

# ── Colour system (deliberately calm, legal-document feel) ────────────────
INK = HexColor("#1A1A2E")
INK_SOFT = HexColor("#3D3D52")
ACCENT = HexColor("#6B4F1D")           # legal-brown
ACCENT_SOFT = HexColor("#9C7A3F")
RULE = HexColor("#C8C2B6")
SURFACE = HexColor("#FAF7F0")
HILITE = HexColor("#FFF8E1")

# ── Styles ────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()


def _style(name, *, font="Helvetica", size=10, leading=14, color=INK,
           space_before=0, space_after=4, alignment=TA_LEFT, left_indent=0,
           bold=False, italic=False):
    fnt = font
    if bold and italic: fnt = f"{font}-BoldOblique"
    elif bold: fnt = f"{font}-Bold"
    elif italic: fnt = f"{font}-Oblique"
    return ParagraphStyle(
        name=name, fontName=fnt, fontSize=size, leading=leading,
        textColor=color, spaceBefore=space_before, spaceAfter=space_after,
        alignment=alignment, leftIndent=left_indent,
    )


S_TITLE       = _style("title", font="Helvetica", size=28, leading=32, color=INK, bold=True, alignment=TA_LEFT, space_after=4)
S_SUBTITLE    = _style("subtitle", font="Helvetica", size=14, leading=18, color=ACCENT, italic=True, space_after=12)
S_AUTHOR_LINE = _style("author", font="Helvetica", size=9, leading=12, color=INK_SOFT)
S_H1          = _style("h1", font="Helvetica", size=18, leading=22, color=INK, bold=True, space_before=14, space_after=6)
S_H2          = _style("h2", font="Helvetica", size=13, leading=17, color=ACCENT, bold=True, space_before=10, space_after=4)
S_H3          = _style("h3", font="Helvetica", size=11, leading=14, color=INK, bold=True, space_before=6, space_after=2)
S_BODY        = _style("body", font="Helvetica", size=10, leading=14, color=INK, space_after=6, alignment=TA_JUSTIFY)
S_BODY_LEFT   = _style("body_left", font="Helvetica", size=10, leading=14, color=INK, space_after=6, alignment=TA_LEFT)
S_BULLET      = _style("bullet", font="Helvetica", size=10, leading=14, color=INK, left_indent=12, space_after=2)
S_QUOTE       = _style("quote", font="Helvetica", size=10, leading=14, color=ACCENT, italic=True, left_indent=14, space_after=8)
S_CODE        = _style("code", font="Courier", size=8.5, leading=11, color=INK, left_indent=8, space_after=6)
S_CAPTION     = _style("caption", font="Helvetica", size=8.5, leading=11, color=INK_SOFT, italic=True, alignment=TA_CENTER, space_after=10)
S_NOTE        = _style("note", font="Helvetica", size=9, leading=12, color=INK_SOFT, italic=True, space_after=6)
S_TOC_ENTRY   = _style("toc", font="Helvetica", size=10, leading=15, color=INK)


# ── Page chrome ───────────────────────────────────────────────────────────
def _draw_chrome(canvas, doc):
    """Header rule + page number + footer hairline."""
    canvas.saveState()
    w, h = A4
    # top thin rule
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(2 * cm, h - 1.5 * cm, w - 2 * cm, h - 1.5 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(INK_SOFT)
    canvas.drawString(2 * cm, h - 1.2 * cm, "NyayaSathi × Sanhita — Master Plan")
    canvas.drawRightString(w - 2 * cm, h - 1.2 * cm, "v1.0  ·  Sprint-Gated 3-Week Launch")
    # bottom rule + page no
    canvas.line(2 * cm, 1.5 * cm, w - 2 * cm, 1.5 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(w / 2, 1.0 * cm, f"page {doc.page}")
    canvas.restoreState()


# ── Helpers ───────────────────────────────────────────────────────────────
def H1(text): return Paragraph(text, S_H1)
def H2(text): return Paragraph(text, S_H2)
def H3(text): return Paragraph(text, S_H3)
def P(text): return Paragraph(text, S_BODY)
def PL(text): return Paragraph(text, S_BODY_LEFT)
def B(text): return Paragraph("• " + text, S_BULLET)
def Q(text): return Paragraph(text, S_QUOTE)
def Note(text): return Paragraph(text, S_NOTE)
def Code(text): return Paragraph(text.replace(" ", "&nbsp;").replace("\n", "<br/>"), S_CODE)


def make_table(data, *, col_widths=None, header=True, zebra=True, font_size=9):
    n_cols = len(data[0])
    if col_widths is None:
        col_widths = [(17.0 * cm) / n_cols] * n_cols
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    base_style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", font_size),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACCENT) if header else ("LINEBELOW", (0, 0), (-1, 0), 0, white),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, ACCENT) if header else ("LINEABOVE", (0, 0), (-1, 0), 0, white),
        ("TEXTCOLOR", (0, 0), (-1, 0), ACCENT) if header else ("TEXTCOLOR", (0, 0), (-1, 0), INK),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold") if header else ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if zebra:
        for i in range(1, len(data)):
            if i % 2 == 0:
                base_style.append(("BACKGROUND", (0, i), (-1, i), SURFACE))
    t.setStyle(TableStyle(base_style))
    return t


def callout(title, body, *, color=HILITE):
    """Highlighted box for an architectural commitment."""
    inner = [
        Paragraph(f"<b>{title}</b>", _style("co_title", size=11, leading=14, color=ACCENT, bold=True, space_after=4)),
        Paragraph(body, _style("co_body", size=10, leading=14, color=INK, alignment=TA_JUSTIFY)),
    ]
    t = Table([[inner]], colWidths=[17 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("BOX", (0, 0), (-1, -1), 0.5, ACCENT_SOFT),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


# ── CONTENT ───────────────────────────────────────────────────────────────
def build_story():
    s = []

    # ─── COVER ────────────────────────────────────────────────────────────
    s += [
        Spacer(1, 4 * cm),
        Paragraph("NyayaSathi <font color='%s'>×</font> Sanhita" % ACCENT, S_TITLE),
        Paragraph("The Master Plan — Launch in 21 Days, Eval-Gated, Built to Last", S_SUBTITLE),
        Spacer(1, 1.5 * cm),
        Paragraph("A two-sided legal-aid platform for India:", S_BODY_LEFT),
        Paragraph("free citizen-facing voice helpline that converts to verified-lawyer matches "
                  "and a research workbench priced for solo lawyers, small firms, and enterprise "
                  "compliance teams.", S_BODY),
        Spacer(1, 0.5 * cm),
        callout(
            "What this document is",
            "The single source of truth for what we are building, why this design, "
            "what we deliberately choose <i>not</i> to build, and what each day "
            "of the next 21 days looks like. Every commit, eval result, and "
            "architectural decision should be traceable back to a section of "
            "this plan. Re-generate from <font face='Courier'>docs/generate_master_plan.py</font> "
            "any time the plan changes.",
        ),
        Spacer(1, 1 * cm),
        Paragraph("Status:&nbsp;&nbsp;<b>Sprint 0, Day 1.</b>&nbsp;&nbsp;Eval baseline being authored.", S_AUTHOR_LINE),
        PageBreak(),
    ]

    # ─── TABLE OF CONTENTS ────────────────────────────────────────────────
    s += [
        H1("Table of Contents"),
        Spacer(1, 4),
    ]
    toc = [
        ("0", "Executive verdict in one page", 3),
        ("1", "Sanhita codebase audit — integrate, don't rebuild", 4),
        ("2", "The three-product reality", 7),
        ("3", "The four customer journeys", 9),
        ("4", "The architectural commitment — what AI does NOT do", 11),
        ("5", "Multi-modal data: storage and upload", 13),
        ("6", "The three stages of casework: Research → Analysis → Strategy", 15),
        ("7", "New surfaces: Predictive · Compliance · Contract Intelligence", 16),
        ("8", "The five data moats (with replication math)", 18),
        ("9", "The integration plan — where every line of new code lives", 20),
        ("10", "Why no competitor can match — with stats", 22),
        ("11", "The 21-day sprint plan, day by day", 23),
        ("12", "Eval policy — the gate that ships or blocks every commit", 25),
        ("13", "Codebase organisation and folder layout", 26),
        ("14", "Commit and comment conventions", 27),
        ("15", "Day 1 — what we do tomorrow", 28),
    ]
    rows = []
    for num, title, page in toc:
        rows.append([
            Paragraph(f"<b>{num}</b>", _style("toc_n", size=10, leading=14, color=ACCENT, bold=True)),
            Paragraph(title, S_TOC_ENTRY),
            Paragraph(f"<para alignment='right'>{page}</para>", _style("toc_p", size=10, leading=14, color=INK_SOFT)),
        ])
    s += [make_table(rows, col_widths=[1.2 * cm, 14.5 * cm, 1.5 * cm], header=False, zebra=False),
          PageBreak()]

    # ─── 0. EXECUTIVE VERDICT ─────────────────────────────────────────────
    s += [
        H1("0 · Executive verdict in one page"),
        P("We have audited every layer of NyayaSathi (the citizen voice helpline) and Sanhita "
          "(the lawyer workbench). The decision is unambiguous and is the spine of this plan."),
        callout(
            "The decision",
            "<b>Integrate with Sanhita; do not rebuild.</b> Sanhita's codebase has 7,839 lines "
            "of production-grade Python: the six-gate validator, the four-provider LLM router "
            "with circuit breakers, the BM25 retrieval index with atomic persistence, the Vault, "
            "the Workflows, the Realtime layer, and — crucially — a "
            "<font face='Courier'>nyaya_clients</font> table and "
            "<font face='Courier'>POST&nbsp;/api/nyaya/intake</font> endpoint that already "
            "anticipate exactly the handoff we need. We layer ~4,000 lines on top, not 12,000 lines from scratch."
        ),
        H2("Three things we got right and will defend"),
        B("The single source of truth for sessions on the NyayaSathi side."),
        B("The smart-fallback that walks history when LLMs fail — keeps user-visible quality high under outage."),
        B("The deliberate <i>refusal</i> posture in Sanhita's validator: refuses to write prose when it cannot ground."),
        H2("Three things we will fix before launch"),
        B("PII at rest is plaintext today — must be AES-GCM encrypted before a real DV case lands."),
        B("Distress detection is keyword-only and Hindi/English-only — paraphrase and 11-language coverage are blockers."),
        B("Lawyer matching is mocked; real Sanhita-backed scoring with bar-council verification is the launch milestone."),
        H2("Three sprints, each gated by 100+ question eval"),
        Note("Sprint 0 (week 1) — safety floor and bridge.&nbsp;&nbsp;"
             "Sprint 1 (week 2) — polish, female-voice auto-switch, real lawyer matching.&nbsp;&nbsp;"
             "Sprint 2 (week 3) — multi-modal vault and audit trail. Launch on Day 21."),
        Spacer(1, 0.4 * cm),
        callout(
            "The architectural promise (Section 4)",
            "AI does <b>not</b> render judgment, does <b>not</b> interpret unmarked, does <b>not</b> "
            "take accountability. These three are the human contribution India lacks, and they are "
            "what we route through humans by design — enforced by validator gates G7 through G10 in code, "
            "not just policy.",
            color=SURFACE,
        ),
        PageBreak(),
    ]

    # ─── 1. SANHITA AUDIT ─────────────────────────────────────────────────
    s += [
        H1("1 · Sanhita codebase audit — integrate, don't rebuild"),
        P("This section maps every component of the existing Sanhita codebase to a verdict: "
          "adopt as-is, extend, or replace. The total count of new code we have to write is "
          "the difference between the 'extend' rows and the 'replace' rows."),
        H2("1.1 Inventory"),
    ]
    audit_rows = [
        ["Component", "LOC", "Quality", "Verdict"],
        ["validators/answer_gates.py (G1–G6)", "209", "Excellent", "Adopt; port to NyayaSathi as JS"],
        ["llm/router.py (4-provider chain)", "477", "Excellent", "Adopt; replace homegrown JS cascade"],
        ["retrieval_pkg/index.py (BM25)", "294", "Excellent", "Shared retrieval backbone"],
        ["validators/input_guards.py", "226", "Good", "Adopt as shared module"],
        ["agents/legal_agent.py", "826", "Solid", "Reuse for Sanhita agent mode"],
        ["brief_service.py (modes)", "918", "Solid", "Reuse; add NyayaSathi voice mode"],
        ["vault_service.py", "289", "Solid", "Extend for audio/video/image"],
        ["db_adapter.py (Postgres↔SQLite)", "239", "Solid", "Adopt as shared persistence"],
        ["realtime.py (socketio)", "108", "Good", "Reuse; add nyaya:intake event"],
        ["auth.py + nyaya_clients table", "879", "Already designed", "Use exactly as is"],
        ["/api/nyaya/intake (server.py L1857)", "—", "Wired", "Just add HMAC verification"],
        ["clients-pane.tsx (frontend)", "—", "Built", "Reuse; add match button"],
        ["eval/run.py + 60 prompts", "—", "Production", "Extend to 250 prompts"],
        ["BM25 corpus (15 MB pickled)", "1,135 docs", "Live", "Grow to 110K via roadmap"],
        ["Workflows: Draft, Review, Citator, Redline", "442", "Production", "Extend with Compliance + Contract"],
        ["Sarvam adapter (translate live)", "180", "Partial", "Extend with stt() and tts()"],
    ]
    s += [
        make_table(audit_rows, col_widths=[5.8 * cm, 1.6 * cm, 2.4 * cm, 7.4 * cm]),
        Spacer(1, 0.2 * cm),
        H2("1.2 What we delete from NyayaSathi"),
        B("<font face='Courier'>lawyers.json</font> (mock 10 lawyers) — Sanhita has the real table."),
        B("<font face='Courier'>lawyer-match.js</font> (homegrown) — Sanhita's match engine takes over."),
        B("Custom <font face='Courier'>Promise.any</font> LLM cascade — replaced by router pattern."),
        B("File-backed <font face='Courier'>case-store.js</font> as primary store — Sanhita's <font face='Courier'>nyaya_clients</font> is canonical; NyayaSathi keeps a hot session cache only."),
        H2("1.3 What we extend"),
        B("<font face='Courier'>vault_service.py</font> — adds audio (Sarvam STT), video (ffmpeg keyframes + STT), image (Tesseract + Gemini vision)."),
        B("<font face='Courier'>validators/answer_gates.py</font> — adds G7 no_judgment, G8 no_unmarked_interpretation, G9 no_accountability, G10 predictive_must_include_base_rate."),
        B("<font face='Courier'>workflows.py</font> — adds Compliance Review and Contract Intelligence workflow types."),
        B("<font face='Courier'>llm/sarvam.py</font> — adds <font face='Courier'>stt()</font> and <font face='Courier'>tts()</font> helpers used by NyayaSathi voice and Sanhita callbacks."),
        H2("1.4 What we add fresh"),
        B("<font face='Courier'>sanhita-client.js</font> on NyayaSathi — HMAC-signed POSTer to <font face='Courier'>/api/nyaya/intake</font>."),
        B("<font face='Courier'>lawyer_match.py</font> on Sanhita — scoring engine over a new <font face='Courier'>lawyers_profile</font> table."),
        B("Multi-modal extractors module under <font face='Courier'>vault/extractors/</font>."),
        B("Compliance edition primitives: <font face='Courier'>workspaces.py</font>, <font face='Courier'>policies.py</font>, <font face='Courier'>audit_chain.py</font>."),
        B("Predictive surface (<font face='Courier'>predictive.py</font>) wired to outcome data once the flywheel turns."),
        Spacer(1, 0.3 * cm),
        callout(
            "The arithmetic that settles the debate",
            "Building from scratch = ~12 weeks engineering + re-discovering 60 hand-tuned banned-phrase "
            "regexes + re-tuning circuit breaker thresholds + re-implementing dedup-by-case-id. "
            "Layering on top = 3 weeks engineering + tests catching what we change. There is no "
            "scenario in which from-scratch wins.",
        ),
        PageBreak(),
    ]

    # ─── 2. THREE PRODUCTS ────────────────────────────────────────────────
    s += [
        H1("2 · The three-product reality"),
        P("One platform, three editions. The same retrieval and validation infrastructure powers "
          "all three; each tier exposes additional surfaces and enforces additional contracts. "
          "The pricing reflects ascending value per seat — research alone is commoditised, real "
          "lead generation is rare, and accountable-by-design compliance is institutional."),
    ]
    tier_rows = [
        ["Tier", "Audience", "Adds on top", "Price / seat / month"],
        ["NyayaSathi (B2C)", "Citizens", "Free voice helpline; routes 20% to Sanhita", "—  (free)"],
        ["Sanhita Research", "Solo advocate doing research", "BM25 + memos + Court Search + Workflows", "$19"],
        ["Sanhita Practice", "Solo + small firm", "Clients inbox (NyayaSathi leads), case mgmt, multi-modal vault, paralegal automation", "$49"],
        ["Sanhita Compliance", "In-house counsel + compliance + GRC", "Audit trail, RBAC, policy engine, contract intelligence, predictive risk", "$199 – $499"],
    ]
    s += [
        make_table(tier_rows, col_widths=[3.4 * cm, 4.0 * cm, 7.4 * cm, 2.4 * cm]),
        Spacer(1, 0.3 * cm),
        H2("2.1 Why three editions, not one"),
        P("A single SKU forces us to either over-charge solos or under-deliver to enterprise. "
          "A three-tier ladder with distinct value props per rung lets us address India's actual "
          "legal-services market structure — a long tail of solo practitioners and a short head "
          "of enterprise GRC teams who can pay 25× more for the right product."),
        H2("2.2 The 18-month revenue ceiling"),
    ]
    rev_rows = [
        ["Tier", "Target market in India", "Realistic capture (18 mo)", "Monthly", "ARR ceiling"],
        ["Research $19", "200,000 paying advocates", "2,000 seats", "$38,000", "$456,000"],
        ["Practice $49", "50,000 solo + small firm", "1,000 seats", "$49,000", "$588,000"],
        ["Compliance $349 (avg)", "5,000 enterprise legal teams", "200 seats", "$69,800", "$837,600"],
        ["TOTAL", "", "3,200 seats", "$156,800", "$1.88 M"],
    ]
    s += [
        make_table(rev_rows, col_widths=[3.4 * cm, 4.5 * cm, 3.6 * cm, 2.6 * cm, 2.7 * cm]),
        Note("These are realistic 18-month captures, not theoretical ceilings. The compliance tier "
             "alone roughly equals Practice + Research combined — and is where the moat sits because "
             "compliance buyers do not switch every quarter."),
        PageBreak(),
    ]

    # ─── 3. CUSTOMER JOURNEYS ─────────────────────────────────────────────
    s += [
        H1("3 · The four customer journeys"),
        P("Every product decision is justified by tracing it back to one of the four journeys. "
          "If a feature does not visibly help one of these flows, we do not build it now."),
        H2("Journey A — Citizen, problem solved without lawyer (~70% of calls)"),
        Q("Dial 1800 → Hindi greet → DPDP consent → state problem → NyayaSathi RAG answers "
          "with statutory citation, actionable steps, helpline number → user hangs up satisfied."),
        Note("Outcome: zero revenue, ₹6 cost per call. This is the moat-builder. We capture 1 M+ "
             "citizen-legal-event interactions per year that no competitor can buy."),
        H2("Journey B — Citizen → matched to lawyer (~20% of calls)"),
        Q("5+ turn dialogue → AI offers lawyer → user accepts → budget question → "
          "POST /api/nyaya/intake (signed) → match engine ranks → user picks → SMS to lawyer → "
          "Exotel call-masking connects → recorded call → 7-day SMS outcome survey."),
        Note("Outcome: lawyer pays Sanhita seat fee + commission. This funds NyayaSathi's runway."),
        H2("Journey C — Solo lawyer using Sanhita Practice"),
        Q("Morning login → 3 new leads in inbox + 2 pending drafts + court calendar → "
          "open client → read intake summary + transcript + case-state JSON → 'Research' → "
          "7-section memo with [n] cites → 'Draft Notice' → BNSS §482 anticipatory bail draft → "
          "upload client's voice note to Vault → ask Vault 'what date did the FIR say?' → "
          "answered with citation → send draft via WhatsApp → mark in_progress."),
        Note("Paralegal work that used to take 4 hours, done in 30 minutes. $49/month justified "
             "the moment a single matter is processed."),
        H2("Journey D — Compliance team (e.g., bank's legal department)"),
        Q("GC creates 'Bank-X DPDP' workspace → configures policy: any clause referencing EU data "
          "must flag and route to Senior Counsel approval → junior associate uploads 50 vendor "
          "contracts → Sanhita extracts clauses, classifies risk, flags 12 → 12 flags routed to "
          "Senior Counsel queue with audit trail entry → Senior reviews, approves 8, escalates 4 → "
          "compliance report with hash-chain proof → quarterly DPDP audit: export full audit trail, "
          "prove every AI suggestion had a human checkpoint."),
        Note("This is what makes the $499 tier worth it. The product enforces what the user said "
             "India lacks — judgment, interpretation, accountability — by routing them to humans "
             "and proving cryptographically that it did."),
        PageBreak(),
    ]

    # ─── 4. WHAT AI DOES NOT DO ───────────────────────────────────────────
    s += [
        H1("4 · The architectural commitment — what AI does NOT do"),
        P("This is not a footnote. It is the spine of every product decision and the brand "
          "position that competitors will struggle to copy because copying it means giving up "
          "demos that make them look impressive in a pitch deck."),
        callout(
            "The Four Refusals",
            "<b>R1.</b> AI does not render judgment ('you should sue').&nbsp;&nbsp;"
            "<b>R2.</b> AI does not interpret ambiguous law as if it were settled.&nbsp;&nbsp;"
            "<b>R3.</b> AI does not take accountability ('I'll handle it').&nbsp;&nbsp;"
            "<b>R4.</b> AI does not substitute for a human in regulated decisions.",
        ),
        H2("4.1 How each refusal is enforced in code"),
    ]
    refusal_rows = [
        ["Refusal", "Enforcement"],
        ["R1 No judgment", "Validator G7: regex bans 'you should', 'I recommend', 'I advise', 'your best move is'. Output is forced into options + tradeoffs structure."],
        ["R2 No unmarked interpretation", "Validator G8: claims either cite binding precedent (G1+G2) or are wrapped in {interpretation_alert: true}. UI renders these in amber with 'verify with counsel'."],
        ["R3 No accountability", "Validator G9: regex bans 'I'll handle it', 'leave it to me'. Forces 'you_are_responsible' line in user's language at end of every NyayaSathi voice answer."],
        ["R4 No substitution", "Compliance edition: high-risk clause flags hard-route to a human reviewer queue. Output marked requires_human_signoff=true blocks export until signed."],
    ]
    s += [
        make_table(refusal_rows, col_widths=[4.0 * cm, 13.0 * cm]),
        Spacer(1, 0.3 * cm),
        H2("4.2 The branding implication"),
        Q("Sanhita: <i>We do the research. You do the judgment.</i>"),
        Q("NyayaSathi: <i>जानकारी हमसे, फ़ैसला आपका — और ज़रूरत पड़े तो भरोसेमंद वकील भी।</i> "
          "<br/>(\"Information from us, decision yours — and a trusted lawyer if needed.\")"),
        H2("4.3 Why this is the moat"),
        P("Most legal-AI products fail this test by accident — they over-promise to win demos. "
          "Casetext markets 'AI-drafted legal opinions'; DoNotPay marketed 'be your lawyer'; "
          "both have been or will be regulated. Our position is the opposite: under-promise, "
          "over-deliver on retrieval, refuse to write prose when we cannot ground. This is "
          "harder to copy than any feature because it requires giving up the exact demos "
          "that VCs reward."),
        PageBreak(),
    ]

    # ─── 5. MULTI-MODAL DATA ──────────────────────────────────────────────
    s += [
        H1("5 · Multi-modal data: storage and upload"),
        P("Real legal practice in India does not happen in clean PDFs. It happens in WhatsApp "
          "voice notes, photographed FIR copies, court hearing recordings, dashcam video, and "
          "screenshots of cyber-fraud chats. A product that only handles native PDFs is a toy."),
        H2("5.1 What we receive"),
    ]
    modality_rows = [
        ["Modality", "Source", "Volume per case", "Why it matters"],
        ["Voice notes", "WhatsApp from citizen", "30s – 5 min", "Tone matters, not just words"],
        ["Hearing audio", "Lawyer recording with consent", "30 min – 6 hrs", "What the judge said vs. what was minuted"],
        ["Phone calls", "Exotel call-masking", "20 min – 1 hr", "Both-side recorded"],
        ["Video", "Dashcam, CCTV, mobile", "1 min – 1 hr", "Accident reconstruction"],
        ["Document photos", "FIR, registry, deed (mobile)", "1 – 20 pages", "OCR'd citizen-side filing"],
        ["Chat exports", "WhatsApp, email backup", "KB – MB", "Cyber fraud evidence"],
        ["Native PDFs", "Bare acts, court orders, contracts", "5 – 500 pages", "Already handled"],
    ]
    s += [
        make_table(modality_rows, col_widths=[2.8 * cm, 4.4 * cm, 3.2 * cm, 6.6 * cm]),
        Spacer(1, 0.3 * cm),
        H2("5.2 The pipeline"),
        P("Upload → MIME router → modality-specific extractor → structured output → "
          "Sanhita's existing vault_chunks table (BM25 + Gemini embeddings)."),
        Code("audio/*    →  Sarvam Saaras STT (22 langs) + diarization\n"
             "video/*    →  ffmpeg → keyframes (vision) + audio (STT)\n"
             "image/*    →  Tesseract + Gemini vision (layout-aware)\n"
             "application/pdf  →  pdfplumber + layout extraction\n"
             "text/*     →  direct\n\n"
             "ALL paths produce: {text_blocks, metadata, entities, pii_redacted, hash}\n"
             "Originals NEVER hit the LLM. Only redacted, structured extracts do."),
        H2("5.3 Storage strategy"),
        B("<b>Tier-1 hot (Cloudflare R2 or AWS S3 IA, encrypted at rest):</b> originals; ~$0.015/GB/month."),
        B("<b>Tier-2 cold (Glacier Deep Archive after 90 days no access):</b> ~$0.001/GB/month."),
        B("<b>Hot extracts always live</b> in Postgres + pgvector. Originals retrieved by signed URL on demand."),
        H2("5.4 The cost numbers"),
        P("100 lawyers × 100 cases × 50 MB average = 500 GB hot = $7.50/month. "
          "1,000 lawyers × 100 cases = 5 TB hot = $75/month. "
          "5 TB cold = $5/month. Storage is not the bottleneck; trust is."),
        H2("5.5 Why this is hard to replicate"),
        B("Sarvam STT licensing — ~3× better than Whisper for Hindi/Tamil/Bengali. Indian advantage."),
        B("Redaction-before-LLM is uncommon in legal-AI products; most ship raw transcripts to OpenAI/Anthropic. We stay DPDP-compliant by architecture."),
        B("Tiered storage with structured-extract-as-source-of-truth keeps the long tail nearly free."),
        PageBreak(),
    ]

    # ─── 6. THREE STAGES ──────────────────────────────────────────────────
    s += [
        H1("6 · The three stages of casework: Research → Analysis → Strategy"),
        P("Mapping AI's role to each stage, with the AI-does-not-judge principle held strict."),
    ]
    stages_rows = [
        ["Stage", "AI role", "Validation", "Pricing tier"],
        ["1 Research\n'What does law say?'", "Heavy. Hybrid retrieval + 7-section memo with [n] cites + citation graph + 'still good law?' check + multi-jurisdiction comparison.", "G1–G6. Refuses prose when can't ground.", "Research $19"],
        ["2 Analysis\n'How does law map to facts?'", "Structured assistance. Fact extraction from vault, element-by-element matrix (statutory element × evidence × strength), counter-arguments surfaced, evidence gaps flagged.", "G1–G9. Each cell cites source.", "Practice $49"],
        ["3 Strategy\n'What do we do?'", "Minimal. Surface option set with base-rate, time, cost, risk per option. Procedural roadmap. Predictive surface gives outcome probabilities. AI does NOT pick.", "G7 (no judgment) hardest enforced. Lawyer's pick recorded with reason → audit trail.", "Compliance $199–499"],
    ]
    s += [
        make_table(stages_rows, col_widths=[4.0 * cm, 7.0 * cm, 3.5 * cm, 2.5 * cm], font_size=8.5),
        Spacer(1, 0.3 * cm),
        Note("This separation is also the billing structure. Research alone is commodity. Analysis "
             "needs case-state extraction and multi-modal vault. Strategy demands audit-trailed "
             "human checkpoints. Each tier earns its price."),
        PageBreak(),
    ]

    # ─── 7. NEW SURFACES ──────────────────────────────────────────────────
    s += [
        H1("7 · New surfaces: Predictive · Compliance · Contract Intelligence"),
        H2("7.1 Predictive analysis (case outcome probability)"),
        P("Given case facts + jurisdiction + parties, returns probability of favourable outcome "
          "<i>with</i> the comparable past cases it is based on, median time-to-disposition, "
          "median cost, procedural roadmap. Output is never a single number alone — always paired "
          "with the comparables. Validator G10: predictive output must include base rate + "
          "sample size."),
        Note("The data work is the moat: Indian outcome data is fragmented across HC websites, "
             "and aggregation + normalisation takes 6–12 months. Our outcome-SMS flywheel "
             "(7-day citizen survey post-handoff) is the only continuous source."),
        H2("7.2 Compliance edition (the human-in-the-loop architecture)"),
        P("Every AI suggestion in compliance mode passes through configurable human checkpoints "
          "with hash-chained audit trail."),
        Code("Workspaces — tenant boundary (per-org corpus filter, policy set, members)\n"
             "Roles      — Counsel · Senior Counsel · Compliance Officer · Auditor · Read-only\n"
             "Policies   — declarative YAML, e.g.:\n"
             "             policy: bank_x_dpdp\n"
             "             triggers:\n"
             "               - clause_type: data_processing\n"
             "                 jurisdiction: EU\n"
             "                 action: require_role: Senior Counsel\n"
             "               - amount_inr: \"> 1_00_00_000\"\n"
             "                 action: require_role: GC\n"
             "               - banned_clauses: [unlimited_indemnity, perpetual_license]\n"
             "                 action: hard_block\n"
             "Checkpoints — every AI output flips requires_human_signoff = true; queued not delivered\n"
             "Audit trail — append-only events with sha256(prev_event) hash chain"),
        H2("7.3 Contract intelligence (Indian-specific edge)"),
        P("Most contract-AI (Harvey, Spellbook, Ironclad) is US/UK-trained. They get Indian "
          "Stamp Act, Companies Act, FEMA, RBI, GST implications wrong. We do not."),
    ]
    contract_rows = [
        ["Layer", "Output"],
        ["Clause classification", "Typed taxonomy: 'indemnity (limited)', 'IP assignment', 'termination for convenience'"],
        ["IP conflict detection", "Flags clauses contradicting prior agreements vault-wide"],
        ["Indemnity analysis", "Cap, scope, carve-outs vs. industry baseline"],
        ["Limitation of liability", "Enforceability across IN / SG / HK"],
        ["Insurance fit", "Required coverage given clause structure"],
        ["Stamp duty", "Jurisdiction-specific under Indian Stamp Act"],
        ["Governing law / forum", "Conflict-of-law issues, asymmetric forum risk"],
    ]
    s += [
        make_table(contract_rows, col_widths=[5.0 * cm, 12.0 * cm]),
        PageBreak(),
    ]

    # ─── 8. DATA MOATS ────────────────────────────────────────────────────
    s += [
        H1("8 · The five data moats — with replication math"),
        P("These are the assets that compound over time. A competitor with funding can copy "
          "any single one in months. Replicating all five together is an 18-month, $700K+ "
          "exercise — and by then we are 18 months further ahead."),
        H2("8.1 Moat-by-moat"),
    ]
    moat_rows = [
        ["Moat", "Replication time", "Replication cost", "Compounds with..."],
        ["Two-sided structure (NyayaSathi → Sanhita)", "12–18 months", "$200K+", "Citizen acquisition cost approaches zero"],
        ["Outcome-data flywheel", "Cannot shortcut", "Must operate", "Predictive surface accuracy"],
        ["Indian NLP infrastructure", "12 weeks", "$60K", "Every quarter the gap widens"],
        ["Multi-modal Indian legal vault", "6 weeks", "$40K", "Storage tail is nearly free"],
        ["Hash-chained audit trail", "4 weeks", "$25K", "Compliance buyers do not switch"],
        ["Eval-suite-as-asset", "Cannot shortcut", "Must operate", "Every regression locks in quality"],
        ["Sanhita's existing 7,839 LOC", "4–6 months", "$250K", "Already battle-tested"],
    ]
    s += [
        make_table(moat_rows, col_widths=[5.0 * cm, 3.4 * cm, 2.8 * cm, 5.8 * cm]),
        Spacer(1, 0.3 * cm),
        H2("8.2 The Indian-NLP infrastructure budget breakdown"),
    ]
    nlp_rows = [
        ["Asset", "Build cost", "Replication time"],
        ["BNS↔IPC mapping table (600 sections)", "1 week", "Cannot shortcut — requires Indian legal context"],
        ["Hindi legal tokenizer (compounds, sandhi)", "2 weeks", "Cannot shortcut"],
        ["Citation parser (SCC, AIR, JT formats)", "2 weeks", "Cannot shortcut"],
        ["OCR pipeline tuned for e-Courts PDFs", "3 weeks", "Cannot shortcut"],
        ["Section renumbering migration (IPC → BNS, CrPC → BNSS)", "1 week", "Cannot shortcut"],
        ["Helpline registry with quarterly verification", "1 week + ongoing", "Operational discipline"],
        ["DLSA per-state referral table", "2 weeks", "Operational discipline"],
    ]
    s += [
        make_table(nlp_rows, col_widths=[6.5 * cm, 2.5 * cm, 8.0 * cm]),
        Note("12 weeks of Indian-specific work that a foreign competitor cannot replicate without an India team."),
        PageBreak(),
    ]

    # ─── 9. INTEGRATION PLAN ──────────────────────────────────────────────
    s += [
        H1("9 · The integration plan — where every line of new code lives"),
        H2("9.1 On NyayaSathi"),
        B("<font face='Courier'>src/matching/sanhita-client.js</font> — HMAC-signed POST to Sanhita /api/nyaya/intake (~80 LOC)"),
        B("<font face='Courier'>src/validators/answer-gates.js</font> — port of Sanhita's G1–G6 (~250 LOC port)"),
        B("<font face='Courier'>src/llm/router.js</font> — port of Sanhita's 4-provider router (~400 LOC port)"),
        B("<font face='Courier'>src/voice/gender-detector.js</font> — F0 estimation; female-voice auto-switch (~150 LOC)"),
        B("<font face='Courier'>src/safety/distress-llm.js</font> — paraphrase classifier in parallel with keyword scan (~150 LOC)"),
        B("DPDP + recording disclosure first 10s of every call (~100 LOC)"),
        B("Distress copy translated to all 11 languages (data work)"),
        H2("9.2 On Sanhita"),
        B("<font face='Courier'>HMAC verification middleware</font> on /api/nyaya/intake (~30 LOC)"),
        B("<font face='Courier'>lawyer_match.py</font> — scoring engine over a new lawyers_profile table (~250 LOC + 1 migration)"),
        B("New endpoints: <font face='Courier'>/api/match/run</font> and <font face='Courier'>/api/match/accept</font> (~150 LOC)"),
        B("<font face='Courier'>vault/extractors/{audio,video,image}.py</font> — multi-modal pipeline (~500 LOC)"),
        B("<font face='Courier'>workspaces.py</font>, <font face='Courier'>policies.py</font>, <font face='Courier'>audit_chain.py</font> — compliance primitives (~800 LOC)"),
        B("<font face='Courier'>workflows.py</font> — Contract Intelligence workflow (~600 LOC)"),
        B("<font face='Courier'>predictive.py</font> — outcome probability surface (~400 LOC)"),
        B("<font face='Courier'>validators/answer_gates.py</font> — extend with G7 G8 G9 G10 (~150 LOC)"),
        H2("9.3 Shared infrastructure"),
        B("<font face='Courier'>llm/sarvam.py</font> — extend with stt() and tts() helpers"),
        B("Audit chain library reused by both products"),
        B("Eval harness — Sanhita's <font face='Courier'>eval/run.py</font> extended to cover NyayaSathi voice prompts"),
        Spacer(1, 0.3 * cm),
        H2("9.4 Total accounting"),
        callout(
            "≈4,000 lines of new code on top of 7,839 lines of stable Python.",
            "Three sprints. Three weeks. Eval-gated. Day-21 launch.",
        ),
        PageBreak(),
    ]

    # ─── 10. WHY NO COMPETITOR MATCHES ────────────────────────────────────
    s += [
        H1("10 · Why no competitor can match — with stats"),
        H2("10.1 The aggregate replication math"),
    ]
    rep_rows = [
        ["Asset", "Replication time", "Replication cost"],
        ["Two-sided structure", "12–18 months", "$200K+"],
        ["Three-edition tiering", "6 months", "$150K"],
        ["Outcome-data flywheel", "Cannot shortcut", "Must operate"],
        ["Indian NLP infra (BNS map, citation parser, OCR, etc.)", "12 weeks", "$60K"],
        ["Multi-modal Indian legal vault", "6 weeks", "$40K"],
        ["Hash-chained audit trail (compliance)", "4 weeks", "$25K"],
        ["AI-does-NOT-judge architectural enforcement", "4 weeks", "Brand-defining"],
        ["Eval-as-compounding-asset", "Cannot shortcut", "Must operate"],
        ["Sanhita's existing 7,839 LOC", "4–6 months", "$250K"],
        ["TOTAL minimum to match", "18+ months", "$700K+"],
    ]
    s += [
        make_table(rep_rows, col_widths=[8.5 * cm, 3.5 * cm, 5.0 * cm]),
        Spacer(1, 0.3 * cm),
        H2("10.2 The compounding argument"),
        P("By the time a fast-follower has month-6 quality, we have month-24 quality, "
          "month-24 outcome data, month-24 eval suite, and month-24 Indian-corpus depth. "
          "The gap widens, it does not shrink, because every NyayaSathi call we take is "
          "a data point they cannot retroactively acquire."),
        H2("10.3 What is uncopyable on principle"),
        B("The architectural commitment to refuse judgment: copying it requires giving up demos."),
        B("The outcome flywheel: requires operating the helpline; cannot be bought."),
        B("The eval suite as time-series: requires sustained operation; cannot be bought."),
        PageBreak(),
    ]

    # ─── 11. SPRINT PLAN ──────────────────────────────────────────────────
    s += [
        H1("11 · The 21-day sprint plan, day by day"),
        callout(
            "The shipping rhythm",
            "Every change runs through the eval gate. Every fail becomes a permanent regression test. "
            "The eval suite never shrinks. Sprint targets are floors, not ceilings — we ship at "
            "≥ target or we fix and re-run."
        ),
        H2("Sprint 0 — Bridge + safety floor (Days 1–7)"),
    ]
    sprint0_rows = [
        ["Day", "What ships", "Eval cut"],
        ["1", "Eval harness (CLI runner) + 30 baseline questions + first run against current NyayaSathi", "baseline.json"],
        ["2", "HMAC bridge: NyayaSathi → Sanhita /api/nyaya/intake. sanhita-client.js. Add 20 questions on handoff flow.", "sprint0_d2.json"],
        ["3", "DPDP + recording + AI disclosure first 10s of every call. 11-language disclosure templates.", "sprint0_d3.json"],
        ["4", "Distress copy translated to 11 languages + LLM tier-1.5 paraphrase classifier in parallel with keyword.", "sprint0_d4.json"],
        ["5", "Sentry + safety-event file writer + Slack/email duty-officer alert pipeline.", "sprint0_d5.json"],
        ["6", "Fix worst 5 fail patterns from D5 eval. Author 30 more questions covering safety paraphrases.", "sprint0_d6.json"],
        ["7", "Final Sprint 0 eval against 100 questions. Block if < 85%. Tag v0.5.0.", "sprint0_final.json"],
    ]
    s += [
        make_table(sprint0_rows, col_widths=[1.2 * cm, 12.8 * cm, 3.0 * cm]),
        Spacer(1, 0.3 * cm),
        H2("Sprint 1 — Polish + real lawyer matching (Days 8–14)"),
    ]
    sprint1_rows = [
        ["Day", "What ships", "Eval cut"],
        ["8", "lawyers_profile schema + migration on Sanhita. Bar-council number field with verification stub.", "sprint1_d8.json"],
        ["9", "lawyer_match.py with full scoring formula (specialty + lang + loc + budget + rating + outcome + load-balance + gender-pref).", "sprint1_d9.json"],
        ["10", "/api/match/run + /api/match/accept endpoints. Wire end-to-end into NyayaSathi.", "sprint1_d10.json"],
        ["11", "Female-voice auto-switch (F0 + override + DV-force-female). Emotion-mapped TTS.", "sprint1_d11.json"],
        ["12", "G7 G8 G9 banned-phrase extension. Cache-key fix (include case.type). PII at-rest encryption.", "sprint1_d12.json"],
        ["13", "Token-budget guard + extractor race-fix + buildSystemPrompt cleanup.", "sprint1_d13.json"],
        ["14", "Final Sprint 1 eval against 150 questions. Block if < 90%. Tag v0.7.0.", "sprint1_final.json"],
    ]
    s += [
        make_table(sprint1_rows, col_widths=[1.2 * cm, 12.8 * cm, 3.0 * cm]),
        Spacer(1, 0.3 * cm),
        H2("Sprint 2 — Multi-modal + audit + connection (Days 15–21)"),
    ]
    sprint2_rows = [
        ["Day", "What ships", "Eval cut"],
        ["15", "Multi-modal vault: audio (Sarvam STT), image (Tesseract + Gemini vision).", "sprint2_d15.json"],
        ["16", "Video extractor (ffmpeg keyframes + audio STT). Storage tier-1/tier-2 wiring.", "sprint2_d16.json"],
        ["17", "Hash-chained audit_events table + middleware. Workspaces + RBAC primitives.", "sprint2_d17.json"],
        ["18", "Compliance policy YAML loader + checkpoint queue. G10 predictive-must-include-base-rate gate.", "sprint2_d18.json"],
        ["19", "Exotel call-masking. Lawyer SMS-to-accept (1=accept, 2=reschedule, 3=decline). 5-second audio briefing on connect.", "sprint2_d19.json"],
        ["20", "Per-state DLSA referral table. Right-to-erasure flow. End-to-end production smoke test.", "sprint2_d20.json"],
        ["21", "Final eval against 250 questions. Block if < 92%. End-to-end smoke pass. Tag v1.0.0. LAUNCH.", "sprint2_final.json"],
    ]
    s += [
        make_table(sprint2_rows, col_widths=[1.2 * cm, 12.8 * cm, 3.0 * cm]),
        PageBreak(),
    ]

    # ─── 12. EVAL POLICY ──────────────────────────────────────────────────
    s += [
        H1("12 · Eval policy — the gate that ships or blocks every commit"),
        H2("12.1 Question distribution (matches NALSA 2022 helpline call distribution)"),
    ]
    eval_rows = [
        ["Category", "Count", "Why this volume"],
        ["FIR / police", "25", "28% of NALSA real-world calls"],
        ["Family / DV / divorce / maintenance", "15", "16%; high distress-bypass rate"],
        ["Property / land / rent / RERA", "12", "13%"],
        ["Cheque bounce / fraud / cyber", "10", "11%; growing post-pandemic"],
        ["Bail / arrest", "8", "9%; high-urgency"],
        ["Salary / employment / labour", "8", "8%"],
        ["Consumer / refund", "7", "7%"],
        ["Motor accident", "5", "5%"],
        ["Caste / SC-ST atrocity", "5", "Rural-rich, underserved"],
        ["RTI / govt scheme", "3", "Niche but high-value"],
        ["Distress / safety bypass", "2", "Tested separately — must pass at 100%"],
    ]
    s += [
        make_table(eval_rows, col_widths=[7.0 * cm, 1.5 * cm, 8.5 * cm]),
        Spacer(1, 0.3 * cm),
        H2("12.2 Block rules (non-negotiable)"),
    ]
    block_rows = [
        ["Rule", "Threshold", "Action if violated"],
        ["Distress questions pass rate", "100%", "Hard fail. No deploy."],
        ["Citation hallucination on any question", "0", "Hard fail."],
        ["Banned phrase ('as an AI', weasel) anywhere", "0", "Hard fail."],
        ["Safety-bypass time-to-first-byte", "< 200 ms p95", "Hard fail."],
        ["Overall pass rate", "≥ 85% (S0), ≥ 90% (S1), ≥ 92% (S2)", "Hard fail."],
    ]
    s += [
        make_table(block_rows, col_widths=[7.5 * cm, 4.5 * cm, 5.0 * cm]),
        Spacer(1, 0.3 * cm),
        H2("12.3 The regression rule"),
        Q("Every failure becomes a permanent regression test. The eval suite never shrinks; "
          "only grows. v2 = v1 + new regressions found in Sprint 0. By launch we have ~250 prompts "
          "in v4 and a 21-day time series of model behaviour saved as JSON."),
        PageBreak(),
    ]

    # ─── 13. CODEBASE ORGANISATION ────────────────────────────────────────
    s += [
        H1("13 · Codebase organisation and folder layout"),
        P("As the codebase grows from ~3K to ~7K lines, structure prevents the kind of single-file "
          "monoliths (server.js at 2,400 lines today) that resist future maintenance. Discipline "
          "starts now."),
        H2("13.1 Target structure"),
        Code("nyayasathi/                          (was /Users/pranav/Desktop/n)\n"
             "├── README.md\n"
             "├── docs/\n"
             "│   ├── MASTER-PLAN.pdf               this document\n"
             "│   ├── ARCHITECTURE.md               one-page entry point\n"
             "│   ├── CONTRIBUTING.md               commit + comment conventions\n"
             "│   └── CHANGELOG.md                  per-sprint release notes\n"
             "├── src/                              (Sprint 0 Day 5 cleanup)\n"
             "│   ├── server.js                     route mounting + bootstrap only\n"
             "│   ├── routes/                       per-endpoint files\n"
             "│   ├── case/                         store · intent · slots · extractor\n"
             "│   ├── safety/                       distress · helplines\n"
             "│   ├── matching/                     sanhita-client\n"
             "│   ├── validators/                   answer-gates · input-guards\n"
             "│   ├── llm/                          router · gemini · sarvam · circuit-breaker\n"
             "│   ├── voice/                        stt · tts · gender-detector\n"
             "│   ├── rag/                          retrieval · citation-resolver\n"
             "│   ├── audit/                        log · hash-chain\n"
             "│   └── lib/                          security · rules-engine\n"
             "├── public/                           static assets\n"
             "├── data/                             gitignored\n"
             "│   ├── cases/  sessions/  safety-events/\n"
             "├── eval/\n"
             "│   ├── v1/  questions.jsonl  README.md\n"
             "│   ├── harness.js                    CLI runner\n"
             "│   ├── checks/                       citations · banned · grounding · helplines · language\n"
             "│   ├── reports/                      versioned JSON, kept forever\n"
             "│   └── fixtures/                     mock audio · mock transcripts\n"
             "├── tests/                            unit tests\n"
             "├── .gitignore  package.json\n"),
        H2("13.2 Migration sequence"),
        B("Day 1–4: leave src/ flat as-is; build new code into the target structure (eval/, src/safety/, src/matching/)."),
        B("Day 5 of Sprint 0: move existing files into target structure with git mv (preserves history). Update imports. Run eval to confirm no breakage."),
        B("Sprint 1 onward: every new file lives in the target structure."),
        H2("13.3 Why this layout"),
        B("Each folder maps to a section of this plan: a new contributor can find anything in <30 sec."),
        B("Eval is a peer of src/, not buried — it is a first-class artefact, not an afterthought."),
        B("Audit module is its own folder because compliance-edition demands cryptographic separation of concerns."),
        PageBreak(),
    ]

    # ─── 14. COMMIT CONVENTIONS ───────────────────────────────────────────
    s += [
        H1("14 · Commit and comment conventions"),
        H2("14.1 Commit message format (Conventional Commits, lightly extended)"),
        Code("<type>(<scope>): <imperative subject under 60 chars>\n\n"
             "<body — wrap at 72 chars; explain *why* not *what*>\n\n"
             "Refs: section 11 sprint 0 day 1\n"
             "Eval: 84% → 87% (sprint0_d2.json)\n"
             "Co-Authored-By: <name> <email>\n"),
        H2("14.2 Allowed types"),
    ]
    type_rows = [
        ["Type", "Use for"],
        ["feat", "User-visible new capability"],
        ["fix", "Bug fix"],
        ["refactor", "Internal restructuring without behaviour change"],
        ["chore", "Tooling, deps, config"],
        ["docs", "Documentation only"],
        ["test", "Eval question additions, unit tests"],
        ["safety", "Distress, helpline, DPDP, audit-trail changes (always reviewed)"],
        ["data", "Corpus or eval-data changes (additive only)"],
    ]
    s += [
        make_table(type_rows, col_widths=[3.0 * cm, 14.0 * cm]),
        Spacer(1, 0.3 * cm),
        H2("14.3 Allowed scopes"),
        Code("server · case · safety · matching · validators · llm · voice · rag · audit\n"
             "eval · docs · routes · vault · workflows · sanhita-bridge · contract-intel"),
        H2("14.4 Examples"),
        Code("safety(distress): translate critical helpline copy to 11 languages\n\n"
             "Adds Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi,\n"
             "Odia plus existing Hindi/English. Bengali DV victim was getting Hindi response\n"
             "before — empirically zero conversion to dialing 181.\n\n"
             "Refs: section 5 (multi-modal), section 11 day 4\n"
             "Eval: 78% → 84% on distress subset (sprint0_d4.json)\n"),
        Code("feat(matching): replace lawyers.json with sanhita-client HMAC bridge\n\n"
             "Deletes mock lawyers.json + lawyer-match.js. NyayaSathi now POSTs to Sanhita\n"
             "/api/nyaya/intake with HMAC-SHA256 signature. Sanhita's match engine returns\n"
             "ranked candidates from the lawyers_profile table.\n\n"
             "Refs: section 9.1, section 11 day 9\n"
             "Eval: 87% → 89% (sprint1_d9.json)"),
        H2("14.5 Comment style for code"),
        Code("// Phase C.3: extracts entities asynchronously after the user-facing reply\n"
             "// has shipped. Updates {entities, summary, urgency, needs_lawyer}. Skipped\n"
             "// for ≤2-word messages (not worth the API call). 8s timeout — JSON-mode\n"
             "// is sometimes slow on Flash; failures are caught up on the next turn.\n"),
        B("Block comments at the head of each module explain <i>why</i> the module exists, what it owns, and what it does NOT own."),
        B("Inline comments mark non-obvious decisions and trade-offs (timeouts, retry policies, regex carve-outs)."),
        B("Never document <i>what</i> the code does at line level — the code says that already."),
        H2("14.6 Pre-merge checklist"),
        B("Eval ran with pass rate ≥ sprint target."),
        B("New failures added as regression tests in eval/v1/questions.jsonl."),
        B("Conventional Commit message with Refs and Eval lines."),
        B("CHANGELOG.md updated for the current sprint."),
        PageBreak(),
    ]

    # ─── 15. DAY 1 ────────────────────────────────────────────────────────
    s += [
        H1("15 · Day 1 — what we do tomorrow"),
        callout(
            "The first 24 hours",
            "We do not write feature code. We build the gate that every future ship will pass through. "
            "By end of Day 1 we have an eval harness, 30 baseline questions, the first regression "
            "tests, and a baseline.json measuring exactly where we start."
        ),
        H2("15.1 Concrete actions"),
        B("Create eval/harness.js — CLI runner, ~150 LOC, POSTs to localhost:3000, captures, scores, saves report JSON."),
        B("Create eval/v1/questions.jsonl with 30 hand-authored prompts (10 FIR, 8 family, 6 property, 4 cheque-bounce, 2 distress)."),
        B("Create eval/checks/ with 5 modules: citations.js, banned_phrases.js, helplines.js, grounding.js, language.js."),
        B("Run baseline. Save eval/reports/baseline.json. Print summary."),
        B("Pick the worst three fail patterns. File as Day 2 fix targets in CHANGELOG."),
        B("Commit with message: <font face='Courier'>test(eval): bootstrap harness + 30 baseline questions</font>"),
        B("Tag commit: <font face='Courier'>v0.4.0-baseline</font>"),
        H2("15.2 Definition of done for Day 1"),
        B("<font face='Courier'>npm run eval</font> works from project root."),
        B("Pass rate measured (we expect 50–65% baseline)."),
        B("Top three failure patterns named in writing."),
        B("Day 2 plan reflects what the eval revealed, not what we assumed."),
        Spacer(1, 0.5 * cm),
        callout(
            "The promise we are making to ourselves",
            "We do not lower the eval threshold to ship. We do not skip the regression test. "
            "We do not delete failing questions. The discipline of <i>not lowering thresholds</i> "
            "is the entire game — it is what separates a public-good service from yet another "
            "VC-funded chatbot."
        ),
        Spacer(1, 0.5 * cm),
        Paragraph("End of plan. Re-generate after every sprint or whenever scope changes.",
                  _style("end", size=9, leading=12, color=INK_SOFT, italic=True, alignment=TA_CENTER)),
    ]

    return s


# ── Build ─────────────────────────────────────────────────────────────────
def main():
    doc = BaseDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="NyayaSathi × Sanhita — Master Plan",
        author="NyayaSathi Team",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="default", frames=[frame], onPage=_draw_chrome)])

    story = build_story()
    doc.build(story)
    print(f"OK: wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
