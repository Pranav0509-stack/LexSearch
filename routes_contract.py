"""routes_contract.py — FastAPI router for the Contract Workbench.

Mounted into server.py via:
    from routes_contract import router as contract_router
    app.include_router(contract_router)

Endpoints (all under /api/contract):
    GET    /templates                   List available templates (filterable)
    GET    /templates/{tid}             Template details + slot schema + clauses
    GET    /clauses                     Free-text search the clause library
    POST   /matters                     Create a matter
    GET    /matters                     List user's matters
    GET    /matters/{mid}               Matter + drafts
    POST   /draft                       Generate a draft (multi-pass)
    POST   /draft/stream                Generate w/ SSE (token streaming, D2)
    GET    /drafts/{did}                Fetch a draft + provenance
    POST   /review                      Run review pass on a draft
    POST   /redline                     Diff two drafts (D3)
    POST   /export                      Export to DOCX/PDF (D3)
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Wire the corpus repo onto sys.path so we can import the drafter
_CORPUS = Path("/Users/pranav/Desktop/india-judgments-corpus")
if str(_CORPUS) not in sys.path:
    sys.path.insert(0, str(_CORPUS))

from scripts.contract.draft       import generate as draft_generate  # noqa: E402
from scripts.contract.review      import review as draft_review       # noqa: E402
from scripts.contract.redline     import diff_drafts                  # noqa: E402
from scripts.contract.export      import export as draft_export       # noqa: E402
from scripts.contract.compliance  import run_all as compliance_run    # noqa: E402
from scripts.contract.citations   import citations_for_anchors        # noqa: E402
from scripts.contract.nudges      import run as nudges_run            # noqa: E402
from scripts.contract.quick_edit  import quick_edit                   # noqa: E402

from fastapi.responses import FileResponse

DB_PATH = _CORPUS / "india_courts.db"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH), timeout=60.0)
    c.execute("PRAGMA busy_timeout=60000")
    c.execute("PRAGMA journal_mode=WAL")
    c.row_factory = sqlite3.Row
    return c


router = APIRouter(prefix="/api/contract", tags=["contract"])


# ── Models ───────────────────────────────────────────────────────────────

class TemplateRow(BaseModel):
    id: str
    domain: str
    doc_type: str
    jurisdiction: Optional[str] = None
    title: str
    description: Optional[str] = None
    version: int = 1
    source_authority: Optional[str] = "practice_standard"


class TemplateDetail(TemplateRow):
    slots: list[dict] = []
    anchors: dict = {}
    risk_profile: str = "standard"
    clause_count: int = 0
    source_authority: str = "practice_standard"
    source_url: Optional[str] = None
    verified_by: Optional[str] = None
    provenance_note: Optional[str] = None


class MatterCreate(BaseModel):
    client_name: Optional[str] = None
    counterparty: Optional[str] = None
    domain: Optional[str] = None
    doc_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    tags: Optional[list[str]] = None


class DraftRequest(BaseModel):
    template_id: str
    slots: dict = Field(default_factory=dict)
    matter_id: Optional[str] = None
    jurisdiction: Optional[str] = None
    extra_instructions: Optional[str] = ""
    mode: str = Field(default="multi_pass", pattern="^(deterministic_only|one_pass|multi_pass)$")


class DraftResponse(BaseModel):
    draft_id: str
    matter_id: str
    template_id: str
    body_md: str
    word_count: int
    risk_score: float
    passes: list[dict]
    critique: Optional[dict] = None
    error: Optional[str] = None


# ── Templates ────────────────────────────────────────────────────────────

@router.get("/templates", response_model=list[TemplateRow])
def list_templates(
    domain: Optional[str] = None,
    doc_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    q: Optional[str] = None,
):
    sql = "SELECT id, domain, doc_type, jurisdiction, title, description, version, source_authority FROM contract_templates WHERE 1=1"
    args: list = []
    if domain:
        sql += " AND domain = ?"; args.append(domain)
    if doc_type:
        sql += " AND doc_type = ?"; args.append(doc_type)
    if jurisdiction:
        sql += " AND (jurisdiction = ? OR jurisdiction IS NULL)"; args.append(jurisdiction)
    if q:
        sql += " AND (title LIKE ? OR description LIKE ?)"
        args.extend([f"%{q}%", f"%{q}%"])
    sql += " ORDER BY domain, doc_type"
    with _conn() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


@router.get("/templates/{tid}", response_model=TemplateDetail)
def template_detail(tid: str):
    with _conn() as c:
        r = c.execute(
            """SELECT id, domain, doc_type, jurisdiction, title, description,
                      version, slots_json, anchors_json, risk_profile,
                      source_authority, source_url, verified_by, provenance_note
               FROM contract_templates WHERE id = ?""", (tid,)
        ).fetchone()
        if not r:
            raise HTTPException(404, f"template {tid} not found")
        nclauses = c.execute(
            "SELECT COUNT(*) FROM contract_clauses WHERE template_id = ?", (tid,)
        ).fetchone()[0]
    return TemplateDetail(
        id=r["id"], domain=r["domain"], doc_type=r["doc_type"],
        jurisdiction=r["jurisdiction"], title=r["title"],
        description=r["description"], version=r["version"],
        slots=json.loads(r["slots_json"] or "[]"),
        anchors=json.loads(r["anchors_json"] or "{}"),
        risk_profile=r["risk_profile"] or "standard",
        clause_count=nclauses,
        source_authority=r["source_authority"] or "practice_standard",
        source_url=r["source_url"],
        verified_by=r["verified_by"],
        provenance_note=r["provenance_note"],
    )


@router.get("/clauses")
def search_clauses(
    q: Optional[str] = None,
    name: Optional[str] = None,
    risk_tier: Optional[str] = None,
    template_id: Optional[str] = None,
    limit: int = Query(default=50, le=200),
):
    sql = "SELECT id, template_id, name, risk_tier, substr(body_md,1,400) AS preview, statute_refs FROM contract_clauses WHERE 1=1"
    args: list = []
    if q:
        sql += " AND body_md LIKE ?"; args.append(f"%{q}%")
    if name:
        sql += " AND name LIKE ?"; args.append(f"%{name}%")
    if risk_tier:
        sql += " AND risk_tier = ?"; args.append(risk_tier)
    if template_id:
        sql += " AND template_id = ?"; args.append(template_id)
    sql += " ORDER BY name LIMIT ?"; args.append(limit)
    with _conn() as c:
        rows = [dict(r) for r in c.execute(sql, args).fetchall()]
    for r in rows:
        try:
            r["statute_refs"] = json.loads(r.get("statute_refs") or "[]")
        except Exception:
            r["statute_refs"] = []
    return rows


# ── Matters ──────────────────────────────────────────────────────────────

@router.post("/matters")
def create_matter(body: MatterCreate):
    mid = f"m_{uuid.uuid4().hex[:16]}"
    with _conn() as c:
        c.execute(
            """INSERT INTO matters
               (id, client_name, counterparty, domain, doc_type, jurisdiction,
                status, tags_json)
               VALUES (?,?,?,?,?,?,?,?)""",
            (mid, body.client_name, body.counterparty, body.domain,
             body.doc_type, body.jurisdiction, "draft",
             json.dumps(body.tags or [])),
        )
        c.commit()
    return {"id": mid, "status": "draft"}


@router.get("/matters")
def list_matters(limit: int = 50):
    with _conn() as c:
        rows = c.execute(
            """SELECT id, client_name, counterparty, domain, doc_type,
                      jurisdiction, status, created_at, updated_at
               FROM matters ORDER BY created_at DESC LIMIT ?""", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


@router.get("/matters/{mid}")
def matter_detail(mid: str):
    with _conn() as c:
        m = c.execute("SELECT * FROM matters WHERE id = ?", (mid,)).fetchone()
        if not m:
            raise HTTPException(404, "matter not found")
        drafts = c.execute(
            """SELECT id, version, template_id, word_count, risk_score,
                      gen_mode, created_at
               FROM drafts WHERE matter_id = ? ORDER BY version DESC""",
            (mid,),
        ).fetchall()
    return {"matter": dict(m), "drafts": [dict(d) for d in drafts]}


# ── Draft ────────────────────────────────────────────────────────────────

@router.post("/draft", response_model=DraftResponse)
def draft(req: DraftRequest):
    t0 = time.time()
    res = draft_generate(
        req.template_id,
        req.slots,
        matter_id=req.matter_id,
        jurisdiction=req.jurisdiction,
        extra_instructions=req.extra_instructions or "",
        gen_mode=req.mode,
    )
    if res.error:
        raise HTTPException(400, res.error)
    return DraftResponse(
        draft_id=res.draft_id,
        matter_id=res.matter_id,
        template_id=res.template_id,
        body_md=res.body_md,
        word_count=res.word_count,
        risk_score=res.risk_score,
        passes=res.passes,
        critique=res.critique,
    )


@router.get("/drafts/{did}")
def draft_detail(did: str):
    with _conn() as c:
        d = c.execute("SELECT * FROM drafts WHERE id = ?", (did,)).fetchone()
        if not d:
            raise HTTPException(404, "draft not found")
        prov = c.execute(
            """SELECT paragraph_idx, clause_name, source_type, source_id,
                      statute_act, statute_section, confidence
               FROM draft_provenance WHERE draft_id = ? ORDER BY paragraph_idx""",
            (did,),
        ).fetchall()
    out = dict(d)
    try:
        out["llm_trace"] = json.loads(out.pop("llm_trace_json") or "{}")
        out["slots"]     = json.loads(out.pop("slots_json") or "{}")
    except Exception:
        pass
    out["provenance"] = [dict(p) for p in prov]
    return out


# ── Review ───────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    draft_id: Optional[str] = None
    body_md:  Optional[str] = None
    template_id: Optional[str] = None
    playbook_rules: Optional[dict] = None
    use_llm: bool = True


@router.post("/review")
def review_endpoint(req: ReviewRequest):
    res = draft_review(
        body_md=req.body_md,
        draft_id=req.draft_id,
        template_id=req.template_id,
        playbook_rules=req.playbook_rules,
        use_llm=req.use_llm,
    )
    if res.error:
        raise HTTPException(400, res.error)
    return {
        "review_id":      res.review_id,
        "draft_id":       res.draft_id,
        "risk_score":     res.risk_score,
        "coverage_score": res.coverage_score,
        "findings":       res.findings,
    }


# ── Redline ──────────────────────────────────────────────────────────────

class RedlineRequest(BaseModel):
    base_md: Optional[str] = None
    new_md:  Optional[str] = None
    base_draft_id: Optional[str] = None
    new_draft_id:  Optional[str] = None


@router.post("/redline")
def redline_endpoint(req: RedlineRequest):
    base = req.base_md
    new  = req.new_md
    if req.base_draft_id and not base:
        with _conn() as c:
            r = c.execute("SELECT body_md FROM drafts WHERE id = ?", (req.base_draft_id,)).fetchone()
            if r: base = r["body_md"]
    if req.new_draft_id and not new:
        with _conn() as c:
            r = c.execute("SELECT body_md FROM drafts WHERE id = ?", (req.new_draft_id,)).fetchone()
            if r: new = r["body_md"]
    if not base or not new:
        raise HTTPException(400, "need base_md+new_md or matching draft ids")
    res = diff_drafts(base, new,
                      base_draft_id=req.base_draft_id,
                      new_draft_id=req.new_draft_id,
                      persist=bool(req.base_draft_id and req.new_draft_id))
    return {
        "redline_id":     res.redline_id,
        "severity_score": res.severity_score,
        "stats":          res.stats,
        "diff":           res.diff,
    }


# ── Export ───────────────────────────────────────────────────────────────

class ExportRequest(BaseModel):
    draft_id: Optional[str] = None
    body_md:  Optional[str] = None
    format:   str = Field(default="docx", pattern="^(docx|pdf|html|md)$")
    title:    str = "Document"


@router.post("/export")
def export_endpoint(req: ExportRequest):
    import tempfile, os
    suffix = f".{req.format}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    try:
        p = draft_export(body_md=req.body_md, draft_id=req.draft_id,
                         fmt=req.format, out=Path(tmp.name), title=req.title)
    except Exception as e:
        raise HTTPException(400, str(e))
    media = {
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pdf":  "application/pdf",
        "html": "text/html",
        "md":   "text/markdown",
    }.get(req.format, "application/octet-stream")
    return FileResponse(str(p), media_type=media,
                        filename=f"{req.title.replace(' ','_')}.{req.format}")


# ── Compliance ──────────────────────────────────────────────────────────

class ComplianceRequest(BaseModel):
    draft_id: Optional[str] = None
    body_md:  Optional[str] = None
    doc_type: Optional[str] = ""


@router.post("/compliance")
def compliance_endpoint(req: ComplianceRequest):
    body = req.body_md
    doc_type = req.doc_type
    if req.draft_id and not body:
        with _conn() as c:
            r = c.execute(
                """SELECT d.body_md, t.doc_type
                   FROM drafts d LEFT JOIN contract_templates t ON t.id = d.template_id
                   WHERE d.id = ?""", (req.draft_id,)
            ).fetchone()
            if r:
                body = r["body_md"]
                doc_type = doc_type or r["doc_type"]
    if not body:
        raise HTTPException(400, "body_md or draft_id required")
    findings = compliance_run(body, {"doc_type": doc_type or ""})
    # Annotate every finding + remediation message with BNS/BNSS/BSA labels
    # so lawyers see the post-2024 code numbers wherever IPC/CrPC/IEA appears.
    try:
        from legal_code_mapping import annotate_text   # type: ignore
        for f in findings:
            if hasattr(f, "finding") and f.finding:
                f.finding = annotate_text(f.finding)
            if hasattr(f, "remediation") and f.remediation:
                f.remediation = annotate_text(f.remediation)
    except ImportError:
        pass
    # Persist if linked to a draft
    if req.draft_id:
        with _conn() as c:
            for f in findings:
                c.execute(
                    """INSERT INTO contract_compliance
                       (draft_id, plugin, rule_id, severity, finding, remediation)
                       VALUES (?,?,?,?,?,?)""",
                    (req.draft_id, f.plugin, f.rule_id, f.severity,
                     f.finding, f.remediation),
                )
            c.commit()
    return {
        "draft_id": req.draft_id,
        "count": len(findings),
        "findings": [f.asdict() for f in findings],
    }


# ── Citations (case law for statute anchors) ────────────────────────────

class CitationsRequest(BaseModel):
    anchors: Optional[list[dict]] = None     # explicit [{act, section, note}, ...]
    template_id: Optional[str] = None        # OR fetch anchors from template
    draft_id: Optional[str] = None           # OR fetch anchors from draft's template
    topn: int = Field(default=5, ge=1, le=20)


@router.post("/citations")
def citations_endpoint(req: CitationsRequest):
    anchors = req.anchors
    if not anchors and (req.template_id or req.draft_id):
        with _conn() as c:
            tid = req.template_id
            if not tid and req.draft_id:
                r = c.execute("SELECT template_id FROM drafts WHERE id = ?",
                              (req.draft_id,)).fetchone()
                if r:
                    tid = r["template_id"]
            if tid:
                r = c.execute("SELECT anchors_json FROM contract_templates WHERE id = ?",
                              (tid,)).fetchone()
                if r:
                    try:
                        anchors = (json.loads(r["anchors_json"] or "{}").get("statutes") or [])
                    except Exception:
                        anchors = []
    if not anchors:
        raise HTTPException(400, "anchors / template_id / draft_id required")
    enriched = citations_for_anchors(anchors, topn=req.topn)
    return {
        "count": sum(a["count"] for a in enriched),
        "anchors": enriched,
    }


# ── Nudges (inline contract intelligence) ───────────────────────────────

class NudgesRequest(BaseModel):
    draft_id: Optional[str] = None
    body_md:  Optional[str] = None
    with_cases: bool = True
    topn_per_nudge: int = Field(default=3, ge=1, le=10)


@router.post("/nudges")
def nudges_endpoint(req: NudgesRequest):
    body = req.body_md
    if not body and req.draft_id:
        with _conn() as c:
            r = c.execute("SELECT body_md FROM drafts WHERE id = ?",
                          (req.draft_id,)).fetchone()
            if r:
                body = r["body_md"]
    if not body:
        raise HTTPException(400, "body_md or draft_id required")
    nudges = nudges_run(body, with_cases=req.with_cases,
                        topn_per_nudge=req.topn_per_nudge)
    by_sev = {"high": 0, "warn": 0, "info": 0}
    for n in nudges:
        by_sev[n.get("severity", "info")] = by_sev.get(n.get("severity", "info"), 0) + 1
    return {
        "count": len(nudges),
        "by_severity": by_sev,
        "case_links": sum(len(n.get("cases", [])) for n in nudges),
        "nudges": nudges,
    }


# ── Quick edit (AI as a scalpel) ────────────────────────────────────────

class QuickEditRequest(BaseModel):
    action: str = Field(..., pattern="^(polish|shorten|cite)$")
    text:   str


@router.post("/quick-edit")
def quick_edit_endpoint(req: QuickEditRequest):
    res = quick_edit(req.action, req.text)
    if res.error and res.edited == res.original:
        # Soft-error — return original text and the reason
        return {
            "action": res.action, "edited": res.original,
            "unchanged": True, "reason": res.error,
            "model": res.model, "latency_ms": res.latency_ms,
        }
    return {
        "action": res.action,
        "edited": res.edited,
        "model": res.model,
        "tokens_in": res.tokens_in,
        "tokens_out": res.tokens_out,
        "latency_ms": res.latency_ms,
        "citations_used": [
            {"title": c.get("title"), "court": c.get("court"), "year": c.get("year")}
            for c in (res.citations_used or [])
        ],
        "unchanged": False,
    }


@router.get("/health")
def health():
    with _conn() as c:
        nt = c.execute("SELECT COUNT(*) FROM contract_templates").fetchone()[0]
        nc = c.execute("SELECT COUNT(*) FROM contract_clauses").fetchone()[0]
        nm = c.execute("SELECT COUNT(*) FROM matters").fetchone()[0]
        nd = c.execute("SELECT COUNT(*) FROM drafts").fetchone()[0]
    return {
        "status": "ok",
        "templates": nt, "clauses": nc,
        "matters": nm, "drafts": nd,
    }
