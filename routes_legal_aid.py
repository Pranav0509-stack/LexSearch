"""routes_legal_aid.py — Sanhita for Legal Aid application intake.

Stores incoming applications in a tiny SQLite table (in the auth db so we
don't pollute the corpus). One POST endpoint:

    POST /api/legal-aid/apply  {org_name, org_type, contact_name, role,
                                email, phone, jurisdiction, caseload, why}

Returns {application_id, status: "received"} for the UI.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, Field


# Reuse the auth db file — small, separate from corpus
_LOCAL = Path(__file__).resolve().parent
DB = _LOCAL / "lexsearch.db"


router = APIRouter(prefix="/api/legal-aid", tags=["legal-aid"])


def _ensure_table() -> None:
    with sqlite3.connect(str(DB), timeout=10.0) as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS legal_aid_applications (
            application_id TEXT PRIMARY KEY,
            org_name       TEXT NOT NULL,
            org_type       TEXT,
            contact_name   TEXT,
            role           TEXT,
            email          TEXT,
            phone          TEXT,
            jurisdiction   TEXT,
            caseload       TEXT,
            why            TEXT,
            status         TEXT DEFAULT 'received',
            payload_json   TEXT,
            received_at    INTEGER DEFAULT (strftime('%s','now'))
        )
        """)
        c.commit()


class LegalAidApplication(BaseModel):
    org_name:     str = Field(..., min_length=2, max_length=200)
    org_type:     str = Field(..., min_length=2)
    contact_name: str = Field(..., min_length=2)
    role:         Optional[str] = None
    email:        EmailStr
    phone:        Optional[str] = None
    jurisdiction: str = Field(..., min_length=2)
    caseload:     Optional[str] = None
    why:          str = Field(..., min_length=20)


@router.post("/apply")
def apply(body: LegalAidApplication):
    _ensure_table()
    app_id = f"laa_{uuid.uuid4().hex[:16]}"
    try:
        with sqlite3.connect(str(DB), timeout=10.0) as c:
            c.execute("""
                INSERT INTO legal_aid_applications
                  (application_id, org_name, org_type, contact_name, role,
                   email, phone, jurisdiction, caseload, why, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                app_id, body.org_name, body.org_type, body.contact_name, body.role,
                body.email, body.phone, body.jurisdiction, body.caseload, body.why,
                json.dumps(body.model_dump()),
            ))
            c.commit()
    except sqlite3.Error as e:
        raise HTTPException(500, f"db error: {e}")
    return {"application_id": app_id, "status": "received"}


@router.get("/applications")
def list_applications(limit: int = 50):
    _ensure_table()
    with sqlite3.connect(str(DB), timeout=10.0) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT application_id, org_name, org_type, contact_name, email, "
            "jurisdiction, status, received_at FROM legal_aid_applications "
            "ORDER BY received_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return {"count": len(rows), "applications": [dict(r) for r in rows]}
