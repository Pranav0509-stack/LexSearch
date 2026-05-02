"""
Google Workspace integration — Docs, Gmail (draft only), Sheets, Drive.

Talks to Google's REST APIs directly via httpx (no `google-api-python-client`
dependency, keeps the deploy slim). OAuth tokens live in `auth.google_tokens`.

Public surface (used by `server.py` + `agents.legal_agent`):

    OAuth:
        oauth_authorize_url(state)        -> str
        oauth_exchange_code(code)         -> token dict
        get_valid_access_token(user_id)   -> str   (auto-refreshes)
        revoke(user_id)                   -> bool

    Tool primitives (called from the agent):
        create_doc(user_id, title, content_md)            -> {url, doc_id}
        create_gmail_draft(user_id, to, subject, body_md) -> {draft_id, gmail_url}
        append_sheet_row(user_id, sheet_id, row)          -> {updated_range}
        ensure_tracker_sheet(user_id)                     -> sheet_id
        search_drive(user_id, query, k=10)                -> [{id,name,url,mime}]

Safety: `create_gmail_draft` NEVER sends. The user must click Send in their
Gmail. We only have `gmail.compose` scope, not `gmail.send`.
"""

from __future__ import annotations

import base64
import logging
import os
import time
import urllib.parse
from typing import Any, Optional

import httpx

import auth

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────

CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
REDIRECT_URI = os.environ.get(
    "GOOGLE_REDIRECT_URI",
    "http://localhost:8080/api/google/oauth/callback",
).strip()

# Minimal scopes for v1. We deliberately use `gmail.compose` not `gmail.send`
# — drafts only, user must click Send themselves. Extra paranoia for legal AI.
SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

DOCS_API = "https://docs.googleapis.com/v1"
GMAIL_API = "https://gmail.googleapis.com/gmail/v1"
SHEETS_API = "https://sheets.googleapis.com/v4"
DRIVE_API = "https://www.googleapis.com/drive/v3"


def is_configured() -> bool:
    return bool(CLIENT_ID and CLIENT_SECRET)


# ── OAuth ──────────────────────────────────────────────────────────────────

def oauth_authorize_url(state: str) -> str:
    """Build the consent-screen URL the user is redirected to.

    `state` should be a CSRF-bound nonce we recognise on callback.
    """
    if not is_configured():
        raise RuntimeError("GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not set")
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",     # we want refresh tokens
        "prompt": "consent",           # force refresh_token issuance
        "include_granted_scopes": "true",
        "state": state,
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def oauth_exchange_code(code: str) -> dict[str, Any]:
    """POST the code back to Google to get access + refresh tokens."""
    if not is_configured():
        raise RuntimeError("Google OAuth not configured")
    with httpx.Client(timeout=15.0) as c:
        r = c.post(
            TOKEN_URL,
            data={
                "code": code,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )
        r.raise_for_status()
        token = r.json()

        # Pull the user's email so we can show "Connected as foo@bar.com".
        email = None
        try:
            ur = c.get(
                USERINFO_URL,
                headers={"Authorization": f"Bearer {token['access_token']}"},
            )
            if ur.status_code == 200:
                email = ur.json().get("email")
        except Exception as e:
            logger.warning("userinfo fetch failed: %s", e)
        token["_email"] = email
        return token


def _refresh_access_token(refresh_token: str) -> dict[str, Any]:
    if not is_configured():
        raise RuntimeError("Google OAuth not configured")
    with httpx.Client(timeout=15.0) as c:
        r = c.post(
            TOKEN_URL,
            data={
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "grant_type": "refresh_token",
            },
        )
        r.raise_for_status()
        return r.json()


def get_valid_access_token(user_id: int) -> Optional[str]:
    """Fetch the user's access token, auto-refreshing if expired.

    Returns None if the user hasn't connected Google or the refresh fails.
    """
    row = auth.google_get_tokens(user_id)
    if not row:
        return None
    now = int(time.time())
    # 30s safety margin
    if row["expiry"] > now + 30:
        return row["access_token"]
    rt = row.get("refresh_token")
    if not rt:
        logger.warning("user %s has no refresh_token; needs reconsent", user_id)
        return None
    try:
        new = _refresh_access_token(rt)
        new_expiry = int(time.time()) + int(new.get("expires_in", 3600))
        auth.google_save_tokens(
            user_id,
            access_token=new["access_token"],
            refresh_token=new.get("refresh_token"),  # may be None on refresh
            expiry=new_expiry,
            scopes=row.get("scopes") or " ".join(SCOPES),
            google_email=row.get("google_email"),
        )
        return new["access_token"]
    except Exception as e:
        logger.error("token refresh for user %s failed: %s", user_id, e)
        return None


def revoke(user_id: int) -> bool:
    """Revoke the refresh token at Google + delete local tokens."""
    row = auth.google_get_tokens(user_id)
    if not row:
        return False
    rt = row.get("refresh_token") or row.get("access_token")
    try:
        with httpx.Client(timeout=10.0) as c:
            c.post(REVOKE_URL, data={"token": rt})
    except Exception as e:
        logger.warning("Google revoke failed (will still delete local): %s", e)
    return auth.google_delete_tokens(user_id)


# ── Helpers ────────────────────────────────────────────────────────────────

def _auth_headers(user_id: int) -> dict[str, str]:
    tok = get_valid_access_token(user_id)
    if not tok:
        raise RuntimeError("Google not connected. Connect from Settings first.")
    return {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}


# ── Docs API ──────────────────────────────────────────────────────────────

def create_doc(user_id: int, title: str, content_md: str) -> dict[str, Any]:
    """Create a Google Doc with the given markdown content.

    We do a two-step: (1) create empty doc with title, (2) batchUpdate to
    insert the body. Markdown is flattened to plain text — full MD-to-Docs
    formatting would need a markdown→Docs-AST translator (next iteration).
    """
    headers = _auth_headers(user_id)
    title = (title or "Sanhita Draft").strip()[:200]
    body = (content_md or "").strip()

    with httpx.Client(timeout=20.0) as c:
        # 1) create
        r = c.post(f"{DOCS_API}/documents", headers=headers, json={"title": title})
        r.raise_for_status()
        doc = r.json()
        doc_id = doc["documentId"]

        # 2) insert body. Index 1 is the start of the document body.
        if body:
            r2 = c.post(
                f"{DOCS_API}/documents/{doc_id}:batchUpdate",
                headers=headers,
                json={
                    "requests": [
                        {"insertText": {"location": {"index": 1}, "text": body}}
                    ]
                },
            )
            r2.raise_for_status()

    return {
        "doc_id": doc_id,
        "url": f"https://docs.google.com/document/d/{doc_id}/edit",
        "title": title,
    }


# ── Gmail API (drafts only) ───────────────────────────────────────────────

def create_gmail_draft(
    user_id: int,
    to: str,
    subject: str,
    body_md: str,
) -> dict[str, Any]:
    """Create a Gmail draft. Does NOT send. User must click Send in Gmail.

    RFC 822 message → base64url → wrapped in Gmail's draft envelope.
    """
    headers = _auth_headers(user_id)
    to = (to or "").strip()
    subject = (subject or "Sanhita Draft").strip()[:300]
    body = (body_md or "").strip()

    if not to or "@" not in to:
        return {"error": "Invalid recipient email."}

    rfc822 = (
        f"To: {to}\r\n"
        f"Subject: {subject}\r\n"
        "Content-Type: text/plain; charset=UTF-8\r\n"
        "MIME-Version: 1.0\r\n"
        "\r\n"
        f"{body}\r\n"
    )
    raw = base64.urlsafe_b64encode(rfc822.encode("utf-8")).decode("ascii").rstrip("=")

    with httpx.Client(timeout=15.0) as c:
        r = c.post(
            f"{GMAIL_API}/users/me/drafts",
            headers=headers,
            json={"message": {"raw": raw}},
        )
        r.raise_for_status()
        draft = r.json()
    draft_id = draft.get("id", "")
    return {
        "draft_id": draft_id,
        "to": to,
        "subject": subject,
        "gmail_url": "https://mail.google.com/mail/u/0/#drafts",
        # Note: there's no public per-draft permalink in Gmail; we point
        # the user to their Drafts folder.
    }


# ── Sheets API ────────────────────────────────────────────────────────────

_TRACKER_HEADERS = [
    "Date", "Matter", "Jurisdiction", "Court", "Stage",
    "Action", "Sanhita Doc URL", "Notes",
]


def ensure_tracker_sheet(user_id: int) -> str:
    """Get or create the user's matter-tracker spreadsheet. Returns sheet_id."""
    row = auth.google_get_tokens(user_id)
    if row and row.get("tracker_sheet"):
        return row["tracker_sheet"]

    headers = _auth_headers(user_id)
    with httpx.Client(timeout=15.0) as c:
        r = c.post(
            f"{SHEETS_API}/spreadsheets",
            headers=headers,
            json={
                "properties": {"title": "Sanhita — Matter Tracker"},
                "sheets": [{"properties": {"title": "Matters"}}],
            },
        )
        r.raise_for_status()
        sheet = r.json()
        sheet_id = sheet["spreadsheetId"]

        # Header row
        c.post(
            f"{SHEETS_API}/spreadsheets/{sheet_id}/values/Matters!A1:append",
            headers=headers,
            params={"valueInputOption": "USER_ENTERED"},
            json={"values": [_TRACKER_HEADERS]},
        )

    auth.google_set_tracker_sheet(user_id, sheet_id)
    return sheet_id


def append_matter_row(user_id: int, row: list[str]) -> dict[str, Any]:
    """Append a row to the user's matter-tracker. Auto-creates the sheet."""
    sheet_id = ensure_tracker_sheet(user_id)
    headers = _auth_headers(user_id)
    # Pad/trim to header length
    padded = (list(row) + [""] * len(_TRACKER_HEADERS))[: len(_TRACKER_HEADERS)]
    with httpx.Client(timeout=15.0) as c:
        r = c.post(
            f"{SHEETS_API}/spreadsheets/{sheet_id}/values/Matters!A1:append",
            headers=headers,
            params={"valueInputOption": "USER_ENTERED"},
            json={"values": [padded]},
        )
        r.raise_for_status()
        body = r.json()
    return {
        "sheet_id": sheet_id,
        "sheet_url": f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit",
        "updated_range": body.get("updates", {}).get("updatedRange", ""),
    }


# ── Drive search ──────────────────────────────────────────────────────────

def search_drive(user_id: int, query: str, k: int = 10) -> list[dict[str, Any]]:
    """Full-text search the user's Drive (only files Sanhita has touched —
    `drive.file` scope is per-file). Returns up to `k` matches.
    """
    headers = _auth_headers(user_id)
    safe_q = (query or "").replace("'", "\\'")
    drive_q = f"fullText contains '{safe_q}' and trashed = false"
    with httpx.Client(timeout=15.0) as c:
        r = c.get(
            f"{DRIVE_API}/files",
            headers=headers,
            params={
                "q": drive_q,
                "pageSize": min(max(k, 1), 50),
                "fields": "files(id,name,mimeType,webViewLink,modifiedTime)",
            },
        )
        r.raise_for_status()
        files = r.json().get("files", [])
    return [
        {
            "id": f.get("id"),
            "name": f.get("name"),
            "mime": f.get("mimeType"),
            "url": f.get("webViewLink"),
            "modified": f.get("modifiedTime"),
        }
        for f in files
    ]


# ── Status check (for /api/google/status + Settings pane) ─────────────────

def status_for_user(user_id: int) -> dict[str, Any]:
    if not is_configured():
        return {
            "configured": False,
            "connected": False,
            "reason": "GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not set on the server.",
        }
    row = auth.google_get_tokens(user_id)
    if not row:
        return {"configured": True, "connected": False}
    return {
        "configured": True,
        "connected": True,
        "email": row.get("google_email"),
        "connected_at": row.get("connected_at"),
        "tracker_sheet": row.get("tracker_sheet"),
        "scopes": (row.get("scopes") or "").split(),
    }
