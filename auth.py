"""
LexSearch — authentication & lightweight persistence.

A deliberately small module:
  * SQLite for access requests, issued codes, chat threads/messages.
  * HMAC-signed opaque cookies — no external JWT dep.
  * In-memory token bucket for rate-limiting noisy IPs.

Design notes
------------
We run a gated-beta product, not a public signup funnel. The primitives
below are intentionally minimal: one secret, one SQLite file, one cookie.
Upgrade to Postgres + a real auth provider when we cross ~50 paying firms.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# ── paths + secret ──────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
DB_PATH = Path(os.environ.get("LEXSEARCH_DB_PATH", str(ROOT / "lexsearch.db")))
SECRET_PATH = ROOT / ".secret"

SESSION_COOKIE = "ls_session"
SESSION_TTL_S = 60 * 60 * 24 * 30  # 30 days


def _load_or_create_secret() -> bytes:
    """Secret used to sign session cookies.

    Prefer env var in production. If neither env nor file exists, create a
    random 32-byte secret and persist to `.secret` (gitignored) so cookies
    survive server restarts in dev.
    """
    env = os.environ.get("LEXSEARCH_SECRET_KEY")
    if env:
        return env.encode("utf-8")
    if SECRET_PATH.exists():
        return SECRET_PATH.read_bytes()
    key = secrets.token_bytes(32)
    try:
        SECRET_PATH.write_bytes(key)
        SECRET_PATH.chmod(0o600)
    except Exception as e:
        logger.warning("Could not persist secret to %s: %s", SECRET_PATH, e)
    return key


SECRET = _load_or_create_secret()


# ── sqlite ──────────────────────────────────────────────────────────────────

_db_lock = threading.Lock()


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with _db_lock, db() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS access_requests (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                email       TEXT NOT NULL,
                role        TEXT,
                firm        TEXT,
                bar_no      TEXT,
                note        TEXT,
                status      TEXT NOT NULL DEFAULT 'pending',  -- pending|approved|rejected
                created_at  INTEGER NOT NULL,
                approved_at INTEGER,
                ip          TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_requests_status ON access_requests(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_requests_email  ON access_requests(email);

            CREATE TABLE IF NOT EXISTS access_codes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                code_hash  TEXT NOT NULL UNIQUE,
                email      TEXT NOT NULL,
                name       TEXT,
                request_id INTEGER,
                created_at INTEGER NOT NULL,
                revoked_at INTEGER,
                FOREIGN KEY(request_id) REFERENCES access_requests(id)
            );
            CREATE INDEX IF NOT EXISTS idx_codes_email ON access_codes(email);

            CREATE TABLE IF NOT EXISTS threads (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,    -- access_codes.id
                title      TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_threads_user ON threads(user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id  INTEGER NOT NULL,
                role       TEXT NOT NULL,   -- user|assistant
                content    TEXT NOT NULL,
                citations  TEXT,            -- JSON blob
                created_at INTEGER NOT NULL,
                FOREIGN KEY(thread_id) REFERENCES threads(id)
            );
            CREATE INDEX IF NOT EXISTS idx_msgs_thread ON messages(thread_id, created_at);

            -- ── Vault: per-user document uploads + their chunks ──
            CREATE TABLE IF NOT EXISTS vault_docs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                filename    TEXT NOT NULL,
                mime        TEXT,
                size_bytes  INTEGER,
                n_chunks    INTEGER DEFAULT 0,
                created_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vault_docs_user ON vault_docs(user_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS vault_chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      INTEGER NOT NULL,
                user_id     INTEGER NOT NULL,
                chunk_id    TEXT NOT NULL,
                para_label  TEXT,
                text        TEXT NOT NULL,
                n_tokens    INTEGER,
                FOREIGN KEY(doc_id) REFERENCES vault_docs(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_vault_chunks_user ON vault_chunks(user_id);
            CREATE INDEX IF NOT EXISTS idx_vault_chunks_doc  ON vault_chunks(doc_id);

            -- ── Per-connector API keys (managed via /api/settings/keys) ──
            CREATE TABLE IF NOT EXISTS connector_keys (
                name        TEXT PRIMARY KEY,
                api_key     TEXT NOT NULL,
                set_at      INTEGER NOT NULL,
                set_by      INTEGER,
                note        TEXT
            );

            -- ── Library: curated statutes, contract templates, pleadings ──
            CREATE TABLE IF NOT EXISTS library_docs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                jurisdiction TEXT NOT NULL,
                kind         TEXT NOT NULL,    -- statute | contract | pleading
                title        TEXT NOT NULL,
                body_md      TEXT NOT NULL,
                source_url   TEXT,
                added_at     INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_library_jk ON library_docs(jurisdiction, kind, added_at);

            -- ── Google Workspace OAuth tokens (per-user) ──
            -- One row per (user_id, provider). access_token rotates; refresh_token
            -- is the one we keep long-lived. expiry is unix seconds.
            CREATE TABLE IF NOT EXISTS google_tokens (
                user_id        INTEGER PRIMARY KEY,
                access_token   TEXT NOT NULL,
                refresh_token  TEXT,
                expiry         INTEGER NOT NULL,
                scopes         TEXT NOT NULL,
                google_email   TEXT,
                connected_at   INTEGER NOT NULL,
                tracker_sheet  TEXT
            );

            -- ── NyayaSathi inbound clients ──
            -- Leads pushed in from the NyayaSathi consumer surface (WhatsApp /
            -- voice / web intake form). Each row is a person + their
            -- problem statement. assigned_user_id maps to access_codes.id;
            -- thread_id is set when the lawyer clicks "Open as thread" so
            -- subsequent chat history is bound to this client record.
            CREATE TABLE IF NOT EXISTS nyaya_clients (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                source            TEXT NOT NULL,            -- 'whatsapp' | 'voice' | 'web' | 'manual'
                name              TEXT,
                phone             TEXT,
                email             TEXT,
                language          TEXT,                     -- preferred reply language code
                jurisdiction      TEXT,
                intake_summary    TEXT,                     -- AI-summarised "what the client wants"
                intake_transcript TEXT,                     -- raw call transcript / chat log
                status            TEXT NOT NULL DEFAULT 'new',  -- 'new' | 'in_progress' | 'closed'
                assigned_user_id  INTEGER,
                thread_id         INTEGER,
                arrived_at        INTEGER NOT NULL,
                updated_at        INTEGER NOT NULL,
                notes             TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_nyaya_status ON nyaya_clients(status, arrived_at DESC);
            CREATE INDEX IF NOT EXISTS idx_nyaya_assigned ON nyaya_clients(assigned_user_id, status);
            """
        )

        # ── Idempotent migrations for additive columns ───────────────────
        # SQLite lacks `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, so we
        # introspect PRAGMA table_info and add the column only if it's
        # missing. This keeps existing dbs working through restarts.
        def _ensure_column(table: str, column: str, ddl: str) -> None:
            cols = {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
            if column not in cols:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

        _ensure_column("vault_chunks", "embedding", "BLOB")  # 768f vectors


# ── vault helpers ───────────────────────────────────────────────────────────

def vault_create_doc(user_id: int, filename: str, mime: str, size_bytes: int) -> int:
    with _db_lock, db() as c:
        cur = c.execute(
            "INSERT INTO vault_docs (user_id, filename, mime, size_bytes, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, filename, mime, size_bytes, int(time.time())),
        )
        return cur.lastrowid


def vault_save_chunks(doc_id: int, user_id: int, chunks: list[dict]) -> int:
    """Persist chunks. If a chunk has an "embedding" key (list[float]),
    we serialize it as a contiguous float32 BLOB so semantic search can
    load it back without per-chunk JSON parse cost."""
    if not chunks:
        return 0
    import struct
    def _enc(emb):
        if not emb:
            return None
        # float32 little-endian. 768 floats × 4B = 3072B per chunk —
        # cheap. SQLite indexes by rowid; BLOB lookup is O(1).
        return struct.pack(f"<{len(emb)}f", *emb)
    with _db_lock, db() as c:
        c.executemany(
            "INSERT INTO vault_chunks (doc_id, user_id, chunk_id, para_label, text, n_tokens, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(doc_id, user_id, ch.get("chunk_id", ""), ch.get("para_label", ""),
              ch.get("text", ""), ch.get("n_tokens", 0), _enc(ch.get("embedding")))
             for ch in chunks],
        )
        c.execute("UPDATE vault_docs SET n_chunks = ? WHERE id = ?", (len(chunks), doc_id))
        return len(chunks)


def _decode_embedding(blob: Optional[bytes]) -> Optional[list[float]]:
    """Inverse of the float32 packing in vault_save_chunks. Returns None
    when the chunk has no embedding (legacy rows / embed step failed)."""
    if not blob:
        return None
    import struct
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def vault_list_docs(user_id: int) -> list[dict]:
    with db() as c:
        rows = c.execute(
            "SELECT id, filename, mime, size_bytes, n_chunks, created_at FROM vault_docs WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def vault_load_chunks(user_id: int, doc_ids: Optional[list[int]] = None) -> list[dict]:
    """Load chunks for ranking. If doc_ids is None, load all of the user's chunks.
    Each returned dict includes a decoded "embedding" list[float] when one
    is stored (None for legacy rows uploaded before semantic-search shipped)."""
    with db() as c:
        if doc_ids:
            qs = ",".join("?" * len(doc_ids))
            rows = c.execute(
                f"SELECT vc.id, vc.doc_id, vc.chunk_id, vc.para_label, vc.text, vc.n_tokens, vc.embedding, vd.filename "
                f"FROM vault_chunks vc JOIN vault_docs vd ON vd.id = vc.doc_id "
                f"WHERE vc.user_id = ? AND vc.doc_id IN ({qs})",
                [user_id, *doc_ids],
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT vc.id, vc.doc_id, vc.chunk_id, vc.para_label, vc.text, vc.n_tokens, vc.embedding, vd.filename "
                "FROM vault_chunks vc JOIN vault_docs vd ON vd.id = vc.doc_id "
                "WHERE vc.user_id = ?",
                (user_id,),
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            d["title"] = d.pop("filename")
            d["embedding"] = _decode_embedding(d.get("embedding"))
            out.append(d)
        return out


def vault_delete_doc(user_id: int, doc_id: int) -> bool:
    with _db_lock, db() as c:
        cur = c.execute("DELETE FROM vault_docs WHERE id = ? AND user_id = ?", (doc_id, user_id))
        c.execute("DELETE FROM vault_chunks WHERE doc_id = ? AND user_id = ?", (doc_id, user_id))
        return cur.rowcount > 0


# ── access requests ─────────────────────────────────────────────────────────

def create_access_request(
    name: str, email: str, role: str, firm: str, bar_no: str, note: str, ip: str
) -> int:
    now = int(time.time())
    with db() as c:
        cur = c.execute(
            """INSERT INTO access_requests
               (name, email, role, firm, bar_no, note, created_at, ip)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (name.strip(), email.strip().lower(), role, firm, bar_no, note, now, ip),
        )
        return int(cur.lastrowid)


def list_access_requests(status: Optional[str] = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM access_requests"
    args: tuple[Any, ...] = ()
    if status:
        sql += " WHERE status = ?"
        args = (status,)
    sql += " ORDER BY created_at DESC"
    with db() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def approve_request(request_id: int) -> Optional[dict[str, Any]]:
    """Approve a request, generate a fresh access code, return it (plaintext
    is returned ONCE — we only store the sha256)."""
    with db() as c:
        row = c.execute(
            "SELECT * FROM access_requests WHERE id = ? AND status = 'pending'",
            (request_id,),
        ).fetchone()
        if not row:
            return None
        code = _generate_code()
        code_hash = _hash_code(code)
        now = int(time.time())
        c.execute(
            """INSERT INTO access_codes (code_hash, email, name, request_id, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (code_hash, row["email"], row["name"], request_id, now),
        )
        c.execute(
            "UPDATE access_requests SET status='approved', approved_at=? WHERE id=?",
            (now, request_id),
        )
        return {
            "request_id": request_id,
            "email": row["email"],
            "name": row["name"],
            "access_code": code,  # plaintext — show ONCE
        }


def reject_request(request_id: int) -> bool:
    with db() as c:
        cur = c.execute(
            "UPDATE access_requests SET status='rejected' WHERE id=? AND status='pending'",
            (request_id,),
        )
        return cur.rowcount > 0


# ── access codes ────────────────────────────────────────────────────────────

def _generate_code() -> str:
    """12-char URL-safe, readable. ~72 bits of entropy."""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no I,O,0,1 confusion
    return "-".join(
        "".join(secrets.choice(alphabet) for _ in range(4)) for _ in range(3)
    )


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.strip().upper().encode("utf-8")).hexdigest()


def validate_code(code: str) -> Optional[dict[str, Any]]:
    """Return the user row if the code matches an un-revoked access_code."""
    if not code or len(code) > 64:
        return None
    with db() as c:
        row = c.execute(
            """SELECT id, email, name, created_at, revoked_at
               FROM access_codes WHERE code_hash = ?""",
            (_hash_code(code),),
        ).fetchone()
        if not row or row["revoked_at"]:
            return None
        return dict(row)


# ── signed session cookies ──────────────────────────────────────────────────

def make_session_token(user_id: int) -> str:
    expires = int(time.time()) + SESSION_TTL_S
    payload = f"{user_id}.{expires}".encode("utf-8")
    sig = hmac.new(SECRET, payload, hashlib.sha256).digest()
    blob = payload + b"." + base64.urlsafe_b64encode(sig).rstrip(b"=")
    return blob.decode("utf-8")


def verify_session_token(token: Optional[str]) -> Optional[int]:
    if not token:
        return None
    try:
        uid_s, exp_s, sig_b64 = token.split(".")
        payload = f"{uid_s}.{exp_s}".encode("utf-8")
        expected = hmac.new(SECRET, payload, hashlib.sha256).digest()
        pad = "=" * (-len(sig_b64) % 4)
        given = base64.urlsafe_b64decode(sig_b64 + pad)
        if not hmac.compare_digest(expected, given):
            return None
        if int(exp_s) < int(time.time()):
            return None
        return int(uid_s)
    except (ValueError, IndexError, base64.binascii.Error):
        return None


def get_user(user_id: int) -> Optional[dict[str, Any]]:
    with db() as c:
        row = c.execute(
            "SELECT id, email, name FROM access_codes WHERE id = ? AND revoked_at IS NULL",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None


# ── threads + messages ──────────────────────────────────────────────────────

def create_thread(user_id: int, title: str = "New chat") -> int:
    now = int(time.time())
    with db() as c:
        cur = c.execute(
            """INSERT INTO threads (user_id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?)""",
            (user_id, title, now, now),
        )
        return int(cur.lastrowid)


def append_message(thread_id: int, role: str, content: str, citations: Optional[str]) -> int:
    now = int(time.time())
    with db() as c:
        cur = c.execute(
            """INSERT INTO messages (thread_id, role, content, citations, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (thread_id, role, content, citations, now),
        )
        c.execute("UPDATE threads SET updated_at = ? WHERE id = ?", (now, thread_id))
        return int(cur.lastrowid)


def list_user_threads(user_id: int, limit: int = 20) -> list[dict[str, Any]]:
    with db() as c:
        return [
            dict(r)
            for r in c.execute(
                "SELECT id, title, created_at, updated_at FROM threads "
                "WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        ]


def search_user_messages(user_id: int, query: str, limit: int = 30) -> list[dict[str, Any]]:
    """Fuzzy LIKE-search across this user's message content.

    Returns rows with thread_id, thread_title, role, snippet, created_at —
    sorted newest first. Empty query → 30 most recent assistant turns.
    """
    q = (query or "").strip()
    with db() as c:
        if q:
            rows = c.execute(
                """SELECT m.id, m.thread_id, m.role, m.content, m.created_at, t.title
                   FROM messages m JOIN threads t ON t.id = m.thread_id
                   WHERE t.user_id = ? AND m.content LIKE ?
                   ORDER BY m.created_at DESC LIMIT ?""",
                (user_id, f"%{q}%", limit),
            ).fetchall()
        else:
            rows = c.execute(
                """SELECT m.id, m.thread_id, m.role, m.content, m.created_at, t.title
                   FROM messages m JOIN threads t ON t.id = m.thread_id
                   WHERE t.user_id = ? ORDER BY m.created_at DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
    out = []
    for r in rows:
        body = r["content"] or ""
        # Build a 240-char snippet around the match if there is one.
        snippet = body[:240]
        if q:
            i = body.lower().find(q.lower())
            if i >= 0:
                start = max(0, i - 60)
                end = min(len(body), i + len(q) + 180)
                snippet = ("…" if start else "") + body[start:end] + ("…" if end < len(body) else "")
        out.append({
            "message_id": r["id"],
            "thread_id": r["thread_id"],
            "thread_title": r["title"] or "Untitled thread",
            "role": r["role"],
            "snippet": snippet,
            "created_at": r["created_at"],
        })
    return out


def get_thread_messages(thread_id: int, user_id: int) -> Optional[list[dict[str, Any]]]:
    with db() as c:
        owner = c.execute(
            "SELECT user_id FROM threads WHERE id = ?", (thread_id,)
        ).fetchone()
        if not owner or owner["user_id"] != user_id:
            return None
        return [
            dict(r)
            for r in c.execute(
                "SELECT role, content, citations, created_at FROM messages "
                "WHERE thread_id = ? ORDER BY created_at",
                (thread_id,),
            ).fetchall()
        ]


# ── connector keys (per-country API keys, DB-backed) ───────────────────────

def set_connector_key(name: str, api_key: str, user_id: Optional[int] = None, note: str = "") -> None:
    """Insert or replace an API key for a named connector.

    `name` is a stable lowercase connector id ("indian_kanoon", "ecourts",
    "serper", "tavily", "lawnet_sg", "hklii", "dubai_pulse", "klri",
    "clj", "jdih"). Plaintext is stored — this is a single-tenant beta;
    upgrade to KMS-encrypted columns when we cross 50 firms.
    """
    name = (name or "").strip().lower()
    api_key = (api_key or "").strip()
    if not name or not api_key:
        raise ValueError("name and api_key required")
    now = int(time.time())
    with _db_lock, db() as c:
        c.execute(
            """INSERT INTO connector_keys (name, api_key, set_at, set_by, note)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                 api_key=excluded.api_key,
                 set_at=excluded.set_at,
                 set_by=excluded.set_by,
                 note=excluded.note""",
            (name, api_key, now, user_id, note or ""),
        )


def get_connector_key(name: str) -> Optional[str]:
    name = (name or "").strip().lower()
    if not name:
        return None
    with db() as c:
        row = c.execute(
            "SELECT api_key FROM connector_keys WHERE name = ?", (name,)
        ).fetchone()
        return row["api_key"] if row else None


def list_connector_keys() -> list[dict[str, Any]]:
    """Return name + masked tail (last 4 chars) — never the plaintext key."""
    with db() as c:
        rows = c.execute(
            "SELECT name, api_key, set_at, note FROM connector_keys ORDER BY name"
        ).fetchall()
    out = []
    for r in rows:
        key = r["api_key"] or ""
        masked = ("…" + key[-4:]) if len(key) >= 4 else "…"
        out.append({
            "name": r["name"],
            "has_key": bool(key),
            "masked_tail": masked,
            "set_at": r["set_at"],
            "note": r["note"] or "",
        })
    return out


def delete_connector_key(name: str) -> bool:
    name = (name or "").strip().lower()
    if not name:
        return False
    with _db_lock, db() as c:
        cur = c.execute("DELETE FROM connector_keys WHERE name = ?", (name,))
        return cur.rowcount > 0


# ── library helpers (statutes / contract templates / pleadings) ────────────

def library_insert(jurisdiction: str, kind: str, title: str, body_md: str, source_url: str = "") -> int:
    now = int(time.time())
    with _db_lock, db() as c:
        cur = c.execute(
            """INSERT INTO library_docs (jurisdiction, kind, title, body_md, source_url, added_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (jurisdiction.upper().strip(), kind.strip().lower(), title.strip(), body_md, source_url, now),
        )
        return int(cur.lastrowid)


def library_count() -> int:
    with db() as c:
        row = c.execute("SELECT COUNT(*) AS n FROM library_docs").fetchone()
        return int(row["n"]) if row else 0


def library_list(jurisdiction: Optional[str] = None, kind: Optional[str] = None) -> list[dict[str, Any]]:
    sql = "SELECT id, jurisdiction, kind, title, source_url, added_at FROM library_docs WHERE 1=1"
    args: list[Any] = []
    if jurisdiction:
        sql += " AND jurisdiction = ?"
        args.append(jurisdiction.upper())
    if kind:
        sql += " AND kind = ?"
        args.append(kind.lower())
    sql += " ORDER BY jurisdiction, kind, title"
    with db() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def library_get(doc_id: int) -> Optional[dict[str, Any]]:
    with db() as c:
        row = c.execute(
            "SELECT id, jurisdiction, kind, title, body_md, source_url, added_at FROM library_docs WHERE id = ?",
            (doc_id,),
        ).fetchone()
        return dict(row) if row else None


# ── rate limiter (in-memory token bucket per IP) ────────────────────────────

_RATE_LOCK = threading.Lock()
_BUCKETS: dict[tuple[str, str], list[float]] = {}


def rate_limit(bucket: str, ip: str, max_hits: int, window_s: int) -> bool:
    """Return True if the request is allowed, False if rate-limited.

    Sliding window over the last `window_s` seconds. In-memory, resets on
    restart — good enough for a single-node beta.
    """
    now = time.time()
    key = (bucket, ip or "")
    with _RATE_LOCK:
        hits = _BUCKETS.setdefault(key, [])
        cutoff = now - window_s
        while hits and hits[0] < cutoff:
            hits.pop(0)
        if len(hits) >= max_hits:
            return False
        hits.append(now)
        return True


# ── Google Workspace OAuth tokens ──────────────────────────────────────────

def google_save_tokens(
    user_id: int,
    access_token: str,
    refresh_token: Optional[str],
    expiry: int,
    scopes: str,
    google_email: Optional[str] = None,
) -> None:
    """Upsert Google OAuth tokens for a user. `refresh_token` is only sent
    by Google on first consent (or when access_type=offline+prompt=consent
    is forced) — preserve the existing one on subsequent refresh-only
    updates.
    """
    with _db_lock, db() as c:
        existing = c.execute(
            "SELECT refresh_token FROM google_tokens WHERE user_id = ?", (user_id,)
        ).fetchone()
        rt = refresh_token or (existing["refresh_token"] if existing else None)
        c.execute(
            """INSERT INTO google_tokens
               (user_id, access_token, refresh_token, expiry, scopes, google_email, connected_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   access_token=excluded.access_token,
                   refresh_token=COALESCE(excluded.refresh_token, google_tokens.refresh_token),
                   expiry=excluded.expiry,
                   scopes=excluded.scopes,
                   google_email=COALESCE(excluded.google_email, google_tokens.google_email),
                   connected_at=excluded.connected_at""",
            (user_id, access_token, rt, expiry, scopes, google_email, int(time.time())),
        )


def google_get_tokens(user_id: int) -> Optional[dict[str, Any]]:
    with db() as c:
        row = c.execute(
            """SELECT user_id, access_token, refresh_token, expiry, scopes,
                      google_email, connected_at, tracker_sheet
               FROM google_tokens WHERE user_id = ?""",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None


def google_delete_tokens(user_id: int) -> bool:
    with _db_lock, db() as c:
        cur = c.execute("DELETE FROM google_tokens WHERE user_id = ?", (user_id,))
        return cur.rowcount > 0


def google_set_tracker_sheet(user_id: int, sheet_id: str) -> None:
    """Remember the matter-tracker spreadsheet ID per user so
    `append_matter_row` doesn't need it explicitly each call."""
    with _db_lock, db() as c:
        c.execute(
            "UPDATE google_tokens SET tracker_sheet = ? WHERE user_id = ?",
            (sheet_id, user_id),
        )


# ── nyaya_clients helpers ──────────────────────────────────────────────────
# Inbound consumer leads from the NyayaSathi WhatsApp/voice surface land
# here. The lawyer's "Clients" pane is a thin viewer over these helpers.

def nyaya_create_client(
    *,
    source: str,
    name: Optional[str],
    phone: Optional[str],
    email: Optional[str],
    language: Optional[str],
    jurisdiction: Optional[str],
    intake_summary: Optional[str],
    intake_transcript: Optional[str] = None,
    assigned_user_id: Optional[int] = None,
    notes: Optional[str] = None,
) -> int:
    now = int(time.time())
    src = (source or "manual").strip().lower()
    if src not in {"whatsapp", "voice", "web", "manual"}:
        src = "manual"
    with _db_lock, db() as c:
        cur = c.execute(
            """INSERT INTO nyaya_clients
               (source, name, phone, email, language, jurisdiction,
                intake_summary, intake_transcript, status,
                assigned_user_id, thread_id, arrived_at, updated_at, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'new', ?, NULL, ?, ?, ?)""",
            (src, name, phone, email, language, jurisdiction,
             intake_summary, intake_transcript,
             assigned_user_id, now, now, notes),
        )
        return cur.lastrowid


def nyaya_list_clients(
    *,
    status: Optional[str] = None,
    assigned_user_id: Optional[int] = None,
    q: Optional[str] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    sql = (
        "SELECT id, source, name, phone, email, language, jurisdiction, "
        "intake_summary, status, assigned_user_id, thread_id, "
        "arrived_at, updated_at "
        "FROM nyaya_clients WHERE 1=1"
    )
    args: list[Any] = []
    if status and status in {"new", "in_progress", "closed"}:
        sql += " AND status = ?"
        args.append(status)
    if assigned_user_id is not None:
        sql += " AND (assigned_user_id = ? OR assigned_user_id IS NULL)"
        args.append(assigned_user_id)
    if q:
        like = f"%{q.strip()}%"
        sql += " AND (name LIKE ? OR phone LIKE ? OR email LIKE ? OR intake_summary LIKE ?)"
        args.extend([like, like, like, like])
    sql += " ORDER BY arrived_at DESC LIMIT ?"
    args.append(limit)
    with db() as c:
        rows = c.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def nyaya_get_client(client_id: int) -> Optional[dict[str, Any]]:
    with db() as c:
        row = c.execute(
            "SELECT * FROM nyaya_clients WHERE id = ?", (client_id,)
        ).fetchone()
    return dict(row) if row else None


def nyaya_update_client(
    client_id: int,
    *,
    status: Optional[str] = None,
    assigned_user_id: Optional[int] = None,
    notes: Optional[str] = None,
    thread_id: Optional[int] = None,
) -> bool:
    """Patch update — only writes fields that are passed in. Returns True
    iff a row was updated."""
    fields: list[str] = []
    args: list[Any] = []
    if status is not None and status in {"new", "in_progress", "closed"}:
        fields.append("status = ?")
        args.append(status)
    if assigned_user_id is not None:
        fields.append("assigned_user_id = ?")
        args.append(assigned_user_id)
    if notes is not None:
        fields.append("notes = ?")
        args.append(notes)
    if thread_id is not None:
        fields.append("thread_id = ?")
        args.append(thread_id)
    if not fields:
        return False
    fields.append("updated_at = ?")
    args.append(int(time.time()))
    args.append(client_id)
    with _db_lock, db() as c:
        cur = c.execute(
            f"UPDATE nyaya_clients SET {', '.join(fields)} WHERE id = ?",
            args,
        )
        return cur.rowcount > 0


def nyaya_count_by_status(assigned_user_id: Optional[int] = None) -> dict[str, int]:
    """Used by the sidebar badge — how many `new` are sitting in the inbox."""
    sql = "SELECT status, COUNT(*) AS n FROM nyaya_clients"
    args: list[Any] = []
    if assigned_user_id is not None:
        sql += " WHERE (assigned_user_id = ? OR assigned_user_id IS NULL)"
        args.append(assigned_user_id)
    sql += " GROUP BY status"
    with db() as c:
        rows = c.execute(sql, args).fetchall()
    out = {"new": 0, "in_progress": 0, "closed": 0}
    for r in rows:
        out[r["status"]] = r["n"]
    return out
