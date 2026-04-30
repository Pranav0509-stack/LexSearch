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
            """
        )


# ── vault helpers ───────────────────────────────────────────────────────────

def vault_create_doc(user_id: int, filename: str, mime: str, size_bytes: int) -> int:
    with _db_lock, db() as c:
        cur = c.execute(
            "INSERT INTO vault_docs (user_id, filename, mime, size_bytes, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, filename, mime, size_bytes, int(time.time())),
        )
        return cur.lastrowid


def vault_save_chunks(doc_id: int, user_id: int, chunks: list[dict]) -> int:
    if not chunks:
        return 0
    with _db_lock, db() as c:
        c.executemany(
            "INSERT INTO vault_chunks (doc_id, user_id, chunk_id, para_label, text, n_tokens) VALUES (?, ?, ?, ?, ?, ?)",
            [(doc_id, user_id, ch.get("chunk_id", ""), ch.get("para_label", ""),
              ch.get("text", ""), ch.get("n_tokens", 0)) for ch in chunks],
        )
        c.execute("UPDATE vault_docs SET n_chunks = ? WHERE id = ?", (len(chunks), doc_id))
        return len(chunks)


def vault_list_docs(user_id: int) -> list[dict]:
    with db() as c:
        rows = c.execute(
            "SELECT id, filename, mime, size_bytes, n_chunks, created_at FROM vault_docs WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def vault_load_chunks(user_id: int, doc_ids: Optional[list[int]] = None) -> list[dict]:
    """Load chunks for ranking. If doc_ids is None, load all of the user's chunks."""
    with db() as c:
        if doc_ids:
            qs = ",".join("?" * len(doc_ids))
            rows = c.execute(
                f"SELECT vc.id, vc.doc_id, vc.chunk_id, vc.para_label, vc.text, vc.n_tokens, vd.filename "
                f"FROM vault_chunks vc JOIN vault_docs vd ON vd.id = vc.doc_id "
                f"WHERE vc.user_id = ? AND vc.doc_id IN ({qs})",
                [user_id, *doc_ids],
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT vc.id, vc.doc_id, vc.chunk_id, vc.para_label, vc.text, vc.n_tokens, vd.filename "
                "FROM vault_chunks vc JOIN vault_docs vd ON vd.id = vc.doc_id "
                "WHERE vc.user_id = ?",
                (user_id,),
            ).fetchall()
        return [{"title": r["filename"], **dict(r)} for r in rows]


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
