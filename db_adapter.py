"""DB adapter — switch between Postgres (NeonDB) and SQLite at runtime.

Design constraint: the existing `auth.py` and friends speak SQLite via
`auth.db()` returning a `sqlite3.Connection`. We don't want to rewrite
every call site. So this adapter exposes the *same* `db()` context-
manager API but routes to Postgres (`psycopg.Connection`) when
`DATABASE_URL` looks like a Postgres URL — otherwise it falls through
to the existing SQLite path.

Differences psycopg ↔ sqlite3 we paper over here:

  • placeholder syntax: psycopg uses `%s`, sqlite3 uses `?`. We expose
    a `q()` helper that rewrites `?` → `%s` when needed.
  • `INTEGER PRIMARY KEY` becomes `BIGSERIAL PRIMARY KEY` on PG.
  • `last_insert_rowid()` becomes `RETURNING id` on PG.
  • boolean columns are TEXT in our SQLite schema; on PG we map to
    BOOLEAN. Most columns are already INTEGER unix-timestamps so this
    rarely bites.

This first cut wires the *connection layer* only. Migration of every
schema is a Phase-2 follow-up; for now Postgres mode runs alongside
the existing SQLite app and is used by the new `/api/dashboard/*`
endpoints, which create their own tables. Existing endpoints keep
running on SQLite via `auth.db()`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
USE_POSTGRES = DATABASE_URL.startswith(("postgres://", "postgresql://"))

# Default SQLite path — co-located with auth.py's so tooling that
# `sqlite3 lexsearch.db ...` from the project root keeps working.
SQLITE_PATH = Path(os.environ.get("LEXSEARCH_DB_PATH", "lexsearch.db"))

# Lazy-initialised connection pool. Postgres benefits from one (saves
# the ~50ms TLS handshake on every request); SQLite doesn't, so we
# open one connection per `db()` call there as before.
_pg_pool: Any = None
_pg_lock = threading.Lock()


def _get_pg_pool() -> Any:
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool
    with _pg_lock:
        if _pg_pool is not None:
            return _pg_pool
        try:
            from psycopg_pool import ConnectionPool  # type: ignore
            _pg_pool = ConnectionPool(
                conninfo=DATABASE_URL,
                min_size=1,
                max_size=8,
                timeout=10,
                kwargs={"autocommit": False},
            )
            _pg_pool.wait()
            logger.info("psycopg pool ready (Postgres mode, max_size=8)")
        except ImportError:
            # psycopg_pool is in the binary package, but if we ever
            # ship without it, fall through to one-off connections.
            logger.warning("psycopg_pool unavailable, falling through to per-request connections")
            _pg_pool = "no-pool"
    return _pg_pool


@contextlib.contextmanager
def db() -> Iterator[Any]:
    """Yield a DB connection. Commits on clean exit, rolls back on
    exception. Same shape as auth.db() so call sites can swap in.

    On SQLite: rows are dict-accessible because we set row_factory.
    On Postgres: rows are dicts via `dict_row` factory.
    """
    if USE_POSTGRES:
        pool = _get_pg_pool()
        if pool == "no-pool":
            import psycopg
            from psycopg.rows import dict_row
            with psycopg.connect(DATABASE_URL, autocommit=False, row_factory=dict_row) as conn:
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
        else:
            from psycopg.rows import dict_row
            with pool.connection() as conn:
                conn.row_factory = dict_row
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
    else:
        conn = sqlite3.connect(SQLITE_PATH, timeout=20)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def q(sql: str) -> str:
    """Rewrite `?` placeholders to `%s` when running on Postgres.
    Quoted strings inside SQL are *not* re-scanned — keep your `?`
    placeholders out of literal text."""
    if USE_POSTGRES:
        return sql.replace("?", "%s")
    return sql


def fetch_all(conn: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """Cross-driver `SELECT … fetchall()` returning list-of-dicts."""
    cur = conn.execute(q(sql), params)
    rows = cur.fetchall()
    if USE_POSTGRES:
        return list(rows)  # already dict-shaped via dict_row
    return [dict(r) for r in rows]


def fetch_one(conn: Any, sql: str, params: tuple = ()) -> dict[str, Any] | None:
    cur = conn.execute(q(sql), params)
    row = cur.fetchone()
    if row is None:
        return None
    if USE_POSTGRES:
        return dict(row)
    return dict(row)


def execute(conn: Any, sql: str, params: tuple = ()) -> Any:
    """Cross-driver execute; returns the cursor for `lastrowid` / etc.
    For inserts that need the new id on Postgres, use `... RETURNING id`
    explicitly and `fetch_one`."""
    return conn.execute(q(sql), params)


def status() -> dict[str, Any]:
    """Lightweight health snapshot for the dashboard's System widget."""
    info: dict[str, Any] = {
        "mode": "postgres" if USE_POSTGRES else "sqlite",
        "url_host": _safe_host(DATABASE_URL) if USE_POSTGRES else str(SQLITE_PATH),
        "ok": False,
    }
    try:
        with db() as conn:
            cur = conn.execute("SELECT 1")
            cur.fetchone()
        info["ok"] = True
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)
    return info


def _safe_host(url: str) -> str:
    """Strip user:pass from a Postgres URL — only return scheme + host."""
    if not url:
        return ""
    try:
        from urllib.parse import urlparse
        u = urlparse(url)
        host = u.hostname or "?"
        return f"{u.scheme}://{host}"
    except Exception:  # noqa: BLE001
        return "postgres://?"


# ─────────────────────────────────────────────────────────────────────
# Dashboard-owned schema — created on first import. Lives alongside
# auth.py's tables; unique table-name prefix `dash_` so we never
# collide with the legacy schema even when both run on the same DB.
# ─────────────────────────────────────────────────────────────────────


def init_dashboard_schema() -> None:
    """Idempotent — safe to call on every server boot."""
    sqlite_ddl = [
        # Audit log of admin-impacting writes — every dashboard mutation
        # appends a row so we can show "X minutes ago, demo@sanhita.ai
        # revoked access for foo@bar.com" in the Activity widget.
        """CREATE TABLE IF NOT EXISTS dash_activity (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            actor       TEXT,                -- email or 'system'
            action      TEXT NOT NULL,       -- e.g. 'revoke_access', 'set_key'
            target      TEXT,                -- subject of the action (email, key name, …)
            payload     TEXT,                -- optional JSON detail
            created_at  INTEGER NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_dash_activity_created ON dash_activity(created_at DESC)""",
        # Dashboard-only key/value settings — feature flags, motd, etc.
        """CREATE TABLE IF NOT EXISTS dash_settings (
            key         TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            updated_at  INTEGER NOT NULL,
            updated_by  TEXT
        )""",
    ]
    pg_ddl = [
        """CREATE TABLE IF NOT EXISTS dash_activity (
            id          BIGSERIAL PRIMARY KEY,
            actor       TEXT,
            action      TEXT NOT NULL,
            target      TEXT,
            payload     TEXT,
            created_at  BIGINT NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_dash_activity_created ON dash_activity(created_at DESC)""",
        """CREATE TABLE IF NOT EXISTS dash_settings (
            key         TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            updated_at  BIGINT NOT NULL,
            updated_by  TEXT
        )""",
    ]
    statements = pg_ddl if USE_POSTGRES else sqlite_ddl
    with db() as conn:
        for stmt in statements:
            conn.execute(stmt)
    logger.info("dashboard schema ready (mode=%s)", "postgres" if USE_POSTGRES else "sqlite")
