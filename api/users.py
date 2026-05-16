"""
Hermes Web UI -- Multi-user data model.

SQLite-backed users / quotas / usage tables, password verification (reusing
api.auth._hash_password), and per-request user resolution via thread-local.

Schema and design: see ``C:\\Users\\lzb\\.claude\\plans\\agent-1-2-wise-badger.md``.

This module is deliberately self-contained: only stdlib + api.config + api.auth
(for the hash helper). Importing api.routes / api.profiles here would create
cycles, so per-request profile pinning is done by callers (server.py).
"""
from __future__ import annotations

import hmac
import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from api.config import STATE_DIR

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
# Username pattern: lowercase alphanumeric + _ - , 3-32 chars, starts alnum.
# Matches the existing _PROFILE_ID_RE in api/profiles.py so derived
# profile_name = "user_<username>" is always a valid profile id.
USERNAME_RE = re.compile(r'^[a-z0-9][a-z0-9_-]{2,31}$')

# Reserve the literal "default" so adopting/renaming doesn't collide.
PROFILE_NAME_PREFIX = "user_"

DEFAULT_MAX_TURNS_PER_DAY = 200
DEFAULT_MAX_CONCURRENT_SESSIONS = 5
DEFAULT_MAX_STORAGE_MB = 2048

DB_PATH = STATE_DIR / "users.db"

# ── Thread-local request → user binding ───────────────────────────────────
# server.py sets this at the top of each request after resolving the session
# cookie to a user row, and clears it in the finally block.
_request_user_tls = threading.local()


def bind_request_user(user: Optional[dict]) -> None:
    """Bind the current thread to *user* for the duration of the request."""
    _request_user_tls.user = user


def clear_request_user() -> None:
    """Drop the per-request user binding for this thread."""
    _request_user_tls.user = None


def user_for_request_thread() -> Optional[dict]:
    """Return the user bound to the current request thread, or None.

    Returns ``None`` for cron-triggered turns and other out-of-band callers
    that did not pass through the HTTP layer.
    """
    return getattr(_request_user_tls, 'user', None)


# ── DB connection (single shared connection, WAL, lock-guarded) ───────────
# RLock so check_and_reserve() can call helpers (get_turns_used_today,
# register_active_session, ...) inside its `with lock:` block without
# self-deadlocking.
_DB_LOCK = threading.RLock()
_DB_CONN: Optional[sqlite3.Connection] = None
_SCHEMA_READY = False
_HAS_ANY_USER_CACHE: Optional[bool] = None  # None = unknown; bool once read.


def _connect() -> sqlite3.Connection:
    """Return the process-wide SQLite connection, opening on first call."""
    global _DB_CONN
    if _DB_CONN is not None:
        return _DB_CONN
    with _DB_LOCK:
        if _DB_CONN is not None:
            return _DB_CONN
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(DB_PATH),
            check_same_thread=False,
            isolation_level=None,  # autocommit; explicit BEGIN ... COMMIT only
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row
        # WAL is essential — every chat turn writes audit + usage_daily rows.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _DB_CONN = conn
        return conn


def _exec(sql: str, params: tuple = ()) -> sqlite3.Cursor:
    """Execute *sql* under the shared DB lock and return the cursor.

    Safe for fire-and-forget writes (INSERT / UPDATE / DELETE) where the
    caller doesn't read rows back. For SELECTs use ``_fetchone`` /
    ``_fetchall`` instead — those hold the lock through the fetch so a
    concurrent thread can't move the shared connection's cursor state and
    corrupt the row data mid-read. (#review-fix: race surfaced as
    ``IndexError: tuple index out of range`` from sqlite3.Row.keys()
    under 20-thread check_and_reserve hammering.)
    """
    conn = _connect()
    with _DB_LOCK:
        return conn.execute(sql, params)


def _fetchone(sql: str, params: tuple = ()):
    """Execute + fetchone under one continuous _DB_LOCK acquisition."""
    conn = _connect()
    with _DB_LOCK:
        return conn.execute(sql, params).fetchone()


def _fetchall(sql: str, params: tuple = ()) -> list:
    """Execute + fetchall under one continuous _DB_LOCK acquisition."""
    conn = _connect()
    with _DB_LOCK:
        return conn.execute(sql, params).fetchall()


def _executescript(script: str) -> None:
    conn = _connect()
    with _DB_LOCK:
        conn.executescript(script)


# ── Schema ─────────────────────────────────────────────────────────────────
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin','user')),
    profile_name TEXT UNIQUE NOT NULL,
    created_at INTEGER NOT NULL,
    disabled INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS quotas (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    max_turns_per_day INTEGER NOT NULL DEFAULT 200,
    max_concurrent_sessions INTEGER NOT NULL DEFAULT 5,
    max_storage_mb INTEGER NOT NULL DEFAULT 2048
);
CREATE TABLE IF NOT EXISTS usage_daily (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    day TEXT NOT NULL,
    turns_used INTEGER NOT NULL DEFAULT 0,
    last_updated INTEGER NOT NULL,
    PRIMARY KEY (user_id, day)
);
CREATE TABLE IF NOT EXISTS usage_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    actor_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    ts INTEGER NOT NULL,
    session_id TEXT,
    event TEXT NOT NULL,
    meta TEXT
);
CREATE TABLE IF NOT EXISTS sessions_active (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_audit_user_ts ON usage_audit(user_id, ts);
CREATE INDEX IF NOT EXISTS idx_audit_session ON usage_audit(session_id);
"""


def ensure_schema() -> None:
    """Create the schema if missing. Idempotent. Called once at startup."""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    _executescript(_SCHEMA_SQL)
    _SCHEMA_READY = True


# ── User-existence cache (for init-admin gate fast-path) ──────────────────
def _invalidate_has_any_user_cache() -> None:
    global _HAS_ANY_USER_CACHE
    _HAS_ANY_USER_CACHE = None


def has_any_user() -> bool:
    """Cheap check used by the auth middleware to decide whether to
    redirect to /init-admin. Cached in-process; invalidated on user create."""
    global _HAS_ANY_USER_CACHE
    if _HAS_ANY_USER_CACHE is not None:
        return _HAS_ANY_USER_CACHE
    try:
        ensure_schema()
        row = _fetchone("SELECT 1 FROM users LIMIT 1")
    except sqlite3.Error:
        logger.exception("has_any_user query failed")
        return False
    _HAS_ANY_USER_CACHE = row is not None
    return _HAS_ANY_USER_CACHE


# ── Password helpers ───────────────────────────────────────────────────────
def _hash(plain: str) -> str:
    """Hash a plaintext password using the same algorithm as legacy auth."""
    from api.auth import _hash_password
    return _hash_password(plain)


def _verify_hash(plain: str, stored_hash: str) -> bool:
    return hmac.compare_digest(_hash(plain), stored_hash)


# ── CRUD: users ────────────────────────────────────────────────────────────
def _row_to_dict(row) -> Optional[dict]:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}


def get_user_by_id(user_id: int) -> Optional[dict]:
    ensure_schema()
    row = _fetchone("SELECT * FROM users WHERE id = ?", (user_id,))
    return _row_to_dict(row)


def get_user_by_username(username: str) -> Optional[dict]:
    ensure_schema()
    if not username:
        return None
    row = _fetchone(
        "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
        (username,),
    )
    return _row_to_dict(row)


def get_user_by_profile_name(profile_name: str) -> Optional[dict]:
    """Reverse lookup: profile → user. Used by cron and out-of-band callers
    to charge a turn to the profile's owner."""
    ensure_schema()
    if not profile_name:
        return None
    row = _fetchone(
        "SELECT * FROM users WHERE profile_name = ?",
        (profile_name,),
    )
    return _row_to_dict(row)


def verify(username: str, plain: str) -> Optional[dict]:
    """Return the user row if credentials match and account is enabled."""
    user = get_user_by_username(username)
    if not user or user.get('disabled'):
        return None
    if not _verify_hash(plain, user['password_hash']):
        return None
    return user


def list_users() -> list[dict]:
    ensure_schema()
    rows = _fetchall(
        "SELECT u.*, q.max_turns_per_day, q.max_concurrent_sessions, "
        "q.max_storage_mb FROM users u "
        "LEFT JOIN quotas q ON q.user_id = u.id "
        "ORDER BY u.id ASC"
    )
    return [_row_to_dict(r) for r in rows]


def count_admins() -> int:
    ensure_schema()
    row = _fetchone(
        "SELECT COUNT(*) AS c FROM users WHERE role='admin' AND disabled=0"
    )
    return int(row['c']) if row else 0


def _derive_profile_name(username: str) -> str:
    return f"{PROFILE_NAME_PREFIX}{username}"


def create_user(
    username: str,
    password: str,
    role: str = 'user',
    *,
    profile_name: Optional[str] = None,
    quotas: Optional[dict] = None,
) -> dict:
    """Create a new user row + default quota row + (caller is expected to
    create the matching profile directory afterwards).

    Returns the inserted user as a dict. Raises ValueError on bad input,
    sqlite3.IntegrityError on uniqueness violations.
    """
    ensure_schema()
    if not USERNAME_RE.fullmatch(username or ''):
        raise ValueError(
            "username must be 3-32 chars of lowercase letters/digits/_/-, "
            "starting with letter or digit"
        )
    if not password or len(password) < 4:
        raise ValueError("password must be at least 4 characters")
    if role not in ('admin', 'user'):
        raise ValueError("role must be 'admin' or 'user'")
    pname = profile_name or _derive_profile_name(username)
    now = int(time.time())
    conn = _connect()
    with _DB_LOCK:
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "INSERT INTO users(username, password_hash, role, profile_name, created_at) "
                "VALUES(?,?,?,?,?)",
                (username, _hash(password), role, pname, now),
            )
            new_id = cur.lastrowid
            q = quotas or {}
            conn.execute(
                "INSERT INTO quotas(user_id, max_turns_per_day, max_concurrent_sessions, max_storage_mb) "
                "VALUES(?,?,?,?)",
                (
                    new_id,
                    int(q.get('max_turns_per_day', DEFAULT_MAX_TURNS_PER_DAY)),
                    int(q.get('max_concurrent_sessions', DEFAULT_MAX_CONCURRENT_SESSIONS)),
                    int(q.get('max_storage_mb', DEFAULT_MAX_STORAGE_MB)),
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
    _invalidate_has_any_user_cache()
    return get_user_by_id(new_id)


def update_user(
    user_id: int,
    *,
    password: Optional[str] = None,
    role: Optional[str] = None,
    disabled: Optional[bool] = None,
    profile_name: Optional[str] = None,
    quotas: Optional[dict] = None,
) -> dict:
    """Patch any subset of user fields / quotas. Atomic. Returns updated row.

    Raises ValueError on invalid input or last-admin demotion attempts.
    """
    ensure_schema()
    existing = get_user_by_id(user_id)
    if existing is None:
        raise ValueError("user not found")
    new_role = role if role is not None else existing['role']
    if new_role not in ('admin', 'user'):
        raise ValueError("role must be 'admin' or 'user'")
    if existing['role'] == 'admin' and new_role != 'admin':
        # Demoting an admin — must leave at least one admin standing.
        if count_admins() <= 1:
            raise ValueError("cannot demote the last admin")
    if existing['role'] == 'admin' and disabled is True and existing['disabled'] == 0:
        if count_admins() <= 1:
            raise ValueError("cannot disable the last admin")
    if profile_name is not None and not re.fullmatch(r'^[a-z0-9][a-z0-9_-]{0,63}$', profile_name):
        raise ValueError("invalid profile_name")

    conn = _connect()
    with _DB_LOCK:
        try:
            conn.execute("BEGIN IMMEDIATE")
            sets: list[str] = []
            args: list = []
            if password is not None:
                if len(password) < 4:
                    raise ValueError("password must be at least 4 characters")
                sets.append("password_hash = ?")
                args.append(_hash(password))
            if role is not None:
                sets.append("role = ?")
                args.append(role)
            if disabled is not None:
                sets.append("disabled = ?")
                args.append(1 if disabled else 0)
            if profile_name is not None:
                sets.append("profile_name = ?")
                args.append(profile_name)
            if sets:
                args.append(user_id)
                conn.execute(
                    f"UPDATE users SET {', '.join(sets)} WHERE id = ?",
                    tuple(args),
                )
            if quotas:
                conn.execute(
                    "UPDATE quotas SET "
                    "max_turns_per_day = COALESCE(?, max_turns_per_day), "
                    "max_concurrent_sessions = COALESCE(?, max_concurrent_sessions), "
                    "max_storage_mb = COALESCE(?, max_storage_mb) "
                    "WHERE user_id = ?",
                    (
                        quotas.get('max_turns_per_day'),
                        quotas.get('max_concurrent_sessions'),
                        quotas.get('max_storage_mb'),
                        user_id,
                    ),
                )
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
    return get_user_by_id(user_id)


def delete_user(user_id: int) -> dict:
    """Remove a user row (CASCADE removes quotas/usage/audit/sessions_active).

    Caller is responsible for the filesystem side (archive/hard-delete the
    profile directory). Refuses to delete the last admin.
    """
    ensure_schema()
    existing = get_user_by_id(user_id)
    if existing is None:
        raise ValueError("user not found")
    if existing['role'] == 'admin' and count_admins() <= 1:
        raise ValueError("cannot delete the last admin")
    conn = _connect()
    with _DB_LOCK:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return existing


# ── Quotas ────────────────────────────────────────────────────────────────
def get_quota(user_id: int) -> dict:
    ensure_schema()
    row = _fetchone(
        "SELECT * FROM quotas WHERE user_id = ?", (user_id,),
    )
    if row is None:
        return {
            'user_id': user_id,
            'max_turns_per_day': DEFAULT_MAX_TURNS_PER_DAY,
            'max_concurrent_sessions': DEFAULT_MAX_CONCURRENT_SESSIONS,
            'max_storage_mb': DEFAULT_MAX_STORAGE_MB,
        }
    return _row_to_dict(row)


# ── Usage audit (write-only convenience) ──────────────────────────────────
def write_audit(
    user_id: int,
    event: str,
    *,
    session_id: Optional[str] = None,
    actor_user_id: Optional[int] = None,
    meta: Optional[dict] = None,
) -> None:
    """Append one audit row. Never raises (logging-class concern)."""
    if user_id is None or not event:
        return
    try:
        ensure_schema()
        _exec(
            "INSERT INTO usage_audit(user_id, actor_user_id, ts, session_id, event, meta) "
            "VALUES(?,?,?,?,?,?)",
            (
                int(user_id),
                int(actor_user_id) if actor_user_id is not None else None,
                int(time.time()),
                session_id,
                event,
                json.dumps(meta, ensure_ascii=False) if meta else None,
            ),
        )
    except Exception:
        logger.debug("write_audit failed", exc_info=True)


def recent_audit(user_id: int, limit: int = 100) -> list[dict]:
    ensure_schema()
    rows = _fetchall(
        "SELECT * FROM usage_audit WHERE user_id = ? ORDER BY ts DESC LIMIT ?",
        (user_id, int(limit)),
    )
    return [_row_to_dict(r) for r in rows]


def daily_usage(user_id: int, days: int = 30) -> list[dict]:
    """Return up to *days* most recent daily usage rows, newest first."""
    ensure_schema()
    rows = _fetchall(
        "SELECT day, turns_used, last_updated FROM usage_daily "
        "WHERE user_id = ? ORDER BY day DESC LIMIT ?",
        (user_id, int(days)),
    )
    return [_row_to_dict(r) for r in rows]


def last_activity_ts(user_id: int) -> int:
    ensure_schema()
    row = _fetchone(
        "SELECT MAX(ts) AS t FROM usage_audit WHERE user_id = ?",
        (user_id,),
    )
    return int(row['t']) if row and row['t'] is not None else 0


# ── Active session tracking ───────────────────────────────────────────────
def count_active_sessions(user_id: int) -> int:
    ensure_schema()
    row = _fetchone(
        "SELECT COUNT(*) AS c FROM sessions_active WHERE user_id = ?",
        (user_id,),
    )
    return int(row['c']) if row else 0


def register_active_session(user_id: int, session_id: str) -> None:
    if not session_id:
        return
    ensure_schema()
    _exec(
        "INSERT OR IGNORE INTO sessions_active(user_id, session_id, started_at) "
        "VALUES(?,?,?)",
        (int(user_id), session_id, int(time.time())),
    )


def drop_active_session(session_id: str) -> None:
    """Called when a session is deleted/closed. Removes all matching rows
    (defensive: a session_id should belong to one user, but enforced by FK)."""
    if not session_id:
        return
    try:
        ensure_schema()
        _exec("DELETE FROM sessions_active WHERE session_id = ?", (session_id,))
    except Exception:
        logger.debug("drop_active_session failed", exc_info=True)


def sweep_stale_active_sessions(max_age_seconds: int = 24 * 3600) -> int:
    """Startup hygiene: drop rows older than *max_age_seconds*.

    Called once at boot from api/startup.py. Returns rows removed.
    """
    ensure_schema()
    cutoff = int(time.time()) - int(max_age_seconds)
    cur = _exec(
        "DELETE FROM sessions_active WHERE started_at < ?", (cutoff,),
    )
    return cur.rowcount or 0


# ── Daily turn counter (used by quotas.py) ────────────────────────────────
def _utc_day() -> str:
    return time.strftime('%Y-%m-%d', time.gmtime())


def get_turns_used_today(user_id: int) -> int:
    ensure_schema()
    row = _fetchone(
        "SELECT turns_used FROM usage_daily WHERE user_id=? AND day=?",
        (user_id, _utc_day()),
    )
    return int(row['turns_used']) if row else 0


def increment_turns_used_today(user_id: int) -> int:
    """Atomic UPSERT + increment. Held under _DB_LOCK (RLock) so the
    UPSERT and the follow-up SELECT see consistent state even when called
    by quotas.check_and_reserve() which already holds the lock — RLock is
    reentrant so re-acquisition is cheap and deadlock-free."""
    day = _utc_day()
    now = int(time.time())
    conn = _connect()
    with _DB_LOCK:
        conn.execute(
            "INSERT INTO usage_daily(user_id, day, turns_used, last_updated) VALUES(?,?,1,?) "
            "ON CONFLICT(user_id, day) DO UPDATE SET "
            "turns_used = turns_used + 1, last_updated = excluded.last_updated",
            (user_id, day, now),
        )
        row = conn.execute(
            "SELECT turns_used FROM usage_daily WHERE user_id=? AND day=?",
            (user_id, day),
        ).fetchone()
    return int(row['turns_used']) if row else 0


def db_lock():
    """Expose the lock for quotas.check_and_reserve atomic blocks."""
    return _DB_LOCK


def db_connection():
    """Expose the shared connection for quotas.check_and_reserve."""
    return _connect()
