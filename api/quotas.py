"""
Hermes Web UI -- Per-user quota enforcement.

Hooked once from api/streaming.py at the top of _run_agent_streaming(). Three
quota dimensions:
  - max_turns_per_day  (rolling UTC daily window)
  - max_concurrent_sessions (live active session count)
  - max_storage_mb (walked du of profile dir, 5-min cache)

Atomic check + reserve under api.users._DB_LOCK with BEGIN IMMEDIATE so that
a burst of parallel turns from one user cannot exceed the cap by N-1.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from api import users

logger = logging.getLogger(__name__)

# ── Storage du cache (per profile path, TTL'd) ────────────────────────────
_DU_CACHE: dict[str, tuple[float, int]] = {}  # path → (computed_at, bytes)
_DU_CACHE_LOCK = threading.Lock()
_DU_TTL_SECONDS = 300  # 5 minutes

# Subdirectory names skipped during the disk-usage walk. These are
# regenerable / VCS / build artifacts that bloat the walk by orders of
# magnitude on real workspaces but aren't user data the operator cares
# about counting against quota.
_DU_SKIP_DIRS = frozenset({
    '.git', '.svn', '.hg', '.bzr',
    '__pycache__', '.mypy_cache', '.pytest_cache', '.ruff_cache', '.tox',
    'node_modules', '.next', '.nuxt', '.cache',
    '.venv', 'venv', 'env', '.gradle', '.idea', '.vscode',
    'target', 'dist', 'build', '.terraform',
    '.DS_Store',
})

# Hard cap on walk wall-time. After this many seconds, return what we
# have so far rather than blocking the chat turn indefinitely on a giant
# workspace. 3s on a cold cache → degrade quota to "not enforced this
# turn" instead of tolling the user.
_DU_WALK_TIMEOUT_SECONDS = 3.0


def _cached_du(path: Path) -> int:
    """Return total size in bytes for *path*, walked and cached for 5 min.

    Skips regenerable / VCS / build dirs (see ``_DU_SKIP_DIRS``) so a
    workspace with ``node_modules`` doesn't make the quota gate take
    seconds. Capped at ``_DU_WALK_TIMEOUT_SECONDS`` of wall time; on
    timeout returns the partial sum and caches it (admin can still see a
    sensible MB number, just slightly low).

    Returns 0 if the path doesn't exist or is unreadable — quota should
    fail-open so a transient FS hiccup doesn't lock users out.
    """
    key = str(path)
    now = time.time()
    with _DU_CACHE_LOCK:
        hit = _DU_CACHE.get(key)
        if hit and (now - hit[0]) < _DU_TTL_SECONDS:
            return hit[1]
    total = 0
    deadline = now + _DU_WALK_TIMEOUT_SECONDS
    try:
        if not path.exists():
            return 0
        # os.walk lets us prune directories in-place (rglob can't), which is
        # essential to skip multi-GB node_modules trees cheaply.
        import os as _os
        for root, dirs, files in _os.walk(str(path)):
            # In-place prune so we don't descend into skipped dirs.
            dirs[:] = [d for d in dirs if d not in _DU_SKIP_DIRS]
            for fname in files:
                if fname in _DU_SKIP_DIRS:
                    continue
                try:
                    total += _os.stat(_os.path.join(root, fname)).st_size
                except OSError:
                    continue
            if time.time() > deadline:
                logger.debug("du walk for %s hit %.1fs timeout; partial sum %d",
                             path, _DU_WALK_TIMEOUT_SECONDS, total)
                break
    except Exception:
        logger.debug("du walk failed for %s", path, exc_info=True)
        return 0
    with _DU_CACHE_LOCK:
        _DU_CACHE[key] = (now, total)
    return total


def invalidate_du_cache(path: Optional[Path] = None) -> None:
    """Drop a cache entry (or all). Call after admin changes a user's quota
    or after a known large write so the next turn sees fresh numbers."""
    with _DU_CACHE_LOCK:
        if path is None:
            _DU_CACHE.clear()
        else:
            _DU_CACHE.pop(str(path), None)


def storage_mb_for(profile_home: Path) -> int:
    return int(_cached_du(profile_home) // (1024 * 1024))


# ── Main check+reserve entry point ────────────────────────────────────────
def check_and_reserve(
    user: dict,
    session_id: Optional[str],
    profile_home: Path,
    *,
    actor_user_id: Optional[int] = None,
) -> Optional[dict]:
    """Atomically verify the user is within all quotas and reserve a turn.

    Returns ``None`` on success (turn reserved, counter incremented).
    Returns ``{'reason': 'turns' | 'concurrency' | 'storage', 'detail': ...}``
    on block — caller should NOT proceed and should emit an error to the
    client. Audit row is written in either case.

    *actor_user_id* differs from user['id'] only when admin is impersonating.
    """
    if not user:
        return None  # no user binding → out-of-band caller, e.g. cron init
    user_id = int(user['id'])
    quota = users.get_quota(user_id)
    max_turns = int(quota.get('max_turns_per_day') or 0)
    max_concur = int(quota.get('max_concurrent_sessions') or 0)
    max_storage_mb = int(quota.get('max_storage_mb') or 0)

    # Storage check is cheap (cached); do it outside the atomic block so we
    # don't hold the DB lock through filesystem IO on a cache miss.
    if max_storage_mb > 0:
        used_mb = storage_mb_for(profile_home)
        if used_mb >= max_storage_mb:
            users.write_audit(
                user_id, 'quota_blocked', session_id=session_id,
                actor_user_id=actor_user_id,
                meta={'reason': 'storage', 'used_mb': used_mb, 'limit_mb': max_storage_mb},
            )
            return {'reason': 'storage', 'used_mb': used_mb, 'limit_mb': max_storage_mb}

    conn = users.db_connection()
    lock = users.db_lock()
    with lock:
        try:
            conn.execute("BEGIN IMMEDIATE")

            # Turns/day check
            if max_turns > 0:
                used = users.get_turns_used_today(user_id)
                if used >= max_turns:
                    conn.execute("ROLLBACK")
                    users.write_audit(
                        user_id, 'quota_blocked', session_id=session_id,
                        actor_user_id=actor_user_id,
                        meta={'reason': 'turns', 'used': used, 'limit': max_turns},
                    )
                    return {'reason': 'turns', 'used': used, 'limit': max_turns}

            # Concurrency check (after registering this session, count must be ≤ cap)
            if session_id and max_concur > 0:
                users.register_active_session(user_id, session_id)
                active = users.count_active_sessions(user_id)
                if active > max_concur:
                    # Roll back: drop the row we just inserted iff this session
                    # wasn't already counted before us. Cheapest path is to
                    # leave it (idempotent INSERT OR IGNORE) only if it was
                    # already there. We check by ts.
                    conn.execute(
                        "DELETE FROM sessions_active WHERE user_id=? AND session_id=? "
                        "AND started_at >= ?",
                        (user_id, session_id, int(time.time()) - 2),
                    )
                    conn.execute("ROLLBACK")
                    users.write_audit(
                        user_id, 'quota_blocked', session_id=session_id,
                        actor_user_id=actor_user_id,
                        meta={'reason': 'concurrency', 'active': active, 'limit': max_concur},
                    )
                    return {'reason': 'concurrency', 'active': active, 'limit': max_concur}

            # All checks passed — reserve the turn.
            new_count = users.increment_turns_used_today(user_id)
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            logger.exception("quota check_and_reserve failed")
            return None  # fail-open: don't block users on internal errors

    users.write_audit(
        user_id, 'turn_start', session_id=session_id,
        actor_user_id=actor_user_id,
        meta={'turns_today': new_count},
    )
    return None


def record_turn_end(
    user_id: Optional[int],
    session_id: Optional[str],
    *,
    status: str = 'ok',
    tokens: Optional[int] = None,
    actor_user_id: Optional[int] = None,
) -> None:
    if not user_id:
        return
    users.write_audit(
        user_id, 'turn_end', session_id=session_id,
        actor_user_id=actor_user_id,
        meta={'status': status, 'tokens': tokens} if tokens is not None else {'status': status},
    )
