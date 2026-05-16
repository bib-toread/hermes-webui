"""
Hermes Web UI -- Admin endpoints: user CRUD, usage view, init-admin bootstrap.

All endpoints (except /api/init-admin/*) require role=admin. The route
dispatch in api/routes.py forwards to the handler functions below.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs

from api import users, quotas
from api.helpers import bad, j, read_body

logger = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────────────────
def _require_admin(handler) -> Optional[dict]:
    """Return the admin user dict, or send 403 and return None."""
    from api.auth import current_user
    user = current_user(handler)
    if not user or user.get('role') != 'admin':
        bad(handler, "admin only", status=403)
        return None
    return user


def _serialize_user(u: dict) -> dict:
    """Strip password_hash before sending to client."""
    if not u:
        return {}
    out = {k: v for k, v in u.items() if k != 'password_hash'}
    return out


def _augment_with_usage(u: dict) -> dict:
    """Decorate a user dict with today's turns + last activity ts."""
    u = _serialize_user(u)
    uid = u.get('id')
    if uid:
        u['turns_used_today'] = users.get_turns_used_today(uid)
        u['last_activity_ts'] = users.last_activity_ts(uid)
        u['active_sessions'] = users.count_active_sessions(uid)
        u['quota'] = users.get_quota(uid)
    return u


# ── Init-admin (public, only when no users exist) ─────────────────────────
def handle_init_admin_status(handler, parsed) -> bool:
    """GET /api/init-admin/status — public, reports whether init is needed."""
    from api.profiles import list_profiles_api
    needs = not users.has_any_user()
    profiles: list[dict] = []
    if needs:
        try:
            profiles = [
                {'name': p['name'], 'path': p.get('path')}
                for p in list_profiles_api()
            ]
        except Exception:
            logger.debug("list_profiles_api failed in init-admin status", exc_info=True)
            profiles = []
    return j(handler, {'needs_init': needs, 'existing_profiles': profiles})


def handle_init_admin_create(handler, parsed, body) -> bool:
    """POST /api/init-admin/create — public iff no users exist; race-safe."""
    if users.has_any_user():
        return bad(handler, "init already complete", status=409)
    if not body:
        return bad(handler, "request body required")
    username = str(body.get('username', '')).strip().lower()
    password = str(body.get('password', ''))
    adopt = str(body.get('adopt_profile', '__new__')).strip() or '__new__'

    if not users.USERNAME_RE.fullmatch(username):
        return bad(handler, "invalid username (3-32 chars, [a-z0-9_-], must start alnum)")
    if len(password) < 4:
        return bad(handler, "password must be at least 4 characters")

    # Decide the profile to associate.
    if adopt == '__new__':
        profile_name = users._derive_profile_name(username)
        try:
            from api.profiles import create_profile_api, _PROFILE_ID_RE
            if not _PROFILE_ID_RE.fullmatch(profile_name):
                return bad(handler, f"derived profile name {profile_name!r} is invalid")
            create_profile_api(profile_name)
        except FileExistsError:
            # Already exists on disk — adopt it instead of failing.
            pass
        except Exception as exc:
            logger.exception("init-admin create_profile_api failed")
            return bad(handler, f"failed to create profile: {exc}", status=500)
    else:
        profile_name = adopt
        # Validate that profile actually exists in the list.
        try:
            from api.profiles import list_profiles_api
            available = {p['name'] for p in list_profiles_api()}
            if profile_name not in available and profile_name != 'default':
                return bad(handler, f"profile {profile_name!r} does not exist")
        except Exception:
            logger.debug("list_profiles_api failed during adopt validation", exc_info=True)

    # Create the admin user row (atomic; race-safe via UNIQUE on username + INSERT).
    try:
        new_user = users.create_user(
            username=username, password=password, role='admin',
            profile_name=profile_name,
        )
    except ValueError as exc:
        return bad(handler, str(exc))
    except Exception as exc:
        logger.exception("init-admin create_user failed")
        # If we created a profile dir above but failed here, leave it — admin
        # can re-attempt and it will adopt instead.
        return bad(handler, f"failed to create user: {exc}", status=500)

    # Multi-user: snapshot the admin's profile config to the global root so
    # future user creations and propagation flows have a source of truth.
    try:
        from api import global_config as _gc
        _gc.snapshot_admin_to_global(new_user['profile_name'])
    except Exception:
        logger.debug("snapshot_admin_to_global on init failed", exc_info=True)

    # Seed a default workspace pointer so the first admin lands in their
    # own profile workspace rather than the process-global DEFAULT_WORKSPACE
    # (which would still resolve to a path likely outside the multi-user
    # tenancy model — e.g. ~/workspace shared with every user).
    _seed_default_workspace(new_user['profile_name'],
                            display_name=f"Home ({new_user['username']})")

    # Auto-login the new admin by setting a session cookie.
    from api.auth import create_session_for_user, set_auth_cookie, _security_headers_safe
    cookie_val = create_session_for_user(new_user['id'])
    handler.send_response(200)
    handler.send_header('Content-Type', 'application/json')
    handler.send_header('Cache-Control', 'no-store')
    _security_headers_safe(handler)
    set_auth_cookie(handler, cookie_val)
    handler.end_headers()
    handler.wfile.write(json.dumps({'ok': True, 'user': _serialize_user(new_user)}).encode())
    return True


# ── Admin user CRUD ───────────────────────────────────────────────────────
_USER_ID_PATH_RE = re.compile(r'^/api/admin/users/(\d+)$')
_USAGE_ID_PATH_RE = re.compile(r'^/api/admin/usage/(\d+)$')


def _parse_user_id(path: str) -> Optional[int]:
    m = _USER_ID_PATH_RE.match(path)
    return int(m.group(1)) if m else None


def handle_admin_users_get(handler, parsed) -> bool:
    """GET /api/admin/users — list, /api/admin/users/<id> — detail.

    The list response also includes ``orphan_profiles`` — profile
    directories on disk that no user row claims. Surfaced so admin can
    reassign them (PATCH .profile_name) or archive them.
    """
    if _require_admin(handler) is None:
        return True
    uid = _parse_user_id(parsed.path)
    if uid is not None:
        u = users.get_user_by_id(uid)
        if not u:
            return bad(handler, "user not found", status=404)
        return j(handler, _augment_with_usage(u))
    rows = users.list_users()
    return j(handler, {
        'users': [_augment_with_usage(r) for r in rows],
        'orphan_profiles': _list_orphan_profiles(rows),
    })


def _seed_default_workspace(profile_name: str, display_name: str = "Home") -> None:
    """Seed a fresh user's profile with a default workspace pointer.

    Writes two per-profile files under ``<profile>/webui_state/``:
      - ``last_workspace.txt`` → ``<profile>/workspace/`` so the new user
        lands in their own isolated workspace on first login (not the
        WebUI's process-global DEFAULT_WORKSPACE which would leak shared
        state across users).
      - ``workspaces.json`` → a single picker entry labeled ``Home``
        pointing at the same dir so the workspace switcher in the right
        rail isn't empty on first open.

    Idempotent — re-running on an already-seeded profile just refreshes
    the pointer files; never deletes user-added workspace entries.
    """
    if not profile_name:
        return
    try:
        from api.profiles import _resolve_profile_home_for_name
        profile_home = _resolve_profile_home_for_name(profile_name)
        ws_dir = profile_home / 'webui_state'
        workspace_dir = profile_home / 'workspace'
        try:
            workspace_dir.mkdir(parents=True, exist_ok=True)
            ws_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.debug("workspace seed mkdir failed for %s", profile_name, exc_info=True)
            return
        # last_workspace.txt — overwrite is fine (admin-side seed).
        try:
            (ws_dir / 'last_workspace.txt').write_text(
                str(workspace_dir.resolve()), encoding='utf-8',
            )
        except OSError:
            logger.debug("failed to write last_workspace.txt for %s", profile_name, exc_info=True)
        # workspaces.json — only write if missing, so we don't trash any
        # entries the user (or a previous admin seed) already added.
        ws_list_file = ws_dir / 'workspaces.json'
        if not ws_list_file.exists():
            try:
                import json as _json
                ws_list_file.write_text(
                    _json.dumps([{
                        'name': display_name,
                        'path': str(workspace_dir.resolve()),
                    }], ensure_ascii=False, indent=2),
                    encoding='utf-8',
                )
            except OSError:
                logger.debug("failed to write workspaces.json for %s", profile_name, exc_info=True)
        # config.yaml — set terminal.cwd to the per-profile workspace so the
        # agent's runtime TERMINAL_CWD env var lands inside the user's own
        # tree even when a stale session.workspace points elsewhere. This is
        # the safety net that catches the case where get_last_workspace()
        # would otherwise return a bad value (e.g. global /root/workspace
        # leaked into last_workspace.txt by a prior install / picker bug).
        try:
            _seed_terminal_cwd_in_config(profile_home / 'config.yaml',
                                         str(workspace_dir.resolve()))
        except Exception:
            logger.debug("failed to seed terminal.cwd for %s", profile_name, exc_info=True)
    except Exception:
        logger.debug("_seed_default_workspace failed for %s", profile_name, exc_info=True)


def _seed_terminal_cwd_in_config(config_yaml_path: Path, cwd: str) -> None:
    """Ensure ``terminal.cwd`` in profile config.yaml points at *cwd*.

    Preserves all other keys in the file. Safe to call repeatedly. No-op
    when PyYAML is unavailable (admin will need to set terminal.cwd by
    hand in that environment). 'terminal' is NOT in GLOBAL_CONFIG_KEYS so
    this value survives later admin-side global config cascades.
    """
    try:
        import yaml as _yaml
    except ImportError:
        logger.debug("PyYAML unavailable; skip terminal.cwd seed for %s", config_yaml_path)
        return
    data: dict = {}
    if config_yaml_path.exists():
        try:
            loaded = _yaml.safe_load(config_yaml_path.read_text(encoding='utf-8'))
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            logger.debug("config.yaml unreadable at %s; will overwrite", config_yaml_path, exc_info=True)
    term = data.get('terminal') if isinstance(data.get('terminal'), dict) else {}
    term['cwd'] = cwd
    data['terminal'] = term
    try:
        # Atomic write so a crash mid-write doesn't truncate the user's
        # config.yaml — the agent reads this on every chat turn.
        # (#review-fix bug_034: atomic write for operator-facing config)
        from api.global_config import _atomic_write_text
        _atomic_write_text(
            config_yaml_path,
            _yaml.dump(data, default_flow_style=False, allow_unicode=True),
        )
    except OSError:
        logger.debug("failed to write %s", config_yaml_path, exc_info=True)


def _list_orphan_profiles(rows: list) -> list[dict]:
    """Return profiles on disk that no user row claims.

    Lets admin spot stale profile directories left over from upgrades or
    deleted-without-hard users. Each entry: ``{name, path}``.
    """
    try:
        from api.profiles import list_profiles_api
        claimed = {(r.get('profile_name') or '').lower() for r in (rows or [])}
        orphans: list[dict] = []
        for p in list_profiles_api():
            name = (p.get('name') or '').lower()
            if not name or name in claimed:
                continue
            # Skip the root/default unless admin explicitly adopted it.
            if name == 'default':
                continue
            orphans.append({'name': p.get('name'), 'path': p.get('path')})
        return orphans
    except Exception:
        logger.debug("orphan profile listing failed", exc_info=True)
        return []


def handle_admin_users_post(handler, parsed, body) -> bool:
    """POST /api/admin/users — create a user."""
    actor = _require_admin(handler)
    if actor is None:
        return True
    if not body:
        return bad(handler, "request body required")
    username = str(body.get('username', '')).strip().lower()
    password = str(body.get('password', ''))
    role = str(body.get('role', 'user')).strip().lower()
    quotas_in = body.get('quotas') or {}

    if not users.USERNAME_RE.fullmatch(username):
        return bad(handler, "invalid username")
    try:
        # Create the profile directory first; if user_create fails we delete it.
        from api.profiles import create_profile_api
        profile_name = users._derive_profile_name(username)
        try:
            create_profile_api(profile_name)
        except FileExistsError:
            return bad(handler, f"profile {profile_name!r} already exists; pick a different username")
        # Multi-user: seed the brand-new profile from the admin-curated
        # global config so the user starts with all approved providers
        # and the default model already wired up.
        try:
            from api import global_config as _gc
            _gc.seed_user_profile_from_global(profile_name)
        except Exception:
            logger.debug("seed_user_profile_from_global failed for %s", profile_name, exc_info=True)
        try:
            new_user = users.create_user(
                username=username, password=password, role=role,
                profile_name=profile_name, quotas=quotas_in,
            )
        except Exception:
            # Clean up the orphan profile dir.
            try:
                from api.profiles import delete_profile_api
                delete_profile_api(profile_name)
            except Exception:
                logger.debug("failed to clean orphan profile %s", profile_name, exc_info=True)
            raise
    except ValueError as exc:
        return bad(handler, str(exc))
    except Exception as exc:
        logger.exception("create user failed")
        return bad(handler, str(exc), status=500)
    # Seed a default per-profile workspace so bob's right-rail file browser
    # opens at his own ~/.hermes/profiles/user_bob/workspace/ on first login,
    # not at whatever last_workspace.txt the OS happened to have lying around.
    _seed_default_workspace(new_user['profile_name'],
                            display_name=f"Home ({new_user['username']})")

    users.write_audit(new_user['id'], 'user_created', actor_user_id=actor['id'])
    return j(handler, _serialize_user(new_user), status=201)


def handle_admin_users_patch(handler, parsed, body) -> bool:
    """PATCH /api/admin/users/<id> — update role/password/disabled/quotas/profile."""
    actor = _require_admin(handler)
    if actor is None:
        return True
    uid = _parse_user_id(parsed.path)
    if uid is None:
        return bad(handler, "user id required in path", status=400)
    body = body or {}
    try:
        updated = users.update_user(
            uid,
            password=body.get('password'),
            role=body.get('role'),
            disabled=body.get('disabled'),
            profile_name=body.get('profile_name'),
            quotas=body.get('quotas'),
        )
    except ValueError as exc:
        return bad(handler, str(exc), status=409 if 'last admin' in str(exc) else 400)
    except Exception as exc:
        logger.exception("update user failed")
        return bad(handler, str(exc), status=500)
    users.write_audit(
        uid, 'user_updated', actor_user_id=actor['id'],
        meta={k: v for k, v in body.items() if k != 'password'},
    )
    return j(handler, _serialize_user(updated))


def handle_admin_users_delete(handler, parsed, body) -> bool:
    """DELETE /api/admin/users/<id>?archive=1 (default) | ?hard=1"""
    actor = _require_admin(handler)
    if actor is None:
        return True
    uid = _parse_user_id(parsed.path)
    if uid is None:
        return bad(handler, "user id required in path", status=400)
    qs = parse_qs(parsed.query)
    hard = qs.get('hard', ['0'])[0].strip().lower() in ('1', 'true', 'yes')

    existing = users.get_user_by_id(uid)
    if not existing:
        return bad(handler, "user not found", status=404)

    try:
        users.delete_user(uid)
    except ValueError as exc:
        return bad(handler, str(exc), status=409)

    # SECURITY: invalidate the deleted user's sessions ONLY after the
    # row is removed. (The dispatcher used to do this unconditionally
    # before the admin check, which let any authenticated user force-
    # logout any other user by issuing DELETE /api/admin/users/<id>.)
    try:
        from api.auth import invalidate_sessions_for_user
        invalidate_sessions_for_user(uid)
    except Exception:
        logger.debug("session invalidation after delete failed", exc_info=True)

    profile_name = existing.get('profile_name')
    if profile_name:
        from api.profiles import _DEFAULT_HERMES_HOME, _PROFILE_ID_RE, _resolve_profile_home_for_name
        # Use the canonical resolver so root-alias names are handled the same
        # everywhere (admin_users, streaming, global_config).
        profile_dir = _resolve_profile_home_for_name(profile_name)
        # Defense in depth: never archive/delete the base ~/.hermes itself.
        if (
            profile_dir != _DEFAULT_HERMES_HOME
            and profile_dir.exists()
            and _PROFILE_ID_RE.fullmatch(profile_name)
        ):
            if hard:
                try:
                    shutil.rmtree(str(profile_dir))
                except Exception:
                    logger.exception("hard delete of profile dir failed")
            else:
                archive_root = _DEFAULT_HERMES_HOME / 'archive'
                try:
                    archive_root.mkdir(parents=True, exist_ok=True)
                    dest = archive_root / f"{existing['username']}-{int(time.time())}"
                    shutil.move(str(profile_dir), str(dest))
                except Exception:
                    logger.exception("archive of profile dir failed")
    users.write_audit(
        uid, 'user_deleted', actor_user_id=actor['id'],
        meta={'hard': hard, 'profile': profile_name},
    )
    return j(handler, {'ok': True, 'archived': not hard})


# ── Admin usage view ──────────────────────────────────────────────────────
def handle_admin_usage(handler, parsed) -> bool:
    """GET /api/admin/usage — all-users summary.
    GET /api/admin/usage/<user_id>?days=30 — per-user detail."""
    if _require_admin(handler) is None:
        return True
    m = _USAGE_ID_PATH_RE.match(parsed.path)
    if m:
        uid = int(m.group(1))
        u = users.get_user_by_id(uid)
        if not u:
            return bad(handler, "user not found", status=404)
        qs = parse_qs(parsed.query)
        try:
            days = int(qs.get('days', ['30'])[0])
        except ValueError:
            days = 30
        days = max(1, min(days, 365))
        # Storage from profile dir (single canonical resolver).
        from api.profiles import _resolve_profile_home_for_name
        profile_dir = _resolve_profile_home_for_name(u['profile_name'])
        return j(handler, {
            'user': _serialize_user(u),
            'quota': users.get_quota(uid),
            'turns_used_today': users.get_turns_used_today(uid),
            'daily_usage': users.daily_usage(uid, days=days),
            'recent_audit': users.recent_audit(uid, limit=100),
            'last_activity_ts': users.last_activity_ts(uid),
            'active_sessions': users.count_active_sessions(uid),
            'storage_mb': quotas.storage_mb_for(profile_dir),
        })
    # Summary table.
    rows = users.list_users()
    return j(handler, {'users': [_augment_with_usage(r) for r in rows]})
