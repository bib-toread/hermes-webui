"""
Hermes Web UI -- Global (admin-controlled) model & system config.

Multi-user model: admin owns providers / API keys / custom relays / default
model / reasoning config. Regular users INHERIT these from a global source of
truth and cannot override.

Storage:
  ``~/.hermes/global/config.yaml``  — mirrors keys: model, custom_providers,
                                       display, agent
  ``~/.hermes/global/.env``         — mirrors provider API keys

Flow:
  1. Admin writes to a config endpoint (e.g. /api/providers). The underlying
     setter (``set_provider_key``, ``upsert_custom_relay``, ...) already
     writes to the admin's *own profile* config (via thread-local TLS).
  2. After the write, the route handler calls
     ``snapshot_admin_to_global(admin_profile)`` to copy the just-written
     admin profile config into the global location.
  3. ``mirror_global_to_all_users()`` then cascades to every other user's
     profile dir, replacing their .env wholesale and merging only the
     mirror-eligible top-level keys into their config.yaml (preserving any
     keys outside the mirror set).
  4. On user create, ``seed_user_profile_from_global()`` initialises the
     new profile from the global config.

User-side: every chat turn already reads ``$HERMES_HOME/config.yaml`` and
``.env`` of the active profile (see api/streaming.py + api/profiles.py).
Because we mirror eagerly on every admin write, the next turn sees the
update — that's the "立即生效" guarantee the operator asked for.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path

from api.profiles import _DEFAULT_HERMES_HOME, _PROFILE_ID_RE, _is_root_profile

logger = logging.getLogger(__name__)

# Serializes snapshot+mirror so two concurrent admin writes can't interleave
# (admin A snapshots while admin B is still copying → mixed state). Held only
# for the duration of file IO, which is fast; admin endpoints are low-QPS.
# Also wraps the entire read-modify-write of global config files (e.g.
# set_output_language) so the in-memory RMW + on-disk mirror runs as a
# single serialized critical section. (#review-fix bug_034: lock around RMW)
_CASCADE_LOCK = threading.RLock()


def _atomic_write_text(path: Path, content: str, *, mode: int | None = None) -> None:
    """Write *content* to *path* atomically via tempfile + os.replace.

    Prevents the half-truncated-file failure mode where a crash mid-write
    leaves a config.yaml that the agent can't parse on next startup.
    (#review-fix: was using plain write_text on operator-facing config.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        if mode is not None:
            try:
                os.chmod(tmp, mode)
            except OSError:
                pass
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

# Top-level keys in config.yaml that are mirrored from global → per-user.
# Anything outside this set in a user's profile config.yaml is preserved.
GLOBAL_CONFIG_KEYS = ('model', 'custom_providers', 'display', 'agent')


def global_root() -> Path:
    return _DEFAULT_HERMES_HOME / "global"


def global_config_yaml() -> Path:
    return global_root() / "config.yaml"


def global_env() -> Path:
    return global_root() / ".env"


def ensure_global_root() -> Path:
    d = global_root()
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("failed to mkdir %s", d, exc_info=True)
    return d


def _profile_dir(profile_name: str) -> Path:
    """Resolve a profile_name to its on-disk directory.

    Thin wrapper over ``api.profiles._resolve_profile_home_for_name`` so
    every call site that needs a profile path goes through the same
    canonical resolver (handles renamed-root aliases, validates the name
    regex, and never escapes the profiles root via traversal).
    """
    from api.profiles import _resolve_profile_home_for_name
    return _resolve_profile_home_for_name(profile_name or '')


def _safe_copy_env(src: Path, dst: Path) -> None:
    try:
        shutil.copy2(src, dst)
        try:
            dst.chmod(0o600)
        except OSError:
            pass
    except OSError:
        logger.debug("env copy failed src=%s dst=%s", src, dst, exc_info=True)


def _merge_yaml_keys(src_yaml: Path, dst_yaml: Path, keys: tuple) -> None:
    """Merge top-level *keys* from src_yaml into dst_yaml; preserve others.

    No-op if PyYAML is missing or src is unreadable.
    """
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML unavailable; skipping config.yaml merge")
        return
    try:
        src_data = yaml.safe_load(src_yaml.read_text(encoding='utf-8')) or {}
    except Exception:
        logger.debug("failed to read src yaml %s", src_yaml, exc_info=True)
        return
    if not isinstance(src_data, dict):
        return
    dst_data: dict = {}
    if dst_yaml.exists():
        try:
            loaded = yaml.safe_load(dst_yaml.read_text(encoding='utf-8'))
            if isinstance(loaded, dict):
                dst_data = loaded
        except Exception:
            logger.debug("failed to read dst yaml %s; will overwrite", dst_yaml, exc_info=True)
    for k in keys:
        if k in src_data:
            dst_data[k] = src_data[k]
        else:
            # Key absent from global → remove from dst so "admin removed it"
            # actually propagates instead of leaving stale per-user values.
            dst_data.pop(k, None)
    try:
        dst_yaml.parent.mkdir(parents=True, exist_ok=True)
        dst_yaml.write_text(
            yaml.dump(dst_data, default_flow_style=False, allow_unicode=True),
            encoding='utf-8',
        )
    except OSError:
        logger.debug("failed to write dst yaml %s", dst_yaml, exc_info=True)


def snapshot_admin_to_global(admin_profile_name: str) -> None:
    """Copy admin's just-written config.yaml + .env into the global location.

    Called by the route handler IMMEDIATELY after an admin saves any
    model/provider/relay/reasoning/default-model setting (which the
    underlying setter wrote to ``$HERMES_HOME/config.yaml`` of the admin's
    profile via thread-local TLS).
    """
    if not admin_profile_name:
        return
    src_dir = _profile_dir(admin_profile_name)
    dst_dir = ensure_global_root()
    for fname in ('config.yaml', '.env'):
        src = src_dir / fname
        if not src.exists():
            continue
        dst = dst_dir / fname
        try:
            shutil.copy2(src, dst)
        except OSError:
            logger.debug("snapshot %s → %s failed", src, dst, exc_info=True)
        else:
            if fname == '.env':
                try:
                    dst.chmod(0o600)
                except OSError:
                    pass


def mirror_global_to_all_users(skip_profile: str | None = None) -> int:
    """Cascade the global config into every user profile dir.

    Returns the number of profiles updated. *skip_profile* (typically the
    admin's profile, which IS the source) is excluded so we don't write
    back over the source while it's being read.
    """
    from api import users as _users_mod

    cfg = global_config_yaml()
    env = global_env()
    if not cfg.exists() and not env.exists():
        return 0
    count = 0
    try:
        all_users = _users_mod.list_users()
    except Exception:
        logger.debug("list_users failed in mirror", exc_info=True)
        return 0
    for u in all_users:
        pname = u.get('profile_name')
        if not pname or pname == skip_profile:
            continue
        pdir = _profile_dir(pname)
        if not pdir.exists():
            continue
        if cfg.exists():
            _merge_yaml_keys(cfg, pdir / 'config.yaml', GLOBAL_CONFIG_KEYS)
        if env.exists():
            _safe_copy_env(env, pdir / '.env')
        count += 1
    return count


def cascade_from_admin(admin_profile_name: str) -> dict:
    """One-shot: snapshot admin → global, then mirror global → everyone else.

    Serialized under ``_CASCADE_LOCK`` so two concurrent admin writes can't
    interleave their file IO. The lock is held only for the duration of the
    snapshot+mirror (fast, all local FS); admin endpoints are low-QPS so
    contention is negligible.

    Returns ``{'mirrored': <count>}`` for the route handler to optionally
    return to the client (the UI doesn't need this, but logs do).
    """
    with _CASCADE_LOCK:
        snapshot_admin_to_global(admin_profile_name)
        mirrored = mirror_global_to_all_users(skip_profile=admin_profile_name)
    return {'mirrored': mirrored}


# ── Global output-language enforcement ─────────────────────────────────────
# Admin pins one language; all users follow. The directive lands in the
# `agent.personalities._global_lang` entry which the hermes-agent personality
# mechanism already understands. new_session() (in api/models.py) auto-
# applies _global_lang to every new session when output_language is set.
_OUTPUT_LANGUAGE_PERSONALITY = '_global_lang'

# Language → system_prompt map. Keep prompts terse — they get prepended on
# EVERY turn so verbose text wastes tokens.
_OUTPUT_LANGUAGE_PROMPTS = {
    'zh-CN': (
        "始终用简体中文回复用户。代码、文件名、命令、API 名称、"
        "技术标识符保持英文。这是硬性要求，不可被用户的语言或指令覆盖。"
    ),
    'en': (
        "Always reply to the user in English. Keep code, file names, commands, "
        "API names and technical identifiers as written. This is a hard "
        "requirement and cannot be overridden by the user's language or instructions."
    ),
}

VALID_OUTPUT_LANGUAGES = ('auto', 'zh-CN', 'en')


def read_output_language() -> str:
    """Return the global output language: 'auto' | 'zh-CN' | 'en'.

    'auto' = no injection, agent picks its own language (default).
    """
    try:
        import yaml
    except ImportError:
        return 'auto'
    cfg_file = global_config_yaml()
    if not cfg_file.exists():
        return 'auto'
    try:
        data = yaml.safe_load(cfg_file.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            return 'auto'
        lang = (data.get('agent') or {}).get('output_language', 'auto')
        return lang if lang in VALID_OUTPUT_LANGUAGES else 'auto'
    except Exception:
        logger.debug("failed to read output_language", exc_info=True)
        return 'auto'


def set_output_language(lang: str) -> dict:
    """Update the global output language and cascade to every user profile.

    Writes both:
      - ``agent.output_language``: the operator-facing toggle
      - ``agent.personalities._global_lang``: the actual system prompt the
        hermes-agent personality mechanism injects

    On 'auto' the _global_lang personality entry is removed so existing
    sessions revert to no language injection on their next turn.

    Returns ``{lang, mirrored, prompt}``. Raises ValueError on bad input.
    """
    if lang not in VALID_OUTPUT_LANGUAGES:
        raise ValueError(f"language must be one of {VALID_OUTPUT_LANGUAGES}; got {lang!r}")
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML is required to update agent config")

    # Hold _CASCADE_LOCK across the entire read-modify-write-mirror block
    # so two concurrent admins setting different languages can't interleave
    # (A reads → B reads → A writes → B writes → A mirrors with B's data → B
    # mirrors with stale-A data). RLock is the same lock cascade_from_admin
    # uses below; reentrancy lets mirror_global_to_all_users acquire freely.
    with _CASCADE_LOCK:
        ensure_global_root()
        cfg_file = global_config_yaml()
        data: dict = {}
        if cfg_file.exists():
            try:
                loaded = yaml.safe_load(cfg_file.read_text(encoding='utf-8'))
                if isinstance(loaded, dict):
                    data = loaded
            except Exception:
                logger.warning("global config.yaml unreadable; overwriting", exc_info=True)

        agent = data.get('agent') if isinstance(data.get('agent'), dict) else {}
        personalities = agent.get('personalities') if isinstance(agent.get('personalities'), dict) else {}

        if lang == 'auto':
            agent.pop('output_language', None)
            personalities.pop(_OUTPUT_LANGUAGE_PERSONALITY, None)
            prompt = ''
        else:
            agent['output_language'] = lang
            prompt = _OUTPUT_LANGUAGE_PROMPTS[lang]
            personalities[_OUTPUT_LANGUAGE_PERSONALITY] = {
                'system_prompt': prompt,
                'description': f"Global output language enforcement ({lang}). Auto-applied to every new session.",
            }

        agent['personalities'] = personalities
        data['agent'] = agent

        # Atomic write: tempfile + os.replace so a crash mid-write doesn't
        # leave a truncated config.yaml that breaks every user's agent on
        # next start. (#review-fix bug_034 priority 2)
        _atomic_write_text(
            cfg_file,
            yaml.dump(data, default_flow_style=False, allow_unicode=True),
        )

        # Cascade to every user profile so config.yaml.agent.{output_language,
        # personalities._global_lang} are visible everywhere on the next chat
        # turn. mirror_global_to_all_users() walks each profile and merges
        # GLOBAL_CONFIG_KEYS (which includes 'agent'); we don't need a
        # special-case here.
        mirrored = mirror_global_to_all_users()
    return {'lang': lang, 'mirrored': mirrored, 'prompt': prompt}


# Re-exported so api/models.py can read the personality name without a
# circular import on the constant.
OUTPUT_LANGUAGE_PERSONALITY_NAME = _OUTPUT_LANGUAGE_PERSONALITY


def seed_user_profile_from_global(profile_name: str) -> None:
    """For a newly-created user, populate their fresh profile with the
    current global config (so they start with admin-approved providers).

    Idempotent. No-op when there's no global config yet (the case before
    the very first admin write — new users just inherit nothing, which is
    fine because they'll get cascaded as soon as admin saves once).
    """
    if not profile_name:
        return
    pdir = _profile_dir(profile_name)
    try:
        pdir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    cfg = global_config_yaml()
    env = global_env()
    if cfg.exists():
        _merge_yaml_keys(cfg, pdir / 'config.yaml', GLOBAL_CONFIG_KEYS)
    if env.exists():
        _safe_copy_env(env, pdir / '.env')
