"""
Hermes Web UI -- Global (admin-curated) skills directory.

Per-user profiles keep their own ``<profile>/skills/`` dir (the existing
``_active_skills_dir()`` in api/routes.py). Global skills live at
``~/.hermes/global/skills/`` and are surfaced to every user as read-only
unless the requester is an admin.

This module is intentionally tiny — just the path + scope helpers. The
list/save/delete endpoint handlers stay in api/routes.py (where the rest of
the skills logic lives).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from api.profiles import _resolve_base_hermes_home, _resolve_profile_home_for_name

logger = logging.getLogger(__name__)


def global_skills_dir() -> Path:
    """Return ``~/.hermes/global/skills/`` (always recomputed; cheap)."""
    return Path(_resolve_base_hermes_home()) / "global" / "skills"


def ensure_global_skills_dir() -> Path:
    """Create the global skills directory if missing. Returns the path."""
    d = global_skills_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("failed to mkdir %s", d, exc_info=True)
    return d


def is_global_skill_path(p: Path) -> bool:
    """True if *p* resolves inside the global skills directory."""
    try:
        p.resolve().relative_to(global_skills_dir().resolve())
        return True
    except (OSError, ValueError):
        return False


def scope_of_path(p: Path) -> str:
    """Return ``'global'`` or ``'user'`` based on where *p* lives."""
    return 'global' if is_global_skill_path(p) else 'user'


def sync_profile_skills_to_global(profile_name: str) -> dict:
    """Copy every skill from *profile_name*'s skills/ dir into the global
    skills root, overwriting same-named entries.

    Walks the standard layout (``<skills>/<name>/SKILL.md`` and
    ``<skills>/<category>/<name>/SKILL.md``), copies each skill's entire
    subdirectory (so linked files / sub-resources travel with SKILL.md),
    and reports what was synced.

    Merge semantics: skills that exist in the global dir but NOT in the
    source profile are left alone — this is "push my stuff" not "make
    global match exactly". Admin can `rm -rf` global skills they no
    longer want from the file system.

    Returns ``{synced: [<category/name or name>, ...], skipped: [...],
    count: N}``. Never raises; per-skill failures are caught + reported.
    """
    src = _resolve_profile_home_for_name(profile_name) / 'skills'
    if not src.exists() or not src.is_dir():
        return {'synced': [], 'skipped': [], 'count': 0}
    dst_root = ensure_global_skills_dir()
    synced: list[str] = []
    skipped: list[dict] = []

    for entry in sorted(src.iterdir()):
        if not entry.is_dir():
            continue
        # Direct skill at root: <skills>/<name>/SKILL.md
        if (entry / 'SKILL.md').is_file():
            _copy_skill(entry, dst_root / entry.name, entry.name, synced, skipped)
            continue
        # Category dir: <skills>/<category>/<name>/SKILL.md
        for sub in sorted(entry.iterdir()):
            if not sub.is_dir() or not (sub / 'SKILL.md').is_file():
                continue
            rel = f"{entry.name}/{sub.name}"
            _copy_skill(sub, dst_root / entry.name / sub.name, rel, synced, skipped)

    return {'synced': synced, 'skipped': skipped, 'count': len(synced)}


def _copy_skill(src_dir: Path, dst_dir: Path, label: str,
                synced: list, skipped: list) -> None:
    """Copy one skill's full subdirectory (SKILL.md + any linked files).

    Always overwrites — that's the intent of "push to global". On any IO
    failure, records the skill in ``skipped`` with the error and moves on.
    """
    try:
        if dst_dir.exists():
            shutil.rmtree(str(dst_dir))
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(src_dir), str(dst_dir))
        synced.append(label)
    except Exception as exc:
        logger.exception("failed to sync skill %s → global", label)
        skipped.append({'name': label, 'error': str(exc)})
