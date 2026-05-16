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


def list_profile_skills(profile_name: str) -> list[dict]:
    """List all skills in *profile_name*'s skills/ dir.

    Returns ``[{label, name, category}]`` for each discovered SKILL.md.
    *label* is the rel path used to identify the skill in sync calls
    (``"<name>"`` or ``"<category>/<name>"``); the UI uses *category*
    + *name* for display, the API uses *label* for selection.
    """
    src = _resolve_profile_home_for_name(profile_name) / 'skills'
    if not src.exists() or not src.is_dir():
        return []
    out: list[dict] = []
    for entry in sorted(src.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / 'SKILL.md').is_file():
            out.append({'label': entry.name, 'name': entry.name, 'category': None})
            continue
        for sub in sorted(entry.iterdir()):
            if not sub.is_dir() or not (sub / 'SKILL.md').is_file():
                continue
            out.append({
                'label': f"{entry.name}/{sub.name}",
                'name': sub.name,
                'category': entry.name,
            })
    return out


def sync_profile_skills_to_global(profile_name: str,
                                   only: list | None = None) -> dict:
    """Copy skills from *profile_name*'s skills/ dir into the global
    skills root, overwriting same-named entries.

    Walks the standard layout (``<skills>/<name>/SKILL.md`` and
    ``<skills>/<category>/<name>/SKILL.md``), copies each skill's entire
    subdirectory (so linked files / sub-resources travel with SKILL.md),
    and reports what was synced.

    *only* — when None (default), every skill in the profile is synced.
    When a list of labels (``"name"`` or ``"category/name"``) is given,
    ONLY those skills are pushed; unknown labels land in ``skipped`` with
    ``reason: 'not found in source profile'``.

    Merge semantics: skills that exist in the global dir but NOT in the
    pushed set are left alone — this is "push these", not "make global
    match exactly". Admin can `rm -rf` global skills they no longer want
    from the file system.

    Returns ``{synced: [<category/name or name>, ...], skipped: [...],
    count: N}``. Never raises; per-skill failures are caught + reported.
    """
    synced: list[str] = []
    skipped: list[dict] = []

    only_set: set | None = None
    if only is not None:
        if not isinstance(only, list):
            return {'synced': [], 'skipped': [], 'count': 0,
                    'error': 'only must be a list of skill labels'}
        only_set = set(s for s in only if isinstance(s, str) and s)
        if not only_set:
            return {'synced': [], 'skipped': [], 'count': 0}

    src = _resolve_profile_home_for_name(profile_name) / 'skills'
    if not src.exists() or not src.is_dir():
        # Source profile has no skills/ dir. If specific labels were
        # requested, report them all as not-found so the caller knows
        # the push silently dropped them.
        if only_set is not None:
            for missing in sorted(only_set):
                skipped.append({'name': missing, 'error': 'not found in source profile'})
        return {'synced': [], 'skipped': skipped, 'count': 0}
    dst_root = ensure_global_skills_dir()

    seen_labels: set = set()

    for entry in sorted(src.iterdir()):
        if not entry.is_dir():
            continue
        # Direct skill at root: <skills>/<name>/SKILL.md
        if (entry / 'SKILL.md').is_file():
            label = entry.name
            seen_labels.add(label)
            if only_set is not None and label not in only_set:
                continue
            _copy_skill(entry, dst_root / entry.name, label, synced, skipped)
            continue
        # Category dir: <skills>/<category>/<name>/SKILL.md
        for sub in sorted(entry.iterdir()):
            if not sub.is_dir() or not (sub / 'SKILL.md').is_file():
                continue
            label = f"{entry.name}/{sub.name}"
            seen_labels.add(label)
            if only_set is not None and label not in only_set:
                continue
            _copy_skill(sub, dst_root / entry.name / sub.name, label, synced, skipped)

    # Report any requested labels that didn't exist in the source profile.
    if only_set is not None:
        for missing in sorted(only_set - seen_labels):
            skipped.append({'name': missing, 'error': 'not found in source profile'})

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
