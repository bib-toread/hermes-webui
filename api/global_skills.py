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
from pathlib import Path

from api.profiles import _resolve_base_hermes_home

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
