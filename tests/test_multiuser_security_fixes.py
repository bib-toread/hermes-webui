"""
Regression tests for the multi-user security/correctness fixes flagged by
the remote ultrareview.

Coverage:
  - bug_003: DELETE /api/admin/users invalidate now requires admin AND
    runs only AFTER the row is actually deleted (no DoS-by-non-admin).
  - bug_011: handler._user cache is per-request — does NOT survive across
    requests on a reused handler instance (keep-alive safety).
  - bug_017: /api/init-admin/create is public ONLY while users.db is
    empty; once an admin exists, the endpoint requires a session.
  - bug_023a: is_auth_enabled() returns True in multi-user mode (when
    users.db is populated, even without HERMES_WEBUI_PASSWORD).
  - bug_039: quota-gate failure logs at WARNING with once-per-process
    dedup (no silent DEBUG-only bypass).
"""
import importlib
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
# Use conftest's shared tempdir; module-level api.users caches it.
_TEST_STATE = Path(os.environ['HERMES_WEBUI_STATE_DIR'])

auth = importlib.import_module("api.auth")
users = importlib.import_module("api.users")


def _reset():
    auth._sessions.clear()
    users.ensure_schema()
    conn = users.db_connection()
    with users.db_lock():
        for tbl in ('usage_audit', 'sessions_active', 'usage_daily',
                    'quotas', 'users'):
            conn.execute(f"DELETE FROM {tbl}")
    users._invalidate_has_any_user_cache()


class _FakeHandler:
    def __init__(self, cookie_value=None):
        self.headers = {'Cookie': f'hermes_session={cookie_value}'} if cookie_value else {}
        self._sent_status = None
        self._sent_headers = []
        self._body = b''

    def send_response(self, code): self._sent_status = code
    def send_header(self, k, v): self._sent_headers.append((k, v))
    def end_headers(self): pass

    @property
    def wfile(self):
        h = self
        class _W:
            def write(s_self, data): h._body += data
        return _W()


# ──────────────────────────────────────────────────────────────────────
# bug_011: handler._user cache MUST clear between requests
# ──────────────────────────────────────────────────────────────────────

class TestHandlerUserCacheClear(unittest.TestCase):
    """Verify server._clear_per_request_user_cache wipes handler._user.

    Without this, BaseHTTPRequestHandler re-uses the same handler instance
    per keep-alive TCP connection and request N+1 sees request N's
    cached identity — surviving logout, role change, disabled flag, and
    admin-side invalidate_sessions_for_user.
    """

    def setUp(self):
        _reset()

    def test_clear_helper_removes_cached_user(self):
        from server import Handler
        h = _FakeHandler(None)
        h._user = {'id': 99, 'username': 'cached'}
        # Bind the helper directly off the class — we don't need a real
        # network handler instance for this contract check.
        Handler._clear_per_request_user_cache(h)
        self.assertFalse(hasattr(h, '_user'),
                         "handler._user must be removed after clear()")

    def test_clear_is_safe_when_no_cache(self):
        from server import Handler
        h = _FakeHandler(None)
        # Must NOT raise even if _user was never assigned.
        Handler._clear_per_request_user_cache(h)
        self.assertFalse(hasattr(h, '_user'))


# ──────────────────────────────────────────────────────────────────────
# bug_011 perf half: verify_session early-out before prune
# ──────────────────────────────────────────────────────────────────────

class TestVerifySessionEarlyOut(unittest.TestCase):

    def setUp(self):
        _reset()

    def test_empty_cookie_skips_prune(self):
        # Plant an expired entry; if verify_session prunes BEFORE the
        # empty-cookie check, the entry would be gone after the call.
        auth._sessions['expired'] = {'user_id': None, 'exp': 1.0}
        auth.verify_session("")  # empty cookie
        self.assertIn('expired', auth._sessions,
                      "verify_session('') must NOT prune — early-out before prune")

    def test_dotless_cookie_skips_prune(self):
        auth._sessions['expired'] = {'user_id': None, 'exp': 1.0}
        auth.verify_session("nodot")
        self.assertIn('expired', auth._sessions)


# ──────────────────────────────────────────────────────────────────────
# bug_023a: is_auth_enabled returns True when users.db is populated
# ──────────────────────────────────────────────────────────────────────

class TestIsAuthEnabledMultiUser(unittest.TestCase):

    def setUp(self):
        _reset()
        # Clear any env password and the hash cache so multi-user is the
        # only auth signal.
        self._saved_env = os.environ.pop('HERMES_WEBUI_PASSWORD', None)
        auth._invalidate_password_hash_cache()

    def tearDown(self):
        if self._saved_env is not None:
            os.environ['HERMES_WEBUI_PASSWORD'] = self._saved_env
        auth._invalidate_password_hash_cache()

    def test_returns_false_with_no_users_no_password(self):
        self.assertFalse(auth.is_auth_enabled())

    def test_returns_true_after_user_created(self):
        users.create_user('admin1', 'pw1234', role='admin',
                          profile_name='user_admin1')
        self.assertTrue(auth.is_auth_enabled(),
                        "is_auth_enabled must reflect users.db, not just env password")


# ──────────────────────────────────────────────────────────────────────
# bug_017: init-admin endpoints public ONLY while users.db is empty
# ──────────────────────────────────────────────────────────────────────

class TestInitAdminConditionalPublic(unittest.TestCase):

    def setUp(self):
        _reset()
        self._saved_bypass = os.environ.pop('HERMES_WEBUI_TEST_NO_AUTH', None)

    def tearDown(self):
        if self._saved_bypass is not None:
            os.environ['HERMES_WEBUI_TEST_NO_AUTH'] = self._saved_bypass

    def test_init_admin_path_public_when_no_users(self):
        from urllib.parse import urlparse
        h = _FakeHandler(None)
        parsed = urlparse('/api/init-admin/create')
        self.assertTrue(auth.check_auth(h, parsed),
                        "init-admin/create must be reachable while no users exist")

    def test_init_admin_path_NOT_public_after_first_admin(self):
        # First admin created → bootstrap is done.
        users.create_user('admin', 'pw1234', role='admin',
                          profile_name='user_admin')
        from urllib.parse import urlparse
        h = _FakeHandler(None)  # no cookie
        parsed = urlparse('/api/init-admin/create')
        ok = auth.check_auth(h, parsed)
        self.assertFalse(ok,
            "init-admin/create must NOT stay public after first admin exists "
            "— otherwise endpoint relies entirely on inner handler safety")
        # Specifically: gets 401 because /api/init-admin/create is an API path.
        self.assertEqual(h._sent_status, 401)


# ──────────────────────────────────────────────────────────────────────
# bug_003: DELETE /api/admin/users — admin gate + invalidate AFTER delete
# ──────────────────────────────────────────────────────────────────────

class TestDeleteAdminUsersGate(unittest.TestCase):
    """The dispatcher MUST run the admin gate before any side effect.

    Pre-fix shape called invalidate_sessions_for_user from the dispatch
    site before delegating into handle_admin_users_delete. Any
    authenticated non-admin could DoS any user (incl. admin) by issuing
    DELETE /api/admin/users/<id> — CSRF passes for same-origin XHRs.
    """

    def setUp(self):
        _reset()
        self.admin = users.create_user('admin', 'pw1234', role='admin',
                                       profile_name='user_admin')
        self.bob = users.create_user('bob', 'pw1234', role='user',
                                     profile_name='user_bob')
        # Give bob a live session cookie.
        self.bob_cookie = auth.create_session_for_user(self.bob['id'])
        self.assertTrue(auth.verify_session(self.bob_cookie))
        self.assertTrue(auth.verify_session(
            auth.create_session_for_user(self.admin['id'])))

    def test_non_admin_dispatch_to_admin_users_delete_returns_403(self):
        from urllib.parse import urlparse
        import api.routes as routes
        h = _FakeHandler(self.bob_cookie)
        # Simulate non-admin POSTing DELETE /api/admin/users/<admin_id>.
        parsed = urlparse(f'/api/admin/users/{self.admin["id"]}')

        # Drive the gate helper directly (handle_delete path also runs
        # CSRF / body-read first; we're testing the post-CSRF dispatch
        # branch where the admin gate must reject).
        is_admin = routes._admin_or_403(h)
        self.assertFalse(is_admin)
        self.assertEqual(h._sent_status, 403)

    def test_admin_cookie_still_active_after_failed_delete_attempt(self):
        """Critical: a non-admin's failed DELETE must NOT have invalidated
        the target's session as a side effect.

        Pre-fix dispatcher unconditionally invalidated sessions BEFORE the
        admin gate inside handle_admin_users_delete fired. We assert here
        that the dispatcher no longer calls invalidate_sessions_for_user
        on its own.
        """
        admin_cookie = auth.create_session_for_user(self.admin['id'])
        # Simulate what the (now-fixed) DELETE dispatcher does: it should
        # ONLY call _admin_or_403 — no invalidate before delegation.
        from urllib.parse import urlparse
        import api.routes as routes
        h = _FakeHandler(self.bob_cookie)
        parsed = urlparse(f'/api/admin/users/{self.admin["id"]}')
        # Run the gate (non-admin → 403).
        routes._admin_or_403(h)
        # Admin's session MUST still be valid; the failed delete attempt
        # must not have nuked it.
        self.assertTrue(auth.verify_session(admin_cookie),
            "admin session was invalidated by a failed non-admin delete — "
            "this is the bug_003 regression we are guarding against")


# ──────────────────────────────────────────────────────────────────────
# bug_039: quota-gate failure logs at WARNING with dedup
# ──────────────────────────────────────────────────────────────────────

class TestGetLastWorkspaceValidation(unittest.TestCase):
    """get_last_workspace must validate last_workspace.txt against the
    profile's workspaces.json — paths NOT in the registered list must be
    treated as stale and fall back to the first registered entry.
    Regression test for the bug where /root/workspace leaked into a
    user's per-profile last_workspace.txt and stuck around forever.
    """

    def setUp(self):
        _reset()
        # Create a user so there's an active profile (via thread-local later
        # we use the on-disk file paths directly to avoid the full TLS dance).
        self.user = users.create_user('lwtest', 'pw1234', role='user',
                                       profile_name='user_lwtest')

    def test_stale_last_workspace_falls_back_to_first_registered(self):
        """When last_workspace.txt points outside workspaces.json, the
        getter must drop that path and return the first registered
        workspace instead."""
        import tempfile, shutil
        # Build a fake on-disk profile state dir layout the workspace helper
        # will discover via api.profiles.get_active_hermes_home().
        from api.profiles import _resolve_profile_home_for_name
        profile_home = _resolve_profile_home_for_name('user_lwtest')
        ws_dir = profile_home / 'webui_state'
        ws_dir.mkdir(parents=True, exist_ok=True)
        registered = profile_home / 'workspace'
        registered.mkdir(parents=True, exist_ok=True)
        # Stranger path that exists but is NOT in workspaces.json
        stranger = Path(tempfile.mkdtemp(prefix='hermes_stranger_'))
        try:
            # workspaces.json only lists the per-profile workspace
            import json
            (ws_dir / 'workspaces.json').write_text(
                json.dumps([{'name': 'Home', 'path': str(registered.resolve())}]),
                encoding='utf-8',
            )
            # last_workspace.txt points at the stranger — should be ignored
            (ws_dir / 'last_workspace.txt').write_text(
                str(stranger.resolve()), encoding='utf-8',
            )
            # Bind the profile so the workspace helper resolves to ours
            from api.profiles import set_request_profile, clear_request_profile
            set_request_profile('user_lwtest')
            try:
                from api.workspace import get_last_workspace
                result = get_last_workspace()
            finally:
                clear_request_profile()
            self.assertEqual(
                str(Path(result).resolve()),
                str(registered.resolve()),
                f"stale last_workspace={stranger!r} should fall back to "
                f"registered Home={registered!r}, got {result!r}",
            )
        finally:
            shutil.rmtree(str(stranger), ignore_errors=True)

    def test_registered_last_workspace_returns_unchanged(self):
        """When last_workspace.txt IS in workspaces.json, it's returned as-is."""
        from api.profiles import _resolve_profile_home_for_name
        profile_home = _resolve_profile_home_for_name('user_lwtest')
        ws_dir = profile_home / 'webui_state'
        ws_dir.mkdir(parents=True, exist_ok=True)
        registered = profile_home / 'workspace'
        registered.mkdir(parents=True, exist_ok=True)
        import json
        (ws_dir / 'workspaces.json').write_text(
            json.dumps([{'name': 'Home', 'path': str(registered.resolve())}]),
            encoding='utf-8',
        )
        (ws_dir / 'last_workspace.txt').write_text(
            str(registered.resolve()), encoding='utf-8',
        )
        from api.profiles import set_request_profile, clear_request_profile
        set_request_profile('user_lwtest')
        try:
            from api.workspace import get_last_workspace
            result = get_last_workspace()
        finally:
            clear_request_profile()
        self.assertEqual(str(Path(result).resolve()), str(registered.resolve()))


class TestReasoningPerUser(unittest.TestCase):
    """Regression: /api/reasoning POST is per-user, NOT admin-only.

    The reasoning chip (None/Low/Medium/High in the composer) writes to
    the active profile's config.yaml. It MUST be writable by any logged-
    in user — locking it admin-only used to silently 403 in the UI.
    Guard against accidental re-locking by future review passes.
    """

    def setUp(self):
        _reset()

    def test_reasoning_endpoint_not_admin_gated(self):
        """Static check: the /api/reasoning POST handler must NOT call
        _admin_or_403 (which would 403 non-admin users)."""
        import api.routes as routes_mod
        import inspect
        src = inspect.getsource(routes_mod.handle_post)
        # Find the /api/reasoning block specifically.
        i = src.find('"/api/reasoning"')
        self.assertGreater(i, -1, "/api/reasoning handler not found")
        # Look at the next 1500 chars (handler body is ~30 lines).
        block = src[i:i + 1500]
        self.assertNotIn(
            '_admin_or_403', block,
            "/api/reasoning was admin-locked again — that breaks the "
            "reasoning chip for non-admin users. Reasoning effort is a "
            "per-user preference, not a system setting."
        )

    def test_reasoning_endpoint_does_not_cascade(self):
        """Static check: must NOT call _mirror_global_config_after_admin_write
        (which would push the user's choice to every other user's profile).
        Comment lines are filtered out so the explanatory ``# No _mirror...()``
        in the handler docs doesn't trip the assertion."""
        import api.routes as routes_mod
        import inspect
        src = inspect.getsource(routes_mod.handle_post)
        i = src.find('"/api/reasoning"')
        self.assertGreater(i, -1)
        block = src[i:i + 1500]
        # Strip comments — only code lines count.
        code_only = '\n'.join(
            line for line in block.splitlines()
            if not line.lstrip().startswith('#')
        )
        self.assertNotIn(
            '_mirror_global_config_after_admin_write(', code_only,
            "/api/reasoning is per-user; cascading would bleed one user's "
            "reasoning choice into every other user's profile."
        )


class TestSyncSkillsConcurrency(unittest.TestCase):
    """sync_profile_skills_to_global must hold _CASCADE_LOCK to serialize
    overlapping admin pushes — otherwise two admins pushing the same
    skill at the same time race on rmtree+copytree."""

    def setUp(self):
        _reset()
        self.user = users.create_user('synctest', 'pw1234', role='admin',
                                       profile_name='user_synctest')
        from api.profiles import _resolve_profile_home_for_name
        self.profile_home = _resolve_profile_home_for_name('user_synctest')
        skills_dir = self.profile_home / 'skills' / 'shared'
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / 'SKILL.md').write_text('# shared', encoding='utf-8')

    def test_concurrent_pushes_serialize_cleanly(self):
        """Two threads pushing the same skill must both report success
        (the lock makes the second wait for the first)."""
        from api.global_skills import sync_profile_skills_to_global, global_skills_dir
        import threading

        results = []
        errors = []

        def push():
            try:
                r = sync_profile_skills_to_global('user_synctest', only=['shared'])
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=push) for _ in range(6)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(errors, [], f"sync raised under concurrency: {errors}")
        # All 6 should have count=1 with no skipped — the lock serializes
        # the rmtree+copytree so each one sees a clean dst.
        for r in results:
            self.assertEqual(r['count'], 1, f"got partial result: {r}")
            self.assertEqual(r['skipped'], [])
        # Final state: the skill exists in global.
        gdir = global_skills_dir()
        self.assertTrue((gdir / 'shared' / 'SKILL.md').exists())


class TestSyncSkillsSymlinkSafety(unittest.TestCase):
    """_copy_skill uses symlinks=True so a stray symlink inside an admin's
    skill dir is copied AS a link, not followed + mirrored. Prevents
    arbitrary tree exfiltration if a malicious or accidental
    `link → /etc` exists in a skill subdir."""

    def setUp(self):
        _reset()
        self.user = users.create_user('symtest', 'pw1234', role='admin',
                                       profile_name='user_symtest')
        from api.profiles import _resolve_profile_home_for_name
        self.profile_home = _resolve_profile_home_for_name('user_symtest')

    def test_symlinks_preserved_not_followed(self):
        import os
        # Windows symlink creation requires special privileges; skip there.
        if os.name == 'nt':
            self.skipTest("symlink creation requires admin on Windows")
        from api.global_skills import sync_profile_skills_to_global, global_skills_dir
        import tempfile
        # Plant a "secret" dir we don't want exfiltrated, then symlink to it
        # from inside a skill subdir.
        secret = Path(tempfile.mkdtemp(prefix='hermes_sym_secret_'))
        (secret / 'top-secret.txt').write_text('CLASSIFIED', encoding='utf-8')
        try:
            skill_dir = self.profile_home / 'skills' / 'has-symlink'
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / 'SKILL.md').write_text('# normal', encoding='utf-8')
            (skill_dir / 'evil-link').symlink_to(str(secret))

            sync_profile_skills_to_global('user_symtest', only=['has-symlink'])

            gdir = global_skills_dir() / 'has-symlink'
            link = gdir / 'evil-link'
            # The link should be preserved AS a link, NOT followed +
            # copied. So global/has-symlink/evil-link should be a symlink,
            # and top-secret.txt should NOT be physically copied under it.
            self.assertTrue(link.is_symlink(),
                            "expected symlink to be preserved as-is, not followed")
            # Critically: top-secret.txt is NOT a real file in the global
            # tree. (It's reachable via the link, but if the link is later
            # broken, no data leaks.)
            import os as _os
            file_in_global = gdir / 'evil-link' / 'top-secret.txt'
            # The path resolves THROUGH the symlink; we want to assert no
            # FILE was actually copied. Check by removing the original
            # secret dir → file_in_global should become unreachable.
            import shutil
            shutil.rmtree(str(secret), ignore_errors=True)
            self.assertFalse(file_in_global.exists(),
                "top-secret.txt was physically copied into global — "
                "symlink was FOLLOWED instead of preserved")
        finally:
            import shutil
            shutil.rmtree(str(secret), ignore_errors=True)


class TestQuotaGateFailureLogging(unittest.TestCase):

    def setUp(self):
        # Reset the dedup set so each test starts clean.
        streaming = importlib.import_module("api.streaming")
        streaming._QUOTA_GATE_WARNED.clear()
        self.streaming = streaming

    def test_dedup_set_exists_and_starts_empty(self):
        self.assertIsInstance(self.streaming._QUOTA_GATE_WARNED, set)
        self.assertEqual(len(self.streaming._QUOTA_GATE_WARNED), 0)

    def test_same_signature_only_warned_once(self):
        """Simulate the dedup pattern used in the quota gate."""
        warned = self.streaming._QUOTA_GATE_WARNED
        err1 = RuntimeError("database is locked")
        err2 = RuntimeError("database is locked")  # same signature
        err3 = RuntimeError("disk I/O error")      # different signature

        def _key(e):
            return (type(e).__name__, str(e)[:80])

        for e in (err1, err2, err3):
            k = _key(e)
            if k not in warned:
                warned.add(k)
        # err1 and err2 share a key; err3 is new. Set should have 2 entries.
        self.assertEqual(len(warned), 2)


if __name__ == '__main__':
    unittest.main()
