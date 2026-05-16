"""
Unit tests for api.users — the multi-user data layer.

Covers:
  - schema bootstrap + has_any_user cache
  - create_user with username regex / password length validation
  - verify (correct/wrong password, disabled user)
  - update_user including the "last admin" demotion + disable + delete guards
  - quota CRUD via update_user
  - active session tracking (register/drop/sweep)
  - daily turn counter UPSERT semantics
  - audit log roundtrip

All tests run in an isolated tempdir via HERMES_WEBUI_STATE_DIR so they do
not touch the host's ~/.hermes — same pattern as the existing
tests/test_auth_sessions.py.
"""
import importlib
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
# conftest.py already pinned HERMES_WEBUI_STATE_DIR / HERMES_HOME /
# HERMES_BASE_HOME to a per-repo isolated tempdir. We MUST NOT override
# those here — every test file does, and api.users (which caches STATE_DIR
# at module load) would lock onto whichever file imported it first, leaving
# the others writing DB rows that the cached connection can't read.
_TEST_STATE = Path(os.environ['HERMES_WEBUI_STATE_DIR'])

users = importlib.import_module("api.users")


def _reset_db():
    """Drop & recreate every table — fresh slate per test class."""
    conn = users.db_connection()
    with users.db_lock():
        for tbl in ('usage_audit', 'sessions_active', 'usage_daily',
                    'quotas', 'users'):
            conn.execute(f"DELETE FROM {tbl}")
    users._invalidate_has_any_user_cache()


class TestSchemaBootstrap(unittest.TestCase):
    """ensure_schema is idempotent and has_any_user toggles correctly."""

    def setUp(self):
        users.ensure_schema()
        _reset_db()

    def test_ensure_schema_idempotent(self):
        users.ensure_schema()
        users.ensure_schema()  # second call must not raise

    def test_has_any_user_empty(self):
        self.assertFalse(users.has_any_user())

    def test_has_any_user_after_create(self):
        users.create_user('alpha', 'pw1234', role='admin',
                          profile_name='user_alpha')
        self.assertTrue(users.has_any_user())

    def test_has_any_user_cache_invalidated_on_create(self):
        # First call populates cache (False)
        self.assertFalse(users.has_any_user())
        users.create_user('beta', 'pw1234', role='admin',
                          profile_name='user_beta')
        # Cache must have been busted by create_user
        self.assertTrue(users.has_any_user())


class TestCreateUserValidation(unittest.TestCase):

    def setUp(self):
        users.ensure_schema()
        _reset_db()

    def test_rejects_short_username(self):
        with self.assertRaises(ValueError):
            users.create_user('ab', 'pw1234', role='user',
                              profile_name='user_ab')

    def test_rejects_uppercase_username(self):
        with self.assertRaises(ValueError):
            users.create_user('Alice', 'pw1234', role='user',
                              profile_name='user_alice')

    def test_rejects_username_starting_with_underscore(self):
        with self.assertRaises(ValueError):
            users.create_user('_foo', 'pw1234', role='user',
                              profile_name='user__foo')

    def test_rejects_short_password(self):
        with self.assertRaises(ValueError):
            users.create_user('alice', 'abc', role='user',
                              profile_name='user_alice')

    def test_rejects_invalid_role(self):
        with self.assertRaises(ValueError):
            users.create_user('alice', 'pw1234', role='superuser',
                              profile_name='user_alice')

    def test_default_profile_name_derivation(self):
        u = users.create_user('zoe', 'pw1234', role='user')
        self.assertEqual(u['profile_name'], 'user_zoe')

    def test_unique_username_enforced(self):
        users.create_user('dup', 'pw1234', role='user', profile_name='user_dup')
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            users.create_user('dup', 'pw5678', role='user',
                              profile_name='user_dup2')

    def test_quota_defaults_applied(self):
        u = users.create_user('quota1', 'pw1234', role='user',
                              profile_name='user_quota1')
        q = users.get_quota(u['id'])
        self.assertEqual(q['max_turns_per_day'], users.DEFAULT_MAX_TURNS_PER_DAY)
        self.assertEqual(q['max_concurrent_sessions'], users.DEFAULT_MAX_CONCURRENT_SESSIONS)
        self.assertEqual(q['max_storage_mb'], users.DEFAULT_MAX_STORAGE_MB)

    def test_quota_overrides_applied(self):
        u = users.create_user('quota2', 'pw1234', role='user',
                              profile_name='user_quota2',
                              quotas={'max_turns_per_day': 99,
                                      'max_storage_mb': 1})
        q = users.get_quota(u['id'])
        self.assertEqual(q['max_turns_per_day'], 99)
        self.assertEqual(q['max_storage_mb'], 1)


class TestVerify(unittest.TestCase):

    def setUp(self):
        users.ensure_schema()
        _reset_db()
        self.u = users.create_user('alice', 'correct1', role='user',
                                   profile_name='user_alice')

    def test_correct_password(self):
        out = users.verify('alice', 'correct1')
        self.assertIsNotNone(out)
        self.assertEqual(out['username'], 'alice')

    def test_wrong_password(self):
        self.assertIsNone(users.verify('alice', 'wrongpw'))

    def test_unknown_user(self):
        self.assertIsNone(users.verify('nobody', 'correct1'))

    def test_case_insensitive_username(self):
        out = users.verify('ALICE', 'correct1')
        self.assertIsNotNone(out)

    def test_disabled_user_cannot_verify(self):
        users.update_user(self.u['id'], disabled=True)
        self.assertIsNone(users.verify('alice', 'correct1'))


class TestLastAdminGuards(unittest.TestCase):
    """Cannot demote / disable / delete the last admin."""

    def setUp(self):
        users.ensure_schema()
        _reset_db()
        self.admin = users.create_user('soloadmin', 'pw1234', role='admin',
                                       profile_name='user_soloadmin')

    def test_cannot_demote_last_admin(self):
        with self.assertRaises(ValueError) as ctx:
            users.update_user(self.admin['id'], role='user')
        self.assertIn('last admin', str(ctx.exception))

    def test_cannot_disable_last_admin(self):
        with self.assertRaises(ValueError) as ctx:
            users.update_user(self.admin['id'], disabled=True)
        self.assertIn('last admin', str(ctx.exception))

    def test_cannot_delete_last_admin(self):
        with self.assertRaises(ValueError) as ctx:
            users.delete_user(self.admin['id'])
        self.assertIn('last admin', str(ctx.exception))

    def test_can_demote_when_second_admin_exists(self):
        users.create_user('admin2', 'pw1234', role='admin',
                          profile_name='user_admin2')
        updated = users.update_user(self.admin['id'], role='user')
        self.assertEqual(updated['role'], 'user')


class TestUpdateUser(unittest.TestCase):

    def setUp(self):
        users.ensure_schema()
        _reset_db()
        # Need a co-admin so last-admin guard never fires.
        users.create_user('admin', 'pw1234', role='admin',
                          profile_name='user_admin')
        self.bob = users.create_user('bob', 'oldpw', role='user',
                                     profile_name='user_bob')

    def test_password_update(self):
        users.update_user(self.bob['id'], password='newpw99')
        self.assertIsNone(users.verify('bob', 'oldpw'))
        self.assertIsNotNone(users.verify('bob', 'newpw99'))

    def test_password_too_short_rejected(self):
        with self.assertRaises(ValueError):
            users.update_user(self.bob['id'], password='x')

    def test_role_promotion(self):
        updated = users.update_user(self.bob['id'], role='admin')
        self.assertEqual(updated['role'], 'admin')

    def test_quota_partial_update(self):
        users.update_user(self.bob['id'],
                          quotas={'max_turns_per_day': 7})
        q = users.get_quota(self.bob['id'])
        self.assertEqual(q['max_turns_per_day'], 7)
        # The fields we didn't touch must keep their defaults.
        self.assertEqual(q['max_concurrent_sessions'],
                         users.DEFAULT_MAX_CONCURRENT_SESSIONS)

    def test_profile_name_change(self):
        updated = users.update_user(self.bob['id'], profile_name='user_robert')
        self.assertEqual(updated['profile_name'], 'user_robert')

    def test_profile_name_invalid_rejected(self):
        with self.assertRaises(ValueError):
            users.update_user(self.bob['id'], profile_name='UPPERCASE')

    def test_update_nonexistent(self):
        with self.assertRaises(ValueError):
            users.update_user(99999, role='admin')


class TestActiveSessions(unittest.TestCase):

    def setUp(self):
        users.ensure_schema()
        _reset_db()
        self.u = users.create_user('chuck', 'pw1234', role='user',
                                   profile_name='user_chuck')

    def test_register_and_count(self):
        users.register_active_session(self.u['id'], 's1')
        users.register_active_session(self.u['id'], 's2')
        self.assertEqual(users.count_active_sessions(self.u['id']), 2)

    def test_register_is_idempotent(self):
        users.register_active_session(self.u['id'], 's1')
        users.register_active_session(self.u['id'], 's1')
        self.assertEqual(users.count_active_sessions(self.u['id']), 1)

    def test_drop_active_session(self):
        users.register_active_session(self.u['id'], 's1')
        users.register_active_session(self.u['id'], 's2')
        users.drop_active_session('s1')
        self.assertEqual(users.count_active_sessions(self.u['id']), 1)

    def test_sweep_drops_stale(self):
        users.register_active_session(self.u['id'], 'old')
        # Backdate manually
        conn = users.db_connection()
        with users.db_lock():
            conn.execute(
                "UPDATE sessions_active SET started_at = ? "
                "WHERE user_id = ? AND session_id = ?",
                (int(time.time()) - 30 * 24 * 3600, self.u['id'], 'old'),
            )
        users.register_active_session(self.u['id'], 'fresh')
        dropped = users.sweep_stale_active_sessions(max_age_seconds=24 * 3600)
        self.assertEqual(dropped, 1)
        self.assertEqual(users.count_active_sessions(self.u['id']), 1)


class TestDailyTurns(unittest.TestCase):

    def setUp(self):
        users.ensure_schema()
        _reset_db()
        self.u = users.create_user('counter', 'pw1234', role='user',
                                   profile_name='user_counter')

    def test_upsert_increments(self):
        self.assertEqual(users.get_turns_used_today(self.u['id']), 0)
        n = users.increment_turns_used_today(self.u['id'])
        self.assertEqual(n, 1)
        n = users.increment_turns_used_today(self.u['id'])
        self.assertEqual(n, 2)
        self.assertEqual(users.get_turns_used_today(self.u['id']), 2)


class TestAuditLog(unittest.TestCase):

    def setUp(self):
        users.ensure_schema()
        _reset_db()
        self.u = users.create_user('auditee', 'pw1234', role='user',
                                   profile_name='user_auditee')

    def test_write_and_read_audit(self):
        users.write_audit(self.u['id'], 'login')
        users.write_audit(self.u['id'], 'turn_start',
                          session_id='s1', meta={'turns_today': 1})
        rows = users.recent_audit(self.u['id'], limit=10)
        self.assertEqual(len(rows), 2)
        # Newest first
        self.assertEqual(rows[0]['event'], 'turn_start')
        self.assertEqual(rows[1]['event'], 'login')
        # meta JSON roundtrip
        import json
        self.assertEqual(json.loads(rows[0]['meta'])['turns_today'], 1)

    def test_last_activity_ts(self):
        before = int(time.time())
        users.write_audit(self.u['id'], 'login')
        after = int(time.time())
        ts = users.last_activity_ts(self.u['id'])
        self.assertGreaterEqual(ts, before)
        self.assertLessEqual(ts, after)


class TestProfileReverseLookup(unittest.TestCase):

    def setUp(self):
        users.ensure_schema()
        _reset_db()

    def test_reverse_lookup(self):
        users.create_user('dave', 'pw1234', role='user', profile_name='user_dave')
        u = users.get_user_by_profile_name('user_dave')
        self.assertIsNotNone(u)
        self.assertEqual(u['username'], 'dave')

    def test_reverse_lookup_missing(self):
        self.assertIsNone(users.get_user_by_profile_name('user_ghost'))


class TestSeedDefaultWorkspace(unittest.TestCase):
    """admin_users._seed_default_workspace pre-populates a new user's
    profile with a default workspace pointer so first login lands them
    in their own isolated workspace (not the process-global DEFAULT).
    """

    def setUp(self):
        users.ensure_schema()
        _reset_db()
        # Build the same per-profile dir layout create_profile_api would.
        # Wipe any leftovers from a prior test so workspaces.json starts
        # absent (the helper is intentionally non-destructive when the
        # file already exists; that's covered by a dedicated test).
        import shutil
        from api.profiles import _resolve_profile_home_for_name
        self.profile_home = _resolve_profile_home_for_name('user_seedtest')
        if self.profile_home.exists():
            shutil.rmtree(str(self.profile_home), ignore_errors=True)
        (self.profile_home / 'workspace').mkdir(parents=True, exist_ok=True)

    def test_writes_last_workspace_pointer(self):
        from api.admin_users import _seed_default_workspace
        _seed_default_workspace('user_seedtest')
        lw = self.profile_home / 'webui_state' / 'last_workspace.txt'
        self.assertTrue(lw.exists())
        # Should point at the profile's own workspace dir, resolved.
        self.assertEqual(
            lw.read_text(encoding='utf-8').strip(),
            str((self.profile_home / 'workspace').resolve()),
        )

    def test_writes_workspaces_json_with_one_entry(self):
        from api.admin_users import _seed_default_workspace
        _seed_default_workspace('user_seedtest', display_name='Home (seedtest)')
        wf = self.profile_home / 'webui_state' / 'workspaces.json'
        self.assertTrue(wf.exists())
        import json as _json
        data = _json.loads(wf.read_text(encoding='utf-8'))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['name'], 'Home (seedtest)')
        self.assertEqual(data[0]['path'], str((self.profile_home / 'workspace').resolve()))

    def test_preserves_existing_workspaces_json(self):
        """Re-seeding must not clobber user-added workspace entries."""
        from api.admin_users import _seed_default_workspace
        wf = self.profile_home / 'webui_state'
        wf.mkdir(parents=True, exist_ok=True)
        import json as _json
        (wf / 'workspaces.json').write_text(
            _json.dumps([{'name': 'My Project', 'path': '/some/path'}]),
            encoding='utf-8',
        )
        _seed_default_workspace('user_seedtest')
        data = _json.loads((wf / 'workspaces.json').read_text(encoding='utf-8'))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['name'], 'My Project',
                         "re-seed must not destroy user-added workspaces")

    def test_empty_profile_name_is_noop(self):
        from api.admin_users import _seed_default_workspace
        _seed_default_workspace('')  # must not raise

    def test_seeds_terminal_cwd_in_config_yaml(self):
        """terminal.cwd in config.yaml MUST be set so the agent's runtime
        TERMINAL_CWD env var lands inside the user's profile even when a
        stale session.workspace points at the global default. (#fix for
        '/root/workspace leaks into tool execution' bug.)"""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        from api.admin_users import _seed_default_workspace
        _seed_default_workspace('user_seedtest')
        cfg = self.profile_home / 'config.yaml'
        self.assertTrue(cfg.exists(), "config.yaml should be created")
        data = yaml.safe_load(cfg.read_text(encoding='utf-8'))
        self.assertIsInstance(data, dict)
        self.assertIn('terminal', data)
        self.assertEqual(
            data['terminal'].get('cwd'),
            str((self.profile_home / 'workspace').resolve()),
        )

    def test_terminal_cwd_seed_preserves_other_keys(self):
        """Re-seeding must not blow away existing top-level keys in config.yaml
        (e.g. model, custom_providers that came from global mirror)."""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        cfg = self.profile_home / 'config.yaml'
        cfg.write_text(
            "model:\n  default: gpt-5-mini\ncustom_providers:\n  relay1:\n    url: x\n",
            encoding='utf-8',
        )
        from api.admin_users import _seed_default_workspace
        _seed_default_workspace('user_seedtest')
        data = yaml.safe_load(cfg.read_text(encoding='utf-8'))
        # model + custom_providers must survive
        self.assertEqual(data['model']['default'], 'gpt-5-mini')
        self.assertIn('relay1', data['custom_providers'])
        # terminal.cwd now also set
        self.assertEqual(
            data['terminal']['cwd'],
            str((self.profile_home / 'workspace').resolve()),
        )


if __name__ == '__main__':
    unittest.main()
