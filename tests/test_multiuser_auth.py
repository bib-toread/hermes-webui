"""
Unit tests for the multi-user pieces of api.auth.

Covers:
  - create_session_for_user binds a user_id into the session payload
  - resolve_session_user_id returns the bound id (or None for legacy/unbound)
  - current_user(handler) looks up the user via cookie and caches on handler
  - require_admin sends 403 for non-admin / no-user
  - invalidate_sessions_for_user drops every cookie belonging to one user
  - legacy bare-float session entries are no longer accepted (multi-user
    refactor)
  - init-admin gate: check_auth redirects to /init-admin when users table
    is empty
"""
import importlib
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

# Use conftest's shared tempdir — see note in test_multiuser_users.py.
_TEST_STATE = Path(os.environ['HERMES_WEBUI_STATE_DIR'])

sys.path.insert(0, str(Path(__file__).parent.parent))

auth = importlib.import_module("api.auth")
users = importlib.import_module("api.users")


def _reset():
    auth._sessions.clear()
    users.ensure_schema()  # idempotent; tables must exist before DELETE
    conn = users.db_connection()
    with users.db_lock():
        for tbl in ('usage_audit', 'sessions_active', 'usage_daily',
                    'quotas', 'users'):
            conn.execute(f"DELETE FROM {tbl}")
    users._invalidate_has_any_user_cache()


# conftest.py sets HERMES_WEBUI_TEST_NO_AUTH=1 so the legacy live-server
# test fleet keeps working. The init-admin tests below need the gate to
# actually fire, so they unset/restore the var on a per-test basis instead
# of at module load (which would strip it before the server subprocess
# fixture inherits the environment).
class _NoBypassMixin:
    def _disable_bypass(self):
        self._saved_bypass = os.environ.pop('HERMES_WEBUI_TEST_NO_AUTH', None)

    def _restore_bypass(self):
        if getattr(self, '_saved_bypass', None) is not None:
            os.environ['HERMES_WEBUI_TEST_NO_AUTH'] = self._saved_bypass


class _FakeHandler:
    """Minimal handler stub for check_auth / require_admin / current_user."""

    def __init__(self, cookie_value=None):
        self.headers = {'Cookie': f'hermes_session={cookie_value}'} if cookie_value else {}
        self._sent_status = None
        self._sent_headers = []
        self._body = b''
        self._ended = False

    def send_response(self, code):
        self._sent_status = code

    def send_header(self, k, v):
        self._sent_headers.append((k, v))

    def end_headers(self):
        self._ended = True

    @property
    def wfile(self):
        h = self
        class _W:
            def write(s_self, data):
                h._body += data
        return _W()


class TestCreateSessionForUser(_NoBypassMixin, unittest.TestCase):

    def setUp(self):
        self._disable_bypass()
        _reset()
        self.u = users.create_user('alpha', 'pw1234', role='user',
                                   profile_name='user_alpha')

    def tearDown(self):
        self._restore_bypass()

    def test_cookie_binds_user_id(self):
        cookie = auth.create_session_for_user(self.u['id'])
        self.assertTrue(auth.verify_session(cookie))
        uid = auth.resolve_session_user_id(cookie)
        self.assertEqual(uid, self.u['id'])

    def test_legacy_create_session_has_no_user_id(self):
        cookie = auth.create_session()  # legacy entry point — None user_id
        self.assertTrue(auth.verify_session(cookie))
        self.assertIsNone(auth.resolve_session_user_id(cookie))

    def test_invalid_cookie_returns_none(self):
        self.assertIsNone(auth.resolve_session_user_id("garbage.sig"))
        self.assertIsNone(auth.resolve_session_user_id(""))
        self.assertIsNone(auth.resolve_session_user_id(None))


class TestCurrentUser(_NoBypassMixin, unittest.TestCase):

    def setUp(self):
        self._disable_bypass()
        _reset()
        self.u = users.create_user('beta', 'pw1234', role='user',
                                   profile_name='user_beta')
        self.cookie = auth.create_session_for_user(self.u['id'])

    def tearDown(self):
        self._restore_bypass()

    def test_returns_user_for_valid_cookie(self):
        h = _FakeHandler(self.cookie)
        u = auth.current_user(h)
        self.assertIsNotNone(u)
        self.assertEqual(u['username'], 'beta')

    def test_caches_on_handler(self):
        h = _FakeHandler(self.cookie)
        first = auth.current_user(h)
        # Mutate the cache so we can detect re-fetching.
        h._user = {'username': 'cached-sentinel'}
        second = auth.current_user(h)
        self.assertEqual(second['username'], 'cached-sentinel',
                         "current_user must reuse handler._user cache")

    def test_no_cookie_returns_none(self):
        h = _FakeHandler(None)
        self.assertIsNone(auth.current_user(h))

    def test_disabled_user_returns_none(self):
        users.update_user(self.u['id'], disabled=True)
        h = _FakeHandler(self.cookie)
        self.assertIsNone(auth.current_user(h))


class TestRequireAdmin(_NoBypassMixin, unittest.TestCase):

    def setUp(self):
        self._disable_bypass()
        _reset()
        self.admin = users.create_user('adm', 'pw1234', role='admin',
                                       profile_name='user_adm')
        self.user = users.create_user('reg', 'pw1234', role='user',
                                      profile_name='user_reg')

    def tearDown(self):
        self._restore_bypass()

    def test_admin_passes(self):
        h = _FakeHandler(auth.create_session_for_user(self.admin['id']))
        self.assertTrue(auth.require_admin(h))
        self.assertIsNone(h._sent_status, "must not send a response for admin")

    def test_regular_user_gets_403(self):
        h = _FakeHandler(auth.create_session_for_user(self.user['id']))
        self.assertFalse(auth.require_admin(h))
        self.assertEqual(h._sent_status, 403)

    def test_no_cookie_gets_403(self):
        h = _FakeHandler(None)
        self.assertFalse(auth.require_admin(h))
        self.assertEqual(h._sent_status, 403)


class TestInvalidateSessionsForUser(_NoBypassMixin, unittest.TestCase):

    def setUp(self):
        self._disable_bypass()
        _reset()
        self.u = users.create_user('victim', 'pw1234', role='user',
                                   profile_name='user_victim')

    def tearDown(self):
        self._restore_bypass()

    def test_drops_all_user_sessions(self):
        c1 = auth.create_session_for_user(self.u['id'])
        c2 = auth.create_session_for_user(self.u['id'])
        self.assertTrue(auth.verify_session(c1))
        self.assertTrue(auth.verify_session(c2))
        n = auth.invalidate_sessions_for_user(self.u['id'])
        self.assertEqual(n, 2)
        self.assertFalse(auth.verify_session(c1))
        self.assertFalse(auth.verify_session(c2))

    def test_leaves_other_users_alone(self):
        other = users.create_user('survivor', 'pw1234', role='user',
                                  profile_name='user_survivor')
        my_cookie = auth.create_session_for_user(self.u['id'])
        their_cookie = auth.create_session_for_user(other['id'])
        n = auth.invalidate_sessions_for_user(self.u['id'])
        self.assertEqual(n, 1)
        self.assertFalse(auth.verify_session(my_cookie))
        self.assertTrue(auth.verify_session(their_cookie))


class TestLegacyBareFloatRejected(_NoBypassMixin, unittest.TestCase):
    """Multi-user refactor: bare-float session entries (legacy single-user
    payload shape) must be treated as invalid by both prune and verify."""

    def setUp(self):
        self._disable_bypass()
        _reset()

    def tearDown(self):
        self._restore_bypass()

    def test_bare_float_treated_as_expired(self):
        # Even far-future bare-float entries get pruned because the payload
        # isn't a dict — see _prune_expired_sessions.
        auth._sessions['legacy'] = time.time() + 86400  # not a dict
        auth._prune_expired_sessions()
        self.assertNotIn('legacy', auth._sessions)


class TestInitAdminGate(_NoBypassMixin, unittest.TestCase):
    """check_auth must redirect to /init-admin when no users exist."""

    def setUp(self):
        self._disable_bypass()
        _reset()

    def tearDown(self):
        self._restore_bypass()

    def test_no_users_redirects_to_init_admin(self):
        from urllib.parse import urlparse
        h = _FakeHandler(None)
        parsed = urlparse('/api/sessions')  # arbitrary non-public path
        ok = auth.check_auth(h, parsed)
        self.assertFalse(ok)
        self.assertEqual(h._sent_status, 401)
        # The body must point at the init-admin path.
        self.assertIn(b'init-admin', h._body)

    def test_no_users_non_api_redirects_302(self):
        from urllib.parse import urlparse
        h = _FakeHandler(None)
        parsed = urlparse('/')
        ok = auth.check_auth(h, parsed)
        self.assertFalse(ok)
        self.assertEqual(h._sent_status, 302)
        # Location header points to /init-admin
        locs = [v for k, v in h._sent_headers if k == 'Location']
        self.assertEqual(locs, ['/init-admin'])

    def test_init_admin_path_itself_is_public(self):
        from urllib.parse import urlparse
        h = _FakeHandler(None)
        parsed = urlparse('/init-admin')
        ok = auth.check_auth(h, parsed)
        self.assertTrue(ok, "/init-admin must be reachable when no users exist")

    def test_with_users_proceeds_to_normal_auth_api(self):
        """No cookie + users exist + API path → 401 JSON (not init-admin)."""
        users.create_user('adm', 'pw1234', role='admin', profile_name='user_adm')
        from urllib.parse import urlparse
        h = _FakeHandler(None)
        parsed = urlparse('/api/sessions')
        self.assertFalse(auth.check_auth(h, parsed))
        self.assertEqual(h._sent_status, 401)
        # Body must NOT carry the init-admin pointer (we're past init).
        self.assertNotIn(b'init-admin', h._body)

    def test_with_users_proceeds_to_normal_auth_page(self):
        """No cookie + users exist + page path → 302 /login (not /init-admin)."""
        users.create_user('adm', 'pw1234', role='admin', profile_name='user_adm')
        from urllib.parse import urlparse
        h = _FakeHandler(None)
        parsed = urlparse('/some/page')
        self.assertFalse(auth.check_auth(h, parsed))
        self.assertEqual(h._sent_status, 302)
        locs = [v for k, v in h._sent_headers if k == 'Location']
        self.assertTrue(any('login' in l for l in locs),
                        f"expected login redirect, got {locs}")
        self.assertFalse(any('init-admin' in l for l in locs))


if __name__ == '__main__':
    unittest.main()
