"""
Unit tests for api.global_config — admin → global → per-user cascade.

Covers:
  - snapshot_admin_to_global copies admin's config.yaml + .env to global root
  - mirror_global_to_all_users propagates only GLOBAL_CONFIG_KEYS to each user
  - admin profile is correctly skipped during mirror (we'd be writing over the
    source while reading from it otherwise)
  - removed top-level keys cascade as deletions (admin drops custom_providers →
    every user's config.yaml loses it too)
  - seed_user_profile_from_global initialises a brand-new user
  - per-user keys outside the mirror set are preserved
  - cascade_from_admin is serialized under _CASCADE_LOCK so concurrent admin
    writes from many threads don't interleave their file IO

Isolation: per-test tempdir under HERMES_BASE_HOME so the resolver in
api.profiles picks up our test root.
"""
import importlib
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
# Use conftest's shared tempdir — see note in test_multiuser_users.py.
_TEST_STATE = Path(os.environ['HERMES_BASE_HOME'])

users = importlib.import_module("api.users")
gc = importlib.import_module("api.global_config")


def _reset():
    """Wipe DB rows + on-disk profile dirs + global dir before each test."""
    conn = users.db_connection()
    with users.db_lock():
        for tbl in ('usage_audit', 'sessions_active', 'usage_daily',
                    'quotas', 'users'):
            conn.execute(f"DELETE FROM {tbl}")
    users._invalidate_has_any_user_cache()
    profiles_root = _TEST_STATE / 'profiles'
    if profiles_root.exists():
        shutil.rmtree(str(profiles_root), ignore_errors=True)
    profiles_root.mkdir(parents=True, exist_ok=True)
    global_root = _TEST_STATE / 'global'
    if global_root.exists():
        shutil.rmtree(str(global_root), ignore_errors=True)


def _profile_dir(name):
    d = _TEST_STATE / 'profiles' / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _has_yaml():
    try:
        import yaml  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_yaml(), "PyYAML not installed")
class TestSnapshotToGlobal(unittest.TestCase):

    def setUp(self):
        users.ensure_schema()
        _reset()
        self.admin = users.create_user('admin', 'pw1234', role='admin',
                                       profile_name='user_admin')
        admin_dir = _profile_dir('user_admin')
        (admin_dir / 'config.yaml').write_text(
            "model:\n  default: gpt-5-mini\n  api_key: SECRET\n",
            encoding='utf-8',
        )
        (admin_dir / '.env').write_text(
            "OPENAI_API_KEY=sk-test-A\n", encoding='utf-8',
        )

    def test_snapshot_creates_global_files(self):
        gc.snapshot_admin_to_global('user_admin')
        self.assertTrue(gc.global_config_yaml().exists())
        self.assertTrue(gc.global_env().exists())

    def test_snapshot_content_matches(self):
        gc.snapshot_admin_to_global('user_admin')
        yaml_txt = gc.global_config_yaml().read_text(encoding='utf-8')
        self.assertIn('gpt-5-mini', yaml_txt)
        env_txt = gc.global_env().read_text(encoding='utf-8')
        self.assertIn('sk-test-A', env_txt)


@unittest.skipUnless(_has_yaml(), "PyYAML not installed")
class TestMirrorToUsers(unittest.TestCase):

    def setUp(self):
        users.ensure_schema()
        _reset()
        users.create_user('admin', 'pw1234', role='admin',
                          profile_name='user_admin')
        users.create_user('bob', 'pw1234', role='user',
                          profile_name='user_bob')
        users.create_user('carol', 'pw1234', role='user',
                          profile_name='user_carol')
        _profile_dir('user_admin')
        _profile_dir('user_bob')
        _profile_dir('user_carol')

        # Seed global with model + relays.
        gc.ensure_global_root()
        gc.global_config_yaml().write_text(
            "model:\n  default: gpt-5-mini\n"
            "custom_providers:\n  relay1:\n    base_url: https://x.example\n",
            encoding='utf-8',
        )
        gc.global_env().write_text(
            "OPENAI_API_KEY=sk-G\n", encoding='utf-8',
        )

    def test_mirror_writes_to_each_user(self):
        n = gc.mirror_global_to_all_users()
        self.assertEqual(n, 3)  # admin + bob + carol — admin is included unless skip
        for p in ('user_admin', 'user_bob', 'user_carol'):
            cfg = (_TEST_STATE / 'profiles' / p / 'config.yaml')
            env = (_TEST_STATE / 'profiles' / p / '.env')
            self.assertTrue(cfg.exists(), f'{p} config.yaml missing')
            self.assertTrue(env.exists(), f'{p} .env missing')
            self.assertIn('gpt-5-mini', cfg.read_text(encoding='utf-8'))

    def test_mirror_skips_named_profile(self):
        n = gc.mirror_global_to_all_users(skip_profile='user_admin')
        self.assertEqual(n, 2)
        # admin's profile shouldn't be touched
        admin_cfg = _TEST_STATE / 'profiles' / 'user_admin' / 'config.yaml'
        self.assertFalse(admin_cfg.exists(),
                         "skip_profile should leave the admin's own config untouched")

    def test_mirror_preserves_user_only_keys(self):
        bob_cfg = _TEST_STATE / 'profiles' / 'user_bob' / 'config.yaml'
        bob_cfg.write_text(
            "workspace:\n  path: /home/bob/work\n"  # NOT a GLOBAL_CONFIG_KEY
            "model:\n  default: gpt-3.5\n",         # WILL be replaced
            encoding='utf-8',
        )
        gc.mirror_global_to_all_users()
        out = bob_cfg.read_text(encoding='utf-8')
        self.assertIn('gpt-5-mini', out)        # mirrored
        self.assertNotIn('gpt-3.5', out)         # overwritten
        self.assertIn('/home/bob/work', out)     # preserved (workspace is not mirrored)

    def test_remove_cascade(self):
        gc.mirror_global_to_all_users()
        bob_cfg = _TEST_STATE / 'profiles' / 'user_bob' / 'config.yaml'
        self.assertIn('relay1', bob_cfg.read_text(encoding='utf-8'))
        # Admin removes custom_providers from global
        gc.global_config_yaml().write_text(
            "model:\n  default: gpt-5-mini\n", encoding='utf-8',
        )
        gc.mirror_global_to_all_users()
        self.assertNotIn('relay1', bob_cfg.read_text(encoding='utf-8'))


@unittest.skipUnless(_has_yaml(), "PyYAML not installed")
class TestCascadeMutex(unittest.TestCase):
    """cascade_from_admin must serialize so concurrent admin writes don't
    interleave their snapshot+mirror file IO."""

    def setUp(self):
        users.ensure_schema()
        _reset()
        users.create_user('admin', 'pw1234', role='admin',
                          profile_name='user_admin')
        users.create_user('bob', 'pw1234', role='user',
                          profile_name='user_bob')
        _profile_dir('user_admin')
        _profile_dir('user_bob')

    def test_eight_parallel_cascades_complete_cleanly(self):
        errors = []
        results = []
        lock = threading.Lock()

        def hammer(tag):
            try:
                # Each thread sets a unique value in admin's config then casades.
                (_TEST_STATE / 'profiles' / 'user_admin' / 'config.yaml').write_text(
                    f"model:\n  default: gpt-mini-{tag}\n", encoding='utf-8',
                )
                r = gc.cascade_from_admin('user_admin')
                with lock:
                    results.append(r)
            except Exception as e:
                with lock:
                    errors.append((tag, e))

        threads = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [], f"cascade errored: {errors}")
        self.assertEqual(len(results), 8)
        # Bob's config.yaml must be a complete + valid YAML (some thread's
        # full snapshot, not a torn half-write).
        import yaml
        bob_cfg = _TEST_STATE / 'profiles' / 'user_bob' / 'config.yaml'
        parsed = yaml.safe_load(bob_cfg.read_text(encoding='utf-8'))
        self.assertIsInstance(parsed, dict)
        self.assertIn('model', parsed)
        # The value should look like one of the gpt-mini-N strings we wrote.
        default = parsed['model'].get('default', '')
        self.assertRegex(default, r'^gpt-mini-\d+$')


@unittest.skipUnless(_has_yaml(), "PyYAML not installed")
class TestSeedUserProfile(unittest.TestCase):

    def setUp(self):
        users.ensure_schema()
        _reset()
        gc.ensure_global_root()
        gc.global_config_yaml().write_text(
            "model:\n  default: gpt-5-mini\n", encoding='utf-8',
        )
        gc.global_env().write_text(
            "OPENAI_API_KEY=sk-seed\n", encoding='utf-8',
        )

    def test_seed_creates_config_and_env(self):
        _profile_dir('user_new')
        gc.seed_user_profile_from_global('user_new')
        cfg = _TEST_STATE / 'profiles' / 'user_new' / 'config.yaml'
        env = _TEST_STATE / 'profiles' / 'user_new' / '.env'
        self.assertTrue(cfg.exists())
        self.assertTrue(env.exists())
        self.assertIn('gpt-5-mini', cfg.read_text(encoding='utf-8'))
        self.assertIn('sk-seed', env.read_text(encoding='utf-8'))

    def test_seed_with_no_global_is_noop(self):
        # Wipe global
        if gc.global_config_yaml().exists():
            gc.global_config_yaml().unlink()
        if gc.global_env().exists():
            gc.global_env().unlink()
        _profile_dir('user_empty')
        gc.seed_user_profile_from_global('user_empty')  # must not raise
        cfg = _TEST_STATE / 'profiles' / 'user_empty' / 'config.yaml'
        # No global → no config copy
        self.assertFalse(cfg.exists())


if __name__ == '__main__':
    unittest.main()
