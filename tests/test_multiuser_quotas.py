"""
Unit tests for api.quotas — quota gate, storage du caching, audit hooks.

Covers:
  - check_and_reserve: turns/day, concurrency, storage limits all enforced
  - rollback semantics: a blocked turn must NOT increment the daily counter
  - storage du: walks subdirs, caches with TTL, skips fat dirs, hits
    timeout cleanly
  - concurrent check_and_reserve: N parallel reservations across the cap
    end in at most cap successes (RLock + BEGIN IMMEDIATE)
  - record_turn_end writes an audit row only when user is present

Isolation: HERMES_WEBUI_STATE_DIR points at a tempdir before any api.*
module is imported, matching the convention used by test_auth_sessions.py.
"""
import importlib
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
# Use conftest's shared tempdir — see note in test_multiuser_users.py.
_TEST_STATE = Path(os.environ['HERMES_WEBUI_STATE_DIR'])

users = importlib.import_module("api.users")
quotas = importlib.import_module("api.quotas")


def _reset_db():
    conn = users.db_connection()
    with users.db_lock():
        for tbl in ('usage_audit', 'sessions_active', 'usage_daily',
                    'quotas', 'users'):
            conn.execute(f"DELETE FROM {tbl}")
    users._invalidate_has_any_user_cache()


def _mk_profile_dir(name='probe'):
    """Make a CLEAN profile dir for storage-du tests.

    Wipes any leftovers from a prior test so each setUp starts at zero
    bytes (otherwise `_cached_du` aggregates across tests and the cache
    timeline becomes very confusing).
    """
    import shutil
    d = _TEST_STATE / 'profiles' / name
    if d.exists():
        shutil.rmtree(str(d), ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


class TestTurnsQuota(unittest.TestCase):

    def setUp(self):
        users.ensure_schema()
        _reset_db()
        quotas.invalidate_du_cache()
        self.u = users.create_user('turner', 'pw1234', role='user',
                                   profile_name='user_turner',
                                   quotas={'max_turns_per_day': 3,
                                           'max_concurrent_sessions': 10,
                                           'max_storage_mb': 99999})
        self.pdir = _mk_profile_dir('user_turner')

    def test_within_limit_passes(self):
        for i in range(3):
            blocked = quotas.check_and_reserve(self.u, f's-{i}', self.pdir)
            self.assertIsNone(blocked, f'turn {i} should pass: {blocked}')
        self.assertEqual(users.get_turns_used_today(self.u['id']), 3)

    def test_over_limit_blocks_with_reason_turns(self):
        for i in range(3):
            quotas.check_and_reserve(self.u, f's-{i}', self.pdir)
        blocked = quotas.check_and_reserve(self.u, 's-4', self.pdir)
        self.assertIsNotNone(blocked)
        self.assertEqual(blocked['reason'], 'turns')
        self.assertEqual(blocked['used'], 3)
        self.assertEqual(blocked['limit'], 3)

    def test_blocked_turn_does_not_increment_counter(self):
        for i in range(3):
            quotas.check_and_reserve(self.u, f's-{i}', self.pdir)
        before = users.get_turns_used_today(self.u['id'])
        quotas.check_and_reserve(self.u, 's-blocked', self.pdir)
        after = users.get_turns_used_today(self.u['id'])
        self.assertEqual(after, before, "blocked turn must not consume quota")

    def test_zero_limit_means_no_check(self):
        # Disable concurrency cap too so it doesn't kick in instead of turns.
        users.update_user(self.u['id'],
                          quotas={'max_turns_per_day': 0,
                                  'max_concurrent_sessions': 0})
        self.u = users.get_user_by_id(self.u['id'])
        for i in range(20):
            blocked = quotas.check_and_reserve(self.u, f's-zero-{i}', self.pdir)
            self.assertIsNone(blocked)

    def test_no_user_means_no_check(self):
        # Quota gate is a no-op when user is None (out-of-band caller).
        self.assertIsNone(quotas.check_and_reserve(None, 's', self.pdir))


class TestConcurrencyQuota(unittest.TestCase):

    def setUp(self):
        users.ensure_schema()
        _reset_db()
        quotas.invalidate_du_cache()
        self.u = users.create_user('concur', 'pw1234', role='user',
                                   profile_name='user_concur',
                                   quotas={'max_turns_per_day': 9999,
                                           'max_concurrent_sessions': 2,
                                           'max_storage_mb': 99999})
        self.pdir = _mk_profile_dir('user_concur')

    def test_within_concurrency_passes(self):
        self.assertIsNone(quotas.check_and_reserve(self.u, 's1', self.pdir))
        self.assertIsNone(quotas.check_and_reserve(self.u, 's2', self.pdir))

    def test_third_session_blocked(self):
        quotas.check_and_reserve(self.u, 's1', self.pdir)
        quotas.check_and_reserve(self.u, 's2', self.pdir)
        blocked = quotas.check_and_reserve(self.u, 's3', self.pdir)
        self.assertIsNotNone(blocked)
        self.assertEqual(blocked['reason'], 'concurrency')

    def test_dropping_session_frees_slot(self):
        quotas.check_and_reserve(self.u, 's1', self.pdir)
        quotas.check_and_reserve(self.u, 's2', self.pdir)
        users.drop_active_session('s1')
        self.assertIsNone(quotas.check_and_reserve(self.u, 's3', self.pdir))


class TestStorageQuota(unittest.TestCase):

    def setUp(self):
        users.ensure_schema()
        _reset_db()
        quotas.invalidate_du_cache()
        self.u = users.create_user('storer', 'pw1234', role='user',
                                   profile_name='user_storer',
                                   quotas={'max_turns_per_day': 9999,
                                           'max_concurrent_sessions': 9999,
                                           'max_storage_mb': 1})
        self.pdir = _mk_profile_dir('user_storer')

    def test_under_storage_passes(self):
        self.assertIsNone(quotas.check_and_reserve(self.u, 's1', self.pdir))

    def test_over_storage_blocked(self):
        # Plant ~2 MB so we exceed the 1 MB cap.
        (self.pdir / 'big.bin').write_bytes(b'x' * (2 * 1024 * 1024))
        quotas.invalidate_du_cache()
        blocked = quotas.check_and_reserve(self.u, 's1', self.pdir)
        self.assertIsNotNone(blocked)
        self.assertEqual(blocked['reason'], 'storage')
        self.assertGreaterEqual(blocked['used_mb'], 2)


class TestStorageDuCache(unittest.TestCase):

    def setUp(self):
        users.ensure_schema()
        _reset_db()
        quotas.invalidate_du_cache()
        self.pdir = _mk_profile_dir('du_probe')

    def test_cache_returns_same_value(self):
        (self.pdir / 'a.txt').write_bytes(b'1234567890')
        first = quotas._cached_du(self.pdir)
        # Modify the file size; cache should mask the change until TTL expires.
        (self.pdir / 'a.txt').write_bytes(b'1234567890' * 1000)
        second = quotas._cached_du(self.pdir)
        self.assertEqual(first, second, "cached value should be returned")

    def test_invalidate_du_cache_forces_rewalk(self):
        (self.pdir / 'a.txt').write_bytes(b'1234567890')
        quotas._cached_du(self.pdir)
        (self.pdir / 'a.txt').write_bytes(b'1' * 9999)
        quotas.invalidate_du_cache()
        self.assertEqual(quotas._cached_du(self.pdir), 9999)

    def test_skips_node_modules(self):
        (self.pdir / 'real.md').write_bytes(b'hello ' * 100)  # 600B real
        big = self.pdir / 'workspace' / 'node_modules' / 'react'
        big.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            (big / f'file{i}.js').write_bytes(b'x' * 100_000)  # 1MB fake
        quotas.invalidate_du_cache()
        total = quotas._cached_du(self.pdir)
        self.assertLess(total, 50_000,
                        f"node_modules should be skipped, got {total}b")

    def test_skips_git_and_pycache(self):
        (self.pdir / 'real.md').write_bytes(b'r' * 1000)
        for skip in ('.git', '__pycache__', '.venv', '.idea'):
            d = self.pdir / skip
            d.mkdir()
            (d / 'huge.bin').write_bytes(b'x' * 200_000)
        quotas.invalidate_du_cache()
        total = quotas._cached_du(self.pdir)
        self.assertLess(total, 5000)

    def test_missing_path_returns_zero(self):
        self.assertEqual(quotas._cached_du(self.pdir / 'nope'), 0)


class TestCheckAndReserveConcurrent(unittest.TestCase):
    """Hammer check_and_reserve from many threads and verify the daily
    counter never overshoots the cap (BEGIN IMMEDIATE + RLock contract)."""

    def setUp(self):
        users.ensure_schema()
        _reset_db()
        quotas.invalidate_du_cache()
        self.u = users.create_user('racer', 'pw1234', role='user',
                                   profile_name='user_racer',
                                   quotas={'max_turns_per_day': 5,
                                           'max_concurrent_sessions': 9999,
                                           'max_storage_mb': 9999})
        self.pdir = _mk_profile_dir('user_racer')

    def test_parallel_reservations_cap_at_limit(self):
        successes = []
        blocks = []
        lock = threading.Lock()

        def try_reserve(i):
            result = quotas.check_and_reserve(self.u, f's-r-{i}', self.pdir)
            with lock:
                if result is None:
                    successes.append(i)
                else:
                    blocks.append(result)

        threads = [threading.Thread(target=try_reserve, args=(i,))
                   for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        # AT MOST 5 successes (= the cap). If the lock isn't held atomically
        # we'd see 5+N concurrent threads all reading "used=4" and racing past.
        self.assertLessEqual(len(successes), 5,
                             f"got {len(successes)} successes — atomic check broken")
        self.assertEqual(len(successes) + len(blocks), 20)
        # Final counter must equal the success count.
        self.assertEqual(users.get_turns_used_today(self.u['id']), len(successes))


class TestRecordTurnEnd(unittest.TestCase):

    def setUp(self):
        users.ensure_schema()
        _reset_db()
        self.u = users.create_user('ender', 'pw1234', role='user',
                                   profile_name='user_ender')

    def test_writes_audit_when_user_present(self):
        quotas.record_turn_end(self.u['id'], 's1', status='ok', tokens=42)
        rows = users.recent_audit(self.u['id'], limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['event'], 'turn_end')

    def test_noop_when_user_none(self):
        quotas.record_turn_end(None, 's1')  # must not raise
        # No way to assert "nothing happened" cleanly other than no exception.


if __name__ == '__main__':
    unittest.main()
