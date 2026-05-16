"""
Hermes Web UI -- Optional password authentication.
Off by default. Enable by setting HERMES_WEBUI_PASSWORD env var
or configuring a password in the Settings panel.
"""
import hashlib
import hmac
import http.cookies
import json
import logging
import os
import secrets
import tempfile
import threading
import time

from api.config import STATE_DIR, load_settings

logger = logging.getLogger(__name__)


# Default session TTL — 30 days. Kept as a module-level constant for backwards
# compatibility with downstream code and regression tests that import it.
# At runtime, prefer ``_resolve_session_ttl()`` which honours the env var and
# settings.json overrides; this constant is the floor / fallback.
SESSION_TTL = 86400 * 30  # 30 days


def _resolve_session_ttl() -> int:
    """Resolve session TTL from env > settings > default.

    Priority mirrors get_password_hash(): HERMES_WEBUI_SESSION_TTL env var
    first, then settings.json, falling back to ``SESSION_TTL`` (30 days).
    Clamped to [60s, 1 year] to prevent runaway cookies or self-lockout.
    """
    env_v = os.getenv('HERMES_WEBUI_SESSION_TTL', '').strip()
    if env_v.isdigit():
        val = int(env_v)
        if 60 <= val <= 86400 * 365:
            return val
    s = load_settings()
    v = s.get('session_ttl_seconds')
    if isinstance(v, int) and 60 <= v <= 86400 * 365:
        return v
    return SESSION_TTL


# ── Public paths (no auth required) ─────────────────────────────────────────
PUBLIC_PATHS = frozenset({
    '/login', '/health', '/favicon.ico', '/sw.js',
    '/api/auth/login', '/api/auth/status',
    '/manifest.json', '/manifest.webmanifest',
    '/session/manifest.json', '/session/manifest.webmanifest',
})

# Init-admin paths — reachable WITHOUT a session ONLY while users.db is
# empty. check_auth grants the exemption conditionally. Moved out of the
# static PUBLIC_PATHS set so the endpoints don't stay world-callable
# forever (#review-fix bug_017: e.g. /api/init-admin/create previously
# remained unauthenticated even after the first admin was created;
# protection relied entirely on the inner handler's has_any_user check).
_INIT_ADMIN_PATHS = frozenset({
    '/init-admin',
    '/api/init-admin/status',
    '/api/init-admin/create',
})

COOKIE_NAME = 'hermes_session'

_SESSIONS_FILE = STATE_DIR / '.sessions.json'


def _load_sessions() -> dict[str, dict]:
    """Load persisted sessions from STATE_DIR, pruning expired entries.

    Session payload shape is ``{'user_id': int|None, 'exp': float}``.
    Legacy entries (bare float expiry) are discarded on load — users will
    need to re-authenticate, which is fine because the multi-user model
    requires a user_id binding that legacy sessions don't carry.

    Returns an empty dict on any read or parse error so startup is never
    blocked by a corrupt or missing sessions file.
    """
    try:
        if _SESSIONS_FILE.exists():
            data = json.loads(_SESSIONS_FILE.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                raise ValueError('malformed sessions file — expected dict')
            now = time.time()
            out: dict[str, dict] = {}
            _legacy_dropped = 0
            for token, payload in data.items():
                if not isinstance(token, str):
                    continue
                if isinstance(payload, dict):
                    exp = payload.get('exp')
                    if isinstance(exp, (int, float)) and exp > now:
                        out[token] = {
                            'user_id': payload.get('user_id'),
                            'exp': float(exp),
                        }
                elif isinstance(payload, (int, float)) and payload > now:
                    # Legacy single-user session: bare float expiry, no user_id.
                    # Dropped because multi-user routes require a user binding.
                    _legacy_dropped += 1
            if _legacy_dropped:
                # Loud one-time warning so operators upgrading from single-user
                # mode understand why everyone got logged out.
                logger.warning(
                    "[webui] multi-user upgrade: dropped %d pre-existing single-user "
                    "session(s) from %s. Affected users must log in again via /login "
                    "with the username/password the admin assigned them.",
                    _legacy_dropped, _SESSIONS_FILE,
                )
            return out
    except Exception as e:
        logger.debug("Failed to load sessions file, starting fresh: %s", e)
    return {}


def _save_sessions(sessions: dict[str, dict]) -> None:
    """Atomically persist sessions to STATE_DIR/.sessions.json (0600).

    Uses a temp file + os.replace() so a crash mid-write never leaves a
    truncated file.  Mirrors the same pattern as .signing_key persistence.
    """
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=STATE_DIR, suffix='.sessions.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(sessions, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, _SESSIONS_FILE)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("Failed to persist sessions: %s", e)


# Active sessions: token -> expiry timestamp (persisted across restarts via STATE_DIR)
_sessions = _load_sessions()

# ── Login rate limiter ──────────────────────────────────────────────────────
_LOGIN_ATTEMPTS_FILE = STATE_DIR / '.login_attempts.json'
_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_WINDOW = 60  # seconds


def _load_login_attempts() -> dict[str, list[float]]:
    """Load persisted login attempts from STATE_DIR, pruning expired entries."""
    try:
        if _LOGIN_ATTEMPTS_FILE.exists():
            data = json.loads(_LOGIN_ATTEMPTS_FILE.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                raise ValueError('malformed login-attempts file — expected dict')
            now = time.time()
            attempts: dict[str, list[float]] = {}
            for ip, raw_times in data.items():
                if not isinstance(ip, str) or not isinstance(raw_times, list):
                    continue
                fresh = [
                    float(t)
                    for t in raw_times
                    if isinstance(t, (int, float)) and now - float(t) < _LOGIN_WINDOW
                ]
                if fresh:
                    attempts[ip] = fresh
            return attempts
    except Exception as e:
        logger.debug("Failed to load login attempts file, starting fresh: %s", e)
    return {}


def _save_login_attempts(attempts: dict[str, list[float]]) -> None:
    """Atomically persist login attempts to STATE_DIR/.login_attempts.json (0600)."""
    try:
        _LOGIN_ATTEMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=_LOGIN_ATTEMPTS_FILE.parent, suffix='.login_attempts.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(attempts, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, _LOGIN_ATTEMPTS_FILE)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("Failed to persist login attempts: %s", e)


_login_attempts = _load_login_attempts()  # ip -> [timestamp, ...]
_LOGIN_ATTEMPTS_LOCK = threading.Lock()


def _check_login_rate(ip: str) -> bool:
    """Return True if the IP is allowed to attempt login (thread-safe)."""
    with _LOGIN_ATTEMPTS_LOCK:
        now = time.time()
        attempts = _login_attempts.get(ip, [])
        # Prune old attempts
        attempts = [t for t in attempts if now - t < _LOGIN_WINDOW]
        if attempts:
            _login_attempts[ip] = attempts
        else:
            _login_attempts.pop(ip, None)
        _save_login_attempts(_login_attempts)
        return len(attempts) < _LOGIN_MAX_ATTEMPTS


def _record_login_attempt(ip: str) -> None:
    """Record a login attempt for rate limiting (thread-safe)."""
    with _LOGIN_ATTEMPTS_LOCK:
        now = time.time()
        attempts = _login_attempts.get(ip, [])
        attempts.append(now)
        _login_attempts[ip] = attempts
        _save_login_attempts(_login_attempts)


def _load_key(filename: str) -> bytes:
    """Load a 32-byte key from STATE_DIR, generating and persisting one if missing."""
    key_file = STATE_DIR / filename
    try:
        if key_file.exists():
            raw = key_file.read_bytes()
            if len(raw) >= 32:
                return raw[:32]
    except OSError:
        logger.debug("Failed to read key %s", filename)
    key = secrets.token_bytes(32)
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        key_file.write_bytes(key)
        key_file.chmod(0o600)
    except OSError:
        logger.debug("Failed to persist key %s", filename)
    return key


_PBKDF2_KEY_CACHE: bytes | None = None
_SIGNING_KEY_CACHE: bytes | None = None


def _pbkdf2_key() -> bytes:
    global _PBKDF2_KEY_CACHE
    if _PBKDF2_KEY_CACHE is None:
        _PBKDF2_KEY_CACHE = _load_key('.pbkdf2_key')
    return _PBKDF2_KEY_CACHE


def _signing_key() -> bytes:
    global _SIGNING_KEY_CACHE
    if _SIGNING_KEY_CACHE is None:
        _SIGNING_KEY_CACHE = _load_key('.signing_key')
    return _SIGNING_KEY_CACHE


def _hash_password(password, *, salt: bytes | None = None) -> str:
    """PBKDF2-SHA256 with 600k iterations (OWASP recommendation).
    Salt is the persisted PBKDF2 key, which is secret and unique per
    installation. This keeps the stored hash format a plain hex string
    (no format change to settings.json) while replacing the predictable
    STATE_DIR-derived salt from the original implementation.

    The *salt* parameter exists solely to support transparent migration
    of password hashes that were computed with a different key (e.g. the
    old `.signing_key`). Normal callers should never pass it.
    """
    if salt is None:
        salt = _pbkdf2_key()
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 600_000)
    return dk.hex()


_AUTH_HASH_LOCK = threading.Lock()
_AUTH_HASH_COMPUTED: bool = False
_AUTH_HASH_CACHE: str | None = None


def _invalidate_password_hash_cache() -> None:
    """Invalidate the in-process password hash cache so the next call to
    get_password_hash() re-reads from settings.json or the env var."""
    global _AUTH_HASH_COMPUTED, _AUTH_HASH_CACHE
    with _AUTH_HASH_LOCK:
        _AUTH_HASH_COMPUTED = False
        _AUTH_HASH_CACHE = None


def get_password_hash() -> str | None:
    """Return the active password hash, or None if auth is disabled.
    Priority: env var > settings.json.

    The hash is computed once and cached for the lifetime of the process.
    PBKDF2-600k takes ~1 s and is called on nearly every HTTP request via
    check_auth → is_auth_enabled, so caching avoids wasting a full second
    of CPU per request after the first one.

    Thread-safe: double-checked locking ensures that under a burst of
    concurrent requests only one thread computes PBKDF2, while the fast
    path (after initialisation) requires zero locks.
    """
    global _AUTH_HASH_COMPUTED, _AUTH_HASH_CACHE

    # Fast path — no lock needed once cache is populated.
    if _AUTH_HASH_COMPUTED:
        return _AUTH_HASH_CACHE

    with _AUTH_HASH_LOCK:
        # Re-check inside lock — another thread may have populated while
        # we were waiting to acquire.
        if _AUTH_HASH_COMPUTED:
            return _AUTH_HASH_CACHE

        env_pw = os.getenv('HERMES_WEBUI_PASSWORD', '').strip()
        if env_pw:
            result = _hash_password(env_pw)
        else:
            result = load_settings().get('password_hash') or None

        _AUTH_HASH_CACHE = result
        _AUTH_HASH_COMPUTED = True
        return result


def is_auth_enabled() -> bool:
    """True when ANY auth mechanism is active.

    Returns True if EITHER:
      - a legacy single-password is configured (env var or settings.json), OR
      - the multi-user system has at least one user in users.db.

    Without the multi-user branch, every clean multi-user deployment would
    fail this check (no env password, no settings password) and downstream
    consumers would emit nonsense — e.g. server.py warning operators to
    set HERMES_WEBUI_PASSWORD on a fully-authenticated 0.0.0.0 bind, and
    /api/onboarding/* refusing authenticated admins from non-LAN IPs.
    (#review-fix bug_023a)
    """
    if get_password_hash() is not None:
        return True
    try:
        from api import users as _users_mod
        return _users_mod.has_any_user()
    except Exception:
        return False


def verify_password(plain: str) -> bool:
    """Verify a plaintext password against the stored hash.

    Supports transparent migration of password hashes that were computed
    with the old `.signing_key` salt.  When the two keys differ and the
    legacy-salted hash matches, the password is transparently re-hashed
    with the current `.pbkdf2_key` and persisted to settings.json.
    """
    expected = get_password_hash()
    if not expected:
        return False
    # Fast path: current PBKDF2 key
    if hmac.compare_digest(_hash_password(plain), expected):
        return True
    # Migration: some hashes were computed with `.signing_key` before the
    # PBKDF2 key was separated.  Try the legacy salt; if it matches,
    # transparently upgrade so the next login uses the fast path.
    legacy_salt = _signing_key()
    current_salt = _pbkdf2_key()
    if legacy_salt != current_salt:
        if hmac.compare_digest(_hash_password(plain, salt=legacy_salt), expected):
            from api.config import save_settings

            save_settings({'_set_password': plain})
            # Password re-hashed and persisted to disk using the current salt.
            # Cache invalidation is handled by fix 2/3 (#2192) which adds the
            # _invalidate_password_hash_cache() call inside save_settings().
            return True
    return False


def create_session() -> str:
    """Create a legacy (no user binding) auth session. Kept for backward
    compatibility callers; multi-user flows should use create_session_for_user."""
    return create_session_for_user(None)


def create_session_for_user(user_id: int | None) -> str:
    """Create a session bound to *user_id*. Returns signed cookie value."""
    token = secrets.token_hex(32)
    _sessions[token] = {
        'user_id': int(user_id) if user_id is not None else None,
        'exp': time.time() + _resolve_session_ttl(),
    }
    _save_sessions(_sessions)
    sig = hmac.new(_signing_key(), token.encode(), hashlib.sha256).hexdigest()
    return f"{token}.{sig}"


def _prune_expired_sessions():
    """Remove all expired session entries to prevent unbounded memory growth."""
    now = time.time()
    expired = [t for t, p in _sessions.items() if not isinstance(p, dict) or now > p.get('exp', 0)]
    if expired:
        for token in expired:
            _sessions.pop(token, None)
        _save_sessions(_sessions)


def _validate_cookie_signature(cookie_value: str) -> str | None:
    """Return the unsigned session token if the cookie's HMAC matches, else None."""
    if not cookie_value or '.' not in cookie_value:
        return None
    token, sig = cookie_value.rsplit('.', 1)
    full_sig = hmac.new(_signing_key(), token.encode(), hashlib.sha256).hexdigest()
    valid = hmac.compare_digest(sig, full_sig) or (
        len(sig) == 32 and hmac.compare_digest(sig, full_sig[:32])
    )
    return token if valid else None


def verify_session(cookie_value: str) -> bool:
    """Verify a signed session cookie. Returns True if valid and not expired.

    Early-out on empty / dotless cookies BEFORE the prune call so requests
    that carry a malformed Cookie header don't trigger an O(N) prune on
    every hit. (#review-fix bug_011 perf half.)
    """
    if not cookie_value or '.' not in cookie_value:
        return False
    _prune_expired_sessions()
    token = _validate_cookie_signature(cookie_value)
    if not token:
        return False
    payload = _sessions.get(token)
    if not isinstance(payload, dict):
        return False
    exp = payload.get('exp', 0)
    if not exp or time.time() > exp:
        _sessions.pop(token, None)
        return False
    return True


def resolve_session_user_id(cookie_value: str) -> int | None:
    """Return the user_id bound to *cookie_value*, or None if invalid/expired."""
    if not verify_session(cookie_value):
        return None
    token = _validate_cookie_signature(cookie_value)
    payload = _sessions.get(token) if token else None
    if not isinstance(payload, dict):
        return None
    uid = payload.get('user_id')
    return int(uid) if uid is not None else None


def current_user(handler) -> dict | None:
    """Return the user dict bound to the request's auth cookie, or None.

    Result is cached on the handler (``handler._user``) so repeated calls
    inside a single request don't re-query the DB.
    """
    cached = getattr(handler, '_user', _UNSET)
    if cached is not _UNSET:
        return cached
    user: dict | None = None
    try:
        cookie_val = parse_cookie(handler)
        if cookie_val:
            uid = resolve_session_user_id(cookie_val)
            if uid is not None:
                from api import users as users_mod
                user = users_mod.get_user_by_id(uid)
                if user and user.get('disabled'):
                    user = None
    except Exception:
        logger.debug("current_user lookup failed", exc_info=True)
    handler._user = user
    return user


def require_admin(handler) -> bool:
    """Return True if request is from an admin; else send 403 and return False."""
    user = current_user(handler)
    if user and user.get('role') == 'admin':
        return True
    handler.send_response(403)
    handler.send_header('Content-Type', 'application/json')
    handler.end_headers()
    handler.wfile.write(b'{"error":"admin only"}')
    return False


_UNSET = object()


# Re-export a name admin_users.py imports for cookie/security helpers.
def _security_headers_safe(handler) -> None:
    """Thin wrapper so api/admin_users.py doesn't need to import api/helpers."""
    try:
        from api.helpers import _security_headers
        _security_headers(handler)
    except Exception:
        logger.debug("_security_headers unavailable", exc_info=True)


def invalidate_session(cookie_value) -> None:
    """Remove a session token."""
    if cookie_value and '.' in cookie_value:
        token = cookie_value.rsplit('.', 1)[0]
        if token in _sessions:
            _sessions.pop(token, None)
            _save_sessions(_sessions)


def invalidate_sessions_for_user(user_id: int) -> int:
    """Drop every active session belonging to *user_id*. Returns count removed.

    Called after admin disables/deletes a user so existing cookies stop working.
    """
    if user_id is None:
        return 0
    dropped = [t for t, p in _sessions.items() if isinstance(p, dict) and p.get('user_id') == int(user_id)]
    for t in dropped:
        _sessions.pop(t, None)
    if dropped:
        _save_sessions(_sessions)
    return len(dropped)


def parse_cookie(handler) -> str | None:
    """Extract the auth cookie from the request headers."""
    cookie_header = handler.headers.get('Cookie', '')
    if not cookie_header:
        return None
    cookie = http.cookies.SimpleCookie()
    try:
        cookie.load(cookie_header)
    except http.cookies.CookieError:
        return None
    morsel = cookie.get(COOKIE_NAME)
    return morsel.value if morsel else None


def check_auth(handler, parsed) -> bool:
    """Check if request is authorized. Returns True if OK.
    If not authorized, sends 401 (API) or 302 redirect (page) and returns False.

    Multi-user gate (added):
      - If the users table is empty AND this is not an init-admin / static path,
        redirect to /init-admin so the operator can create the first admin.
      - Otherwise require a valid session cookie that resolves to an enabled
        user. Legacy single-password mode (HERMES_WEBUI_PASSWORD) is honoured
        only when there are no users yet and no init-admin work in progress.

    Test escape hatch: ``HERMES_WEBUI_TEST_NO_AUTH=1`` bypasses both the
    init-admin gate and the cookie check, returning True for every path.
    Used by the live-server test fleet (tests/conftest.py) so existing tests
    that don't care about multi-user keep working without per-test setup.
    Never set this in production — it disables ALL auth.
    """
    if os.environ.get('HERMES_WEBUI_TEST_NO_AUTH', '').strip() in ('1', 'true', 'yes'):
        return True

    # Public + static paths always allowed.
    if parsed.path in PUBLIC_PATHS or parsed.path.startswith('/static/') or parsed.path.startswith('/session/static/'):
        return True

    # Init-admin gate: if no users yet, route everything else to /init-admin.
    try:
        from api import users as users_mod
        any_user = users_mod.has_any_user()
    except Exception:
        logger.debug("has_any_user lookup failed", exc_info=True)
        any_user = True  # fail-closed: don't expose init-admin on DB errors

    # Init-admin paths are public ONLY while bootstrap is incomplete. Once
    # any_user is True, these endpoints require a session like any other —
    # otherwise /api/init-admin/create stays unauthenticated forever and
    # protection collapses to whatever the inner handler bothers to check.
    if parsed.path in _INIT_ADMIN_PATHS and not any_user:
        return True
    if not any_user:
        if parsed.path.startswith('/api/'):
            handler.send_response(401)
            handler.send_header('Content-Type', 'application/json')
            handler.end_headers()
            handler.wfile.write(b'{"error":"setup required","next":"/init-admin"}')
        else:
            handler.send_response(302)
            handler.send_header('Location', '/init-admin')
            handler.end_headers()
        return False

    # Normal multi-user auth path.
    cookie_val = parse_cookie(handler)
    if cookie_val:
        uid = resolve_session_user_id(cookie_val)
        if uid is not None:
            from api import users as users_mod
            user = users_mod.get_user_by_id(uid)
            if user and not user.get('disabled'):
                return True
    # Not authorized
    if parsed.path.startswith('/api/'):
        handler.send_response(401)
        handler.send_header('Content-Type', 'application/json')
        handler.end_headers()
        handler.wfile.write(b'{"error":"Authentication required"}')
    else:
        handler.send_response(302)
        # Pass the original path as ?next= so login.js redirects back after auth.
        # SECURITY/CORRECTNESS: the inner `?` and `&` MUST be percent-encoded
        # when stuffed into the outer `?next=` parameter, otherwise:
        #   (a) multi-param query strings get truncated at the first inner `&`
        #       (e.g. `/api/sessions?limit=50&offset=0` would round-trip as
        #       just `/api/sessions?limit=50` after the browser parses the
        #       outer URL — `offset=0` becomes a separate top-level query
        #       parameter that the login page ignores).
        #   (b) attacker-controlled paths could inject a second `next=`
        #       parameter; per RFC 3986 the duplicate behaviour is undefined
        #       and parsers diverge (Python's parse_qs returns last-match,
        #       URLSearchParams returns first-match), opening a query-pollution
        #       footgun even though _safeNextPath() rejects most malicious
        #       shapes downstream.
        # Encoding the entire `path?query` blob with quote(safe='/') turns
        # `?` → `%3F` and `&` → `%26`, so the outer parameter holds exactly
        # one path-with-query string and `searchParams.get('next')` returns
        # the full original URL (the browser auto-decodes once).
        # (Opus pre-release advisor finding for v0.50.258.)
        import urllib.parse as _urlparse
        _path_with_query = parsed.path or '/'
        if parsed.query:
            _path_with_query += '?' + parsed.query
        # safe='/' keeps path separators readable; everything else (including
        # `?`, `&`, `=`) gets percent-encoded.
        _next = _urlparse.quote(_path_with_query, safe='/')
        handler.send_header('Location', 'login?next=' + _next)
        handler.end_headers()
    return False


def _is_secure_context(handler=None) -> bool:
    """Return True if cookies should carry the Secure flag.

    Behaviour is overridable via HERMES_WEBUI_SECURE env var for
    reverse-proxy setups where TLS terminates at a frontend proxy
    (nginx, Cloudflare, etc.) and Python only sees plain HTTP.
    1/true/yes → force Secure on; 0/false/no → force Secure off.
    When unset, fall back to heuristics: direct TLS socket (getpeercert)
    or X-Forwarded-Proto header from the request.

    .. warning::
       The ``X-Forwarded-Proto`` header is only trustworthy when a
       reverse proxy (nginx, Cloudflare, etc.) is deployed in front
       of the application.  Without a proxy, any client can forge the
       header and cause the Secure flag to be set on plain HTTP.
    """
    env = os.getenv('HERMES_WEBUI_SECURE', '').strip().lower()
    if env in ('1', 'true', 'yes'):
        return True
    if env in ('0', 'false', 'no'):
        return False
    if handler is not None:
        if getattr(handler.request, 'getpeercert', None) is not None:
            return True
        if handler.headers.get('X-Forwarded-Proto', '') == 'https':
            return True
    return False


def set_auth_cookie(handler, cookie_value) -> None:
    """Set the auth cookie on the response."""
    cookie = http.cookies.SimpleCookie()
    cookie[COOKIE_NAME] = cookie_value
    cookie[COOKIE_NAME]['httponly'] = True
    cookie[COOKIE_NAME]['samesite'] = 'Lax'
    cookie[COOKIE_NAME]['path'] = '/'
    cookie[COOKIE_NAME]['max-age'] = str(_resolve_session_ttl())
    if _is_secure_context(handler):
        cookie[COOKIE_NAME]['secure'] = True
    handler.send_header('Set-Cookie', cookie[COOKIE_NAME].OutputString())


def clear_auth_cookie(handler) -> None:
    """Clear the auth cookie on the response."""
    cookie = http.cookies.SimpleCookie()
    cookie[COOKIE_NAME] = ''
    cookie[COOKIE_NAME]['httponly'] = True
    cookie[COOKIE_NAME]['path'] = '/'
    cookie[COOKIE_NAME]['max-age'] = '0'
    handler.send_header('Set-Cookie', cookie[COOKIE_NAME].OutputString())
