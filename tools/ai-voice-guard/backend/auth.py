"""Session-based login. Credentials come from environment variables only —
loaded from a local .env file that is gitignored and never committed. There
is no hardcoded fallback credential anywhere in this module; if the
required variables aren't set, the app refuses to start (see main.py).
"""

import hashlib
import os
import secrets

from fastapi import Cookie, HTTPException

SESSION_COOKIE_NAME = "voiceguard_session"
_active_sessions: set[str] = set()


def credentials_configured() -> bool:
    return bool(os.environ.get("APP_USERNAME")) and bool(
        os.environ.get("APP_PASSWORD") or os.environ.get("APP_PASSWORD_HASH")
    )


def _expected_password_hash() -> str:
    if "APP_PASSWORD_HASH" in os.environ:
        return os.environ["APP_PASSWORD_HASH"]
    return hashlib.sha256(os.environ["APP_PASSWORD"].encode()).hexdigest()


def verify_credentials(username: str, password: str) -> bool:
    expected_username = os.environ["APP_USERNAME"]
    expected_hash = _expected_password_hash()
    given_hash = hashlib.sha256(password.encode()).hexdigest()
    # constant-time comparisons so a timing side-channel can't leak how much
    # of the guess was correct
    username_ok = secrets.compare_digest(username, expected_username)
    password_ok = secrets.compare_digest(given_hash, expected_hash)
    return username_ok and password_ok


def create_session() -> str:
    token = secrets.token_urlsafe(32)
    _active_sessions.add(token)
    return token


def destroy_session(token: str) -> None:
    _active_sessions.discard(token)


def require_session(voiceguard_session: str | None = Cookie(default=None)) -> str:
    if not voiceguard_session or voiceguard_session not in _active_sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return voiceguard_session
