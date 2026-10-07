"""Authentication: bcrypt password hashing and signed session cookies.

Sessions are stateless: an itsdangerous-signed cookie carrying the user id,
valid SESSION_MAX_AGE_S seconds. The signing key comes from
VROOM_SECRET_KEY; without it (dev only) a random per-process key is used,
so sessions reset on restart — set it in deploy/.env for real use.
"""

from __future__ import annotations

import os
import secrets

import bcrypt
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from vroom.db import User, db_session

SESSION_COOKIE = "vroom_session"
SESSION_MAX_AGE_S = 30 * 24 * 3600

_serializer: URLSafeTimedSerializer | None = None


def _get_serializer() -> URLSafeTimedSerializer:
    global _serializer
    if _serializer is None:
        key = os.environ.get("VROOM_SECRET_KEY")
        if not key:
            key = secrets.token_hex(32)  # dev fallback: sessions die on restart
        _serializer = URLSafeTimedSerializer(key, salt="vroom-session")
    return _serializer


def reset_serializer() -> None:
    """Testing hook: forget the serializer so the next call re-reads the env."""
    global _serializer
    _serializer = None


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


def make_session_token(user_id: int) -> str:
    return _get_serializer().dumps({"uid": user_id})


def user_from_token(token: str | None) -> User | None:
    if not token:
        return None
    try:
        data = _get_serializer().loads(token, max_age=SESSION_MAX_AGE_S)
    except (BadSignature, SignatureExpired):
        return None
    with db_session() as session:
        return session.get(User, data.get("uid"))
