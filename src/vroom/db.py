"""App database: users and invite codes, SQLite via SQLAlchemy.

The file lives at data/vroom.sqlite (override with VROOM_DB_PATH); a later
Postgres move is a connection-string change. Model data stays in parquet —
this database holds only web-app state. Schema is league-ready: nothing
here precludes adding user-submitted predictions later.
"""

from __future__ import annotations

import os
import secrets
from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from f1pred.config import DATA_DIR


def _db_url() -> str:
    path = os.environ.get("VROOM_DB_PATH", str(DATA_DIR / "vroom.sqlite"))
    return f"sqlite:///{path}"


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    is_admin: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class InviteCode(Base):
    __tablename__ = "invite_codes"

    code: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    used_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


_engine = None


def get_engine():
    """Process-wide engine, created lazily so env overrides apply in tests."""
    global _engine
    if _engine is None:
        _engine = create_engine(_db_url(), connect_args={"check_same_thread": False})
        Base.metadata.create_all(_engine)
    return _engine


def reset_engine() -> None:
    """Testing hook: forget the engine so the next call re-reads the env."""
    global _engine
    _engine = None


def db_session() -> Session:
    return Session(get_engine())


def new_invite_code() -> str:
    return secrets.token_urlsafe(9)
