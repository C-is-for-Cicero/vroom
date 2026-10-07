"""Management CLI for the web app.

    python -m vroom.manage create-admin <username>   # prompts for password
    python -m vroom.manage invite [-n COUNT]         # mint invite codes
    python -m vroom.manage list-users
"""

from __future__ import annotations

import argparse
import getpass

from sqlalchemy import select

from vroom.auth import hash_password
from vroom.db import InviteCode, User, db_session, new_invite_code


def create_admin(username: str) -> int:
    password = getpass.getpass("password: ")
    if len(password) < 8:
        print("password must be at least 8 characters")
        return 1
    if password != getpass.getpass("repeat: "):
        print("passwords do not match")
        return 1
    with db_session() as session:
        if session.scalar(select(User).where(User.username == username)):
            print(f"user {username!r} already exists")
            return 1
        session.add(User(username=username, password_hash=hash_password(password), is_admin=True))
        session.commit()
    print(f"admin {username!r} created")
    return 0


def invite(count: int) -> int:
    with db_session() as session:
        codes = [new_invite_code() for _ in range(count)]
        session.add_all([InviteCode(code=c) for c in codes])
        session.commit()
    for c in codes:
        print(c)
    return 0


def list_users() -> int:
    with db_session() as session:
        for user in session.scalars(select(User).order_by(User.id)):
            role = "admin" if user.is_admin else "viewer"
            print(f"{user.id:4d}  {user.username:24s} {role}  joined {user.created_at:%Y-%m-%d}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vroom.manage", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_admin = sub.add_parser("create-admin")
    p_admin.add_argument("username")
    p_invite = sub.add_parser("invite")
    p_invite.add_argument("-n", type=int, default=1, dest="count")
    sub.add_parser("list-users")
    args = parser.parse_args(argv)

    if args.command == "create-admin":
        return create_admin(args.username)
    if args.command == "invite":
        return invite(args.count)
    return list_users()


if __name__ == "__main__":
    raise SystemExit(main())
