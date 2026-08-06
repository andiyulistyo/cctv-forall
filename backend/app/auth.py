"""JWT authentication for a single admin user."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .database import SessionLocal
from .models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)


def hash_password(password: str) -> str:
    # bcrypt operates on <=72 bytes; truncate defensively.
    pw = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8")[:72], hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def ensure_admin_user() -> None:
    """Create/update the single admin from settings on startup."""
    db: Session = SessionLocal()
    try:
        user = db.scalar(select(User).where(User.username == settings.admin_username))
        if user is None:
            user = User(
                username=settings.admin_username,
                password_hash=hash_password(settings.admin_password),
            )
            db.add(user)
        else:
            # Keep the stored hash in sync if the env password changed.
            if not verify_password(settings.admin_password, user.password_hash):
                user.password_hash = hash_password(settings.admin_password)
        db.commit()
    finally:
        db.close()


def authenticate(username: str, password: str) -> User | None:
    db: Session = SessionLocal()
    try:
        user = db.scalar(select(User).where(User.username == username))
        if user and verify_password(password, user.password_hash):
            return user
        return None
    finally:
        db.close()


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def validate_token(token: str | None) -> str:
    """Validate a raw token string (e.g. from a query param) or raise 401."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
    )
    if not token:
        raise credentials_exc
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        subject = payload.get("sub")
        if subject is None:
            raise credentials_exc
        return subject
    except jwt.PyJWTError:
        raise credentials_exc


def get_current_user(token: str | None = Depends(oauth2_scheme)) -> str:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exc
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        subject = payload.get("sub")
        if subject is None:
            raise credentials_exc
        return subject
    except jwt.PyJWTError:
        raise credentials_exc
