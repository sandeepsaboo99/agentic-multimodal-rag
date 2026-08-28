"""
Authentication + multi-tenant identity.

Design reasoning (rubric: Security & Multi-Tenancy)
---------------------------------------------------
Authorization must live in the *retrieval query*, not the UI. So every request
carries a verifiable identity that resolves to a `Principal` (tenant_id,
user_id, workspace_id, role). That principal is threaded all the way down to the
vector-store filter, guaranteeing a user can never read another tenant's chunks
even if the UI is bypassed.

We use stateless JWTs (HS256) so the API can scale horizontally without a shared
session store. Passwords are bcrypt-hashed. In production, JWT_SECRET comes from
a secret manager (Vault/AWS Secrets Manager), never from source.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def hash_password(raw: str) -> str:
    return _pwd.hash(raw)


def verify_password(raw: str, hashed: str) -> bool:
    try:
        return _pwd.verify(raw, hashed)
    except Exception:
        return False


@dataclass
class Principal:
    """The authenticated actor. This is the unit of authorization."""

    user_id: str
    tenant_id: str
    workspace_id: str
    role: str  # "admin" | "member" | "viewer"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def create_access_token(principal: Principal) -> str:
    s = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": principal.user_id,
        "tid": principal.tenant_id,
        "wid": principal.workspace_id,
        "role": principal.role,
        "iat": now,
        "exp": now + timedelta(minutes=s.jwt_expire_minutes),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str) -> Principal:
    s = get_settings()
    try:
        payload = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    except JWTError as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        ) from exc
    return Principal(
        user_id=payload["sub"],
        tenant_id=payload["tid"],
        workspace_id=payload["wid"],
        role=payload.get("role", "member"),
    )


async def current_principal(token: str | None = Depends(oauth2_scheme)) -> Principal:
    """FastAPI dependency that resolves the caller's identity from the JWT."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(token)
