"""Auth endpoints: register (creates tenant+admin) and login (issues JWT)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from pydantic import BaseModel

from app.core.security import Principal, create_access_token, hash_password, verify_password
from app.models.db import Tenant, User, session_scope
from app.models.schemas import RegisterRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest) -> TokenResponse:
    with session_scope() as db:
        if db.query(User).filter(User.email == req.email).first():
            raise HTTPException(400, "Email already registered")
        tenant = Tenant(name=req.tenant_name)
        db.add(tenant)
        db.flush()
        user = User(
            tenant_id=tenant.id, workspace_id="default", email=req.email,
            password_hash=hash_password(req.password), role=req.role,
        )
        db.add(user)
        db.flush()
        principal = Principal(user.id, tenant.id, "default", user.role)
        token = create_access_token(principal)
        return TokenResponse(access_token=token, tenant_id=tenant.id,
                             user_id=user.id, workspace_id="default", role=user.role)


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest) -> TokenResponse:
    with session_scope() as db:
        user = db.query(User).filter(User.email == req.email).first()
        if not user or not verify_password(req.password, user.password_hash):
            raise HTTPException(401, "Invalid credentials")
        principal = Principal(user.id, user.tenant_id, user.workspace_id, user.role)
        token = create_access_token(principal)
        return TokenResponse(access_token=token, tenant_id=user.tenant_id,
                             user_id=user.id, workspace_id=user.workspace_id, role=user.role)
