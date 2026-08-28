"""
Relational metadata model (the assignment's "Metadata / ACL Database").

Design reasoning
----------------
Qdrant stores vectors + payloads for retrieval; the *system of record* for
identity, ownership, jobs, versions, feedback, and traces is a relational DB.
This separation is deliberate:
  - Postgres/SQLite gives transactional integrity for ownership & billing state.
  - Qdrant gives ANN retrieval; its payload is a denormalized projection of the
    ACL fields (tenant_id/workspace_id/document_id) needed to *filter* at query
    time.

Entities map 1:1 to the assignment's required data model: users, tenants/
workspaces, documents, versions, jobs, permissions — plus feedback and traces
for observability.

The default engine is SQLite for zero-setup local runs. In production, set
DATABASE_URL to Postgres; the ORM code is unchanged.
"""
from __future__ import annotations

import datetime as dt
import uuid
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import get_settings

Base = declarative_base()


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    plan = Column(String, default="free")  # free | pro | enterprise
    monthly_token_quota = Column(Integer, default=2_000_000)
    created_at = Column(DateTime, default=_now)


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=_uuid)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    workspace_id = Column(String, nullable=False, default="default")
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="member")  # admin | member | viewer
    created_at = Column(DateTime, default=_now)


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=_uuid)
    tenant_id = Column(String, nullable=False)
    workspace_id = Column(String, nullable=False, default="default")
    user_id = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    object_key = Column(String, nullable=False)  # path in object storage
    content_hash = Column(String, nullable=False, index=True)  # for versioning
    version = Column(Integer, default=1)
    status = Column(String, default="uploaded")  # uploaded|parsing|chunking|indexing|ready|failed
    n_chunks = Column(Integer, default=0)
    n_tables = Column(Integer, default=0)
    n_images = Column(Integer, default=0)
    error = Column(Text, default="")
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)


class Job(Base):
    """An async ingestion job. Mirrors the Upload->...->ready state machine."""

    __tablename__ = "jobs"
    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    tenant_id = Column(String, nullable=False)
    status = Column(String, default="queued")  # queued|running|succeeded|failed
    stage = Column(String, default="")  # parsing|chunking|indexing|...
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    error = Column(Text, default="")
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(String, primary_key=True, default=_uuid)
    tenant_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    request_id = Column(String, index=True)
    vote = Column(Integer)  # +1 / -1
    note = Column(Text, default="")
    created_at = Column(DateTime, default=_now)


class TraceRecord(Base):
    """Persisted request trace — powers the analytics deep-dive tab."""

    __tablename__ = "traces"
    id = Column(String, primary_key=True, default=_uuid)
    request_id = Column(String, index=True)
    tenant_id = Column(String, index=True)
    user_id = Column(String)
    route = Column(String, index=True)  # DIRECT_LLM | RAG | WEB
    route_reason = Column(Text, default="")
    model = Column(String, default="")
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_latency_ms = Column(Float, default=0.0)
    cost_usd = Column(Float, default=0.0)
    payload = Column(JSON)  # full trace.as_dict() for step-level drill-down
    created_at = Column(DateTime, default=_now, index=True)


# ---- Engine / session --------------------------------------------------------
_settings = get_settings()
_engine = create_engine(
    _settings.database_url,
    connect_args={"check_same_thread": False} if _settings.database_url.startswith("sqlite") else {},
    future=True,
)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(_engine)


@contextmanager
def session_scope() -> Iterator["SessionLocal"]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
