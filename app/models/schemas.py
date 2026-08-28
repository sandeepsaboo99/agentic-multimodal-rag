"""Pydantic request/response contracts for the API layer."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

JudgeMode = Literal["proxy", "llm"]


# ---- Auth ----
class RegisterRequest(BaseModel):
    email: str
    password: str
    tenant_name: str = "My Org"
    role: str = "admin"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant_id: str
    user_id: str
    workspace_id: str
    role: str


# ---- Documents ----
class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    version: int
    n_chunks: int
    n_tables: int
    n_images: int
    error: str = ""


class JobOut(BaseModel):
    id: str
    document_id: str
    status: str
    stage: str
    attempts: int
    error: str = ""


# ---- Chat ----
class ChatRequest(BaseModel):
    query: str
    force_route: Optional[Literal["DIRECT_LLM", "RAG", "WEB"]] = None
    history: list[dict[str, str]] = Field(default_factory=list)


class Citation(BaseModel):
    document_id: str
    filename: str
    page: Optional[int] = None
    modality: str = "text"  # text | table | image
    score: float = 0.0
    snippet: str = ""


class ChatResponse(BaseModel):
    request_id: str
    route: str
    route_reason: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace: dict[str, Any]


# ---- Feedback ----
class FeedbackRequest(BaseModel):
    request_id: str
    vote: int  # +1 / -1
    note: str = ""


# ---- Evaluation ----
class EvalItem(BaseModel):
    question: str
    expected_source: str = ""
    expected_answer: str = ""


class EvalRunRequest(BaseModel):
    items: list[EvalItem]
    # "proxy" = deterministic lexical metrics (free, CI-stable)
    # "llm"   = LLM-as-judge scoring via Groq (falls back to proxy if no key)
    judge: JudgeMode = "proxy"


class EvalReport(BaseModel):
    n: int
    judge: str  # judge actually used ("proxy" | "llm")
    retrieval_recall: float
    groundedness: float
    answer_correctness: float
    citation_correctness: float
    avg_latency_ms: float
    avg_cost_usd: float
    failure_rate: float
    details: list[dict[str, Any]]
