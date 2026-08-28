"""Chat + feedback endpoints — the agentic answer path."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.security import Principal, current_principal
from app.models.db import Feedback, session_scope
from app.models.schemas import ChatRequest, ChatResponse, FeedbackRequest
from app.services.agent import answer_query

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, principal: Principal = Depends(current_principal)) -> ChatResponse:
    result = answer_query(
        principal, req.query, history=req.history, force_route=req.force_route)
    return ChatResponse(**result)


@router.post("/feedback")
def feedback(req: FeedbackRequest, principal: Principal = Depends(current_principal)) -> dict:
    with session_scope() as db:
        db.add(Feedback(tenant_id=principal.tenant_id, user_id=principal.user_id,
                        request_id=req.request_id, vote=req.vote, note=req.note))
    return {"ok": True}
