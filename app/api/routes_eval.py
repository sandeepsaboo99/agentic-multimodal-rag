"""Evaluation endpoint — run the offline eval harness on demand.

Accepts a judge selector: "proxy" (deterministic lexical metrics) or "llm"
(LLM-as-judge via Groq). The response reports which judge actually ran, so a
missing key transparently downgrades to proxy instead of failing.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.security import Principal, current_principal
from app.models.schemas import EvalReport, EvalRunRequest
from app.services.evaluation import run_evaluation

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.post("/run", response_model=EvalReport)
def run(req: EvalRunRequest, principal: Principal = Depends(current_principal)) -> EvalReport:
    return run_evaluation(principal, req.items, judge=req.judge)
