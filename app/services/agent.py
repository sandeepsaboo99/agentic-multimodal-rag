"""
Agent orchestrator: the request-time brain that turns a query into a grounded,
cited, fully-traced answer.

Flow
----
    corpus check -> AgenticRouter.decide -> execute(route) -> generate -> trace persist

`execute` is a small dispatch over the three tools the agent can use:
  RAG        : reuse the evidence the router already retrieved (no double work).
  WEB        : run web_search, format results as numbered evidence.
  DIRECT_LLM : no evidence; answer from parametric knowledge.

Every path writes token usage + cost + per-step timings into the Trace, which is
persisted so the analytics tab can aggregate performance over time.
"""
from __future__ import annotations

import uuid

from app.core.logging import get_logger
from app.core.observability import Trace, estimate_cost_usd
from app.core.security import Principal
from app.models.db import Document, TraceRecord, session_scope
from app.models.schemas import Citation
from app.services.generation import get_generation_service
from app.services.retrieval import Evidence
from app.services.router import get_router
from app.services.websearch import web_search

log = get_logger(__name__)


def _has_corpus(principal: Principal) -> bool:
    with session_scope() as db:
        return db.query(Document).filter(
            Document.tenant_id == principal.tenant_id,
            Document.workspace_id == principal.workspace_id,
            Document.status == "ready",
        ).count() > 0


def _format_rag_evidence(ev: list[Evidence]) -> tuple[str, list[Citation]]:
    blocks, cites = [], []
    with session_scope() as db:
        for i, e in enumerate(ev, 1):
            doc = db.get(Document, e.document_id)
            fname = doc.filename if doc else e.document_id
            tag = {"table": "TABLE", "image": "IMAGE"}.get(e.modality, "TEXT")
            blocks.append(f"[{i}] ({tag}, {fname} p.{e.page}) {e.content}")
            cites.append(Citation(
                document_id=e.document_id, filename=fname, page=e.page,
                modality=e.modality, score=e.score, snippet=e.content[:240],
            ))
    return "\n\n".join(blocks), cites


def _format_web_evidence(results) -> tuple[str, list[Citation]]:
    blocks, cites = [], []
    for i, r in enumerate(results, 1):
        blocks.append(f"[{i}] {r.title} — {r.url}\n{r.snippet}")
        cites.append(Citation(
            document_id=r.url, filename=r.title, page=None,
            modality="web", score=0.0, snippet=r.snippet[:240],
        ))
    return "\n\n".join(blocks), cites


def answer_query(
    principal: Principal,
    query: str,
    history: list[dict] | None = None,
    force_route: str | None = None,
    document_ids: list[str] | None = None,
) -> dict:
    request_id = uuid.uuid4().hex
    trace = Trace(request_id=request_id, query=query,
                  tenant_id=principal.tenant_id, user_id=principal.user_id)
    gen = get_generation_service()

    has_corpus = _has_corpus(principal)
    decision = get_router().decide(
        principal, query, trace, has_corpus, force_route, document_ids)
    trace.route = decision.route
    trace.route_reason = decision.reason

    # ---- execute chosen tool ----
    evidence_block, citations = "", []
    if decision.route == "RAG":
        evidence_block, citations = _format_rag_evidence(decision.evidence)
    elif decision.route == "WEB":
        with trace.step("web_search") as m:
            results = web_search(query)
            m["results"] = len(results)
        evidence_block, citations = _format_web_evidence(results)

    # ---- generate ----
    with trace.step("generation", model=gen._settings.groq_text_model) as m:
        result = gen.generate(query, decision.route, evidence_block, history)
        trace.prompt_tokens = result.prompt_tokens
        trace.completion_tokens = result.completion_tokens
        trace.model = result.model
        m.update(prompt_tokens=result.prompt_tokens,
                 completion_tokens=result.completion_tokens)
    trace.answer_preview = result.text[:280]

    cost = estimate_cost_usd(trace.model, trace.prompt_tokens, trace.completion_tokens)
    _persist_trace(trace, cost)

    return {
        "request_id": request_id,
        "route": decision.route,
        "route_reason": decision.reason,
        "answer": result.text,
        "citations": [c.model_dump() for c in citations],
        "trace": {**trace.as_dict(), "cost_usd": cost},
    }


def _persist_trace(trace: Trace, cost: float) -> None:
    with session_scope() as db:
        db.add(TraceRecord(
            request_id=trace.request_id, tenant_id=trace.tenant_id,
            user_id=trace.user_id, route=trace.route, route_reason=trace.route_reason,
            model=trace.model, prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            total_latency_ms=trace.total_latency_ms, cost_usd=cost,
            payload=trace.as_dict(),
        ))
