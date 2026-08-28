"""
Analytics endpoints — the data behind the "deep-dive" performance tab.

Design reasoning (user ask: inference time, memory, all relevant analytics)
---------------------------------------------------------------------------
Because every request persists a full Trace, we can compute operational and
quality analytics without any extra instrumentation:

  /analytics/summary   - fleet-level KPIs: request count, route mix, p50/p95
                         latency, token + $ cost, feedback score, live process
                         memory (RSS) and per-stage latency/memory breakdown.
  /analytics/traces    - recent raw traces for step-level drill-down.
  /analytics/health    - which subsystems are live vs in fallback (embeddings,
                         reranker, LLM, vector store) so you can see WHY numbers
                         look the way they do.

All queries are tenant-scoped: an admin sees only their tenant's telemetry.
"""
from __future__ import annotations

from collections import defaultdict

import psutil
from fastapi import APIRouter, Depends

from app.core.security import Principal, current_principal
from app.models.db import Feedback, TraceRecord, session_scope

router = APIRouter(prefix="/analytics", tags=["analytics"])

_PROC = psutil.Process()


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return round(xs[idx], 2)


@router.get("/summary")
def summary(limit: int = 500, principal: Principal = Depends(current_principal)) -> dict:
    with session_scope() as db:
        rows = (db.query(TraceRecord)
                .filter(TraceRecord.tenant_id == principal.tenant_id)
                .order_by(TraceRecord.created_at.desc()).limit(limit).all())
        fb = db.query(Feedback).filter(Feedback.tenant_id == principal.tenant_id).all()

    latencies = [r.total_latency_ms for r in rows]
    route_mix: dict[str, int] = defaultdict(int)
    tokens = cost = 0
    stage_lat: dict[str, list[float]] = defaultdict(list)
    stage_mem: dict[str, list[float]] = defaultdict(list)

    for r in rows:
        route_mix[r.route] += 1
        tokens += (r.prompt_tokens + r.completion_tokens)
        cost += (r.cost_usd or 0.0)
        for step in (r.payload or {}).get("steps", []):
            stage_lat[step["name"]].append(step["latency_ms"])
            stage_mem[step["name"]].append(step.get("heap_peak_mb", 0.0))

    up = sum(1 for f in fb if f.vote > 0)
    total_fb = len(fb)

    return {
        "requests": len(rows),
        "route_mix": dict(route_mix),
        "latency_ms": {
            "p50": _pct(latencies, 0.50),
            "p95": _pct(latencies, 0.95),
            "avg": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "tokens_total": tokens,
        "cost_usd_total": round(cost, 5),
        "feedback": {"up": up, "down": total_fb - up, "score": round(up / total_fb, 3) if total_fb else None},
        "stage_latency_avg_ms": {k: round(sum(v) / len(v), 2) for k, v in stage_lat.items()},
        "stage_heap_peak_avg_mb": {k: round(sum(v) / len(v), 3) for k, v in stage_mem.items()},
        "process_memory_mb": round(_PROC.memory_info().rss / (1024 * 1024), 1),
        "cpu_percent": _PROC.cpu_percent(interval=0.0),
    }


@router.get("/traces")
def traces(limit: int = 50, principal: Principal = Depends(current_principal)) -> list[dict]:
    with session_scope() as db:
        rows = (db.query(TraceRecord)
                .filter(TraceRecord.tenant_id == principal.tenant_id)
                .order_by(TraceRecord.created_at.desc()).limit(limit).all())
        return [{
            "request_id": r.request_id, "route": r.route, "route_reason": r.route_reason,
            "model": r.model, "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens, "total_latency_ms": r.total_latency_ms,
            "cost_usd": r.cost_usd, "created_at": r.created_at.isoformat(),
            "steps": (r.payload or {}).get("steps", []),
            "query": (r.payload or {}).get("query", ""),
        } for r in rows]


@router.get("/health")
def health(principal: Principal = Depends(current_principal)) -> dict:
    from app.services.embeddings import EmbeddingProvider
    from app.services.generation import get_generation_service
    from app.services.reranker import Reranker

    emb = EmbeddingProvider.instance()
    rr = Reranker.instance()
    gen = get_generation_service()
    return {
        "embedding_model": emb.name,
        "embedding_fallback": emb.is_fallback,
        "reranker_fallback": rr.is_fallback,
        "llm_available": gen.available,
        "llm_model": gen._settings.groq_text_model,
    }
