"""
FastAPI application entrypoint — the API Gateway of the target architecture.

Responsibilities
----------------
- Wire the service-oriented routes (auth, documents, chat, analytics, eval).
- Initialize the metadata DB on startup.
- CORS so the Streamlit frontend (a separate origin) can call it.
- A /health liveness probe for orchestrators (k8s/ECS).

This is the ONLY process the frontend talks to; it fans out to the ingestion,
retrieval, and generation services internally. That boundary is what lets each
concern scale and deploy independently (assignment 3.1).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_analytics, routes_auth, routes_chat, routes_documents, routes_eval
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.db import init_db

log = get_logger(__name__)


def create_app() -> FastAPI:
    get_settings()
    init_db()
    app = FastAPI(
        title="InsightRAG API",
        version="1.0.0",
        description="Agentic, multimodal, production-grade RAG service.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten to the frontend origin in prod
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(routes_auth.router)
    app.include_router(routes_documents.router)
    app.include_router(routes_chat.router)
    app.include_router(routes_analytics.router)
    app.include_router(routes_eval.router)

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        return {"status": "ok", "service": "insightrag-api"}

    log.info("InsightRAG API ready.")
    return app


app = create_app()
