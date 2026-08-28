"""
Centralized configuration.

Design reasoning
----------------
A single, typed settings object (Pydantic Settings) is the seam between the
12-factor environment and the code. Every service (ingestion, retrieval,
generation, router) reads from here rather than calling os.getenv directly, so
that tuning knobs (chunk size, top-k, thresholds, model names) are declared in
exactly one place and are trivially overridable per environment without code
changes.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- LLM (Groq) ----
    groq_api_key: str = ""
    groq_text_model: str = "llama-3.3-70b-versatile"
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_router_model: str = "llama-3.1-8b-instant"

    # ---- Embeddings / reranker ----
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    allow_embedding_fallback: bool = True

    # ---- Vector store ----
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_local_path: str = "./storage/qdrant"
    qdrant_collection: str = "insightrag_chunks"

    # ---- Metadata DB ----
    database_url: str = "sqlite:///./storage/insightrag.db"

    # ---- Object storage ----
    object_storage_path: str = "./storage/objects"

    # ---- Auth ----
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 720

    # ---- Retrieval tuning ----
    chunk_size: int = 900
    chunk_overlap: int = 150
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 5
    rerank_score_floor: float = 0.15

    # ---- Router ----
    router_use_llm: bool = True
    router_rag_confidence: float = 0.35

    # ---- Misc ----
    backend_url: str = "http://localhost:8000"
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        """Create local storage directories on boot (idempotent)."""
        for p in (self.object_storage_path, self.qdrant_local_path, "./storage"):
            Path(p).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
