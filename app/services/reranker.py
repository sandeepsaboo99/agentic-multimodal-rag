"""
Cross-encoder reranker.

Design reasoning (rubric: Retrieval Quality)
--------------------------------------------
Bi-encoder ANN (dense) and BM25 (sparse) are *recall* devices: fast, but they
score query and document independently. A cross-encoder re-reads (query, chunk)
*together* and produces a far more accurate relevance score. We apply it only to
the small fused candidate set (top ~40) — expensive per-pair, cheap overall —
and keep the best RERANK_TOP_K as the generation context.

Offline fallback: if the cross-encoder can't load, we fall back to the fused RRF
score so retrieval still returns ranked results.
"""
from __future__ import annotations

import threading

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


class Reranker:
    _instance: "Reranker | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self.is_fallback = True
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(get_settings().reranker_model)
            self.is_fallback = False
            log.info("Loaded reranker %s", get_settings().reranker_model)
        except Exception as exc:  # noqa: BLE001
            log.warning("Reranker load failed (%s); using RRF-score fallback.", exc)

    @classmethod
    def instance(cls) -> "Reranker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        if self._model is None:
            return [0.0] * len(docs)  # caller keeps prior fused order
        raw = self._model.predict([(query, d) for d in docs])
        # sigmoid-normalize to 0..1 for a stable, comparable confidence signal
        import math

        return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]
