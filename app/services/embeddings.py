"""
Dense embedding provider.

Design reasoning
----------------
Groq serves LLMs but not embeddings, so we embed locally with an open-source
sentence-transformers model (all-MiniLM-L6-v2: 384-d, fast, CPU-friendly). This
keeps embeddings free, private (no data leaves the box), and low-latency.

Robustness: model weights download on first use. In an air-gapped or offline
environment that download fails, which would otherwise make the whole app
unusable. So we provide a deterministic *hashing* fallback embedder that maps
text into the same dimensionality via feature hashing. It is obviously weaker
semantically, but it keeps the entire pipeline runnable end-to-end for demos and
CI. Controlled by ALLOW_EMBEDDING_FALLBACK.

The provider is a lazily-initialized singleton so the (expensive) model load
happens once per process.
"""
from __future__ import annotations

import hashlib
import threading

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

_FALLBACK_DIM = 384


class _HashingEmbedder:
    """Deterministic bag-of-words feature-hashing embedder (offline fallback)."""

    dim = _FALLBACK_DIM
    is_fallback = True
    name = "hashing-fallback-384"

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
                vecs[i, idx] += sign
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class _STEmbedder:
    is_fallback = False

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        self.name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


class EmbeddingProvider:
    _instance: "EmbeddingProvider | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        s = get_settings()
        try:
            self._impl = _STEmbedder(s.embedding_model)
            log.info("Loaded embedding model %s (dim=%d)", self._impl.name, self._impl.dim)
        except Exception as exc:  # noqa: BLE001 - fall back deliberately
            if not s.allow_embedding_fallback:
                raise
            log.warning("Embedding model load failed (%s); using hashing fallback.", exc)
            self._impl = _HashingEmbedder()

    @classmethod
    def instance(cls) -> "EmbeddingProvider":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def dim(self) -> int:
        return self._impl.dim

    @property
    def name(self) -> str:
        return self._impl.name

    @property
    def is_fallback(self) -> bool:
        return self._impl.is_fallback

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._impl.encode(texts)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
