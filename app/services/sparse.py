"""
Sparse (lexical) encoder for native Qdrant hybrid search.

Design reasoning
----------------
We replaced the in-process BM25 (which scanned the whole ACL-scoped corpus on
every query — O(corpus) and blind to Qdrant's index) with NATIVE Qdrant sparse
vectors. Each chunk is stored with a sparse vector of term frequencies; the
collection's sparse index carries `Modifier.IDF`, so Qdrant computes the IDF
weighting **server-side at query time** — i.e. true BM25-style lexical scoring
inside the vector DB, fused with dense results by Qdrant itself.

This encoder turns text into `(indices, values)`:
  - token -> stable uint32 id via a hash (Qdrant sparse indices are u32)
  - value -> raw term frequency (IDF is applied by Qdrant's modifier)

No model download is required, so it works everywhere (unlike SPLADE/BM42, which
we note as the upgrade path). The hashing collides rarely at this width and is
deterministic across processes, which is what lets query- and index-time vectors
line up.
"""
from __future__ import annotations

import hashlib
import threading
from collections import Counter

_U32 = 0xFFFFFFFF


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def _token_id(token: str) -> int:
    # 32-bit stable id; Qdrant sparse-vector indices must be non-negative u32.
    return int(hashlib.md5(token.encode()).hexdigest()[:8], 16) & _U32


class SparseEncoder:
    """Deterministic bag-of-words term-frequency encoder. Thread-safe singleton."""

    _instance: "SparseEncoder | None" = None
    _lock = threading.Lock()
    name = "bm25-idf-hashing"

    @classmethod
    def instance(cls) -> "SparseEncoder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        counts = Counter(_token_id(t) for t in _tokenize(text))
        if not counts:
            return [], []
        indices = list(counts.keys())
        values = [float(v) for v in counts.values()]
        return indices, values
