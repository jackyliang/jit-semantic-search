"""Cross-encoder reranker for high-quality semantic reranking.

Bi-encoders encode query and document SEPARATELY, then compare via dot
product.  This misses cross-attention between query and document tokens.

Cross-encoders process (query, document) as a PAIR through a full
transformer, giving much better semantic matching — but at O(n) cost
since you cannot pre-compute document embeddings.

In a cascade pipeline the cross-encoder only sees ~50 candidates that
survived the cheap pre-filter stages, so total latency is ~200 ms.
"""

from __future__ import annotations

import threading
from typing import Any

from jit_search.core import SearchResult


class CrossEncoderReranker:
    """Cross-encoder reranker using a small cross-encoder model.

    Uses the 'cross-encoder/ms-marco-MiniLM-L-6-v2' model via the
    sentence-transformers library's CrossEncoder class.

    The model is lazily loaded on first use and shared across all
    instances (class-level, thread-safe via double-checked locking).
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    _model: Any = None
    _model_lock: threading.Lock = threading.Lock()

    @classmethod
    def _get_model(cls) -> Any:
        """Lazily initialise and return the shared CrossEncoder model.

        Uses double-checked locking so that only the first caller pays
        the initialisation cost and subsequent callers are lock-free.
        """
        if cls._model is None:
            with cls._model_lock:
                if cls._model is None:
                    from sentence_transformers import CrossEncoder

                    cls._model = CrossEncoder(cls.MODEL_NAME)
        return cls._model

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Score each (query, document) pair and return the top-k results.

        Parameters
        ----------
        query : str
            The search query.
        documents : list[str]
            Candidate documents to rerank.
        top_k : int
            Number of top results to return.

        Returns
        -------
        list[SearchResult]
            Top-k results sorted by descending cross-encoder score.
        """
        if not documents:
            return []

        top_k = min(top_k, len(documents))

        model = self._get_model()

        # Build (query, document) pairs for the cross-encoder
        pairs = [(query, doc) for doc in documents]
        scores = model.predict(pairs)

        # Build results and sort by score descending
        scored = sorted(
            enumerate(scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        results: list[SearchResult] = []
        for idx, score in scored[:top_k]:
            results.append(
                SearchResult(
                    index=idx,
                    score=float(score),
                    document=documents[idx],
                )
            )

        return results
