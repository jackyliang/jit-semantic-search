"""Streaming neural search strategy using fastembed sentence transformers.

This is the high-quality baseline strategy. It embeds documents on-the-fly
in a streaming fashion, maintaining only a top-k min-heap in memory rather
than storing all embeddings. Supports early termination when scores stabilize.
"""

from __future__ import annotations

import heapq
import threading
from typing import Any

import numpy as np

from jit_search.core import JITSearch, SearchResult, SearchStrategy


@JITSearch.register("neural")
class NeuralSearchStrategy(SearchStrategy):
    """Streaming neural search using fastembed with BGE-small.

    Core approach:
      1. Embed the query once.
      2. Stream documents through the embedder in batches.
      3. Compute cosine similarity (via normalized dot product) per batch.
      4. Maintain a min-heap of size top_k for the best results.
      5. Discard embeddings after comparison -- O(k) memory, not O(n).
      6. Optionally terminate early if top-k scores stabilize.
    """

    _model: Any = None
    _model_lock: threading.Lock = threading.Lock()

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        batch_size: int = 256,
        patience: int = 3,
        parallel: int | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.patience = patience
        self.parallel = parallel

    @classmethod
    def _get_model(cls) -> Any:
        """Lazily initialize and return the shared fastembed model.

        Uses double-checked locking so that only the first caller pays
        the initialization cost and subsequent callers are lock-free.
        Uses CPUExecutionProvider explicitly for optimized ONNX inference.
        """
        if cls._model is None:
            with cls._model_lock:
                # Double-check after acquiring lock.
                if cls._model is None:
                    from fastembed import TextEmbedding

                    cls._model = TextEmbedding(
                        model_name=cls.MODEL_NAME,
                        providers=["CPUExecutionProvider"],
                    )
        return cls._model

    def _embed_batch_direct(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts using direct ONNX calls, returning a
        normalized 2D array.

        Bypasses the generator overhead of model.embed() by calling
        onnx_embed + _post_process_onnx_output directly, which returns
        a numpy matrix without per-row generator yield.
        """
        model = self._get_model()
        inner = model.model
        onnx_output = inner.onnx_embed(texts)
        matrix = inner._post_process_onnx_output(onnx_output)
        # Ensure float32 and 2D.
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(list(matrix), dtype=np.float32)
        elif matrix.dtype != np.float32:
            matrix = matrix.astype(np.float32)
        # Normalize each row in-place.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        matrix /= norms
        return matrix

    def _embed_batch_with_parallel(self, texts: list[str]) -> np.ndarray:
        """Embed a batch using model.embed() with the parallel parameter.

        Used when self.parallel is set, to leverage multi-process embedding.
        """
        model = self._get_model()
        embeddings = np.stack(
            list(model.embed(texts, batch_size=self.batch_size, parallel=self.parallel))
        )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embeddings /= norms
        return embeddings

    def search(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[SearchResult]:
        if not documents:
            return []

        # --- Filter out empty documents, remembering original indices ---
        indexed_docs: list[tuple[int, str]] = [
            (i, doc) for i, doc in enumerate(documents) if doc and doc.strip()
        ]

        if not indexed_docs:
            return []

        effective_k = min(top_k, len(indexed_docs))
        total = len(indexed_docs)

        # --- First batch: co-embed query with documents to avoid a
        #     separate ONNX session call for the query alone. ---
        first_batch_pairs = indexed_docs[: self.batch_size]
        first_batch_texts = [doc for _, doc in first_batch_pairs]

        # Embed [query] + first_batch together, then split.
        combined_texts = [query] + first_batch_texts
        if self.parallel is not None:
            combined_matrix = self._embed_batch_with_parallel(combined_texts)
        else:
            combined_matrix = self._embed_batch_direct(combined_texts)

        query_vec = combined_matrix[0]  # normalized query vector
        first_matrix = combined_matrix[1:]  # normalized doc vectors

        # --- Streaming search with min-heap ---
        heap: list[tuple[float, int]] = []
        processed = 0

        # Early termination bookkeeping.
        prev_min_score: float | None = None
        stable_batches = 0

        # --- Process the first batch (already embedded) ---
        first_indices = [idx for idx, _ in first_batch_pairs]
        similarities = first_matrix @ query_vec

        # Use numpy argpartition for the first batch to seed the heap
        # efficiently: pick the top effective_k candidates directly.
        batch_len = len(similarities)
        if batch_len <= effective_k:
            # All go into the heap.
            for j in range(batch_len):
                heapq.heappush(heap, (float(similarities[j]), first_indices[j]))
        else:
            # Use argpartition to find top effective_k without full sort.
            top_indices = np.argpartition(similarities, -effective_k)[-effective_k:]
            for j in top_indices:
                heapq.heappush(heap, (float(similarities[j]), first_indices[j]))

        processed += batch_len

        # Early termination check for the first batch.
        if len(heap) == effective_k:
            prev_min_score = heap[0][0]

        # --- Process remaining batches ---
        for batch_start in range(self.batch_size, total, self.batch_size):
            batch_pairs = indexed_docs[batch_start : batch_start + self.batch_size]
            batch_indices = [idx for idx, _ in batch_pairs]
            batch_texts = [doc for _, doc in batch_pairs]

            # Embed this batch.
            if self.parallel is not None:
                batch_matrix = self._embed_batch_with_parallel(batch_texts)
            else:
                batch_matrix = self._embed_batch_direct(batch_texts)

            # Cosine similarity via dot product (both sides normalized).
            similarities = batch_matrix @ query_vec

            # Use numpy argpartition to find candidates that could enter
            # the heap, avoiding Python-loop overhead for the full batch.
            current_min = heap[0][0] if heap else -float("inf")
            above_mask = similarities > current_min
            candidate_positions = np.flatnonzero(above_mask)

            if len(candidate_positions) > 0:
                if len(candidate_positions) > effective_k:
                    # Narrow to top effective_k candidates via argpartition.
                    candidate_sims = similarities[candidate_positions]
                    top_within = np.argpartition(candidate_sims, -effective_k)[
                        -effective_k:
                    ]
                    candidate_positions = candidate_positions[top_within]

                for j in candidate_positions:
                    score = float(similarities[j])
                    orig_idx = batch_indices[j]
                    if len(heap) < effective_k:
                        heapq.heappush(heap, (score, orig_idx))
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, (score, orig_idx))

            processed += len(batch_texts)

            # --- Early termination check ---
            if len(heap) == effective_k:
                current_min = heap[0][0]
                if prev_min_score is not None and current_min == prev_min_score:
                    stable_batches += 1
                else:
                    stable_batches = 1
                prev_min_score = current_min

                if (
                    stable_batches >= self.patience
                    and processed >= total * 0.5
                ):
                    break

        # --- Build results sorted by score descending ---
        results = [
            SearchResult(
                index=orig_idx,
                score=score,
                document=documents[orig_idx],
            )
            for score, orig_idx in heap
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results
