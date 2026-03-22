"""Cascade v2 search strategy — cross-encoder reranking.

Replaces the bi-encoder neural reranking stage from cascade v1 with a
cross-encoder, which processes (query, document) pairs jointly through a
full transformer.  This breaks the "semantic ceiling" of bi-encoder models
at modest cost (~200 ms on 50 candidates).

Pipeline
--------
For small corpora (< sps_direct_threshold):
    SPS on full corpus -> top stage2_k -> cross-encoder rerank -> top_k

For large corpora:
    Stage 1: BM25 + SPS pre-filter (union candidates) -> top stage2_k
    Stage 2: Cross-encoder rerank -> top_k
"""

from __future__ import annotations

from jit_search.core import JITSearch, SearchResult, SearchStrategy


@JITSearch.register("cascade_v2")
class CascadeSearchV2(SearchStrategy):
    """Cascade with cross-encoder reranking.

    Parameters
    ----------
    stage1_k : int
        Candidates from each pre-filter strategy (default 200).
    stage2_k : int
        Candidates passed to cross-encoder reranker (default 50).
    sps_direct_threshold : int
        If corpus size <= this, skip BM25 and run SPS directly (default 5000).
    """

    name = "cascade_v2"

    def __init__(
        self,
        *,
        stage1_k: int = 200,
        stage2_k: int = 50,
        sps_direct_threshold: int = 5000,
    ) -> None:
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k
        self.sps_direct_threshold = sps_direct_threshold

        self._lexical = None
        self._projection = None
        self._reranker = None

    def _get_lexical(self):
        if self._lexical is None:
            from jit_search.lexical import LexicalSearch
            self._lexical = LexicalSearch()
        return self._lexical

    def _get_projection(self):
        if self._projection is None:
            from jit_search.projection import SemanticProjectionSearch
            self._projection = SemanticProjectionSearch()
        return self._projection

    def _get_reranker(self):
        if self._reranker is None:
            from jit_search.reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def search(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[SearchResult]:
        if not documents:
            return []

        n_docs = len(documents)
        top_k = min(top_k, n_docs)

        # Scale stage2_k with corpus size: at least stage2_k, up to 5% of corpus
        effective_stage2_k = max(self.stage2_k, min(n_docs // 20, 500))

        # -----------------------------------------------------------------
        # Small corpora: SPS can handle the full set directly
        # -----------------------------------------------------------------
        if n_docs <= self.sps_direct_threshold:
            sps_k = min(effective_stage2_k, n_docs)
            sps_results = self._get_projection().search(query, documents, sps_k)

            # Cross-encoder rerank the SPS candidates
            return self._cross_encoder_rerank(query, documents, sps_results, top_k)

        # -----------------------------------------------------------------
        # Large corpora: BM25 + SPS pre-filter union, then cross-encoder
        # -----------------------------------------------------------------
        s1_k = min(self.stage1_k, n_docs)

        # Get candidates from both BM25 and SPS
        bm25_results = self._get_lexical().search(query, documents, s1_k)
        sps_results = self._get_projection().search(query, documents, s1_k)

        # Union candidates (deduplicate by index, keep best score from either)
        candidate_map: dict[int, float] = {}
        for r in bm25_results:
            candidate_map[r.index] = r.score
        for r in sps_results:
            if r.index not in candidate_map or r.score > candidate_map[r.index]:
                candidate_map[r.index] = r.score

        # Sort by combined best score, take top stage2_k
        sorted_candidates = sorted(
            candidate_map.items(), key=lambda x: x[1], reverse=True
        )
        candidate_indices = [idx for idx, _ in sorted_candidates[: effective_stage2_k]]
        candidate_docs = [documents[i] for i in candidate_indices]

        # Cross-encoder rerank
        reranked = self._get_reranker().rerank(query, candidate_docs, top_k)

        # Map indices back to the original corpus
        for r in reranked:
            r.index = candidate_indices[r.index]
            r.document = documents[r.index]

        return reranked

    def _cross_encoder_rerank(
        self,
        query: str,
        documents: list[str],
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Rerank candidate results using the cross-encoder."""
        candidate_indices = [r.index for r in candidates]
        candidate_docs = [documents[i] for i in candidate_indices]

        reranked = self._get_reranker().rerank(query, candidate_docs, top_k)

        # Map indices back to the original corpus
        for r in reranked:
            r.index = candidate_indices[r.index]
            r.document = documents[r.index]

        return reranked
