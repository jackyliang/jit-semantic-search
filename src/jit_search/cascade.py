"""Adaptive Cascade search strategy — combines all three strategies.

For small corpora (< sps_direct_threshold):
    SPS directly on full corpus → Neural rerank top candidates

For large corpora:
    Stage 1: BM25 + SPS parallel pre-filter (union candidates)
    Stage 2: Neural rerank top candidates

Adaptive: skips Neural stage if confidence is high enough.
"""

from __future__ import annotations

from jit_search.core import JITSearch, SearchResult, SearchStrategy


@JITSearch.register("cascade")
class CascadeSearch(SearchStrategy):
    """Adaptive cascade with parallel pre-filtering.

    Parameters
    ----------
    stage1_k : int
        Candidates from each pre-filter strategy (default 200).
    stage2_k : int
        Candidates passed to Neural reranker (default 50).
    sps_direct_threshold : int
        If corpus size <= this, skip BM25 and run SPS directly (default 5000).
    confidence_threshold : float
        Score gap threshold to skip Neural stage (default 0.15).
    skip_neural : bool
        Always skip Neural stage for max speed (default False).
    """

    name = "cascade"

    def __init__(
        self,
        *,
        stage1_k: int = 200,
        stage2_k: int = 50,
        sps_direct_threshold: int = 5000,
        confidence_threshold: float = 0.3,
        skip_neural: bool = False,
    ) -> None:
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k
        self.sps_direct_threshold = sps_direct_threshold
        self.confidence_threshold = confidence_threshold
        self.skip_neural = skip_neural

        self._lexical = None
        self._projection = None
        self._neural = None

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

    def _get_neural(self):
        if self._neural is None:
            from jit_search.neural import NeuralSearchStrategy
            self._neural = NeuralSearchStrategy(batch_size=64)
        return self._neural

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

        # For small corpora: SPS can handle the full set directly — no pre-filter needed
        if n_docs <= self.sps_direct_threshold:
            sps_k = min(effective_stage2_k, n_docs)
            sps_results = self._get_projection().search(query, documents, sps_k)

            if self.skip_neural:
                return sps_results[:top_k]

            # Neural rerank the SPS candidates (~100ms for 50 docs — worth it)
            return self._neural_rerank(query, documents, sps_results, top_k)

        # For large corpora: parallel BM25 + SPS pre-filter, then neural rerank
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
        sorted_candidates = sorted(candidate_map.items(), key=lambda x: x[1], reverse=True)
        neural_indices = [idx for idx, _ in sorted_candidates[:effective_stage2_k]]

        if self.skip_neural:
            return [
                SearchResult(index=idx, score=candidate_map[idx], document=documents[idx])
                for idx in neural_indices[:top_k]
            ]

        # Neural rerank
        neural_docs = [documents[i] for i in neural_indices]
        neural_results = self._get_neural().search(query, neural_docs, top_k)

        for r in neural_results:
            r.index = neural_indices[r.index]
            r.document = documents[r.index]

        return neural_results

    def _neural_rerank(
        self,
        query: str,
        documents: list[str],
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Rerank candidate results using the neural strategy."""
        candidate_indices = [r.index for r in candidates]
        candidate_docs = [documents[i] for i in candidate_indices]

        neural_results = self._get_neural().search(query, candidate_docs, top_k)

        for r in neural_results:
            r.index = candidate_indices[r.index]
            r.document = documents[r.index]

        return neural_results

    def _is_confident(self, results: list[SearchResult], top_k: int) -> bool:
        """Check if there's a clear gap at the top-k boundary."""
        if len(results) <= top_k:
            return True

        if not results or results[0].score <= 0:
            return False

        top_score = results[0].score
        kth_score = results[min(top_k - 1, len(results) - 1)].score
        next_score = results[top_k].score if top_k < len(results) else 0.0

        gap = (kth_score - next_score) / top_score if top_score > 0 else 0
        return gap >= self.confidence_threshold
