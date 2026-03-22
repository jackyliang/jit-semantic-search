"""JIT Random Projection Tree (RP-tree) search strategy.

Builds random projection trees on-the-fly at query time for sub-linear
semantic search. Unlike traditional RP-trees (e.g. Spotify's Annoy) which
are built offline, this constructs the trees JIT using fast TF-IDF features.

Algorithm
---------
1. Vectorize all documents with TF-IDF (fast, no pre-fitting needed).
2. Build ``n_trees`` RP-trees: recursively split the document set using
   random hyperplanes (project onto a random unit vector, split at the
   median) until each leaf has <= ``leaf_size`` documents.
3. For each tree, traverse to the query's leaf, then backtrack to nearby
   leaves using a priority queue (margin-based) until ``n_candidates``
   candidates have been gathered.
4. Union candidate sets across all trees, score by cosine similarity,
   return top-k.

Performance target: build + search < 200ms for 10K documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from jit_search.core import JITSearch, SearchResult, SearchStrategy


# ---------------------------------------------------------------------------
# RP-tree data structure
# ---------------------------------------------------------------------------

@dataclass
class RPNode:
    """A node in a random projection tree.

    Internal nodes store a split hyperplane (``projection``) and median
    threshold. Leaf nodes store the indices of documents that landed here.
    """

    # Internal node fields
    projection: Optional[np.ndarray] = None  # random unit vector
    threshold: float = 0.0                   # median split value
    left: Optional["RPNode"] = None
    right: Optional["RPNode"] = None

    # Leaf node field
    indices: Optional[np.ndarray] = None     # doc indices in this leaf

    @property
    def is_leaf(self) -> bool:
        return self.indices is not None


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def _build_rptree(
    X: np.ndarray,
    indices: np.ndarray,
    leaf_size: int,
    rng: np.random.Generator,
) -> RPNode:
    """Recursively build a single RP-tree.

    Parameters
    ----------
    X : sparse or dense matrix, shape (n_docs, n_features)
        The full document-term matrix (read-only, shared across recursion).
    indices : 1-D int array
        Indices into X for the documents at this node.
    leaf_size : int
        Maximum number of documents in a leaf.
    rng : numpy Generator
        Random number generator for reproducible projections.
    """
    if len(indices) <= leaf_size:
        return RPNode(indices=indices)

    # Pick a random unit vector in feature space.
    n_features = X.shape[1]
    v = rng.standard_normal(n_features).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-10

    # Project the subset of documents onto v.  X may be sparse, so use
    # X[indices] @ v which works for both sparse and dense.
    scores = np.asarray(X[indices] @ v).ravel()

    # Split at the median.
    median = float(np.median(scores))

    left_mask = scores <= median
    right_mask = ~left_mask

    # Guard against degenerate splits where all docs end up on one side
    # (can happen when many docs are identical).
    if left_mask.all() or right_mask.all():
        return RPNode(indices=indices)

    left_child = _build_rptree(X, indices[left_mask], leaf_size, rng)
    right_child = _build_rptree(X, indices[right_mask], leaf_size, rng)

    return RPNode(
        projection=v,
        threshold=median,
        left=left_child,
        right=right_child,
    )


# ---------------------------------------------------------------------------
# Tree search with priority-queue backtracking
# ---------------------------------------------------------------------------

def _search_rptree(
    node: RPNode,
    query_vec: np.ndarray,
    n_candidates: int,
) -> np.ndarray:
    """Search a single RP-tree using margin-based priority backtracking.

    Returns an array of candidate document indices (up to n_candidates).
    """
    # Priority queue: (neg_margin, node).  We use negative margin so that
    # nodes closer to the decision boundary (smaller margin) are popped
    # first -- they are most likely to contain relevant results.
    import heapq

    candidates: list[int] = []
    # (priority, unique_counter, node)
    counter = 0
    pq: list[tuple[float, int, RPNode]] = [(0.0, counter, node)]

    while pq and len(candidates) < n_candidates:
        _, _, current = heapq.heappop(pq)

        if current.is_leaf:
            candidates.extend(current.indices.tolist())
            continue

        # Compute the query's projection score for this split.
        score = float(query_vec @ current.projection)
        margin = abs(score - current.threshold)

        # Traverse towards the side the query falls on.
        if score <= current.threshold:
            primary, secondary = current.left, current.right
        else:
            primary, secondary = current.right, current.left

        # Always explore the primary child with priority 0 (best).
        counter += 1
        heapq.heappush(pq, (0.0, counter, primary))

        # Also enqueue the sibling with priority = margin (higher margin
        # means less likely to contain the nearest neighbour).
        counter += 1
        heapq.heappush(pq, (margin, counter, secondary))

    return np.array(candidates[:n_candidates], dtype=np.intp)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@JITSearch.register("rptree")
class RPTreeSearch(SearchStrategy):
    """JIT Random Projection Tree search.

    Builds RP-trees on the fly using TF-IDF features. Suitable for
    medium-to-large document sets where brute-force cosine search is
    too slow but neural embeddings are overkill or unavailable.

    Parameters
    ----------
    n_trees : int
        Number of RP-trees to build (more trees = better recall).
    leaf_size : int
        Maximum documents per leaf node.
    n_candidates : int
        Maximum total candidates to examine across all trees.
    max_features : int
        Number of TF-IDF features (vocabulary size cap).
    """

    def __init__(
        self,
        n_trees: int = 3,
        leaf_size: int = 50,
        n_candidates: int = 200,
        max_features: int = 5000,
    ) -> None:
        self.n_trees = n_trees
        self.leaf_size = leaf_size
        self.n_candidates = n_candidates
        self.max_features = max_features

    def search(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[SearchResult]:
        # --- Edge cases ------------------------------------------------
        if not documents or not query or not query.strip():
            return []

        # Filter empty documents, remembering original indices.
        indexed_docs: list[tuple[int, str]] = [
            (i, doc) for i, doc in enumerate(documents) if doc and doc.strip()
        ]
        if not indexed_docs:
            return []

        orig_indices = np.array([i for i, _ in indexed_docs], dtype=np.intp)
        texts = [doc for _, doc in indexed_docs]
        n_docs = len(texts)
        effective_k = min(top_k, n_docs)

        # --- Step 1: TF-IDF vectorization ------------------------------
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        # fit_transform on documents, then transform query separately
        doc_matrix = vectorizer.fit_transform(texts)     # sparse (n_docs, F)
        query_sparse = vectorizer.transform([query])     # sparse (1, F)
        query_vec = np.asarray(query_sparse.todense()).ravel().astype(np.float32)

        # --- Brute-force fallback for small corpora --------------------
        if n_docs <= self.n_candidates:
            return self._brute_force(
                doc_matrix, query_vec, orig_indices, documents, effective_k,
            )

        # --- Step 2 & 3: Build trees and search ------------------------
        rng = np.random.default_rng(42)
        all_indices = np.arange(n_docs, dtype=np.intp)
        per_tree_budget = max(1, self.n_candidates // self.n_trees)

        candidate_set: set[int] = set()
        for _ in range(self.n_trees):
            tree = _build_rptree(doc_matrix, all_indices, self.leaf_size, rng)
            tree_candidates = _search_rptree(tree, query_vec, per_tree_budget)
            candidate_set.update(tree_candidates.tolist())

        # --- Step 4: Score candidates by cosine similarity -------------
        candidate_indices = np.array(sorted(candidate_set), dtype=np.intp)
        return self._score_candidates(
            doc_matrix, query_vec, candidate_indices, orig_indices,
            documents, effective_k,
        )

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _cosine_similarities(
        doc_matrix: np.ndarray,
        query_vec: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between query and selected docs."""
        subset = doc_matrix[indices]
        # For sparse matrices, convert the dot product result to a 1-D array.
        dots = np.asarray(subset @ query_vec).ravel()

        # Norms -- handle sparse and dense uniformly.
        from scipy.sparse import issparse

        if issparse(subset):
            doc_norms = np.sqrt(np.asarray(subset.multiply(subset).sum(axis=1)).ravel())
        else:
            doc_norms = np.linalg.norm(subset, axis=1)

        query_norm = np.linalg.norm(query_vec)
        denom = doc_norms * query_norm
        denom = np.maximum(denom, 1e-10)
        return dots / denom

    def _brute_force(
        self,
        doc_matrix,
        query_vec: np.ndarray,
        orig_indices: np.ndarray,
        documents: list[str],
        effective_k: int,
    ) -> list[SearchResult]:
        """Brute-force cosine search over all documents."""
        all_idx = np.arange(doc_matrix.shape[0], dtype=np.intp)
        return self._score_candidates(
            doc_matrix, query_vec, all_idx, orig_indices, documents, effective_k,
        )

    def _score_candidates(
        self,
        doc_matrix,
        query_vec: np.ndarray,
        candidate_indices: np.ndarray,
        orig_indices: np.ndarray,
        documents: list[str],
        effective_k: int,
    ) -> list[SearchResult]:
        """Score candidate documents and return top-k results."""
        sims = self._cosine_similarities(doc_matrix, query_vec, candidate_indices)

        # Pick top-k from candidates.
        if len(sims) <= effective_k:
            top_pos = np.argsort(sims)[::-1]
        else:
            top_pos = np.argpartition(sims, -effective_k)[-effective_k:]
            top_pos = top_pos[np.argsort(sims[top_pos])[::-1]]

        results: list[SearchResult] = []
        for pos in top_pos:
            local_idx = candidate_indices[pos]
            orig_idx = int(orig_indices[local_idx])
            results.append(
                SearchResult(
                    index=orig_idx,
                    score=float(sims[pos]),
                    document=documents[orig_idx],
                )
            )
        return results
