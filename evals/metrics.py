"""Evaluation metrics for search quality and performance."""

from __future__ import annotations

import math
from dataclasses import dataclass

from jit_search.core import SearchResult


@dataclass
class EvalResult:
    """Evaluation result for a single query."""
    query: str
    strategy: str
    ndcg_at_k: float
    mrr: float
    recall_at_k: float
    precision_at_k: float
    latency_ms: float
    top_k: int


def ndcg_at_k(results: list[SearchResult], relevant: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, result in enumerate(results[:k]):
        rel = 1.0 if result.index in relevant else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG: all relevant docs at top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(results: list[SearchResult], relevant: set[int]) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first relevant result."""
    for i, result in enumerate(results):
        if result.index in relevant:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(results: list[SearchResult], relevant: set[int], k: int) -> float:
    """Recall at k — fraction of relevant docs found in top k."""
    if not relevant:
        return 0.0
    found = sum(1 for r in results[:k] if r.index in relevant)
    return found / len(relevant)


def precision_at_k(results: list[SearchResult], relevant: set[int], k: int) -> float:
    """Precision at k — fraction of top k results that are relevant."""
    if k == 0:
        return 0.0
    found = sum(1 for r in results[:k] if r.index in relevant)
    return found / k


def evaluate_single(
    query: str,
    results: list[SearchResult],
    relevant: set[int],
    latency_ms: float,
    strategy_name: str,
    k: int = 10,
) -> EvalResult:
    """Evaluate a single query's results."""
    return EvalResult(
        query=query,
        strategy=strategy_name,
        ndcg_at_k=ndcg_at_k(results, relevant, k),
        mrr=mrr(results, relevant),
        recall_at_k=recall_at_k(results, relevant, k),
        precision_at_k=precision_at_k(results, relevant, k),
        latency_ms=latency_ms,
        top_k=k,
    )
