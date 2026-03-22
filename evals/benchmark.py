"""Benchmark runner — BEIR-aligned evaluation methodology.

Primary metric: NDCG@10 (industry standard per BEIR/MTEB)
Secondary: Recall@10, Recall@100, MRR@10, Precision@10, Hit Rate@10
Efficiency: Latency p50, p95, mean

References:
  - BEIR (Thakur et al., NeurIPS 2021): NDCG@10 as primary
  - MTEB (Muennighoff et al., 2022): adopts BEIR methodology
  - ANN-Benchmarks: Recall + QPS tradeoff curves
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jit_search.core import JITSearch, SearchResult
from evals.dataset import generate_dataset, BenchmarkDataset
from evals.metrics import ndcg_at_k, mrr, recall_at_k, precision_at_k


@dataclass
class StrategyResult:
    """Aggregated evaluation results for one strategy."""
    strategy: str
    num_queries: int
    num_documents: int

    # Primary (BEIR Tier 1)
    ndcg_10: float = 0.0
    recall_10: float = 0.0
    recall_100: float = 0.0

    # Secondary (Tier 2)
    mrr_10: float = 0.0
    precision_10: float = 0.0
    hit_rate_10: float = 0.0

    # Efficiency
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0

    # Per-query breakdown
    per_query: list[dict] = field(default_factory=list)

    error: str | None = None


def _hit_rate(results: list[SearchResult], relevant: set[int], k: int) -> float:
    """Binary: did at least one relevant doc appear in top-k?"""
    return 1.0 if any(r.index in relevant for r in results[:k]) else 0.0


def _evaluate_strategy(
    strategy_name: str,
    dataset: BenchmarkDataset,
    num_runs: int = 3,
) -> StrategyResult:
    """Evaluate a single strategy against the benchmark dataset."""
    try:
        searcher = JITSearch(strategy=strategy_name)
    except Exception as e:
        print(f"    [ERROR] {strategy_name}: {e}")
        return StrategyResult(
            strategy=strategy_name,
            num_queries=len(dataset.queries),
            num_documents=len(dataset.documents),
            error=str(e),
        )

    # We need top-100 results for Recall@100
    top_k_fetch = min(100, len(dataset.documents))

    # Collect per-query metrics
    ndcg_scores = []
    recall_10_scores = []
    recall_100_scores = []
    mrr_scores = []
    precision_scores = []
    hit_rate_scores = []
    latencies = []
    per_query_data = []

    for query_idx, query in enumerate(dataset.queries):
        relevant = dataset.relevance[query_idx]

        # Run multiple times for latency stability
        run_latencies = []
        results = []
        for _ in range(num_runs):
            r, elapsed = searcher.search_timed(query, dataset.documents, top_k_fetch)
            run_latencies.append(elapsed)
            results = r  # keep last run's results

        avg_latency = statistics.mean(run_latencies)
        latencies.append(avg_latency)

        # Quality metrics at k=10
        ndcg_val = ndcg_at_k(results, relevant, 10)
        recall_10_val = recall_at_k(results, relevant, 10)
        recall_100_val = recall_at_k(results, relevant, 100)
        mrr_val = mrr(results, relevant)
        prec_val = precision_at_k(results, relevant, 10)
        hit_val = _hit_rate(results, relevant, 10)

        ndcg_scores.append(ndcg_val)
        recall_10_scores.append(recall_10_val)
        recall_100_scores.append(recall_100_val)
        mrr_scores.append(mrr_val)
        precision_scores.append(prec_val)
        hit_rate_scores.append(hit_val)

        per_query_data.append({
            "query": query,
            "ndcg_10": round(ndcg_val, 4),
            "recall_10": round(recall_10_val, 4),
            "recall_100": round(recall_100_val, 4),
            "mrr": round(mrr_val, 4),
            "precision_10": round(prec_val, 4),
            "hit_rate_10": round(hit_val, 4),
            "latency_ms": round(avg_latency, 2),
        })

    n = len(dataset.queries)
    sorted_lat = sorted(latencies)

    result = StrategyResult(
        strategy=strategy_name,
        num_queries=n,
        num_documents=len(dataset.documents),
        ndcg_10=statistics.mean(ndcg_scores),
        recall_10=statistics.mean(recall_10_scores),
        recall_100=statistics.mean(recall_100_scores),
        mrr_10=statistics.mean(mrr_scores),
        precision_10=statistics.mean(precision_scores),
        hit_rate_10=statistics.mean(hit_rate_scores),
        latency_mean_ms=statistics.mean(latencies),
        latency_p50_ms=sorted_lat[n // 2],
        latency_p95_ms=sorted_lat[int(n * 0.95)],
        per_query=per_query_data,
    )

    return result


def run_benchmark(
    strategies: list[str] | None = None,
    docs_per_cluster: int = 50,
    num_runs: int = 3,
) -> dict[str, StrategyResult]:
    """Run BEIR-aligned benchmark across all strategies."""
    _import_strategies()

    dataset = generate_dataset(docs_per_cluster=docs_per_cluster)
    print(f"\n  Dataset: {len(dataset.documents)} documents, {len(dataset.queries)} queries")

    if strategies is None:
        strategies = list(JITSearch.STRATEGIES.keys())

    results: dict[str, StrategyResult] = {}

    for name in strategies:
        if name not in JITSearch.STRATEGIES:
            print(f"  [SKIP] '{name}' not registered")
            continue

        print(f"\n  Evaluating: {name}")
        result = _evaluate_strategy(name, dataset, num_runs)
        results[name] = result

        if not result.error:
            print(f"    NDCG@10:     {result.ndcg_10:.4f}  (primary)")
            print(f"    Recall@10:   {result.recall_10:.4f}")
            print(f"    Recall@100:  {result.recall_100:.4f}")
            print(f"    MRR@10:      {result.mrr_10:.4f}")
            print(f"    Precision@10:{result.precision_10:.4f}")
            print(f"    Hit Rate@10: {result.hit_rate_10:.4f}")
            print(f"    Latency:     {result.latency_mean_ms:.1f}ms mean, "
                  f"{result.latency_p50_ms:.1f}ms p50, "
                  f"{result.latency_p95_ms:.1f}ms p95")

    return results


def print_comparison(results: dict[str, StrategyResult]) -> str | None:
    """Print BEIR-style comparison table. Returns the winner name."""
    valid = {k: v for k, v in results.items() if not v.error}

    if not valid:
        print("\n  No valid results to compare.")
        return None

    print("\n" + "=" * 110)
    print(f"{'Strategy':<15} {'NDCG@10':>8} {'Recall@10':>10} {'Recall@100':>11} "
          f"{'MRR@10':>8} {'Prec@10':>8} {'HitRate':>8} {'Lat p50':>8} {'Lat p95':>8}")
    print("-" * 110)

    sorted_strategies = sorted(
        valid.items(),
        key=lambda x: x[1].ndcg_10,
        reverse=True,
    )

    for name, r in sorted_strategies:
        print(
            f"{name:<15} {r.ndcg_10:>8.4f} {r.recall_10:>10.4f} {r.recall_100:>11.4f} "
            f"{r.mrr_10:>8.4f} {r.precision_10:>8.4f} {r.hit_rate_10:>8.4f} "
            f"{r.latency_p50_ms:>7.1f}ms {r.latency_p95_ms:>7.1f}ms"
        )

    print("=" * 110)

    winner = sorted_strategies[0][0]
    print(f"\n  Winner (by NDCG@10): {winner}")
    return winner


def save_results(results: dict[str, StrategyResult], path: Path):
    """Save results to JSON for reporting."""
    data = {}
    for name, r in results.items():
        data[name] = {
            "strategy": r.strategy,
            "num_queries": r.num_queries,
            "num_documents": r.num_documents,
            "ndcg_10": round(r.ndcg_10, 4),
            "recall_10": round(r.recall_10, 4),
            "recall_100": round(r.recall_100, 4),
            "mrr_10": round(r.mrr_10, 4),
            "precision_10": round(r.precision_10, 4),
            "hit_rate_10": round(r.hit_rate_10, 4),
            "latency_mean_ms": round(r.latency_mean_ms, 2),
            "latency_p50_ms": round(r.latency_p50_ms, 2),
            "latency_p95_ms": round(r.latency_p95_ms, 2),
            "error": r.error,
            "per_query": r.per_query,
        }
    path.write_text(json.dumps(data, indent=2))
    print(f"\n  Results saved to {path}")


def _import_strategies():
    """Import strategy modules to trigger @JITSearch.register decorators."""
    import importlib
    for module in ["jit_search.lexical", "jit_search.projection", "jit_search.neural", "jit_search.cascade", "jit_search.rptree", "jit_search.cascade_v2"]:
        try:
            importlib.import_module(module)
        except ImportError as e:
            print(f"  [WARN] {module}: {e}")


if __name__ == "__main__":
    print("JIT Semantic Search — BEIR-Aligned Benchmark")
    print("=" * 50)

    results = run_benchmark(docs_per_cluster=50, num_runs=3)
    winner = print_comparison(results)
    save_results(results, Path(__file__).parent / "results.json")
