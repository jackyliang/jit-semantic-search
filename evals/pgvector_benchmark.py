"""Benchmark pgvector (traditional vector search) against JIT strategies.

Pre-processes documents (embed + load into pgvector with HNSW index),
then measures search-only latency for fair comparison.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evals.dataset import generate_dataset
from evals.metrics import ndcg_at_k, mrr, recall_at_k, precision_at_k

# Ghost DB connection
DB_URL = "postgresql://tsdbadmin:sj3bh9u8gi9yyqzw@natihvossi.imbto7p7zf.tsdb.cloud.timescale.com:37389/tsdb"

# Same model as our neural strategy
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts using fastembed (same model as JIT neural strategy)."""
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=EMBED_MODEL)
    embeddings = list(model.embed(texts))
    return np.array(embeddings, dtype=np.float32)


def setup_pgvector(docs_per_cluster: int = 50):
    """Embed all documents and load into pgvector."""
    dataset = generate_dataset(docs_per_cluster=docs_per_cluster)
    print(f"  Dataset: {len(dataset.documents)} documents")

    # Embed all documents
    print("  Embedding all documents (pre-processing)...")
    t0 = time.perf_counter()
    embeddings = embed_texts(dataset.documents)
    embed_time = time.perf_counter() - t0
    print(f"  Embedding took: {embed_time:.1f}s ({embed_time/len(dataset.documents)*1000:.1f}ms/doc)")

    # Connect and load
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # Clear existing data
    cur.execute("TRUNCATE documents RESTART IDENTITY;")
    cur.execute("DROP INDEX IF EXISTS documents_embedding_hnsw;")

    # Bulk insert
    print("  Loading into pgvector...")
    t0 = time.perf_counter()
    for i, (doc, label) in enumerate(zip(dataset.documents, dataset.cluster_labels)):
        vec_str = "[" + ",".join(f"{v:.6f}" for v in embeddings[i]) + "]"
        cur.execute(
            "INSERT INTO documents (content, cluster_label, embedding) VALUES (%s, %s, %s::halfvec(384))",
            (doc, label, vec_str),
        )
    conn.commit()
    load_time = time.perf_counter() - t0
    print(f"  Load took: {load_time:.1f}s")

    # Build HNSW index
    print("  Building HNSW index (m=16, ef_construction=64)...")
    t0 = time.perf_counter()
    cur.execute("""
        CREATE INDEX documents_embedding_hnsw
        ON documents USING hnsw (embedding halfvec_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)
    conn.commit()
    index_time = time.perf_counter() - t0
    print(f"  HNSW index built: {index_time:.1f}s")

    # Verify
    cur.execute("SELECT COUNT(*) FROM documents;")
    count = cur.fetchone()[0]
    print(f"  Verified: {count} documents in pgvector")

    cur.close()
    conn.close()

    return dataset, embeddings, embed_time, load_time, index_time


def benchmark_pgvector(dataset, query_embeddings: np.ndarray, top_k: int = 10, ef_search: int = 100, num_runs: int = 3):
    """Benchmark pgvector search-only performance."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # Set ef_search for recall
    cur.execute(f"SET hnsw.ef_search = {ef_search};")

    results_per_query = []

    for qi, query in enumerate(dataset.queries):
        relevant = dataset.relevance[qi]
        vec_str = "[" + ",".join(f"{v:.6f}" for v in query_embeddings[qi]) + "]"

        latencies = []
        result_ids = []

        for _ in range(num_runs):
            t0 = time.perf_counter()
            cur.execute(
                f"SELECT id, content, embedding <=> %s::halfvec({EMBED_DIM}) AS distance "
                f"FROM documents ORDER BY embedding <=> %s::halfvec({EMBED_DIM}) LIMIT %s",
                (vec_str, vec_str, top_k),
            )
            rows = cur.fetchall()
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)
            # id is 1-indexed in postgres, convert to 0-indexed for eval
            result_ids = [(row[0] - 1, 1.0 - row[2]) for row in rows]  # (index, score)

        avg_lat = sum(latencies) / len(latencies)

        # Build SearchResult-compatible objects for metrics
        from jit_search.core import SearchResult
        search_results = [
            SearchResult(index=idx, score=score, document="")
            for idx, score in result_ids
        ]

        ndcg = ndcg_at_k(search_results, relevant, top_k)
        m = mrr(search_results, relevant)
        r_at_k = recall_at_k(search_results, relevant, top_k)
        p_at_k = precision_at_k(search_results, relevant, top_k)

        results_per_query.append({
            "query": query,
            "ndcg_10": ndcg,
            "mrr": m,
            "recall_10": r_at_k,
            "precision_10": p_at_k,
            "latency_ms": avg_lat,
        })

    cur.close()
    conn.close()

    # Aggregate
    n = len(results_per_query)
    agg = {
        "ndcg_10": sum(r["ndcg_10"] for r in results_per_query) / n,
        "mrr_10": sum(r["mrr"] for r in results_per_query) / n,
        "recall_10": sum(r["recall_10"] for r in results_per_query) / n,
        "precision_10": sum(r["precision_10"] for r in results_per_query) / n,
        "latency_mean_ms": sum(r["latency_ms"] for r in results_per_query) / n,
        "latency_p50_ms": sorted(r["latency_ms"] for r in results_per_query)[n // 2],
        "latency_p95_ms": sorted(r["latency_ms"] for r in results_per_query)[int(n * 0.95)],
    }

    return agg, results_per_query


def run_full_benchmark(docs_per_cluster: int = 50):
    """Run pgvector benchmark and compare with JIT strategies."""
    print("\n" + "=" * 60)
    print("pgvector vs JIT Semantic Search — Benchmark")
    print("=" * 60)

    # Setup pgvector
    print("\n[1/3] Setting up pgvector (pre-processing)...")
    dataset, doc_embeddings, embed_time, load_time, index_time = setup_pgvector(docs_per_cluster)
    total_preprocess = embed_time + load_time + index_time
    print(f"  Total pre-processing: {total_preprocess:.1f}s")

    # Embed queries
    print("\n[2/3] Embedding queries...")
    query_embeddings = embed_texts(dataset.queries)

    # Benchmark pgvector search
    print("\n[3/3] Benchmarking pgvector search (search-only, no pre-processing)...")
    for ef in [40, 100, 200]:
        agg, per_query = benchmark_pgvector(dataset, query_embeddings, top_k=10, ef_search=ef)
        print(f"\n  pgvector (ef_search={ef}):")
        print(f"    NDCG@10:     {agg['ndcg_10']:.4f}")
        print(f"    MRR@10:      {agg['mrr_10']:.4f}")
        print(f"    Recall@10:   {agg['recall_10']:.4f}")
        print(f"    Precision@10:{agg['precision_10']:.4f}")
        print(f"    Latency:     {agg['latency_mean_ms']:.1f}ms mean, "
              f"{agg['latency_p50_ms']:.1f}ms p50, "
              f"{agg['latency_p95_ms']:.1f}ms p95")

    # Compare with JIT strategies
    print("\n\n" + "=" * 60)
    print("Comparison: pgvector vs JIT (search-only latency)")
    print("=" * 60)
    print(f"\n  pgvector pre-processing cost (NOT included in search latency):")
    print(f"    Embedding:   {embed_time:.1f}s")
    print(f"    Load:        {load_time:.1f}s")
    print(f"    HNSW index:  {index_time:.1f}s")
    print(f"    TOTAL:       {total_preprocess:.1f}s")
    print(f"\n  JIT pre-processing cost: 0.0s (by definition)")

    # Save results
    results = {
        "pgvector": {
            "ndcg_10": agg["ndcg_10"],
            "mrr_10": agg["mrr_10"],
            "recall_10": agg["recall_10"],
            "precision_10": agg["precision_10"],
            "latency_p50_ms": agg["latency_p50_ms"],
            "latency_p95_ms": agg["latency_p95_ms"],
            "preprocess_embed_s": round(embed_time, 2),
            "preprocess_load_s": round(load_time, 2),
            "preprocess_index_s": round(index_time, 2),
            "preprocess_total_s": round(total_preprocess, 2),
            "num_documents": len(dataset.documents),
            "ef_search": 100,
        },
        "per_query": per_query,
    }
    out_path = Path(__file__).parent / "pgvector_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run_full_benchmark(docs_per_cluster=50)
