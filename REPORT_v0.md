# JIT Semantic Search — Evaluation Report

## Methodology

Evaluation follows **BEIR benchmark methodology** (Thakur et al., NeurIPS 2021), the industry standard for information retrieval evaluation used by MTEB, Pinecone, Weaviate, and Qdrant.

- **Primary metric**: NDCG@10 (Normalized Discounted Cumulative Gain at k=10)
- **Secondary quality**: Recall@10, Recall@100, MRR@10, Precision@10, Hit Rate@10
- **Efficiency**: Latency p50, p95 (ms)
- **Dataset**: 400 synthetic support tickets across 8 semantic clusters (50 docs/cluster), 24 queries (3 per cluster)
- **Runs**: 3 per query, averaged for latency
- **Hardware**: Apple Silicon (M-series), CPU-only inference

---

## v0 Results (Baseline)

| Strategy | NDCG@10 | Recall@10 | Recall@100 | MRR@10 | Prec@10 | HitRate@10 | Lat p50 | Lat p95 |
|---|---|---|---|---|---|---|---|---|
| **neural** | **0.9743** | **0.1925** | **0.8742** | **1.0000** | **0.9625** | **1.0000** | 1977.6ms | 2049.4ms |
| lexical | 0.6891 | 0.1342 | 0.4325 | 0.7763 | 0.6708 | 0.8750 | **0.4ms** | **0.5ms** |
| projection | 0.4319 | 0.0850 | 0.4342 | 0.5122 | 0.4250 | 0.6667 | 5.6ms | 5.7ms |

**SPS was worse than BM25.** Root causes: insufficient training corpus (~850 sentences), no regularization (underdetermined system), pure projection without lexical signal.

---

## v1 Results (After Improvements)

| Strategy | NDCG@10 | Recall@10 | Recall@100 | MRR@10 | Prec@10 | HitRate@10 | Lat p50 | Lat p95 |
|---|---|---|---|---|---|---|---|---|
| **neural** | **0.9743** | **0.1925** | **0.8742** | **1.0000** | **0.9625** | **1.0000** | 2281.4ms | 2837.6ms |
| projection | 0.7475 | 0.1450 | 0.5550 | 0.8162 | 0.7250 | 0.9167 | 14.6ms | 15.6ms |
| lexical | 0.6891 | 0.1342 | 0.4325 | 0.7763 | 0.6708 | 0.8750 | **0.4ms** | **0.5ms** |

### v0 → v1 Delta (Projection/SPS)

| Metric | v0 | v1 | Change |
|---|---|---|---|
| NDCG@10 | 0.4319 | 0.7475 | **+73%** |
| MRR@10 | 0.5122 | 0.8162 | **+59%** |
| Precision@10 | 0.4250 | 0.7250 | **+71%** |
| Hit Rate@10 | 0.6667 | 0.9167 | **+37%** |
| Recall@100 | 0.4342 | 0.5550 | **+28%** |
| Latency p50 | 5.6ms | 14.6ms | +2.6x (BM25 fusion cost) |

**SPS now beats BM25 on every quality metric.** The hybrid approach (semantic projection + BM25 score fusion) validated the core hypothesis.

---

## Strategy Deep-Dive

### Neural (Streaming fastembed) — Quality Ceiling

**Architecture**: Lazy-loaded BGE-small-en-v1.5 via fastembed → batch embed in streaming fashion → min-heap top-k → early termination.

- Near-perfect retrieval: NDCG@10 = 0.97, MRR = 1.0, Hit Rate = 100%
- Every query's first result is relevant (MRR = 1.0)
- **Bottleneck is pure latency**: ~2.3s for 400 docs, extrapolates to ~50s for 10K docs
- Optimizations applied (co-embed query with first batch, numpy argpartition for heap, explicit ONNX CPU provider) — marginal speedup, embedding inference dominates
- Early termination doesn't kick in on this small dataset

**Verdict**: Gold standard for quality. Impractical for JIT at >1K docs without a cascade in front.

### Projection/SPS (Novel Method) — Best Speed/Quality Tradeoff

**Architecture**: HashingVectorizer (16K features, unigrams+bigrams) → learned linear projection W (16K→384, Ridge regression) → cosine similarity. Fused with BM25 scores at alpha=0.5.

**What fixed it (v0 → v1):**
1. **BM25 score fusion**: Hybrid search (alpha * SPS + (1-alpha) * BM25) — this was the biggest single improvement. Lexical signal covers the cases where the projection is lossy.
2. **Ridge regression** (alpha=1.0): Replaced plain lstsq. The original system was underdetermined (850 samples, 16K features) — Ridge's L2 regularization produces a more generalizable W.
3. **Expanded training corpus**: +320 sentences including 100 paraphrase pairs. Paraphrases are critical — they teach W that different surface forms map to similar embeddings.
4. **Cache invalidation**: Versioned cache path ensures stale matrices aren't loaded.

**Remaining gap to neural**: 0.23 NDCG points. The linear projection fundamentally can't capture compositional semantics ("not good" ≠ "bad" in bag-of-ngrams space). This is an inherent limitation of the linear approach.

**Verdict**: The sweet spot for JIT. 15ms for semantic-quality search that beats BM25. The novel contribution is validated — linear projection from hashing features into embedding space works, especially when fused with lexical scores.

### Lexical (BM25) — Speed Baseline

**Architecture**: Query-only tokenization+stemming → `str.count` substring matching (C-speed) → numpy-vectorized BM25 scoring → argpartition top-k.

- Sub-millisecond: 0.4ms p50 for 400 docs
- NDCG@10 = 0.69: solid for keyword-overlap queries
- 12.5% of queries miss entirely (Hit Rate = 0.875) — purely semantic queries with no lexical overlap
- Clever design: stemming query but using substring match on raw docs gives implicit prefix matching ("custom" matches "customer", "customers")

**Verdict**: Unbeatable for speed. Good enough when queries have keyword overlap with documents. Falls apart on synonym/paraphrase queries.

---

## Tradeoff Map

```
NDCG@10
  1.0 |  * neural (0.97)
      |
  0.8 |        * projection (0.75)
      |  * lexical (0.69)
  0.6 |
      |
  0.4 |        * projection_v0 (0.43)  [before fixes]
      |
  0.2 |
      +---+--------+--------+--------+--------+---> Latency (ms)
         0.1      1        10      100     1000
              lexical    projection          neural
              (0.4ms)    (15ms)              (2281ms)
```

**The gap between projection (0.75) and neural (0.97) at 15ms vs 2281ms is the cascade opportunity.**

---

## v2 Results (Cascade Strategy Added)

| Strategy | NDCG@10 | Recall@10 | Recall@100 | MRR@10 | Prec@10 | HitRate@10 | Lat p50 | Lat p95 |
|---|---|---|---|---|---|---|---|---|
| **neural** | **0.9743** | **0.1925** | **0.8742** | **1.0000** | **0.9625** | **1.0000** | 3208ms | 3525ms |
| **cascade** | **0.9028** | 0.1758 | 0.3983 | 0.9583 | 0.8792 | 0.9583 | **353ms** | **416ms** |
| projection | 0.7475 | 0.1450 | 0.5550 | 0.8162 | 0.7250 | 0.9167 | 15ms | 15ms |
| lexical | 0.6891 | 0.1342 | 0.4325 | 0.7763 | 0.6708 | 0.8750 | 0.4ms | 0.5ms |

### Cascade: The Best JIT Strategy

The cascade (SPS pre-filter → neural rerank top-50) achieves **93% of neural quality at 9x the speed**:
- NDCG@10 = 0.90 (target was > 0.90)
- Latency = 353ms p50 (SPS: 15ms + neural on 50 docs: ~340ms)
- MRR = 0.96: first relevant result is almost always #1
- Architecture: SPS selects top-50 candidates from full corpus → neural model reranks only those 50

**Key parameter**: `stage2_k=50` (candidates from SPS to neural). This parameter controls the quality/speed tradeoff:

| stage2_k | NDCG@10 | Approx Latency |
|---|---|---|
| 20 | 0.82 | ~100ms |
| 30 | 0.88 | ~150ms |
| 50 | 0.90 | ~350ms |
| 100 | 0.93 | ~600ms |

---

## Full Evolution Summary

| Version | Best Strategy | NDCG@10 | Latency | Key Change |
|---|---|---|---|---|
| v0 | neural | 0.97 | 2000ms | Baseline — three strategies built |
| v0 | projection (SPS) | 0.43 | 6ms | SPS worse than BM25 |
| v1 | projection (SPS) | 0.75 | 15ms | BM25 fusion + Ridge + more training data |
| **v2** | **cascade** | **0.90** | **353ms** | SPS pre-filter → neural rerank top-50 |

---

## Tradeoff Map (Final)

```
NDCG@10
  1.0 |  * neural (0.97)
      |       * cascade (0.90)       <-- BEST TRADEOFF
  0.8 |        * projection (0.75)
      |  * lexical (0.69)
  0.6 |
      |        * projection_v0 (0.43)
  0.4 |
      +---+--------+--------+--------+--------+---> Latency (ms)
         0.1      1        10      100     1000
              lexical    projection  cascade  neural
              (0.4ms)    (15ms)     (353ms)  (3208ms)
```

---

## Remaining Improvement Roadmap

### Push cascade quality higher (0.90 → 0.95+)
1. **Better SPS recall**: The cascade is bounded by SPS's ability to surface relevant docs in top-50. Queries like "requesting new functionality" still fail because SPS doesn't find them
2. **Non-linear projection**: 2-layer MLP instead of linear W for SPS
3. **Cross-encoder reranking**: Replace bi-encoder neural with a cross-encoder for Stage 3

### Push cascade speed lower (353ms → <100ms)
1. **Smaller embedding model**: BGE-micro or MiniLM-L6 for the reranking stage
2. **Matryoshka embeddings**: Use 64-dim instead of 384-dim
3. **Adaptive stage2_k**: Use 20 for easy queries, 50 for hard ones

### Research directions
1. **Query-adaptive projection**: Multiple W matrices selected by query type
2. **Matryoshka projection**: Variable-dimension projected embeddings
3. **Learned query expansion**: Use the projection to expand BM25 queries with semantic neighbors

---

## Reproducibility

```bash
# Install
uv venv && uv pip install -e ".[dev]"

# Run benchmark
.venv/bin/python -m evals.benchmark

# Use the library
from jit_search import JITSearch

# Fastest (sub-ms)
searcher = JITSearch(strategy="lexical")

# Best quality/speed tradeoff (~15ms, NDCG 0.75)
searcher = JITSearch(strategy="projection")

# Best JIT quality (~350ms, NDCG 0.90)
searcher = JITSearch(strategy="cascade")

# Maximum quality (~3s, NDCG 0.97)
searcher = JITSearch(strategy="neural")

# Auto mode uses cascade
searcher = JITSearch(strategy="auto")

results = searcher.search("frustrated customers", documents, top_k=10)
for r in results:
    print(f"[{r.score:.3f}] {r.document}")
```
