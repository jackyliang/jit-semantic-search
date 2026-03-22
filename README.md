# jit-semantic-search

> **Note:** This is a research project exploring JIT semantic search techniques. Not production-ready.

`grep` for semantic search. Query any text data without setting up a vector store first.

## When to Use This

If you're running repeated searches on the same data, use pgvector. It's faster at query time and well-suited for that.

But in agentic workflows, you're often pulling data from different APIs, searching it once, and moving on. Standing up a vector store for each of these one-off queries adds complexity and wastes time. You'd be embedding, loading, and indexing data just to run a single search and throw it all away.

That's where JIT semantic search comes in. Some examples:

- An agent pulls 5,000 product listings from a supplier API to find ones matching a spec
- A research workflow fetches hundreds of papers from arXiv and needs to find the relevant ones
- A support agent queries three different internal APIs and needs to search across all the responses together
- A data pipeline pulls customer feedback from Slack, Intercom, and email, and needs to find mentions of a specific issue

No database, no preprocessing pipeline, no cleanup. Just pass in text and get results.

**jit-semantic-search** is the semantic equivalent of grep. Pass in text, get results:

```python
from jit_search import JITSearch

searcher = JITSearch(strategy="cascade")

# products = fetch_from_supplier_api()
results = searcher.search(
    "waterproof bluetooth speaker under 50 dollars",
    [p["title"] + " " + p["description"] for p in products],
    top_k=10,
)
```

## Installation

```bash
uv add jit-semantic-search
```

From source:

```bash
git clone https://github.com/jackyliang/jit-semantic-search.git
cd jit-semantic-search
uv sync
```

## Usage

### Python API

```python
from jit_search import JITSearch

searcher = JITSearch(strategy="cascade")

results = searcher.search("waterproof speaker", documents, top_k=10)
for r in results:
    print(f"[{r.score:.3f}] Doc #{r.index}: {r.document[:80]}")
```

### API Server

```bash
python -m jit_search serve
# Runs on http://localhost:8000
```

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "waterproof speaker", "documents": ["doc1...", "doc2..."], "top_k": 5}'
```

Search structured objects directly:

```bash
curl -X POST http://localhost:8000/search/objects \
  -H "Content-Type: application/json" \
  -d '{
    "query": "waterproof speaker",
    "objects": [{"id": 1, "title": "BT Speaker", "desc": "IPX7 rated..."}],
    "text_fields": ["title", "desc"],
    "top_k": 5
  }'
```

### CLI

```bash
echo '["doc1", "doc2", "doc3"]' | jit-search "search query" --top-k 5

jit-search "search query" --file data.json --text-fields title,description
```

## Strategies

| Strategy | NDCG@10 | Latency (400 docs) | Best For |
|---|---|---|---|
| `cascade` | 0.956 | 228ms | **Default.** Best quality/speed tradeoff |
| `neural` | 0.974 | 1,874ms | Maximum quality, small corpora |
| `cascade_v2` | 0.861 | 55ms | Fast semantic search |
| `projection` | 0.696 | 14ms | Lightweight, no neural model |
| `lexical` | 0.689 | 0.4ms | Keyword search |
| `rptree` | 0.621 | 5ms | Sub-linear TF-IDF search |

## How It Works

The `cascade` strategy combines three techniques:

1. **SPS pre-filter (~15ms)**: Semantic Projection Search maps TF-IDF features through a learned linear projection into embedding space, fused with BM25 scores. Surfaces the top candidates from the full corpus without running a neural model.

2. **Neural reranking (~200ms)**: A streaming bi-encoder (BGE-small via ONNX) embeds only the top candidates on-the-fly and reranks by cosine similarity. Embeddings are discarded after comparison.

3. **Adaptive scaling**: Candidate count adjusts based on corpus size (5% of corpus, up to 500).

No documents are pre-embedded. No index is built. No vector store is needed.

### Performance Notes

The search latencies above are **warm-model** numbers (models already in memory). On the first call, model loading adds 2-5s. Models stay warm for subsequent calls.

These numbers also don't include API fetch time for the source data. A realistic breakdown for 10K documents:

| Step | Time |
|---|---|
| API fetch (network) | varies |
| Model cold start (one-time) | 2-5s |
| Search | ~1s |
| **Subsequent searches** | **~1s** |

## Evaluation

Evaluated using [BEIR methodology](https://github.com/beir-cellar/beir) (NDCG@10 as primary metric).

### vs pgvector

pgvector requires preprocessing (embed + load + build HNSW index) before the first search. JIT doesn't. If you're going to search the same corpus many times, pgvector amortizes that cost and wins. For one-off searches, JIT is faster end-to-end.

**400 documents:**

| | NDCG@10 | Search latency | Preprocessing | Time to first result |
|---|---|---|---|---|
| pgvector | 0.974 | 23ms | 11.9s | 11.9s |
| JIT cascade | 0.956 | 228ms | 0s | 228ms |

**10,000 documents:**

| | NDCG@10 | Search latency | Preprocessing | Time to first result |
|---|---|---|---|---|
| pgvector | 0.833 | 36ms | 356s | 356s |
| JIT cascade | 0.917 | ~1s | 0s | ~1s |

pgvector amortizes its preprocessing cost after roughly 50-400 repeated queries on the same corpus, depending on corpus size. Use pgvector for stable, repeatedly-queried data. Use JIT for one-off searches across dynamic or disparate data sources.

## Related Work

The cascade architecture draws inspiration from [LEANN](https://arxiv.org/abs/2506.08276) (Lee et al., 2025), which uses a two-level search with PQ-compressed approximate distances and on-demand embedding recomputation for edge-device RAG. This library applies a similar multi-fidelity approach in a fully JIT context where no pre-built index exists.

## Areas for Improvement

- **Non-linear projection**: Replace the linear SPS projection with a small MLP
- **Query expansion**: Use an LLM to expand ambiguous queries before search
- **Better embedding models**: Swap BGE-small for faster or higher-quality models
- **Domain-specific cross-encoder**: The cross-encoder reranker (cascade_v2) currently uses ms-marco, which is trained on web search. A domain-tuned model would likely improve quality.
- **JIT Product Quantization**: PQ-encode documents on-the-fly for faster distance computation (from the LEANN paper)

## License

MIT
