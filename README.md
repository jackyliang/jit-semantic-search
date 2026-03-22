# jit-semantic-search

Semantic search without pre-processing. Query any text corpus instantly — no embeddings to pre-compute, no vector store to load, no index to build.

## The Problem

You're building an AI agent that calls the Zendesk API and gets back 10,000 tickets. You need to find the ones about "frustrated customers unable to complete onboarding." With traditional vector search (pgvector, Pinecone, etc.), you'd need to:

1. Chunk the tickets
2. Embed each one (44s for 10K docs)
3. Load into a vector store (5+ minutes)
4. Build an HNSW index
5. *Then* search

That's 6 minutes before your first result. In an agentic context with JIT tool calls, this is unacceptable — you need results in under a second, just like `grep` but with semantic understanding.

**jit-semantic-search** gives you vector-search-quality results with zero preprocessing:

```python
from jit_search import JITSearch

searcher = JITSearch(strategy="cascade")

# tickets = fetch_from_zendesk_api()  # 10K tickets
results = searcher.search(
    "frustrated customers unable to complete onboarding",
    [t["subject"] + " " + t["description"] for t in tickets],
    top_k=10,
)
```

## Installation

```bash
pip install jit-semantic-search
```

Or from source:

```bash
git clone https://github.com/your-org/jit-semantic-search.git
cd jit-semantic-search
uv venv && uv pip install -e .
```

## Usage

### Python API

```python
from jit_search import JITSearch

# Choose your strategy based on speed/quality needs
searcher = JITSearch(strategy="cascade")  # best quality/speed tradeoff

results = searcher.search("angry customer billing issue", documents, top_k=10)
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
  -d '{"query": "billing dispute", "documents": ["doc1...", "doc2..."], "top_k": 5}'
```

Search structured objects (Zendesk tickets, CRM records, etc.):

```bash
curl -X POST http://localhost:8000/search/objects \
  -H "Content-Type: application/json" \
  -d '{
    "query": "frustrated customer",
    "objects": [{"id": 1, "subject": "Login issue", "body": "I cant log in..."}],
    "text_fields": ["subject", "body"],
    "top_k": 5
  }'
```

### CLI

```bash
# Pipe JSON documents
echo '["ticket 1 text", "ticket 2 text"]' | jit-search "billing problem" --top-k 5

# From a file
jit-search "onboarding issue" --file tickets.json --text-fields subject,description
```

## Strategies

| Strategy | NDCG@10 | Latency (400 docs) | Best For |
|---|---|---|---|
| `cascade` | 0.956 | 228ms | **Default.** Best quality/speed tradeoff |
| `neural` | 0.974 | 1,874ms | Maximum quality, small corpora |
| `cascade_v2` | 0.861 | 55ms | Fastest semantic search |
| `projection` | 0.696 | 14ms | Lightweight, no neural model |
| `lexical` | 0.689 | 0.4ms | Instant keyword search |
| `rptree` | 0.621 | 5ms | Sub-linear TF-IDF search |

## How It Works

The `cascade` strategy combines three techniques:

1. **SPS pre-filter (15ms)**: A novel Semantic Projection Search that maps TF-IDF features through a learned linear projection into embedding space, fused with BM25 scores. Surfaces the top-50 candidates from the full corpus.

2. **Neural reranking (~200ms)**: A streaming bi-encoder (BGE-small via ONNX) re-embeds only the 50 candidates on-the-fly and reranks by cosine similarity. Embeddings are discarded after use — O(k) memory.

3. **Adaptive scaling**: Automatically adjusts candidate count based on corpus size (5% of corpus, up to 500).

No documents are pre-embedded. No index is built. No vector store is needed.

## Evaluation Results

Evaluated using **BEIR methodology** (NDCG@10 primary metric) on a synthetic support ticket dataset.

### 400 Documents

| Strategy | NDCG@10 | MRR@10 | Search p50 | Preprocess | Total TTR |
|---|---|---|---|---|---|
| **pgvector (HNSW)** | 0.974 | 1.000 | 23ms | **11.9s** | **11,903ms** |
| **JIT cascade** | 0.956 | 1.000 | 228ms | **0s** | **228ms** |

JIT cascade: **52x faster to first result**, 98% of pgvector quality.

### 10,000 Documents

| Strategy | NDCG@10 | MRR@10 | Search p50 | Preprocess | Total TTR |
|---|---|---|---|---|---|
| **pgvector (HNSW)** | 0.833 | — | 36ms | **356.8s** | **356,877ms** |
| **JIT cascade** | 0.917 | — | 974ms | **0s** | **974ms** |

JIT cascade: **367x faster to first result**, *higher quality* than pgvector at this scale.

### Crossover Analysis

pgvector's pre-processing cost amortizes after repeated queries on the same corpus:
- **400 docs**: ~52 queries to break even
- **10K docs**: ~384 queries to break even

If you're searching the same static corpus hundreds of times, use pgvector. If you're searching dynamic data from API calls in an agentic workflow, use JIT.

## Areas for Improvement

We've identified several research directions that could push this further:

- **Non-linear projection**: Replace the linear projection matrix (SPS) with a small MLP for better semantic quality without neural model inference
- **Query expansion**: Use an LLM to expand ambiguous queries ("wish list for product updates" → "feature request", "product improvement suggestion") before search
- **Better embedding models**: Swap BGE-small for a faster/better model (BGE-micro for speed, E5-large for quality)
- **Cross-encoder with domain fine-tuning**: The cross-encoder reranker underperforms the bi-encoder on our dataset because ms-marco is trained on web search, not support tickets
- **JIT Product Quantization**: PQ-encode documents on-the-fly for hardware-accelerated distance computation (inspired by [LEANN](https://arxiv.org/abs/2506.08276))
- **Streaming graph construction**: Build an approximate k-NN graph incrementally as documents stream in, enabling O(log n) search without a pre-built index

## License

MIT
