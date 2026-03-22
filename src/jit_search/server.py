"""FastAPI server for JIT Semantic Search.

Usage:
    uvicorn jit_search.server:app --host 0.0.0.0 --port 8000
    # or
    python -m jit_search.server
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("jit_search.server")

app = FastAPI(
    title="JIT Semantic Search",
    description="Just-in-time semantic search on arbitrary data without pre-processing",
    version="0.1.0",
)

# CORS middleware — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Timing middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_request_latency(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s %.1fms", request.method, request.url.path, elapsed_ms)
    return response


# ---------------------------------------------------------------------------
# Lazy strategy loader — avoids importing heavy deps at module level
# ---------------------------------------------------------------------------

_searcher_cache: dict[str, Any] = {}


def _get_searcher(strategy: str):
    """Lazily import strategies and return a JITSearch instance."""
    if strategy not in _searcher_cache:
        # First call triggers the strategy imports
        from jit_search import JITSearch  # noqa: registers strategies on import

        _searcher_cache[strategy] = JITSearch(strategy=strategy)
    return _searcher_cache[strategy]


def _available_strategies() -> list[str]:
    """Return registered strategy names (triggers import if needed)."""
    from jit_search import JITSearch

    # Ensure strategies are registered by touching at least one
    return list(JITSearch.STRATEGIES.keys())


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int = Field(default=10, ge=1)
    strategy: str = "cascade"


class SearchResultItem(BaseModel):
    index: int
    score: float
    document: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    strategy: str
    latency_ms: float
    num_documents: int


class ObjectSearchRequest(BaseModel):
    query: str
    objects: list[dict[str, Any]]
    text_fields: list[str]
    top_k: int = Field(default=10, ge=1)
    strategy: str = "cascade"


class ObjectSearchResultItem(BaseModel):
    index: int
    score: float
    document: str
    object: dict[str, Any]


class ObjectSearchResponse(BaseModel):
    results: list[ObjectSearchResultItem]
    strategy: str
    latency_ms: float
    num_documents: int


class HealthResponse(BaseModel):
    status: str


class StrategiesResponse(BaseModel):
    strategies: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Search a list of plain-text documents."""
    searcher = _get_searcher(req.strategy)
    results, elapsed_ms = searcher.search_timed(req.query, req.documents, req.top_k)

    return SearchResponse(
        results=[
            SearchResultItem(index=r.index, score=r.score, document=r.document)
            for r in results
        ],
        strategy=req.strategy,
        latency_ms=round(elapsed_ms, 1),
        num_documents=len(req.documents),
    )


@app.post("/search/objects", response_model=ObjectSearchResponse)
async def search_objects(req: ObjectSearchRequest):
    """Search structured objects by concatenating specified text fields."""
    # Build plain-text documents by concatenating the requested fields
    documents: list[str] = []
    for obj in req.objects:
        parts = [str(obj.get(f, "")) for f in req.text_fields]
        documents.append(" ".join(parts))

    searcher = _get_searcher(req.strategy)
    results, elapsed_ms = searcher.search_timed(req.query, documents, req.top_k)

    return ObjectSearchResponse(
        results=[
            ObjectSearchResultItem(
                index=r.index,
                score=r.score,
                document=r.document,
                object=req.objects[r.index],
            )
            for r in results
        ],
        strategy=req.strategy,
        latency_ms=round(elapsed_ms, 1),
        num_documents=len(req.objects),
    )


@app.get("/strategies", response_model=StrategiesResponse)
async def strategies():
    """Return available search strategies."""
    return StrategiesResponse(strategies=_available_strategies())


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok")


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "jit_search.server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
