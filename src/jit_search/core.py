"""Core interfaces for JIT Semantic Search."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """A single search result."""
    index: int
    score: float
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchStrategy(ABC):
    """Base class for all search strategies."""

    name: str = "base"

    @abstractmethod
    def search(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search documents for the given query. Returns top_k results sorted by relevance."""
        ...

    def search_timed(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> tuple[list[SearchResult], float]:
        """Search and return (results, elapsed_ms)."""
        start = time.perf_counter()
        results = self.search(query, documents, top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return results, elapsed_ms


class JITSearch:
    """Main entry point for JIT Semantic Search.

    Usage:
        searcher = JITSearch(strategy="auto")
        results = searcher.search("frustrated customers", documents, top_k=10)
    """

    STRATEGIES: dict[str, type[SearchStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy."""
        def decorator(strategy_cls: type[SearchStrategy]):
            cls.STRATEGIES[name] = strategy_cls
            strategy_cls.name = name
            return strategy_cls
        return decorator

    def __init__(self, strategy: str = "auto", **kwargs):
        if strategy == "auto":
            strategy = "cascade"

        if strategy not in self.STRATEGIES:
            available = ", ".join(self.STRATEGIES.keys()) or "(none registered)"
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {available}")

        self._strategy = self.STRATEGIES[strategy](**kwargs)

    @property
    def strategy_name(self) -> str:
        return self._strategy.name

    def search(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[SearchResult]:
        return self._strategy.search(query, documents, top_k)

    def search_timed(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> tuple[list[SearchResult], float]:
        return self._strategy.search_timed(query, documents, top_k)
