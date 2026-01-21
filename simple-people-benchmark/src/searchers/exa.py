import os
from typing import Any

import httpx

from .base import Searcher, SearchResponse, SearchResult


class CachingSearcher(Searcher):
    """Wrapper that caches search results by (query, num_results) to ensure
    multiple searchers get exactly the same results for the same query."""

    name = "caching"

    def __init__(self, searcher: Searcher):
        self._searcher = searcher
        self._cache: dict[tuple[str, int], SearchResponse] = {}

    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        key = (query, num_results)
        if key not in self._cache:
            self._cache[key] = await self._searcher.search(query, num_results)
        return self._cache[key]

    async def close(self):
        await self._searcher.close()

    def get_config(self) -> dict[str, Any]:
        return {
            "cached": True,
            "inner": self._searcher.get_config(),
        }

    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Return the number of cached queries."""
        return len(self._cache)


class ExaSearcher(Searcher):
    name = "exa"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.exa.ai",
        include_text: bool = True,
        category: str | None = None,
        search_type: str = "fast",
    ):
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY required - get one at https://exa.ai")

        self.base_url = base_url
        self.include_text = include_text
        self.category = category
        self.search_type = search_type
        self._client = httpx.AsyncClient(timeout=60.0)

    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        payload: dict[str, Any] = {
            "query": query,
            "numResults": num_results,
            "type": self.search_type,
        }

        if self.category:
            payload["category"] = self.category

        if self.include_text:
            payload["contents"] = {"text": True}

        response = await self._client.post(
            f"{self.base_url}/search",
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for r in data.get("results", []):
            results.append(
                SearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    text=r.get("text", ""),
                    metadata={
                        "score": r.get("score"),
                        "published_date": r.get("publishedDate"),
                        "author": r.get("author"),
                    },
                )
            )

        query_stats: dict[str, Any] = {}
        cost_dollars = data.get("costDollars", {})
        if "total" in cost_dollars:
            query_stats["exa_cost_usd"] = cost_dollars["total"]

        return SearchResponse(results=results, query_stats=query_stats)

    async def close(self):
        await self._client.aclose()

    def get_config(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "include_text": self.include_text,
            "category": self.category,
            "search_type": self.search_type,
        }
