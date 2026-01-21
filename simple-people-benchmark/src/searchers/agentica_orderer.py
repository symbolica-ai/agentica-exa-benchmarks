import asyncio
import os
from dataclasses import dataclass
from typing import Any

from agentica import spawn
from agentica.logging import AgentListener
from agentica.logging.loggers import StandardLogger

from .base import Searcher, SearchResponse, SearchResult, estimate_cost


class AgenticaOrderer(Searcher):
    name = "agentica_orderer"
    # Class-level lock to serialize spawn operations (avoids agentica websocket race condition)
    _spawn_lock: asyncio.Lock = asyncio.Lock()

    def __init__(
        self,
        exa_searcher: Searcher,
        model: str | None = None,
        logs_dir: str | None = None,
    ):
        self.model = model or os.getenv(
            "AGENTICA_MODEL", "openrouter:google/gemini-3-flash-preview"
        )
        self._exa_searcher = exa_searcher
        self._logs_dir = logs_dir or "./logs"
        print(f"[agentica_orderer] Using model: {self.model}")

    def _get_listener(self) -> AgentListener:
        return AgentListener(StandardLogger(logs_dir=self._logs_dir))

    async def _search_impl(self, query: str, num_results: int = 10) -> SearchResponse:
        # First, get results from Exa
        exa_response = await self._exa_searcher.search(query, num_results)
        exa_results = exa_response.results
        exa_cost = exa_response.query_stats.get("exa_cost_usd", 0.0)

        @dataclass
        class SearchResults:
            results: list[SearchResult]

            def __post_init__(self):
                if len(self.results) != num_results:
                    raise ValueError(f"Expected {num_results} results, got {len(self.results)}")

            def __str__(self) -> str:
                return "\n".join(
                    f"Result #{i + 1}: URL: {result.url}, Title: {result.title}, Text: {result.text[:500]}"
                    for i, result in enumerate(self.results)
                )

            def __getitem__(self, index: int) -> SearchResult:
                return self.results[index]

            def __len__(self) -> int:
                return len(self.results)

        results = SearchResults(results=exa_results)

        # Serialize only the spawn operation to avoid agentica websocket race condition
        async with AgenticaOrderer._spawn_lock:
            agent = await spawn(
                premise=f"""
You are a search assistant that helps reorder search results by relevance.

# Task
You will be given a query and {num_results} search results for this query. Your goal is to reorder these results by relevance from the most relevant to the least relevant.

Your workflow outline:

```python
print(results) # Use print(results) to inspect the results only, avoid printing fields of `results` object
# NEVER `return` here. `return` from REPL is expected to return final result
```

As many times as needed:
```python
... # analyze the results, reorder
```

Only after all output of code blocks above is inspected and you are sure you got {num_results} results sorted by relevance from the most relevant to the least relevant, return:
```python
return result
```

""",
                model=self.model,
                scope={
                    "query": query,
                    "results": results,
                    "SearchResult": SearchResult,
                },
                listener=self._get_listener,
            )

        # Agent calls can run in parallel
        reordered_results: list[SearchResult] = await agent.call(
            list[SearchResult],
            f"Query: {query}\nReturn all {num_results} results reordered by relevance.",
        )

        # Collect usage stats
        usage = agent.last_usage()
        query_stats: dict[str, Any] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }
        cost = estimate_cost(self.model, usage.input_tokens, usage.output_tokens)
        if cost is not None:
            query_stats["cost_usd"] = cost
        if exa_cost > 0:
            query_stats["exa_cost_usd"] = exa_cost

        return SearchResponse(results=reordered_results[:num_results], query_stats=query_stats)

    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        return await self._search_impl(query, num_results)

    async def close(self):
        pass  # exa_searcher is managed externally

    def get_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "exa_config": self._exa_searcher.get_config(),
        }
