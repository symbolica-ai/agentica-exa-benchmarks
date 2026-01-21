import asyncio
import os
from dataclasses import dataclass
from typing import Any

from agentica import spawn
from agentica.logging import AgentListener
from agentica.logging.loggers import StandardLogger

from .base import Searcher, SearchResponse, SearchResult, estimate_cost
from .exa import ExaSearcher


class AgenticaSearcher(Searcher):
    name = "agentica"
    # Class-level lock to serialize spawn operations (avoids agentica websocket race condition)
    _spawn_lock: asyncio.Lock | None = None

    def __init__(
        self,
        exa_api_key: str | None = None,
        model: str | None = None,
        category: str | None = "people",
        logs_dir: str | None = None,
    ):
        self.exa_api_key = exa_api_key or os.getenv("EXA_API_KEY")
        if not self.exa_api_key:
            raise ValueError("EXA_API_KEY required - get one at https://exa.ai")

        self.model = model or os.getenv(
            "AGENTICA_MODEL", "openrouter:google/gemini-3-flash-preview"
        )
        self.category = category
        self._logs_dir = logs_dir or "./logs"
        self._exa_searcher = ExaSearcher(
            api_key=self.exa_api_key,
            category=self.category,
        )
        if AgenticaSearcher._spawn_lock is None:
            AgenticaSearcher._spawn_lock = asyncio.Lock()
        print(f"[agentica] Using model: {self.model}")

    def _get_listener(self) -> AgentListener:
        return AgentListener(StandardLogger(logs_dir=self._logs_dir))

    async def _search_impl(self, query: str, num_results: int = 10) -> SearchResponse:
        search_call_count = 0
        exa_total_cost = 0.0

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

        async def exa_search(search_query: str) -> SearchResults:
            """
            Search for information using Exa search engine. Can only be called once.
            Search supports Advanced Search Operators and filters.
            This function logically behaves as a websearch tool.

            Args:
                search_query: The search query to find relevant results.

            Returns:
                SearchResults containing exactly {num_results} results with url, title, and text content.
            """
            nonlocal search_call_count, exa_total_cost
            if search_call_count >= 1:
                raise ValueError("Exa search can only be called once.")
            search_call_count += 1
            response = await self._exa_searcher.search(search_query, num_results)
            if "exa_cost_usd" in response.query_stats:
                exa_total_cost += response.query_stats["exa_cost_usd"]
            return SearchResults(results=response.results)

        # Serialize only the spawn operation to avoid agentica websocket race condition
        # The actual agent.call() runs in parallel after spawn completes
        assert self._spawn_lock is not None
        async with self._spawn_lock:
            agent = await spawn(
                premise=f"""
You are a search assistant that helps find people's professional profiles.

# Task
You will be given a query and you need to find the {num_results} most relevant people's professional profiles matching the query.

Your workflow outline:

```python
query_to_search = '...' # construct the query to search based on the query given; you are allowed to use your knowledge (e.g. existing company names) to improve search results relevance
```

```python
result = await search(..) # search function can be called only once
print(result) # Use print(result) to inspect the results only, avoid print(result.results)
# NEVER `return` here. `return` from REPL is expected to return final result
```

As many times as needed:
```python
... # analyze the results, order by relevance from the most relevant to the least relevant
```

Only after all output of code blocks above is inspected and you are sure you got {num_results} results sorted by relevance from the most relevant to the least relevant, return:
```python
return result.results
```
""",
                model=self.model,
                scope={
                    "search": exa_search,
                    "SearchResult": SearchResult,
                },
                listener=self._get_listener,
            )

        # Agent calls can run in parallel
        results: list[SearchResult] = await agent.call(
            list[SearchResult],
            f"Query: {query}\nReturn the {num_results} most relevant results matching the query sorted by relevance.",
        )

        # Collect usage stats
        usage = agent.last_usage()
        query_stats: dict[str, Any] = {
            "search_calls": search_call_count,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }
        cost = estimate_cost(self.model, usage.input_tokens, usage.output_tokens)
        if cost is not None:
            query_stats["cost_usd"] = cost
        if exa_total_cost > 0:
            query_stats["exa_cost_usd"] = exa_total_cost

        return SearchResponse(results=results[:num_results], query_stats=query_stats)

    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        return await self._search_impl(query, num_results)

    async def close(self):
        await self._exa_searcher.close()

    def get_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "category": self.category,
            "exa_config": self._exa_searcher.get_config(),
        }
