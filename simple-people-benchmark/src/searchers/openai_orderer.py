import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .base import Searcher, SearchResponse, SearchResult, estimate_cost


class SearchResultSchema(BaseModel):
    """Pydantic model for validating search result output."""

    url: str
    title: str
    text: str


# Global counter for agent runs
_agent_counter = 0


class OpenAIOrderer(Searcher):
    name = "openai_orderer"

    def __init__(
        self,
        exa_searcher: Searcher,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "google/gemini-3-flash-preview")
        self._exa_searcher = exa_searcher

        # Use OpenAI directly for OpenAI models, OpenRouter for everything else
        is_openai_model = self.model.startswith(("gpt-", "o1-", "o3-", "text-", "davinci"))
        if is_openai_model:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY required")
            self._client = AsyncOpenAI(api_key=api_key)
        else:
            api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY required for non-OpenAI models")
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )

        print(f"[openai_orderer] Using model: {self.model}")

    async def _search_impl(self, query: str, num_results: int = 10) -> SearchResponse:
        # First, get results from Exa
        exa_response = await self._exa_searcher.search(query, num_results)
        exa_results = exa_response.results
        exa_cost = exa_response.query_stats.get("exa_cost_usd", 0.0)

        # Format results for the model
        results_for_model = [
            {
                "index": i,
                "url": r.url,
                "title": r.title,
                "text": r.text if r.text else "",
            }
            for i, r in enumerate(exa_results)
        ]

        system_prompt = f"""
You are a search assistant that helps reorder search results by relevance.

# Task
You will be given a query and {num_results} search results for this query. Your goal is to reorder these results by relevance from the most relevant to the least relevant.

# Output
Respond with a list of 10 integers, where each integer is the index of the result in the original list of {num_results} results, sorted by relevance from the most relevant to the least relevant.
Do NOT list the results in any other format than the list of 10 integers.
"""

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Query: {query}\n\nResults:\n{json.dumps(results_for_model, indent=2)}\n\nReturn the indices of the {num_results} results reordered by relevance as a list of 10 integers.",
            },
        ]

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        # Track token usage
        total_input_tokens = 0
        total_output_tokens = 0
        if response.usage:
            total_input_tokens = response.usage.prompt_tokens
            total_output_tokens = response.usage.completion_tokens

        choice = response.choices[0]
        final_content = choice.message.content or "[]"

        # Log conversation
        messages.append({"role": "assistant", "content": final_content})
        self._log_conversation(query, messages)

        # Parse final response to extract results
        indices = self._parse_results(final_content, exa_results)
        results = [exa_results[i] for i in indices]

        # Collect usage stats
        query_stats: dict[str, Any] = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }
        cost = estimate_cost(self.model, total_input_tokens, total_output_tokens)
        if cost is not None:
            query_stats["cost_usd"] = cost
        if exa_cost > 0:
            query_stats["exa_cost_usd"] = exa_cost

        return SearchResponse(results=results[:num_results], query_stats=query_stats)

    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        return await self._search_impl(query, num_results)

    def _log_conversation(self, query: str, messages: list[dict[str, Any]]) -> None:
        """Log the full conversation to a file."""
        global _agent_counter
        _agent_counter += 1

        logs_dir = Path("./logs")
        logs_dir.mkdir(exist_ok=True)

        log_file = logs_dir / f"openai-orderer-{_agent_counter}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n")
            f.write(f"Model: {self.model}\n")
            f.write("=" * 80 + "\n\n")

            for msg in messages:
                role = msg.get("role", "unknown")
                f.write(f"[{role.upper()}]\n")
                f.write(msg.get("content", "") + "\n")
                f.write("\n" + "-" * 40 + "\n\n")

    def _parse_results(
        self, content: str, original_results: list[SearchResult]
    ) -> list[int]:
        """Parse the response content to extract reordered indices.

        Falls back to original results if parsing fails.
        """
        if not content.startswith("["):
            content = "[" + content
        if not content.endswith("]"):
            content = content + "]"
        try:
            indices = eval(content)
        except Exception as e:
            raise ValueError(f"Invalid indices in response: {e}") from e
        assert len(indices) == 10, f"Expected 10 indices, got {len(indices)}"
        assert all(isinstance(i, int) for i in indices), "Indices must be integers"
        assert all(0 <= i < len(original_results) for i in indices), "Indices must be within range"
        assert len(set(indices)) == len(indices), "Indices must be unique"
        return indices

    async def close(self):
        await self._client.close()

    def get_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "exa_config": self._exa_searcher.get_config(),
        }
