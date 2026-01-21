from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


def get_model_pricing(model: str) -> tuple[float, float] | None:
    """Get pricing per million tokens for a model: (input_cost, output_cost).

    Returns None if model is not found in pricing table.
    """
    # Pricing per million tokens: (input_cost, output_cost)
    pricing = {
        # Google
        "gemini-3-flash-preview": (0.50, 3.00),
        "gemini-2.5-flash-lite": (0.10, 0.40),
        "gemini-2.5-flash": (0.30, 2.50),
        # Meta
        "llama-4-scout": (0.08, 0.30),
        # Qwen
        "qwen3-235b-a22b-thinking-2507": (0.11, 0.60),
        "qwen3-30b-a3b-instruct-2507": (0.08, 0.33),
        # OpenAI
        "gpt-5-nano": (0.05, 0.40),
    }
    for model_key, prices in pricing.items():
        if model_key in model:
            return prices
    return None


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate cost in USD based on model and token counts."""
    pricing = get_model_pricing(model)
    if pricing is None:
        return None
    input_price, output_price = pricing
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


@dataclass
class SearchResult:
    url: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Response from a search query, with optional per-query stats."""

    results: list[SearchResult]
    query_stats: dict[str, Any] = field(default_factory=dict)
    """Optional per-query stats like cost, latency, tokens used, etc."""


class Searcher(ABC):
    name: str = "base"

    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        pass

    def get_config(self) -> dict[str, Any]:
        """Return searcher configuration for logging/results."""
        return {}
