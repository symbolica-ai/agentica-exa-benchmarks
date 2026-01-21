import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .base import Searcher, SearchResponse, SearchResult, estimate_cost
from .exa import ExaSearcher


class SearchResultSchema(BaseModel):
    """Pydantic model for validating search result output."""

    url: str
    title: str
    text: str


# Global counter for agent runs
_agent_counter = 0


# Tool definitions for function calling
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information using Exa search engine. Can only be called once.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "The search query to find relevant results.",
                }
            },
            "required": ["search_query"],
        },
    },
}


class OpenAISearcher(Searcher):
    name = "openai"

    def __init__(
        self,
        exa_api_key: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        category: str | None = "people",
    ):
        self.exa_api_key = exa_api_key or os.getenv("EXA_API_KEY")
        if not self.exa_api_key:
            raise ValueError("EXA_API_KEY required - get one at https://exa.ai")

        self.model = model or os.getenv("OPENAI_MODEL", "google/gemini-3-flash-preview")
        self.category = category

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

        self._exa_searcher = ExaSearcher(
            api_key=self.exa_api_key,
            category=self.category,
        )
        print(f"[openai] Using model: {self.model}")

    async def _search_impl(self, query: str, num_results: int = 10) -> SearchResponse:
        search_call_count = 0
        exa_total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0

        async def exa_search(search_query: str) -> list[dict[str, Any]]:
            """
            Execute search via Exa and return results as dicts. Can only be called once.
            """
            nonlocal search_call_count, exa_total_cost
            search_call_count += 1
            if search_call_count > 1:
                raise ValueError("Exa search can only be called once.")
            response = await self._exa_searcher.search(search_query, num_results)
            if "exa_cost_usd" in response.query_stats:
                exa_total_cost += response.query_stats["exa_cost_usd"]
            # Convert to dicts for JSON serialization
            return [
                {
                    "url": r.url,
                    "title": r.title,
                    "text": r.text if r.text else "",
                }
                for r in response.results
            ]

        system_prompt = f"""
You are a search assistant that helps find people's professional profiles.

# Task
You will be given a query and you need to find the {num_results} most relevant people's professional profiles matching the query.

# Tools
Use the `search` tool to find the results. The `search` tool returns exactly {num_results} results. Inspect the output of the `search` tool systematically.
The `search` tool can only be called once.

# Output
When you have the final results, respond with a JSON array of the {num_results} most relevant results, where each result in the array is an object of the following schema:
{{
  "type": "object",
  "properties": {{
    "url": {{
      "type": "string"
    }},
    "title": {{
      "type": "string"
    }},
    "text": {{
      "type": "string"
    }}
  }},
  "required": ["url", "title", "text"],
  "additionalProperties": false
}}

Do NOT list the results in any other format than the JSON array.
"""

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Query: {query}\nReturn the {num_results} most relevant results matching the query as a JSON array.",
            },
        ]

        # Agentic loop - keep calling until we get a final response
        try:
            while True:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=[SEARCH_TOOL],
                    tool_choice="auto",
                )

                # Track token usage
                if response.usage:
                    total_input_tokens += response.usage.prompt_tokens
                    total_output_tokens += response.usage.completion_tokens

                choice = response.choices[0]
                assistant_message = choice.message

                # Add assistant response to messages
                messages.append(assistant_message.model_dump())

                # Check if we're done (no tool calls)
                if not assistant_message.tool_calls:
                    break

                # Process tool calls
                for tool_call in assistant_message.tool_calls:
                    if tool_call.function.name == "search":
                        args = json.loads(tool_call.function.arguments)
                        search_query = args.get("search_query", query)
                        search_results = await exa_search(search_query)

                        # Add tool result to messages
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(search_results),
                            }
                        )
        finally:
            self._log_conversation(query, messages)

        # Parse final response to extract results
        final_content = assistant_message.content or "[]"
        results = self._parse_results(final_content)

        # Collect usage stats
        query_stats: dict[str, Any] = {
            "search_calls": search_call_count,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }
        cost = estimate_cost(self.model, total_input_tokens, total_output_tokens)
        if cost is not None:
            query_stats["cost_usd"] = cost
        if exa_total_cost > 0:
            query_stats["exa_cost_usd"] = exa_total_cost

        return SearchResponse(results=results[:num_results], query_stats=query_stats)

    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        return await self._search_impl(query, num_results)

    def _log_conversation(self, query: str, messages: list[dict[str, Any]]) -> None:
        """Log the full agent conversation to a file."""
        global _agent_counter
        _agent_counter += 1

        logs_dir = Path("./logs")
        logs_dir.mkdir(exist_ok=True)

        log_file = logs_dir / f"openai-agent-{_agent_counter}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n")
            f.write(f"Model: {self.model}\n")
            f.write("=" * 80 + "\n\n")

            for msg in messages:
                role = msg.get("role", "unknown")
                f.write(f"[{role.upper()}]\n")

                if role == "tool":
                    tool_call_id = msg.get("tool_call_id", "")
                    f.write(f"Tool Call ID: {tool_call_id}\n")
                    # Pretty print the tool result (search results)
                    try:
                        content = json.loads(msg.get("content", "[]"))
                        f.write(json.dumps(content, indent=2))
                    except json.JSONDecodeError:
                        f.write(msg.get("content", ""))
                elif "tool_calls" in msg and msg["tool_calls"]:
                    # Assistant message with tool calls
                    if msg.get("content"):
                        f.write(f"{msg['content']}\n")
                    f.write("Tool Calls:\n")
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        f.write(
                            f"  - {func.get('name', 'unknown')}({func.get('arguments', '{}')})\n"
                        )
                else:
                    f.write(msg.get("content", "") + "\n")

                f.write("\n" + "-" * 40 + "\n\n")

    def _parse_results(self, content: str) -> list[SearchResult]:
        """Parse the final response content to extract SearchResult objects.

        Raises:
            ValueError: If JSON cannot be extracted or doesn't match expected schema.
        """
        # Look for JSON array in the content
        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end <= start:
            raise ValueError(f"No JSON array found in response: {content[:200]}")

        json_str = content[start:end]
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}") from e

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

        results = []
        for i, item in enumerate(data):
            try:
                validated = SearchResultSchema.model_validate(item)
                results.append(
                    SearchResult(
                        url=validated.url,
                        title=validated.title,
                        text=validated.text,
                    )
                )
            except ValidationError as e:
                raise ValueError(f"Invalid result at index {i}: {e}") from e

        return results

    async def close(self):
        await self._exa_searcher.close()
        await self._client.close()

    def get_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "category": self.category,
            "exa_config": self._exa_searcher.get_config(),
        }
