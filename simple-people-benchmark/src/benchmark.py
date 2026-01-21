import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from .graders import PeopleGrader
from .metrics import compute_retrieval_metrics
from .searchers import Searcher, SearchResponse, SearchResult

console = Console()
logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class Query:
    query_id: str
    text: str
    bucket: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    limit: int | None = None
    num_results: int = 10
    output_file: str | None = None
    enrich_exa_contents: bool = False
    no_grading: bool = False


def load_queries(limit: int | None = None) -> list[Query]:
    filepath = DATA_DIR / "people" / "simple_people_search.jsonl"
    if not filepath.exists():
        return []

    queries = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            queries.append(
                Query(
                    query_id=data.get("query_id", ""),
                    text=data.get("text", ""),
                    bucket=data.get("bucket", ""),
                    metadata=data.get("metadata", {}),
                )
            )

    return queries[:limit] if limit else queries


async def fetch_exa_contents(urls: list[str], api_key: str | None = None) -> dict[str, str]:
    api_key = api_key or os.getenv("EXA_API_KEY")
    if not api_key or not urls:
        return {}

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.exa.ai/contents",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={"urls": urls, "text": True, "livecrawl": "fallback"},
        )
        resp.raise_for_status()
        return {
            r["url"]: r["text"]
            for r in resp.json().get("results", [])
            if r.get("url") and r.get("text")
        }


async def enrich_results(results: list[SearchResult]) -> list[SearchResult]:
    try:
        contents = await fetch_exa_contents([r.url for r in results if r.url])
    except BaseException as e:
        logger.warning(f"Content fetch failed: {e}")
        return results
    return [SearchResult(r.url, r.title, contents.get(r.url, r.text), r.metadata) for r in results]


class Benchmark:
    def __init__(self, searchers: list[Searcher], grading_concurrency: int = 50):
        self.searchers = searchers
        self.grader = PeopleGrader()
        self._grade_semaphore = asyncio.Semaphore(grading_concurrency)

    async def _grade(self, query: Query, results: list[SearchResult]) -> list[dict]:
        async def grade_one(rank: int, r: SearchResult) -> dict:
            async with self._grade_semaphore:
                g = await self.grader.grade(query.text, r)
            return {
                "query_id": query.query_id,
                "rank": rank,
                "is_match": g.scores.get("is_match", 0),
            }

        return await asyncio.gather(*[grade_one(i, r) for i, r in enumerate(results, 1)])

    async def _run_searcher(
        self,
        searcher: Searcher,
        queries: list[Query],
        config: BenchmarkConfig,
        progress: Progress,
        task_id: TaskID,
    ) -> tuple[list[dict], list[dict]]:
        """Run searcher and return (grades, query_results).

        - grades: original format for metrics computation
        - query_results: per-query stats for output
        """
        grades: list[dict] = []
        query_results: list[dict] = []
        semaphore = asyncio.Semaphore(20)

        async def process(q: Query):
            try:
                async with semaphore:
                    start_time = time.perf_counter()
                    try:
                        # 5 minute timeout per search
                        response: SearchResponse = await asyncio.wait_for(
                            searcher.search(q.text, config.num_results),
                            timeout=300.0,
                        )
                        results = response.results
                        if config.enrich_exa_contents:
                            results = await enrich_results(results)

                        # Collect grades for metrics (skip if no_grading)
                        if config.no_grading:
                            # Add dummy grades with is_match=0 for metrics
                            query_grades = [
                                {"query_id": q.query_id, "rank": rank, "is_match": 0}
                                for rank in range(1, len(results) + 1)
                            ]
                        else:
                            query_grades = await self._grade(q, results)
                        grades.extend(query_grades)

                        # Collect per-query stats for output
                        query_data: dict[str, Any] = {
                            "query_id": q.query_id,
                            "num_results": len(results),
                        }
                        if response.query_stats:
                            query_data.update(response.query_stats)

                        # Add grading results
                        matches = sum(1 for g in query_grades if g.get("is_match", 0) >= 1.0)
                        query_data["matches"] = matches
                        query_data["grades"] = [
                            {"rank": g["rank"], "is_match": g["is_match"]} for g in query_grades
                        ]

                        query_data["elapsed_s"] = round(time.perf_counter() - start_time, 3)
                        query_results.append(query_data)
                    except asyncio.TimeoutError:
                        console.print(
                            f"[red]Search timed out after 5min for query: '{q.text}'[/red]"
                        )
                        # Add timeout error grade for metrics
                        grades.append(
                            {
                                "query_id": q.query_id,
                                "rank": 0,
                                "is_match": 0,
                                "error": "Search timed out after 5 minutes",
                                "error_type": "TimeoutError",
                            }
                        )
                        # Add timeout error to query_results
                        query_results.append(
                            {
                                "query_id": q.query_id,
                                "num_results": 0,
                                "error": "Search timed out after 5 minutes",
                                "error_type": "TimeoutError",
                                "elapsed_s": round(time.perf_counter() - start_time, 3),
                            }
                        )
                    except BaseException as e:
                        console.print(
                            f"[red]Searcher {searcher.name} failed: '{e}' for query: '{q.text}'[/red]"
                        )
                        # Add error grade for metrics (rank=0, is_match=0)
                        grades.append(
                            {
                                "query_id": q.query_id,
                                "rank": 0,
                                "is_match": 0,
                                "error": str(e),
                                "error_type": type(e).__name__,
                            }
                        )
                        # Add error entry to query_results
                        query_results.append(
                            {
                                "query_id": q.query_id,
                                "num_results": 0,
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "elapsed_s": round(time.perf_counter() - start_time, 3),
                            }
                        )
            finally:
                progress.advance(task_id)

        await asyncio.gather(*[process(q) for q in queries])
        return grades, query_results

    async def run(self, config: BenchmarkConfig | None = None) -> dict[str, Any]:
        config = config or BenchmarkConfig()
        queries = load_queries(config.limit)

        if not queries:
            console.print("[red]No queries found![/red]")
            console.print("Make sure data/people/simple_people_search.jsonl exists.")
            return {}

        console.print("\n[bold]People Search Benchmark[/bold]")
        console.print(f"  Searchers: {[s.name for s in self.searchers]}")
        console.print(f"  Queries: {len(queries)}")
        console.print(f"  Exa enrichment: {'on' if config.enrich_exa_contents else 'off'}")
        console.print()

        results: dict[str, Any] = {
            "config": {
                "limit": config.limit,
                "num_results": config.num_results,
                "enrich_exa_contents": config.enrich_exa_contents,
                "grader": self.grader.get_config(),
            },
            "searchers": {},
        }

        with Progress(
            TextColumn("[cyan]{task.fields[name]:>10}[/cyan]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            tasks = {
                s.name: progress.add_task("", name=s.name, total=len(queries))
                for s in self.searchers
            }

            async def run_one(searcher: Searcher) -> tuple[str, list[dict], list[dict], dict]:
                grades, query_results = await self._run_searcher(
                    searcher, queries, config, progress, tasks[searcher.name]
                )
                return searcher.name, grades, query_results, searcher.get_config()

            all_results = await asyncio.gather(
                *[run_one(s) for s in self.searchers], return_exceptions=True
            )

        for item in all_results:
            if isinstance(item, BaseException):
                console.print(f"[red]Error running searcher {item.searcher.name}: {item}[/red]")
                continue
            name, grades, query_results, searcher_config = item
            results["searchers"][name] = {
                "config": searcher_config,
                "metrics": compute_retrieval_metrics(grades).__dict__,
                "queries": query_results,
            }

        _print_summary(results)

        if config.output_file:
            with open(config.output_file, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"\n[green]Saved to {config.output_file}[/green]")

        return results


def _print_summary(results: dict[str, Any]):
    console.print("\n[bold]Results[/bold]\n")
    searchers = results.get("searchers", {})

    if not searchers:
        return

    t = Table(title="People Search")
    t.add_column("Searcher", style="cyan")
    for col in ["R@1", "R@10", "Precision", "Queries"]:
        t.add_column(col, justify="right")

    for name, data in searchers.items():
        m = data.get("metrics", {})
        t.add_row(
            name,
            f"{m.get('match', 0):.1%}",
            f"{m.get('recall_at_10', 0):.1%}",
            f"{m.get('precision', 0):.1%}",
            str(m.get("num_queries", 0)),
        )

    console.print(t)


def _build_searcher(
    name: str, shared_exa: Searcher | None = None, logs_dir: str | None = None
) -> Searcher | None:
    try:
        if name == "exa":
            from .searchers.exa import ExaSearcher

            return ExaSearcher(category="people")
        if name == "brave":
            from .searchers.brave import BraveSearcher

            return BraveSearcher(site_filter="linkedin.com/in")
        if name == "parallel":
            from .searchers.parallel import ParallelSearcher

            return ParallelSearcher(source_policy={"include_domains": ["linkedin.com"]})
        if name == "agentica":
            from .searchers.agentica import AgenticaSearcher

            return AgenticaSearcher(category="people", logs_dir=logs_dir)
        if name == "agentica_orderer":
            from .searchers.agentica_orderer import AgenticaOrderer

            if shared_exa is None:
                raise ValueError("agentica_orderer requires shared_exa searcher")
            return AgenticaOrderer(exa_searcher=shared_exa, logs_dir=logs_dir)
        if name == "openai":
            from .searchers.openai_searcher import OpenAISearcher

            return OpenAISearcher(category="people")
        if name == "openai_orderer":
            from .searchers.openai_orderer import OpenAIOrderer

            if shared_exa is None:
                raise ValueError("openai_orderer requires shared_exa searcher")
            return OpenAIOrderer(exa_searcher=shared_exa)
    except (ValueError, ImportError) as e:
        console.print(f"[yellow]{name}: {e}[/yellow]")
    return None


def main():
    queries_exist = (DATA_DIR / "people" / "simple_people_search.jsonl").exists()

    if not queries_exist:
        console.print("[red]No benchmark data found![/red]")
        console.print("\nDownload the people dataset:")
        console.print(
            "  curl -L https://github.com/exa-labs/people-benchmark/releases/latest/download/data.tar.gz | tar xz"
        )
        return

    parser = argparse.ArgumentParser(description="People Search Benchmark")
    parser.add_argument("--limit", type=int, help="Limit number of queries")
    parser.add_argument("--num-results", type=int, default=10, help="Results per query")
    parser.add_argument("--output", "-o", help="Output file for results JSON")
    parser.add_argument(
        "--enrich-exa-contents", action="store_true", help="Fetch page contents via Exa API"
    )
    parser.add_argument(
        "--searchers", nargs="+", help="Searchers to use (default: exa, brave, parallel)"
    )
    parser.add_argument(
        "--no-grading",
        action="store_true",
        help="Skip grading (no OpenAI calls, all results scored as 0)",
    )
    args = parser.parse_args()

    searcher_names = args.searchers or ["exa", "brave", "parallel"]

    # Create shared caching searcher for orderers to ensure they get identical results
    orderer_names = {"agentica_orderer", "openai_orderer"}
    has_orderers = any(name in orderer_names for name in searcher_names)
    shared_exa: Searcher | None = None

    if has_orderers:
        from .searchers.exa import CachingSearcher, ExaSearcher

        shared_exa = CachingSearcher(ExaSearcher(category="people"))
        console.print("[dim]Using shared caching searcher for orderers[/dim]")

    # Compute logs_dir from output file: strip extension, use as directory name
    logs_dir: str | None = None
    if args.output:
        logs_dir = "./" + Path(args.output).stem

    searchers = [s for name in searcher_names if (s := _build_searcher(name, shared_exa, logs_dir))]

    if not searchers:
        console.print("[red]No searchers available![/red]")
        if shared_exa:
            asyncio.run(shared_exa.close())
        return

    config = BenchmarkConfig(
        limit=args.limit,
        num_results=args.num_results,
        output_file=args.output,
        enrich_exa_contents=args.enrich_exa_contents,
        no_grading=args.no_grading,
    )

    async def run_benchmark():
        try:
            await Benchmark(searchers).run(config)
        finally:
            if shared_exa:
                await shared_exa.close()

    asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()
