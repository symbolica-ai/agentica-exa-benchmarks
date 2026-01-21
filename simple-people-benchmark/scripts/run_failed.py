#!/usr/bin/env python3
"""Rerun failed queries from a benchmark results file."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from src.benchmark import Query, load_queries
from src.graders import PeopleGrader
from src.metrics import compute_retrieval_metrics
from src.searchers import Searcher
from src.searchers.exa import CachingSearcher, ExaSearcher

console = Console()


def load_results(path: str) -> dict:
    """Load results from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_failed_queries(
    searcher_data: dict,
    exclude_errors: list[str],
) -> list[str]:
    """Get query IDs that failed with errors not in the exclude list."""
    failed = []
    for query in searcher_data.get("queries", []):
        error_type = query.get("error_type")
        if error_type and error_type not in exclude_errors:
            failed.append(query["query_id"])
    return failed


def build_searcher(name: str, config: dict, shared_exa: Searcher | None = None) -> Searcher | None:
    """Build a searcher from its name and config."""
    try:
        if name == "exa":
            from src.searchers.exa import ExaSearcher

            return ExaSearcher(
                category=config.get("category"),
                search_type=config.get("search_type", "fast"),
            )
        if name == "agentica":
            from src.searchers.agentica import AgenticaSearcher

            return AgenticaSearcher(
                model=config.get("model"),
                category=config.get("category"),
            )
        if name == "agentica_orderer":
            from src.searchers.agentica_orderer import AgenticaOrderer

            if shared_exa is None:
                raise ValueError("agentica_orderer requires shared_exa")
            return AgenticaOrderer(
                exa_searcher=shared_exa,
                model=config.get("model"),
            )
        if name == "openai":
            from src.searchers.openai_searcher import OpenAISearcher

            return OpenAISearcher(
                model=config.get("model"),
                category=config.get("category"),
            )
        if name == "openai_orderer":
            from src.searchers.openai_orderer import OpenAIOrderer

            if shared_exa is None:
                raise ValueError("openai_orderer requires shared_exa")
            return OpenAIOrderer(
                exa_searcher=shared_exa,
                model=config.get("model"),
            )
    except (ValueError, ImportError) as e:
        console.print(f"[yellow]Cannot build {name}: {e}[/yellow]")
    return None


async def rerun_queries(
    searcher: Searcher,
    queries: list[Query],
    num_results: int,
    grader: PeopleGrader,
    progress: Progress,
    task_id: Any,
) -> tuple[list[dict], list[dict]]:
    """Rerun queries and return (grades, query_results)."""
    grades: list[dict] = []
    query_results: list[dict] = []
    semaphore = asyncio.Semaphore(20)
    grade_semaphore = asyncio.Semaphore(50)

    async def grade_one(query: Query, rank: int, result: Any) -> dict:
        async with grade_semaphore:
            g = await grader.grade(query.text, result)
        return {"query_id": query.query_id, "rank": rank, "is_match": g.scores.get("is_match", 0)}

    async def process(q: Query):
        try:
            async with semaphore:
                start_time = time.perf_counter()
                try:
                    response = await asyncio.wait_for(
                        searcher.search(q.text, num_results),
                        timeout=300.0,
                    )
                    results = response.results

                    # Grade results
                    query_grades = await asyncio.gather(
                        *[grade_one(q, i, r) for i, r in enumerate(results, 1)]
                    )
                    grades.extend(query_grades)

                    # Collect per-query stats
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
                    console.print(f"[red]Timeout for query: {q.query_id}[/red]")
                    grades.append(
                        {
                            "query_id": q.query_id,
                            "rank": 0,
                            "is_match": 0,
                            "error": "Search timed out after 5 minutes",
                            "error_type": "TimeoutError",
                        }
                    )
                    query_results.append(
                        {
                            "query_id": q.query_id,
                            "num_results": 0,
                            "error": "Search timed out after 5 minutes",
                            "error_type": "TimeoutError",
                            "elapsed_s": round(time.perf_counter() - start_time, 3),
                        }
                    )
                except Exception as e:
                    console.print(f"[red]Error for {q.query_id}: {e}[/red]")
                    grades.append(
                        {
                            "query_id": q.query_id,
                            "rank": 0,
                            "is_match": 0,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        }
                    )
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


async def main_async(args: argparse.Namespace) -> int:
    results_path = Path(args.results_file)
    if not results_path.exists():
        console.print(f"[red]File not found: {results_path}[/red]")
        return 1

    results = load_results(results_path)
    exclude_errors = args.exclude or []

    console.print("[bold]Rerunning failed queries[/bold]")
    console.print(f"  Input: {results_path}")
    console.print(f"  Exclude errors: {exclude_errors}")
    console.print()

    # Load all queries from the dataset
    all_queries = load_queries()
    query_map = {q.query_id: q for q in all_queries}

    config = results.get("config", {})
    num_results = config.get("num_results", 10)
    grader = PeopleGrader()

    # Check if we need shared exa for orderers
    orderer_names = {"agentica_orderer", "openai_orderer"}
    searcher_names = list(results.get("searchers", {}).keys())
    needs_shared_exa = any(name in orderer_names for name in searcher_names)

    shared_exa: Searcher | None = None
    if needs_shared_exa:
        shared_exa = CachingSearcher(ExaSearcher(category="people"))
        console.print("[dim]Using shared caching searcher for orderers[/dim]")

    try:
        # Process each searcher
        for searcher_name, searcher_data in results.get("searchers", {}).items():
            failed_query_ids = get_failed_queries(searcher_data, exclude_errors)

            if not failed_query_ids:
                console.print(f"[green]{searcher_name}: No failed queries to rerun[/green]")
                continue

            console.print(
                f"[cyan]{searcher_name}: {len(failed_query_ids)} failed queries to rerun[/cyan]"
            )

            # Build the searcher
            searcher = build_searcher(searcher_name, searcher_data.get("config", {}), shared_exa)
            if searcher is None:
                console.print(
                    f"[yellow]Skipping {searcher_name}: could not build searcher[/yellow]"
                )
                continue

            # Get Query objects for failed queries
            failed_queries = [query_map[qid] for qid in failed_query_ids if qid in query_map]
            if not failed_queries:
                console.print(
                    f"[yellow]{searcher_name}: Could not find query data for failed queries[/yellow]"
                )
                continue

            # Rerun failed queries with progress
            with Progress(
                TextColumn("[cyan]{task.fields[name]:>15}[/cyan]"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task("", name=searcher_name, total=len(failed_queries))
                new_grades, new_query_results = await rerun_queries(
                    searcher, failed_queries, num_results, grader, progress, task_id
                )

            # Close searcher
            await searcher.close()

            # Merge results: replace failed queries with new results
            new_results_map = {q["query_id"]: q for q in new_query_results}
            merged_queries = []
            for query in searcher_data.get("queries", []):
                query_id = query["query_id"]
                if query_id in new_results_map:
                    merged_queries.append(new_results_map[query_id])
                else:
                    merged_queries.append(query)

            # Update searcher data
            searcher_data["queries"] = merged_queries

            # Recompute metrics from all grades (existing + new)
            # We need to rebuild grades from all query results
            all_grades = []
            for query in merged_queries:
                query_id = query["query_id"]
                grades_list = query.get("grades", [])
                for g in grades_list:
                    all_grades.append(
                        {
                            "query_id": query_id,
                            "rank": g["rank"],
                            "is_match": g["is_match"],
                        }
                    )
                # If query has error, add a dummy grade
                if query.get("error"):
                    all_grades.append(
                        {
                            "query_id": query_id,
                            "rank": 0,
                            "is_match": 0,
                        }
                    )

            # Recompute metrics
            new_metrics = compute_retrieval_metrics(all_grades)
            searcher_data["metrics"] = new_metrics.__dict__

            # Count remaining failures
            remaining_failures = sum(1 for q in merged_queries if q.get("error"))
            console.print(
                f"  [green]Reran {len(failed_queries)} queries, {remaining_failures} still failing[/green]"
            )

    finally:
        if shared_exa:
            await shared_exa.close()

    # Save output
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]Saved to {output_path}[/green]")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Rerun failed queries from benchmark results.")
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to the results JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file for merged results",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=[],
        help="Error types to exclude from rerun (e.g., TimeoutError)",
    )

    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    exit(main())
