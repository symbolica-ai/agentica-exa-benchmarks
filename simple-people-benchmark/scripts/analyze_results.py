#!/usr/bin/env python3
"""Analyze benchmark results and print a summary table."""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


def analyze_results(results_file: str) -> None:
    path = Path(results_file)
    if not path.exists():
        print(f"Error: File not found: {results_file}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    console = Console()
    table = Table(title="Benchmark Results", expand=False)

    # Add columns with minimum widths to prevent truncation
    table.add_column("Searcher", style="cyan", min_width=10)
    table.add_column("R@1", justify="right", min_width=5)
    table.add_column("R@10", justify="right", min_width=5)
    table.add_column("Prec", justify="right", min_width=5)
    table.add_column("Fail", justify="right", min_width=4)
    table.add_column("∑ Time", justify="right", min_width=6)
    table.add_column("Est. Wall", justify="right", min_width=8)
    table.add_column("Min", justify="right", min_width=5)
    table.add_column("Max", justify="right", min_width=5)
    table.add_column("Avg", justify="right", min_width=6)
    table.add_column("LLM Cost", justify="right", min_width=7)
    table.add_column("LLM Cost/q", justify="right", min_width=7)
    table.add_column("Exa Cost", justify="right", min_width=7)
    table.add_column("Exa Cost/q", justify="right", min_width=7)
    table.add_column("Total Cost", justify="right", min_width=7)
    table.add_column("Srch/q", justify="right", min_width=6)
    table.add_column("Tokens/q", justify="right", min_width=12)

    searchers = data.get("searchers", {})

    for name, searcher_data in searchers.items():
        metrics = searcher_data.get("metrics", {})
        queries = searcher_data.get("queries", [])

        # Basic metrics
        r_at_1 = metrics.get("match", 0)
        r_at_10 = metrics.get("recall_at_10", 0)
        precision = metrics.get("precision", 0)
        num_queries = metrics.get("num_queries", 0)

        # Calculate stats from queries
        failed_queries = sum(1 for q in queries if "error" in q)
        elapsed_times = [q.get("elapsed_s", 0) for q in queries]
        successful_elapsed_times = [q.get("elapsed_s", 0) for q in queries if "error" not in q]
        total_elapsed = sum(elapsed_times)
        min_elapsed = min(successful_elapsed_times) if successful_elapsed_times else 0
        max_elapsed = max(successful_elapsed_times) if successful_elapsed_times else 0
        avg_elapsed = (
            sum(successful_elapsed_times) / len(successful_elapsed_times)
            if successful_elapsed_times
            else 0
        )

        # Estimate wall-clock time (assuming concurrency of 20)
        concurrency = 20
        est_wall_time = total_elapsed / concurrency if total_elapsed > 0 else 0

        # LLM cost stats (may not be present for all searchers)
        total_llm_cost = sum(q.get("cost_usd", 0) for q in queries)
        queries_with_llm_cost = [q for q in queries if "cost_usd" in q]
        avg_llm_cost = total_llm_cost / len(queries_with_llm_cost) if queries_with_llm_cost else 0

        # Exa cost stats
        total_exa_cost = sum(q.get("exa_cost_usd", 0) for q in queries)
        queries_with_exa_cost = [q for q in queries if "exa_cost_usd" in q]
        avg_exa_cost = total_exa_cost / len(queries_with_exa_cost) if queries_with_exa_cost else 0

        # Total cost (LLM + Exa)
        grand_total_cost = total_llm_cost + total_exa_cost

        # Search calls (agentica-specific)
        queries_with_searches = [q for q in queries if "search_calls" in q]
        avg_searches = (
            sum(q.get("search_calls", 0) for q in queries_with_searches)
            / len(queries_with_searches)
            if queries_with_searches
            else 0
        )

        # Token stats (agentica-specific)
        queries_with_tokens = [q for q in queries if "input_tokens" in q]
        if queries_with_tokens:
            avg_input = sum(q.get("input_tokens", 0) for q in queries_with_tokens) / len(
                queries_with_tokens
            )
            avg_output = sum(q.get("output_tokens", 0) for q in queries_with_tokens) / len(
                queries_with_tokens
            )
            token_str = f"{avg_input:.0f} / {avg_output:.0f}"
        else:
            token_str = "-"

        # Format wall time as mm:ss if > 60s
        def format_time(seconds: float) -> str:
            if seconds >= 60:
                mins = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{mins}:{secs:02d}"
            return f"{seconds:.1f}s"

        # Format row
        table.add_row(
            name,
            f"{r_at_1:.1%}",
            f"{r_at_10:.1%}",
            f"{precision:.1%}",
            str(failed_queries),
            format_time(total_elapsed),
            format_time(est_wall_time),
            f"{min_elapsed:.1f}s",
            f"{max_elapsed:.1f}s",
            f"{avg_elapsed:.2f}s",
            f"${total_llm_cost:.4f}" if total_llm_cost > 0 else "-",
            f"${avg_llm_cost:.4f}" if avg_llm_cost > 0 else "-",
            f"${total_exa_cost:.4f}" if total_exa_cost > 0 else "-",
            f"${avg_exa_cost:.4f}" if avg_exa_cost > 0 else "-",
            f"${grand_total_cost:.4f}" if grand_total_cost > 0 else "-",
            f"{avg_searches:.1f}" if avg_searches > 0 else "-",
            token_str,
        )

    console.print(table)

    # Print config info
    config = data.get("config", {})
    console.print(
        f"\n[dim]Config: limit={config.get('limit')}, num_results={config.get('num_results')}[/dim]"
    )
    console.print("[dim]Note: Est. Wall assumes concurrency=20[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument(
        "results_file", nargs="?", default="results.json", help="Path to results.json"
    )
    args = parser.parse_args()

    analyze_results(args.results_file)


if __name__ == "__main__":
    main()
