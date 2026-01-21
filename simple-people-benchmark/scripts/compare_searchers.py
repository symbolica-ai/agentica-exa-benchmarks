#!/usr/bin/env python3
"""Compare searchers from benchmark results - per-query analysis."""

import argparse
import json
from pathlib import Path


def load_results(path: str) -> dict:
    """Load results from JSON file."""
    with open(path) as f:
        return json.load(f)


def format_value(value: float | None, decimals: int = 2) -> str:
    """Format a numeric value for display."""
    if value is None:
        return "-"
    if isinstance(value, int) or value == int(value):
        return str(int(value))
    return f"{value:.{decimals}f}"


def compute_query_metrics(query_data: dict) -> dict:
    """Compute R@1, R@10, Precision for a single query from grades."""
    grades = query_data.get("grades", [])
    if not grades:
        return {"r_at_1": None, "r_at_10": None, "precision": None}

    # Sort by rank
    sorted_grades = sorted(grades, key=lambda g: g.get("rank", 0))

    # Find first match
    first_match_rank = None
    for g in sorted_grades:
        if g.get("is_match", 0) >= 1.0:
            first_match_rank = g.get("rank", sorted_grades.index(g) + 1)
            break

    # R@1: 1 if first result is a match
    r_at_1 = 1.0 if first_match_rank == 1 else 0.0

    # R@10: 1 if any result in top 10 is a match
    r_at_10 = 1.0 if first_match_rank and first_match_rank <= 10 else 0.0

    # Precision: fraction of results that are matches
    n_results = len(sorted_grades)
    n_matches = sum(1 for g in sorted_grades if g.get("is_match", 0) >= 1.0)
    precision = n_matches / n_results if n_results > 0 else 0.0

    return {"r_at_1": r_at_1, "r_at_10": r_at_10, "precision": precision}


def print_query_comparison_table(results: dict, searchers_to_compare: list[str] | None = None):
    """Print a per-query comparison table for the specified searchers."""
    searchers_data = results.get("searchers", {})

    # Filter searchers if specified
    if searchers_to_compare:
        searchers_data = {k: v for k, v in searchers_data.items() if k in searchers_to_compare}

    if not searchers_data:
        print("No searchers found to compare.")
        return

    searcher_names = sorted(searchers_data.keys())

    # Build query index: query_id -> {searcher_name -> query_data}
    query_index: dict[str, dict[str, dict]] = {}
    for searcher_name, searcher_data in searchers_data.items():
        for query in searcher_data.get("queries", []):
            query_id = query.get("query_id")
            if query_id:
                if query_id not in query_index:
                    query_index[query_id] = {}
                # Add computed metrics to query data
                query_with_metrics = dict(query)
                query_with_metrics.update(compute_query_metrics(query))
                query_index[query_id][searcher_name] = query_with_metrics

    # Sort query IDs
    sorted_query_ids = sorted(query_index.keys())

    # Define metrics columns
    metrics = [
        ("num_results", "results", 0),
        ("search_calls", "searches", 0),
        ("elapsed_s", "time_s", 2),
        ("r_at_1", "R@1", 1),
        ("r_at_10", "R@10", 1),
        ("precision", "Prec", 2),
    ]

    # Calculate column widths
    query_id_width = max(
        len("query_id"), max(len(qid) for qid in sorted_query_ids) if sorted_query_ids else 10
    )
    col_width = 10

    # Build header - group searchers under each metric
    header_parts = [f"{'query_id':<{query_id_width}}"]
    for field, short_name, _ in metrics:
        for searcher in searcher_names:
            # Abbreviate searcher name if needed
            abbrev = searcher[:col_width]
            header_parts.append(f"{abbrev:>{col_width}}")

    # Build metric group header
    metric_header_parts = [" " * query_id_width]
    for field, short_name, _ in metrics:
        group_width = col_width * len(searcher_names) + 2 * (len(searcher_names) - 1)
        metric_header_parts.append(f"{short_name:^{group_width}}")

    print("  ".join(metric_header_parts))
    header = "  ".join(header_parts)
    print(header)
    print("-" * len(header))

    # Print rows
    totals: dict[str, dict[str, float]] = {s: {} for s in searcher_names}
    counts: dict[str, dict[str, int]] = {s: {} for s in searcher_names}

    for query_id in sorted_query_ids:
        row_parts = [f"{query_id:<{query_id_width}}"]

        for field, _, decimals in metrics:
            for searcher in searcher_names:
                query_data = query_index[query_id].get(searcher, {})
                value = query_data.get(field)
                row_parts.append(f"{format_value(value, decimals):>{col_width}}")

                # Accumulate totals
                if value is not None:
                    if field not in totals[searcher]:
                        totals[searcher][field] = 0
                        counts[searcher][field] = 0
                    totals[searcher][field] += value
                    counts[searcher][field] += 1

        print("  ".join(row_parts))

    # Print totals row
    print("-" * len(header))
    totals_parts = [f"{'TOTAL':<{query_id_width}}"]
    for field, _, decimals in metrics:
        for searcher in searcher_names:
            value = totals[searcher].get(field)
            totals_parts.append(f"{format_value(value, decimals):>{col_width}}")
    print("  ".join(totals_parts))

    # Print averages row
    avg_parts = [f"{'AVG':<{query_id_width}}"]
    for field, _, decimals in metrics:
        for searcher in searcher_names:
            total = totals[searcher].get(field)
            count = counts[searcher].get(field, 0)
            if total is not None and count > 0:
                avg_parts.append(f"{format_value(total / count, decimals):>{col_width}}")
            else:
                avg_parts.append(f"{'-':>{col_width}}")
    print("  ".join(avg_parts))


def main():
    parser = argparse.ArgumentParser(
        description="Compare searchers from benchmark results - per-query analysis."
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to the results JSON file",
    )
    parser.add_argument(
        "--searchers",
        type=str,
        nargs="+",
        help="Searchers to compare (include in the table). If not specified, all searchers are included.",
    )

    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: File not found: {results_path}")
        return 1

    results = load_results(results_path)

    print(f"Results from: {results_path.name}")
    config = results.get("config", {})
    if config:
        print(f"Config: limit={config.get('limit')}, num_results={config.get('num_results')}")
    print()

    print_query_comparison_table(results, args.searchers)

    return 0


if __name__ == "__main__":
    exit(main())
