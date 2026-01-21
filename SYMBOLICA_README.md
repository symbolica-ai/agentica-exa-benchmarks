# Symbolica Agentica - People Search Benchmark Guide

This guide covers how to run the `pbench` benchmark tool with Symbolica Agentica-based searchers and analyze the results.

## Prerequisites

### Environment Variables

```bash
# Required for all searchers
export EXA_API_KEY="your-exa-api-key"

# Required for LLM grading (evaluating results)
export OPENAI_API_KEY="your-openai-api-key"

# Required for agentica searchers
export AGENTICA_API_KEY="your-agentica-api-key"

# Optional: Configure agentica model (default: openrouter:google/gemini-3-flash-preview)
# Model name should be prefixed with `openrouter:`
export AGENTICA_MODEL="openrouter:google/gemini-3-flash-preview"
```

### Installation

```bash
cd simple-people-benchmark

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Running Benchmarks

### Basic Usage

First `source ./venv/bin/activate` or prepend **all** the commands below with `uv run` (i.e. `uv run pbench` instead of `pbench`).

```bash
# Run with default searchers (exa, brave, parallel)
pbench

# Run with specific searchers
pbench --searchers agentica

# Run with multiple searchers for comparison
pbench --searchers exa agentica agentica_orderer

# Limit queries for quick testing
pbench --searchers agentica --limit 10

# Save results to file
pbench --searchers agentica -o results.json
```

NOTE: If you don't specify `-o` path, results **are not going to be saved on the disk** and it won't be possible to retrieve the evaluation details. So it's most likely you'd want to specify `-o` for your runs.

### Available Searchers

| Searcher | Description |
|----------|-------------|
| `exa` | Direct Exa API search with `category="people"` |
| `brave` | Brave Search filtered to `linkedin.com/in` |
| `parallel` | Internal parallel searcher filtered to LinkedIn |
| `agentica` | Agentica agent that constructs and executes Exa search |
| `agentica_orderer` | Agentica agent that only reorders provided Exa results by relevance |
| `openai` | OpenAI-based agent searcher |
| `openai_orderer` | OpenAI-based agent that only reorders provided Exa results by relevance |

### CLI Options

```bash
pbench --help

Options:
  --limit N              Limit number of queries (default: all 1400)
  --num-results N        Results per query (default: 10)
  --output, -o FILE      Save results to JSON file
  --enrich-exa-contents  Fetch full page contents via Exa API
  --searchers NAME...    Searchers to use
  --no-grading           Skip LLM grading (results scored as 0)
```

### Agent Logs

When using `agentica` or `agentica_orderer` searchers, agent conversation logs are saved to a directory:

- **With `-o` option**: Logs go to `./<output_stem>/` (e.g., `-o results.json` → `./results/`)
- **Without `-o` option**: Logs go to `./logs/`

Log files are named `agent-0.log`, `agent-1.log`, etc. and contain the full agent conversation history.

## Analyzing Results

### Summary Table

Use `analyze_results.py` to view a comprehensive summary:

```bash
python scripts/analyze_results.py results.json
```

Output includes:

| Column | Description |
|--------|-------------|
| R@1 | % of queries with correct first result |
| R@10 | % of queries with correct result in top 10 |
| Prec | Precision (% of results that are relevant) |
| Fail | Number of failed queries |
| ∑ Time | Total elapsed time across all queries |
| Est. Wall | Estimated wall-clock time (assumes concurrency=20) |
| Min/Max/Avg | Query time statistics |
| LLM Cost | Total LLM API cost (for agent searchers) |
| Exa Cost | Total Exa API cost (based on public exa cost) |
| Total Cost | Combined cost |
| Srch/q | Average search calls per query |
| Tokens/q | Average input/output tokens per query |

### Per-Query Comparison

Use `compare_searchers.py` for detailed per-query analysis:

```bash
# Compare all searchers
python scripts/compare_searchers.py results.json

# Compare specific searchers
python scripts/compare_searchers.py results.json --searchers exa agentica
```

This shows per-query metrics side-by-side for each searcher.

## Rerunning Failed Queries

If some queries failed (e.g., timeouts), you can rerun just those:

```bash
python scripts/run_failed.py results.json -o results-fixed.json

# Exclude certain error types from rerun, useful when we don't want to rerun queries where agent returned value of incorrect format
python scripts/run_failed.py results.json -o results-fixed.json --exclude ValueError
```

NOTE: If output file specified in `-o` is the same as input file, **failed queries from input file will be overwritten**. So be careful to not loose initial run failing queries details. 

## Example Workflows

### Quick Test Run

```bash
# Test agentica with 5 queries
pbench --searchers agentica --limit 5 -o test.json
python scripts/analyze_results.py test.json
```

### Full Benchmark Comparison

```bash
# Run full benchmark with multiple searchers
pbench --searchers exa agentica agentica_orderer -o comparison.json

# Analyze results
python scripts/analyze_results.py comparison.json

# View per-query breakdown
python scripts/compare_searchers.py comparison.json
```

### Fix Failed Queries

```bash
# Initial run
pbench --searchers agentica -o results.json

# Check for failures
python scripts/analyze_results.py results.json  # Look at "Fail" column, also inspect results.json to see actual errors

# Rerun failed queries
python scripts/run_failed.py results.json -o results-fixed.json

# Verify fixes
python scripts/analyze_results.py results-fixed.json
```

### Skip Grading for Speed

```bash
# Run without LLM grading (useful for testing metrics except scores avoiding expensive grading)
pbench --searchers agentica --limit 50 --no-grading -o no-grade.json
```

## Metrics Explained

| Metric | Formula | Description |
|--------|---------|-------------|
| **R@1** (Match) | `queries_with_match_at_rank_1 / total_queries` | First result is relevant |
| **R@10** (Recall@10) | `queries_with_any_match_in_top_10 / total_queries` | Any relevant result in top 10 |
| **Precision** | `total_relevant_results / total_results` | Overall relevance of returned results |
