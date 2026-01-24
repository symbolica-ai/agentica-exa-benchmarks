import json

INPUT_PRICE_PER_MILLION = 0.50
OUTPUT_PRICE_PER_MILLION = 3.00

with open('results.json', 'r') as f:
    data = json.load(f)

for searcher_name in ['agentica_orderer', 'openai_orderer']:
    queries = data['searchers'][searcher_name]['queries']
    input_tokens = sum(q.get('input_tokens', 0) for q in queries)
    output_tokens = sum(q.get('output_tokens', 0) for q in queries)
    num_queries = len(queries)
    
    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    total_cost = input_cost + output_cost
    
    avg_cost = total_cost / num_queries
    avg_input_tokens = input_tokens / num_queries
    avg_output_tokens = output_tokens / num_queries

    num_errors = sum(1 for q in queries if 'error' in q)
    print(f'{searcher_name} ({num_queries} queries):') 
    print(f'  Avg Input Tokens:  {avg_input_tokens:,.1f}')
    print(f'  Avg Output Tokens: {avg_output_tokens:,.1f}')
    print(f'  Total Cost:        \${total_cost:.4f} ({total_cost * 1000:.4f} per 1K queries)')
    print(f'  Avg Cost/Query:    \${avg_cost:.6f} ({avg_cost * 1000:.4f} per 1K queries)')
    print()