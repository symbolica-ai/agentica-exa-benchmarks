import json

INPUT_PRICE_PER_MILLION = 0.50
OUTPUT_PRICE_PER_MILLION = 3.00

with open('results.json', 'r') as f:
    data = json.load(f)

for searcher_name in ['agentica_orderer', 'openai_orderer']:
    searcher = data['searchers'][searcher_name]
    queries = searcher['queries']
    
    input_tokens = sum(q.get('input_tokens', 0) for q in queries)
    output_tokens = sum(q.get('output_tokens', 0) for q in queries)
    num_queries = len(queries)
    
    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    total_cost = input_cost + output_cost
    
    avg_cost = total_cost / num_queries
    avg_input_tokens = input_tokens / num_queries
    avg_output_tokens = output_tokens / num_queries

    avg_latency = sum(q.get('elapsed_s', 0) for q in queries)
    avg_latency_per_query = avg_latency / num_queries

    # Calculate metrics from query grades
    recall_at_1_sum = 0
    recall_at_10_sum = 0
    precision_sum = 0
    
    for q in queries:
        grades = q.get('grades', [])
        if not grades:
            continue
        
        # Sort by rank to be safe
        grades_sorted = sorted(grades, key=lambda g: g.get('rank', 999))
        
        # Recall@1: is rank-1 a match?
        if grades_sorted and grades_sorted[0].get('is_match', 0) >= 1.0:
            recall_at_1_sum += 1
        
        # Recall@10: is any result in top 10 a match?
        if any(g.get('is_match', 0) >= 1.0 for g in grades_sorted[:10]):
            recall_at_10_sum += 1
        
        # Precision: fraction of results that are matches
        n_matches = sum(1 for g in grades if g.get('is_match', 0) >= 1.0)
        precision_sum += n_matches / len(grades) if grades else 0
    
    recall_at_1 = recall_at_1_sum / num_queries
    recall_at_10 = recall_at_10_sum / num_queries
    precision = precision_sum / num_queries

    print(f'{searcher_name} ({num_queries} queries):') 
    print(f'  Recall@1: {recall_at_1:.1%}')
    print(f'  Recall@10: {recall_at_10:.1%}')
    print(f'  Precision: {precision:.1%}')
    print(f'  Total Input Tokens Cost:  ${input_cost:.4f}')
    print(f'  Total Output Tokens Cost:  ${output_cost:.4f}')
    print(f'  Total Cost:        ${total_cost:.4f}')
    print(f'  Avg Cost/Query:    ${avg_cost:.6f}')
    print(f'  Avg Latency:       {avg_latency_per_query:.2f}s')
    print()