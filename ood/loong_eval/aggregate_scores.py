#!/usr/bin/env python
# coding=utf-8
"""
Aggregate evaluation scores by averaging across multiple generations per original example.
"""
import json
import argparse
import re
from collections import defaultdict
from tqdm import tqdm


def extract_number(text):
    """
    Extract score from evaluation response text.
    Looks for pattern [[score]] or [score].
    """
    if not text:
        return None
    
    # Try [[score]] format first
    match = re.search(r'\[\[([0-9]*\.?[0-9]+)\]\]', text)
    if match:
        return float(match.group(1))
    
    # Try [score] format as fallback
    match = re.search(r'\[([0-9]*\.?[0-9]+)\]', text)
    if match:
        return float(match.group(1))
    
    return None


def aggregate_scores(input_path, output_path):
    """
    Aggregate evaluation scores by averaging per original_id.
    
    Args:
        input_path: Path to evaluation output JSONL file with eval_response scores
        output_path: Path to output JSONL file with aggregated scores
    """
    # Group examples by original_id
    grouped_examples = defaultdict(list)
    
    print(f"Reading evaluation results from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Reading evaluations")):
            if not line.strip():
                continue
            
            try:
                item = json.loads(line.strip())
                original_id = item.get('original_id')
                
                if original_id is None:
                    print(f"Warning: Line {line_num + 1} has no original_id, skipping")
                    continue
                
                grouped_examples[original_id].append(item)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num + 1}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num + 1}: {e}")
                continue
    
    # Aggregate scores for each group
    aggregated_examples = []
    skipped_count = 0
    
    print(f"Aggregating scores for {len(grouped_examples)} original examples")
    for original_id, examples in tqdm(grouped_examples.items(), desc="Aggregating"):
        scores = []
        
        # Extract scores from all generations
        for example in examples:
            eval_response = example.get('eval_response', '')
            score = extract_number(eval_response)
            if score is not None:
                scores.append(score)
        
        if not scores:
            print(f"Warning: No valid scores found for original_id {original_id}, skipping")
            skipped_count += 1
            continue
        
        # Calculate average score
        avg_score = sum(scores) / len(scores)
        
        # Create aggregated example using the first example as base
        aggregated = examples[0].copy()
        
        # Remove generation-specific fields
        aggregated.pop('generation_id', None)
        aggregated.pop('id', None)  # Remove the gen-specific id
        
        # Restore original id
        aggregated['id'] = original_id
        
        # Create aggregated eval_response with average score
        # Format it similar to the original evaluation format
        aggregated['eval_response'] = f"Evaluation evidence: Averaged across {len(scores)} generations.\nRating: [[{avg_score:.2f}]]"
        
        # Store individual scores for reference
        aggregated['individual_scores'] = scores
        aggregated['num_generations'] = len(examples)
        
        aggregated_examples.append(aggregated)
    
    # Write aggregated examples to output file
    print(f"Writing {len(aggregated_examples)} aggregated examples to {output_path}")
    print(f"Skipped {skipped_count} examples with no valid scores")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in tqdm(aggregated_examples, desc="Writing output"):
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Aggregation complete: {len(aggregated_examples)} examples written")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate evaluation scores by averaging per original example')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to evaluation output JSONL file with eval_response scores')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output JSONL file with aggregated scores')
    
    args = parser.parse_args()
    aggregate_scores(args.input_path, args.output_path)

