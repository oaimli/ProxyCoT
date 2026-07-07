#!/usr/bin/env python
# coding=utf-8
"""
Convert input file format for Loong evaluation.
Expands examples with multiple generations into separate examples, one per generation.
"""
import json
import argparse
from tqdm import tqdm


def convert_file(input_path, output_path):
    """
    Convert input JSONL file by expanding generations.
    
    Args:
        input_path: Path to input JSONL file with 'generations' field
        output_path: Path to output JSONL file with 'generate_response' field
    """
    expanded_examples = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Converting examples")):
            if not line.strip():
                continue
                
            try:
                item = json.loads(line.strip())
                
                # Get generations list
                generations = item.get('generations', [])
                if not generations:
                    print(f"Warning: Line {line_num + 1} has no generations field, skipping")
                    continue
                
                # Get original ID
                original_id = item.get('id', f"example_{line_num}")
                
                # Expand each generation into a separate example
                for gen_idx, generation in enumerate(generations):
                    # Create new example
                    new_example = item.copy()
                    
                    # Remove generations field and add generate_response
                    new_example.pop('generations', None)
                    new_example['generate_response'] = generation
                    
                    # Add tracking fields
                    new_example['generation_id'] = gen_idx
                    new_example['original_id'] = original_id
                    
                    # Keep original id as well for compatibility
                    new_example['id'] = f"{original_id}_gen{gen_idx}"
                    
                    expanded_examples.append(new_example)
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num + 1}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num + 1}: {e}")
                continue
    
    # Write expanded examples to output file
    print(f"Writing {len(expanded_examples)} expanded examples to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in tqdm(expanded_examples, desc="Writing output"):
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Conversion complete: {len(expanded_examples)} examples written")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert input file format for Loong evaluation')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input JSONL file with generations field')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output JSONL file with generate_response field')
    
    args = parser.parse_args()
    convert_file(args.input_path, args.output_path)

