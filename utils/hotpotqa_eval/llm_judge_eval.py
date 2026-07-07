import json
import os
import argparse
from openai import OpenAI
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any

prompt_template = """You need to check whether the prediction of a question-answering system
to a question is correct. You should make the judgment based on a list of
ground truth answers provided to you. Your response should be "correct"
if the prediction is correct or "incorrect" if the prediction is wrong.

Question: Who authored The Taming of the Shrew (published in 2002)?
Ground truth: ["William Shakespeare", "Roma Gill"]
Prediction: W Shakespeare
Correctness: correct

Question: Who authored The Taming of the Shrew (published in 2002)?
Ground truth: ["William Shakespeare", "Roma Gill"]
Prediction: Roma Gill and W Shakespeare
Correctness: correct

Question: Who authored The Taming of the Shrew (published in 2002)?
Ground truth: ["William Shakespeare", "Roma Gill"]
Prediction: Roma Shakespeare
Correctness: incorrect

Question: What country is Maharashtra Metro Rail Corporation Limited
located in?
Ground truth: ["India"]
Prediction: Maharashtra
Correctness: incorrect

Question: What's the job of Song Kang-ho in Parasite (2019)?
Ground truth: ["actor"]
Prediction: He plays the role of Kim Ki-taek, the patriarch of the Kim
family.
Correctness: correct

Question: Which era did Michael Oakeshott belong to?
Ground truth: ["20th-century philosophy"]
Prediction: 20th century.
Correctness: correct

Question: Edward Tise (known for Full Metal Jacket (1987)) is in what
department?
Ground truth: ["sound department"]
Prediction: 2nd Infantry Division, United States Army
Correctness: incorrect

Question: What wine region is Finger Lakes AVA a part of?
Ground truth: ["New York wine"]
Prediction: Finger Lakes AVA
Correctness: incorrect

Question: {QUESTION}
Ground truth: {GROUND_TRUTH}
Prediction: {PREDICTION}
Correctness:"""


def format_prompt(question, ground_truth, prediction):
    """Format the prompt template with question, ground truth, and prediction."""
    # Wrap ground truth answer in a list format
    ground_truth_list = json.dumps([ground_truth])
    return prompt_template.format(
        QUESTION=question,
        GROUND_TRUTH=ground_truth_list,
        PREDICTION=prediction
    )


def parse_judgment(response_text):
    """Parse the judge response to extract correct/incorrect judgment."""
    response_text = response_text.strip().lower()
    if "correct" in response_text and "incorrect" not in response_text:
        return True
    elif "incorrect" in response_text:
        return False
    else:
        # Try to find the last occurrence of "correct" or "incorrect"
        if response_text.endswith("correct"):
            return True
        elif response_text.endswith("incorrect"):
            return False
        # Default: if we can't parse, assume incorrect
        print(f"Warning: Could not parse judgment from: {response_text}")
        return False


def call_judge_api(client, prompt):
    """Call GPT-5 mini API to get judgment."""
    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=[{"role": "user", "content": prompt}]
        )
        # Extract text from response
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        elif hasattr(response, "output") and response.output:
            # Try to stitch text parts
            parts = []
            for item in response.output:
                for c in getattr(item, "content", []) or []:
                    if hasattr(c, "text") and c.text:
                        parts.append(c.text)
            return "\n".join(parts) if parts else ""
        return ""
    except Exception as e:
        print(f"Error calling API: {e}")
        return None


def evaluate_sample(client, sample, multiple_judges=False):
    """Evaluate a single sample with all its generations."""
    question = sample["question"]
    ground_truth = sample["answer"]
    generations = sample["generations"]
    
    all_judgments = []
    all_judgment_texts = []
    
    for gen_idx, generation in enumerate(generations):
        prompt = format_prompt(question, ground_truth, generation)
        
        if multiple_judges:
            # Make 3 API calls for this generation
            gen_judgments = []
            gen_judgment_texts = []
            for _ in range(3):
                judgment_text = call_judge_api(client, prompt)
                if judgment_text is None or judgment_text == "":
                    # If API call fails, treat as incorrect
                    judgment = False
                    judgment_text = ""
                else:
                    judgment = parse_judgment(judgment_text)
                gen_judgments.append(judgment)
                gen_judgment_texts.append(judgment_text)
            all_judgments.append(gen_judgments)
            all_judgment_texts.append(gen_judgment_texts)
        else:
            # Single API call for this generation
            judgment_text = call_judge_api(client, prompt)
            if judgment_text is None or judgment_text == "":
                # If API call fails, treat as incorrect
                judgment = False
                judgment_text = ""
            else:
                judgment = parse_judgment(judgment_text)
            all_judgments.append([judgment])
            all_judgment_texts.append([judgment_text])
    
    return all_judgments, all_judgment_texts


def evaluate_samples_parallel(client, samples, multiple_judges=False, max_workers=16):
    """Evaluate all samples with parallel API calls."""
    # Collect all API calls that need to be made with their metadata
    api_tasks = []
    
    for sample_idx, sample in enumerate(samples):
        question = sample["question"]
        ground_truth = sample["answer"]
        generations = sample["generations"]
        
        for gen_idx, generation in enumerate(generations):
            prompt = format_prompt(question, ground_truth, generation)
            
            if multiple_judges:
                # 3 API calls per generation
                for judge_idx in range(3):
                    api_tasks.append({
                        "sample_idx": sample_idx,
                        "gen_idx": gen_idx,
                        "judge_idx": judge_idx,
                        "prompt": prompt
                    })
            else:
                # 1 API call per generation
                api_tasks.append({
                    "sample_idx": sample_idx,
                    "gen_idx": gen_idx,
                    "judge_idx": None,
                    "prompt": prompt
                })
    
    # Process API calls in parallel
    def process_api_call(task_info):
        """Process a single API call and return the result."""
        judgment_text = call_judge_api(client, task_info["prompt"])
        if judgment_text is None or judgment_text == "":
            judgment = False
            judgment_text = ""
        else:
            judgment = parse_judgment(judgment_text)
        return {
            "sample_idx": task_info["sample_idx"],
            "gen_idx": task_info["gen_idx"],
            "judge_idx": task_info["judge_idx"],
            "judgment": judgment,
            "judgment_text": judgment_text
        }
    
    # Use ThreadPoolExecutor to process calls in parallel
    results = []
    completed_samples = set()
    
    # Track API calls per sample to know when a sample is complete
    sample_api_call_counts = {}
    for task in api_tasks:
        sample_idx = task["sample_idx"]
        sample_api_call_counts[sample_idx] = sample_api_call_counts.get(sample_idx, 0) + 1
    
    # Track completed API calls per sample
    sample_completion_counts = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_api_call, task): task
            for task in api_tasks
        }
        
        # Collect results as they complete
        with tqdm(total=len(samples), desc="Evaluating samples") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing task (sample {task['sample_idx']}, gen {task['gen_idx']}): {e}")
                    result = {
                        "sample_idx": task["sample_idx"],
                        "gen_idx": task["gen_idx"],
                        "judge_idx": task["judge_idx"],
                        "judgment": False,
                        "judgment_text": ""
                    }
                    results.append(result)
                
                # Track completion for this sample
                sample_idx = result["sample_idx"]
                sample_completion_counts[sample_idx] = sample_completion_counts.get(sample_idx, 0) + 1
                
                # Check if this sample is now complete
                if (sample_idx not in completed_samples and 
                    sample_completion_counts[sample_idx] >= sample_api_call_counts.get(sample_idx, 0)):
                    completed_samples.add(sample_idx)
                    pbar.update(1)
    
    # Reconstruct results for each sample
    sample_results = []
    for sample_idx in range(len(samples)):
        sample = samples[sample_idx]
        generations = sample["generations"]
        
        all_judgments = []
        all_judgment_texts = []
        
        for gen_idx in range(len(generations)):
            if multiple_judges:
                gen_judgments = []
                gen_judgment_texts = []
                for judge_idx in range(3):
                    # Find the result for this combination
                    result = next(
                        (r for r in results 
                         if r["sample_idx"] == sample_idx 
                         and r["gen_idx"] == gen_idx 
                         and r["judge_idx"] == judge_idx),
                        None
                    )
                    if result:
                        gen_judgments.append(result["judgment"])
                        gen_judgment_texts.append(result["judgment_text"])
                    else:
                        gen_judgments.append(False)
                        gen_judgment_texts.append("")
                all_judgments.append(gen_judgments)
                all_judgment_texts.append(gen_judgment_texts)
            else:
                # Find the result for this sample and generation
                result = next(
                    (r for r in results 
                     if r["sample_idx"] == sample_idx 
                     and r["gen_idx"] == gen_idx 
                     and r["judge_idx"] is None),
                    None
                )
                if result:
                    all_judgments.append([result["judgment"]])
                    all_judgment_texts.append([result["judgment_text"]])
                else:
                    all_judgments.append([False])
                    all_judgment_texts.append([""])
        
        sample_results.append((all_judgments, all_judgment_texts))
    
    return sample_results


def calculate_statistics(judgments, multiple_judges=False):
    """Calculate accuracy and std dev for a sample."""
    accuracies = []
    std_devs = []
    
    for gen_judgments in judgments:
        if multiple_judges:
            # Calculate accuracy (mean) and std dev for this generation
            gen_acc = np.mean(gen_judgments) if gen_judgments else 0.0
            gen_std = np.std(gen_judgments) if len(gen_judgments) > 1 else 0.0
            accuracies.append(gen_acc)
            std_devs.append(gen_std)
        else:
            # Single judgment per generation
            gen_acc = 1.0 if gen_judgments[0] else 0.0
            accuracies.append(gen_acc)
    
    overall_accuracy = np.mean(accuracies) if accuracies else 0.0
    
    result = {
        "accuracies": accuracies,
        "accuracy": overall_accuracy
    }
    
    if multiple_judges:
        result["std_devs"] = std_devs
    
    return result


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-judge evaluation for HotpotQA")
    parser.add_argument(
        "--input-file",
        type=str,
        default="hotpotqa_proxy_qwen3_235b_a22b_instruct_0527_test.jsonl",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to output JSONL file (default: {input_file}_evaluated.jsonl)"
    )
    parser.add_argument(
        "--multiple-judges",
        action="store_true",
        help="Enable 3 judge generations per sample to calculate std dev"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Maximum number of parallel API calls (default: 16)"
    )
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output_file is None:
        base_name = args.input_file.replace(".jsonl", "")
        args.output_file = f"{base_name}_evaluated.jsonl"
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Read input file
    samples = []
    with open(args.input_file, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples from {args.input_file}")
    print(f"Multiple judges mode: {args.multiple_judges}")
    print(f"Parallel workers: {args.max_workers}")
    
    # Evaluate samples in parallel
    print("\nEvaluating samples with parallel API calls...")
    sample_results = evaluate_samples_parallel(
        client, samples, 
        multiple_judges=args.multiple_judges,
        max_workers=args.max_workers
    )
    
    # Process results
    results = []
    all_accuracies = []
    accuracy_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
    all_std_devs = []
    
    for sample, (judgments, judgment_texts) in zip(samples, sample_results):
        # Preserve all original fields
        result = sample.copy()
        
        # Calculate statistics
        stats = calculate_statistics(judgments, args.multiple_judges)
        
        # Add evaluation results
        result["judgments"] = judgments
        result["judgment_texts"] = judgment_texts
        result["accuracies"] = stats["accuracies"]
        result["accuracy"] = stats["accuracy"]
        
        if args.multiple_judges:
            result["std_devs"] = stats["std_devs"]
            all_std_devs.extend(stats["std_devs"])
        
        results.append(result)
        all_accuracies.append(stats["accuracy"])
        
        # Update accuracy distribution (count how many of the 3 generations are correct)
        # A generation is correct if its accuracy >= 0.5
        correct_count = sum(1 for acc in stats["accuracies"] if acc >= 0.5)
        accuracy_distribution[correct_count] = accuracy_distribution.get(correct_count, 0) + 1
    
    # Write results to output file
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\nResults saved to {args.output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total samples evaluated: {len(samples)}")
    print(f"Total generations evaluated: {len(samples) * 3}")
    print(f"Average accuracy across all samples: {np.mean(all_accuracies):.4f}")
    print(f"\nAccuracy distribution:")
    print(f"  0/3 correct: {accuracy_distribution[0]} samples ({accuracy_distribution[0]/len(samples)*100:.2f}%)")
    print(f"  1/3 correct: {accuracy_distribution[1]} samples ({accuracy_distribution[1]/len(samples)*100:.2f}%)")
    print(f"  2/3 correct: {accuracy_distribution[2]} samples ({accuracy_distribution[2]/len(samples)*100:.2f}%)")
    print(f"  3/3 correct: {accuracy_distribution[3]} samples ({accuracy_distribution[3]/len(samples)*100:.2f}%)")
    
    if args.multiple_judges and all_std_devs:
        print(f"\nAverage std dev across all generations: {np.mean(all_std_devs):.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
