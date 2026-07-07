import json
import os
from openai import OpenAI
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import jsonlines


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


def evaluate_sample(client, sample, multiple_judges=True):
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


def evaluate_samples_parallel(client, samples, multiple_judges=True, max_workers=16):
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


def calculate_statistics(judgments, multiple_judges=True):
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


if __name__ == "__main__":
    # export OPENAI_API_KEY=API_KEY
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)

    file_paths = []

    # on GAIL
    # file_paths.append("../testing/hotpotqa_proxy_qwen3_4b_instruct_0527_test.jsonl")
    # file_paths.append("../testing/hotpotqa_none_qwen3_4b_instruct_0527_test.jsonl")
    # file_paths.append("../testing/hotpotqa_full_qwen3_4b_instruct_0527_test.jsonl")
    #
    # file_paths.append("../testing/hotpotqa_proxy_qwen3_235b_a22b_instruct_0527_test.jsonl")
    # file_paths.append("../testing/hotpotqa_full_qwen3_235b_a22b_instruct_0527_test.jsonl")
    # file_paths.append("../testing/hotpotqa_proxy_qwen3_235b_a22b_thinking_0527_test.jsonl")
    # file_paths.append("../testing/hotpotqa_full_qwen3_235b_a22b_thinking_0527_test.jsonl")
    #
    # file_paths.append("../reasoning/openrlhf/hotpotqa_proxy_qwen3_4b_instruct_0527_rl_proxy_test.jsonl")
    # file_paths.append("../reasoning/openrlhf/hotpotqa_full_qwen3_4b_instruct_0527_rl_proxy_test.jsonl")

    # on Spartan
    # file_paths.append("../simple_sft/hotpotqa_proxy_qwen3_4b_instruct_0527_simple_sft_test.jsonl")
    # file_paths.append("../simple_sft/hotpotqa_full_qwen3_4b_instruct_0527_simple_sft_test.jsonl")
    #
    # file_paths.append("../simple_rl_openrlhf/hotpotqa_proxy_qwen3_4b_instruct_0527_simple_rl_test.jsonl")
    # file_paths.append("../simple_rl_openrlhf/hotpotqa_full_qwen3_4b_instruct_0527_simple_rl_test.jsonl")
    #
    # file_paths.append("../grounding/hotpotqa_proxy_qwen3_4b_instruct_0527_reinforcement_5_225_test.jsonl")
    # file_paths.append("../grounding/hotpotqa_full_qwen3_4b_instruct_0527_reinforcement_5_225_test.jsonl")
    #
    # file_paths.append("../more_analysis/possible_proxies/hotpotqa_proxy_qwen3_4b_instruct_0527_random_test.jsonl")
    # file_paths.append("../more_analysis/possible_proxies/hotpotqa_proxy_qwen3_4b_instruct_0527_related_test.jsonl")
    #
    # file_paths.append("../grounding/hotpotqa_full_qwen3_4b_instruct_0527_direct_grounding_test.jsonl")
    # file_paths.append("../grounding/hotpotqa_proxy_qwen3_4b_instruct_0527_direct_grounding_test.jsonl")
    #
    # file_paths.append("../teacher_full/hotpotqa_full_qwen3_4b_instruct_0527_teacher_full_test.jsonl")
    # file_paths.append("../teacher_full/hotpotqa_proxy_qwen3_4b_instruct_0527_teacher_full_test.jsonl")

    file_paths.append("../more_analysis/less_oracle/hotpotqa_proxy_qwen3_4b_instruct_0527_1t1_test.jsonl")
    file_paths.append("../more_analysis/less_oracle/hotpotqa_proxy_qwen3_4b_instruct_0527_1t2_test.jsonl")
    file_paths.append("../more_analysis/less_oracle/hotpotqa_proxy_qwen3_4b_instruct_0527_1t5_test.jsonl")

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")

    for file_path in file_paths:
        if os.path.exists(file_path):
            with jsonlines.open(file_path, "r") as reader:
                samples = []
                for line in reader:
                    samples.append(line)

            print(f"Loaded {len(samples)} samples from {file_path}")

            # Evaluate samples in parallel
            print("\nEvaluating samples with parallel API calls...")
            sample_results = evaluate_samples_parallel(
                client, samples,
                multiple_judges=True,
                max_workers=32
            )

            all_accuracies = []
            all_std_devs = []
            for judgments, judgment_texts in sample_results:
                stats = calculate_statistics(judgments, multiple_judges=True)
                all_std_devs.extend(stats["std_devs"])
                all_accuracies.append(stats["accuracy"])

            print(f"Total samples evaluated: {len(samples)}")
            print(f"Average accuracy across all samples: {np.mean(all_accuracies):.3f}, sample-level std: {np.mean(all_std_devs):.3f}")
        else:
            print(f"The file {file_path} does not exist.")