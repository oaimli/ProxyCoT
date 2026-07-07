import jsonlines
import json
import sys

sys.path.append('../')
from utils.answer_eval import exact_match_single, f1_score_single, pre_processing


def calculate_scores(sample):
    generations = sample["generations"]
    reference = sample["answer"]
    generation_batches_processed, references_processed = pre_processing([generations], [reference])
    generations = generation_batches_processed[0]
    reference = references_processed[0]
    results = []
    for generation in generations:
        results.append({"generation": generation, "em": exact_match_single(generation, reference), "f1": f1_score_single(generation, reference)})
    return results



if __name__ == "__main__":
    samples_qwen3_4b_instruct_proxy = []
    with jsonlines.open("hotpotqa_proxy_qwen3_4b_instruct_0527_test.jsonl") as reader:
        for line in reader:
            samples_qwen3_4b_instruct_proxy.append(line)
    samples_qwen3_4b_thinking_proxy = []
    with jsonlines.open("hotpotqa_proxy_qwen3_4b_thinking_0527_test.jsonl") as reader:
        for line in reader:
            samples_qwen3_4b_thinking_proxy.append(line)
    samples_qwen3_235b_instruct_proxy = []
    with jsonlines.open("hotpotqa_proxy_qwen3_235b_a22b_instruct_0527_test.jsonl") as reader:
        for line in reader:
            samples_qwen3_235b_instruct_proxy.append(line)
    samples_qwen3_235b_thinking_proxy = []
    with jsonlines.open("hotpotqa_proxy_qwen3_235b_a22b_thinking_0527_test.jsonl") as reader:
        for line in reader:
            samples_qwen3_235b_thinking_proxy.append(line)
    samples_qwen3_4b_instruct_full = []
    with jsonlines.open("hotpotqa_full_qwen3_4b_instruct_0527_test.jsonl") as reader:
        for line in reader:
            samples_qwen3_4b_instruct_full.append(line)
    samples_qwen3_4b_thinking_full = []
    with jsonlines.open("hotpotqa_full_qwen3_4b_thinking_0527_test.jsonl") as reader:
        for line in reader:
            samples_qwen3_4b_thinking_full.append(line)

    results = []
    for sample_qwen3_4b_instruct_proxy, sample_qwen3_4b_thinking_proxy, sample_qwen3_235b_instruct_proxy, sample_qwen3_235b_thinking_proxy, sample_qwen3_4b_instruct_full, sample_qwen3_4b_thinking_full in zip(
            samples_qwen3_4b_instruct_proxy, samples_qwen3_4b_thinking_proxy, samples_qwen3_235b_instruct_proxy,
            samples_qwen3_235b_thinking_proxy, samples_qwen3_4b_instruct_full, samples_qwen3_4b_thinking_full):
        assert sample_qwen3_4b_instruct_proxy["question"] == sample_qwen3_4b_thinking_proxy["question"] == \
               sample_qwen3_235b_instruct_proxy["question"] == sample_qwen3_235b_thinking_proxy["question"] == \
               sample_qwen3_4b_instruct_full["question"] == sample_qwen3_4b_thinking_full["question"]
        tmp = {}
        tmp["question"] = sample_qwen3_4b_instruct_proxy["question"]
        tmp["answer"] = sample_qwen3_4b_instruct_proxy["answer"]

        tmp["qwen3_4b_instruct_proxy"] = calculate_scores(sample_qwen3_4b_instruct_proxy)
        tmp["qwen3_4b_thinking_proxy"] = calculate_scores(sample_qwen3_4b_thinking_proxy)
        tmp["qwen3_235b_instruct_proxy"] = calculate_scores(sample_qwen3_235b_instruct_proxy)
        tmp["qwen3_235b_thinking_proxy"] = calculate_scores(sample_qwen3_235b_thinking_proxy)
        tmp["qwen3_4b_instruct_full"] = calculate_scores(sample_qwen3_4b_instruct_full)
        tmp["qwen3_4b_thinking_full"] = calculate_scores(sample_qwen3_4b_thinking_full)
        results.append(tmp)

    with open("eyeballing_hotpotqa.json", "w") as f:
        json.dump(results, f, indent=4)
