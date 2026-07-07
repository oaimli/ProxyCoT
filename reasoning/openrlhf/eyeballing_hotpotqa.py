import jsonlines
import json
import sys

sys.path.append('../../')
from utils.answer_eval import exact_match_single, f1_score_single, pre_processing


def calculate_scores(sample):
    generations = sample["generations"]
    reference = sample["answer"]
    reasonings = sample["reasonings"]
    generation_batches_processed, references_processed = pre_processing([generations], [reference])
    generations = generation_batches_processed[0]
    reference = references_processed[0]
    results = []
    for generation, reasoning in zip(generations, reasonings):
        results.append({"reasoning": reasoning, "generation": generation, "em": exact_match_single(generation, reference), "f1": f1_score_single(generation, reference)})
    return results



if __name__ == "__main__":
    samples_qwen3_4b_instruct_full = []
    with jsonlines.open("hotpotqa_full_qwen3_4b_instruct_0527_rl_proxy_test.jsonl") as reader:
        for line in reader:
            samples_qwen3_4b_instruct_full.append(line)
    samples_qwen3_4b_instruct_proxy = []
    with jsonlines.open("hotpotqa_proxy_qwen3_4b_instruct_0527_rl_proxy_test.jsonl") as reader:
        for line in reader:
            samples_qwen3_4b_instruct_proxy.append(line)

    results = []
    for sample_qwen3_4b_instruct_full, sample_qwen3_4b_instruct_proxy in zip(samples_qwen3_4b_instruct_full, samples_qwen3_4b_instruct_proxy):
        assert sample_qwen3_4b_instruct_full["question"] == sample_qwen3_4b_instruct_proxy["question"]
        tmp = {}
        tmp["question"] = sample_qwen3_4b_instruct_proxy["question"]
        tmp["answer"] = sample_qwen3_4b_instruct_proxy["answer"]

        tmp["qwen3_4b_instruct_proxy"] = calculate_scores(sample_qwen3_4b_instruct_proxy)
        tmp["qwen3_4b_instruct_full"] = calculate_scores(sample_qwen3_4b_instruct_full)
        results.append(tmp)

    with open("eyeballing_hotpotqa.json", "w") as f:
        json.dump(results, f, indent=4)
