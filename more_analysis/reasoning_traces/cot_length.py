import numpy as np
import jsonlines
from transformers import AutoTokenizer


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    # generation_files = [
    #     "../../testing/scitrek_full_qwen3_4b_instruct_0527_test.jsonl",
    #     "../../simple_rl_openrlhf/scitrek_full_qwen3_4b_instruct_0527_simple_rl_test.jsonl",
    #     "../../grounding/scitrek_full_qwen3_4b_instruct_0527_direct_grounding_test.jsonl",
    #     "../../grounding/scitrek_full_qwen3_4b_instruct_0527_reinforcement_5_test.jsonl"
    # ]
    generation_files = [
        "../../teacher_full/scitrek_full_qwen3_4b_instruct_0527_teacher_full_test.jsonl"
    ]
    for generaiton_file in generation_files:
        print(generaiton_file)
        reasonings_all = []
        with jsonlines.open(generaiton_file) as reader:
            for line in reader:
                reasonings_all.extend(line["reasonings"])

        lengths_all = []
        for reasoning in reasonings_all:
            tokenized_reasoning = tokenizer.encode(reasoning)
            lengths_all.append(len(tokenized_reasoning))
        print(np.mean(lengths_all), np.std(lengths_all))