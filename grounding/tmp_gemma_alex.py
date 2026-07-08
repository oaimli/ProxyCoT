import random
import re
import jsonlines
import numpy as np
from vllm import LLM, SamplingParams
import sys
sys.path.append("../")
from utils.answer_eval import exact_match_single, f1_score_single, pre_processing_single


if __name__ == "__main__":
    random.seed(42)
    COMPILED_REGEX = re.compile(r"\\boxed\{([\s\S]*?)\}")
    PATTERN_VALID_NUMBER = r'^(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?$'

    # model_name = "agurung/SFT-gemma-3-4b-it"
    # model_name = "google/gemma-3-4b-it"
    model_name = "./gemma3_4b_it_alex"
    jsonl_path = "scitrek_data_reinforcement_gemma_alex/test.jsonl"

    eval_samples = []
    with jsonlines.open(jsonl_path, "r") as f:
        for line in f:
            eval_samples.append(line)
    eval_samples = random.sample(eval_samples, k=min(len(eval_samples), 100))

    llm = LLM(model=model_name, tensor_parallel_size=4)
    sampling_params = SamplingParams(
        n=3,
        temperature=1.0,
        max_tokens=2048,
        top_k=64,
        top_p=0.95,
    )

    # drop the last assistant message which is the ground truth
    messages_list = [ex["messages"][:-1] for ex in eval_samples]
    conversation_outputs = llm.chat(messages_list, sampling_params=sampling_params)

    em_scores = []
    f1_scores = []

    # Now compute metrics & pretty print sequentially for stable output
    for eval_sample, conversation_output in zip(eval_samples, conversation_outputs):
        messages = eval_sample["messages"]
        answer = messages[-1]['content']

        generations = []
        for output in conversation_output.outputs:
            matches = COMPILED_REGEX.findall(output.text)
            generations.append(matches[-1].strip() if matches else "")

        generations, answer = pre_processing_single(generations, answer)
        print(answer)
        print(generations)

        ems = []
        f1s = []
        for generation in generations:
            em = exact_match_single(generation, answer)
            f1 = f1_score_single(generation, answer)
            ems.append(em)
            f1s.append(f1)

        em_scores.append(np.mean(ems))
        f1_scores.append(np.mean(f1s))

    # ---- Summary metrics ----
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print("em", avg_em)
    print("f1", avg_f1)
