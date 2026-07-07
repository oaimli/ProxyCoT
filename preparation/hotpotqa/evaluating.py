import re
import jsonlines
import numpy as np
import sys

sys.path.append("../../utils")
from answer_eval import exact_match, f1_score, pre_processing


def evaluating(generation_batches, references):
    generation_batches_processed, references_processed = pre_processing(generation_batches, references)
    scores = {}
    scores.update(exact_match(generation_batches_processed, references_processed))
    scores.update(f1_score(generation_batches_processed, references_processed))
    return scores


if __name__ == "__main__":
    model_name_save = "qwen3_4b_instruct_0527"
    save_name = f"hotpotqa_{model_name_save}_test.jsonl"

    COMPILED_REGEX = re.compile(r"\\boxed\{(.*?)\}")
    generation_file = save_name
    print(generation_file)

    answers_all_long = []
    generation_batches_all_long = []

    answers_all_proxy = []
    generation_batches_all_proxy = []

    answers_all_none = []
    generation_batches_all_none = []

    with jsonlines.open(f"{generation_file}") as reader:
        for line in reader:
            if "generations_long" in line.keys():
                answers_all_long.append(line["answer"])
                answers_all_proxy.append(line["answer"])
                answers_all_none.append(line["answer"])
                generation_batches_all_long.append(line["generations_long"])
                generation_batches_all_proxy.append(line["generations_proxy"])
                generation_batches_all_none.append(line["generations_none"])

    print("long", len(answers_all_long), len(generation_batches_all_long))
    scores = evaluating(generation_batches_all_long, answers_all_long)
    metrics = []
    values_all = []
    for metric, values_tmp in scores.items():
        metrics.append(metric)
        values_all.append(str(round(np.mean(values_tmp), 3)))
    print(metrics, "/".join(values_all))

    print("proxy", len(answers_all_proxy), len(generation_batches_all_proxy))
    scores = evaluating(generation_batches_all_proxy, answers_all_proxy)
    metrics = []
    values_all = []
    for metric, values_tmp in scores.items():
        metrics.append(metric)
        values_all.append(str(round(np.mean(values_tmp), 3)))
    print(metrics, "/".join(values_all))

    print("none", len(answers_all_none), len(generation_batches_all_none))
    scores = evaluating(generation_batches_all_none, answers_all_none)
    metrics = []
    values_all = []
    for metric, values_tmp in scores.items():
        metrics.append(metric)
        values_all.append(str(round(np.mean(values_tmp), 3)))
    print(metrics, "/".join(values_all))