import os
import re
import jsonlines
import numpy as np
import sys
sys.path.append("../utils")
from answer_eval import exact_match, f1_score, pre_processing


def evaluating(generation_batches, references):
    # invalid_answers = []
    # for generation_batch in generation_batches:
    #     for generation in generation_batch:
    #         if "\\text{" in generation:
    #             invalid_answers.append(generation)
    # print("invalid answers ratio: {}".format(len(invalid_answers)/(len(generation_batches) * 3)))

    generation_batches_processed, references_processed = pre_processing(generation_batches, references)
    scores = {}
    scores.update(exact_match(generation_batches_processed, references_processed))
    scores.update(f1_score(generation_batches_processed, references_processed))
    return scores


if __name__ == "__main__":
    COMPILED_REGEX = re.compile(r"\\boxed\{(.*?)\}")
    result_folder = "../testing"
    output_files = []
    for file in os.listdir(result_folder):
        if file.startswith("hotpotqa") and file.endswith(".jsonl"):
            output_files.append(file)
    for generation_file in output_files:
        print(generation_file)
        answers_all = []
        generation_batches_all = []
        with jsonlines.open(os.path.join(result_folder, f"{generation_file}")) as reader:
            for line in reader:
                if "generations" in line.keys():
                    answers_all.append(line["answer"])
                    line_generations = line["generations"]
                    generation_batches_all.append(line_generations)

        print("overall", len(answers_all), len(generation_batches_all))
        scores = evaluating(generation_batches_all, answers_all)
        metrics = []
        values_all = []
        for metric, values_tmp in scores.items():
            metrics.append(metric)
            values_all.append(str(round(np.mean(values_tmp), 3)))
        print(metrics, "/".join(values_all))
