import os
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
    COMPILED_REGEX = re.compile(r"\\boxed\{(.*?)\}")
    result_folder = "./"
    output_files = []
    for file in os.listdir(result_folder):
        if file.endswith("_test.jsonl") and file.startswith("scitrek"):
            output_files.append(file)
    for generation_file in output_files:
        print(generation_file)
        answers_all = []
        generation_batches_all = []
        answers_aggregating = []
        generation_batches_aggregating = []
        answers_sorting = []
        generation_batches_sorting = []
        answers_filtering = []
        generation_batches_filtering = []
        answers_filtering_aggregating = []
        generation_batches_filtering_aggregating = []
        answers_filtering_sorting = []
        generation_batches_filtering_sorting = []
        answers_relational_filtering = []
        generation_batches_relational_filtering = []
        with jsonlines.open(os.path.join(result_folder, f"{generation_file}")) as reader:
            for line in reader:
                if "generations" in line.keys():
                    answers_all.append(line["answer"])
                    line_generations = line["generations"]
                    generation_batches_all.append(line_generations)

                    sql_type = line["sql_type"]
                    if sql_type == "multi_ran_organizing":
                        answers_sorting.append(line["answer"])
                        generation_batches_sorting.append(line_generations)
                    elif sql_type == "multi_ran_aggregating":
                        answers_aggregating.append(line["answer"])
                        generation_batches_aggregating.append(line_generations)
                    elif sql_type == "multi_ran_filtering_ofo":
                        answers_filtering.append(line["answer"])
                        generation_batches_filtering.append(line_generations)
                    elif sql_type == "multi_ran_filtering_foa":
                        answers_filtering_aggregating.append(line["answer"])
                        generation_batches_filtering_aggregating.append(line_generations)
                    elif sql_type == "multi_ran_filtering_foo":
                        answers_filtering_sorting.append(line["answer"])
                        generation_batches_filtering_sorting.append(line_generations)
                    elif sql_type == "multi_graph_filtering":
                        answers_relational_filtering.append(line["answer"])
                        generation_batches_relational_filtering.append(line_generations)
                    else:
                        print("incorrect sql type")

        print("overall", len(answers_all), len(generation_batches_all))
        scores = evaluating(generation_batches_all, answers_all)
        metrics = []
        values_all = []
        for metric, values_tmp in scores.items():
            metrics.append(metric)
            values_all.append(str(round(np.mean(values_tmp), 3)))
        print(metrics, "/".join(values_all))

        print("sorting", len(answers_sorting), len(generation_batches_sorting))
        scores = evaluating(generation_batches_sorting, answers_sorting)
        metrics = []
        values_all = []
        for metric, values_tmp in scores.items():
            metrics.append(metric)
            values_all.append(str(round(np.mean(values_tmp), 3)))
        print(metrics, "/".join(values_all))

        print("aggregating", len(answers_aggregating), len(generation_batches_aggregating))
        scores = evaluating(generation_batches_aggregating, answers_aggregating)
        metrics = []
        values_all = []
        for metric, values_tmp in scores.items():
            metrics.append(metric)
            values_all.append(str(round(np.mean(values_tmp), 3)))
        print(metrics, "/".join(values_all))

        print("filtering", len(answers_filtering), len(generation_batches_filtering))
        scores = evaluating(generation_batches_filtering, answers_filtering)
        metrics = []
        values_all = []
        for metric, values_tmp in scores.items():
            metrics.append(metric)
            values_all.append(str(round(np.mean(values_tmp), 3)))
        print(metrics, "/".join(values_all))

        print("filtering+sorting", len(answers_filtering_sorting), len(generation_batches_filtering_sorting))
        scores = evaluating(generation_batches_filtering_sorting, answers_filtering_sorting)
        metrics = []
        values_all = []
        for metric, values_tmp in scores.items():
            metrics.append(metric)
            values_all.append(str(round(np.mean(values_tmp), 3)))
        print(metrics, "/".join(values_all))

        print("filtering+aggregating", len(answers_filtering_aggregating), len(generation_batches_filtering_aggregating))
        scores = evaluating(generation_batches_filtering_aggregating, answers_filtering_aggregating)
        metrics = []
        values_all = []
        for metric, values_tmp in scores.items():
            metrics.append(metric)
            values_all.append(str(round(np.mean(values_tmp), 3)))
        print(metrics, "/".join(values_all))

        print("relational filtering", len(answers_relational_filtering),
              len(generation_batches_relational_filtering))
        scores = evaluating(generation_batches_relational_filtering, answers_relational_filtering)
        metrics = []
        values_all = []
        for metric, values_tmp in scores.items():
            metrics.append(metric)
            values_all.append(str(round(np.mean(values_tmp), 3)))
        print(metrics, "/".join(values_all))


