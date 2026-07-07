import random

import jsonlines
import json

samples = []
with jsonlines.open("/Users/miao4/Downloads/Temporary/scitrek_table_qwen3_235b_a22b_thinking_0527_test.jsonl") as reader:
    for line in reader:
        samples.append(line)

samples_aggregating = []
samples_sorting = []
samples_filtering = []
samples_filtering_aggregating = []
samples_filtering_sorting = []
samples_relational_filtering = []
for sample in samples:
    answer = sample["answer"]
    generations = sample["generations"]
    if answer in generations:
        sql_type = sample["sql_type"]
        if sql_type == "multi_ran_organizing":
            samples_sorting.append(sample)
        elif sql_type == "multi_ran_aggregating":
            samples_aggregating.append(sample)
        elif sql_type == "multi_ran_filtering_ofo":
            samples_filtering.append(sample)
        elif sql_type == "multi_ran_filtering_foa":
            samples_filtering_aggregating.append(sample)
        elif sql_type == "multi_ran_filtering_foo":
            samples_filtering_sorting.append(sample)
        elif sql_type == "multi_graph_filtering":
            samples_relational_filtering.append(sample)
        else:
            print("incorrect sql type")

print("sorting", len(samples_sorting))
print("filtering", len(samples_filtering))
print("aggregating", len(samples_aggregating))
print("filtering aggregating", len(samples_filtering_aggregating))
print("filtering sorting", len(samples_filtering_sorting))
print("relational filtering", len(samples_relational_filtering))

tmp = []
tmp.extend(random.sample(samples_sorting, 10))
tmp.extend(random.sample(samples_filtering, 10))
tmp.extend(random.sample(samples_aggregating, 10))
tmp.extend(random.sample(samples_filtering_aggregating, 10))
tmp.extend(random.sample(samples_filtering_sorting, 10))
tmp.extend(random.sample(samples_relational_filtering, 10))


with open("scitrek.json", "w") as f:
    json.dump(tmp, f, indent=4)