import os
import jsonlines

target_folder = "../../SciTrek/results_test"
generation_files = []
for filename in os.listdir(target_folder):
    if "o4mini_20250416" in filename and ("64k" in filename or "128k" in filename):
        if "full" in filename:
            generation_files.append(filename)

samples = []
for generation_file in generation_files:
    with jsonlines.open(os.path.join(target_folder, generation_file), "r") as reader:
        for line in reader:
            if line["answer"] != "NULL" and line["sql_type"] != "multi_simple":
                samples.append(line)

print(len(samples))
save_name = f"scitrek_full_o4mini_20250416_test.jsonl"
with jsonlines.open(save_name, "w") as writer:
    writer.write_all(samples)

