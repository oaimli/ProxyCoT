import jsonlines


samples = []
with jsonlines.open("scitrek_bridge_qwen3_235b_a22b_thinking_0527_train.jsonl") as reader:
    for line in reader:
        samples.append(line)

for sample in samples:
    generations = sample["generations"]
    reasonings = sample["reasonings"]
    assert len(generations) == len(reasonings) == 3