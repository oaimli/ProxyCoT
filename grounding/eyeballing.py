import jsonlines
import random


if __name__ == "__main__":
    random.seed(42)
    # model_name_save = "gemma3_4b_it_direct_grounding"
    # model_name_save = "gemma3_4b_it_reinforcement_5"
    model_name_save = "qwen3_4b_instruct_0527_reinforcement_5"
    target_mode = "full"
    save_name = f"scitrek_{target_mode}_{model_name_save}_test.jsonl"

    results = []
    with jsonlines.open(save_name) as reader:
        for line in reader:
            results.append(line)
    samples = random.sample(results, 10)

    for sample in samples:
        print("#######################")
        print(sample["question"])
        print(sample["sql"])
        print(sample["answer"])
        print(sample["articles"])
        print(random.sample(sample["reasonings"], 1))