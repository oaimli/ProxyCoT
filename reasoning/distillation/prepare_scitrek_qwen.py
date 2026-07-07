import jsonlines
import os
import random
import sys
sys.path.append("../../")
from preparation.scitrek.data_loading import load_articles, get_proxy

if __name__ == "__main__":
    dataset_dir = "../../../SciTrek/benchmark"
    articles_all = load_articles(articles_folder=dataset_dir + "/article/")
    # read the data from teacher bridge outputs
    samples_train = []
    samples_dev = []
    with jsonlines.open("../../teacher_proxy/scitrek_proxy_qwen3_235b_a22b_thinking_0527_train.jsonl") as reader:
        for line in reader:
            samples_train.append(line)
    with jsonlines.open("../../teacher_proxy/scitrek_proxy_qwen3_235b_a22b_thinking_0527_dev.jsonl") as reader:
        for line in reader:
            samples_dev.append(line)
    print("samples loaded", len(samples_train), len(samples_dev))


    def process_data(samples):
        message_data = []
        for sample in samples:
            question = sample["question"]
            articles = get_proxy(sample, articles_all)
            instruction = open("../../instructions/instruction_proxy_scitrek.txt").read()
            instruction = instruction.replace("<question>", question)
            instruction = instruction.replace("<articles>", articles)
            reasonings = sample["reasonings"]
            generations = sample["generations"]
            answer = sample["answer"]
            target = ""
            for reasoning, generation in zip(reasonings, generations):
                generation = str(generation).strip()
                generation = " ".join(generation.split())
                generation = ", ".join([tmp.strip() for tmp in generation.split(",")])
                if generation == answer:
                    if target == "":
                        target = reasoning
                    else:
                        if len(reasoning) < len(target):
                            target = reasoning
            if target != "":
                messages = [{"role": "user", "content": instruction},
                            {"role": "assistant", "content": target}]
                tmp = {"messages": messages}
                message_data.append(tmp)
        return message_data

    train_message_data = process_data(samples_train)
    dev_message_data = process_data(samples_dev)
    dev_message_data = random.sample(dev_message_data, 100)
    print("training/dev", len(train_message_data), len(dev_message_data))

    data_folder = "scitrek_data_qwen"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    with jsonlines.open(f"{data_folder}/train.jsonl", "w") as writer:
        writer.write_all(train_message_data)
    with jsonlines.open(f"{data_folder}/dev.jsonl", "w") as writer:
        writer.write_all(dev_message_data)


