import jsonlines
import os
import random
import sys
sys.path.append("../../")
from preparation.scitrek.data_loading import load_articles, get_proxy, load_test

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
                tmp = {"instruction": instruction, "reasoning": target, "answer": answer}
                message_data.append(tmp)
        return message_data

    train_message_data = process_data(samples_train)
    dev_message_data = process_data(samples_dev)
    print("training/dev", len(train_message_data), len(dev_message_data))

    data_folder = "scitrek_tmp_qwen"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    with jsonlines.open(f"{data_folder}/train.jsonl", "w") as writer:
        writer.write_all(train_message_data)
    with jsonlines.open(f"{data_folder}/dev.jsonl", "w") as writer:
        writer.write_all(dev_message_data)

    # test data
    samples_test = []
    samples_test.extend(load_test(prefix="64k",
                                  samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True))
    samples_test.extend(load_test(prefix="128k",
                                  samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True))
    print("test", len(samples_test))
    data_test = []
    for sample in samples_test:
        question = sample["question"]
        answer = sample["answer"]
        articles = get_proxy(sample, articles_all)
        instruction = open("../../instructions/instruction_proxy_scitrek.txt").read()
        instruction = instruction.replace("<question>", question)
        instruction = instruction.replace("<articles>", articles)
        tmp = {"instruction": instruction, "answer": answer}
        data_test.append(tmp)
    with jsonlines.open(f"{data_folder}/test.jsonl", "w") as writer:
        writer.write_all(data_test)


