import jsonlines
import os
import random
import sys
sys.path.append("../")
from preparation.scitrek.data_loading import load_articles, get_full_texts
from utils.answer_eval import pre_processing, normalize_text

if __name__ == "__main__":
    random.seed(42)
    dataset_dir = "../../SciTrek/benchmark"
    articles_all = load_articles(articles_folder=dataset_dir + "/article/")
    # read the data from teacher proxy outputs
    samples_train = []
    samples_dev = []

    subset_0 = []
    with jsonlines.open("../teacher_full/scitrek_full_qwen3_235b_a22b_thinking_0527_train_0.jsonl") as reader:
        for line in reader:
            subset_0.append(line)
    subset_1 = []
    with jsonlines.open("../teacher_full/scitrek_full_qwen3_235b_a22b_thinking_0527_train_1.jsonl") as reader:
        for line in reader:
            subset_1.append(line)
    subset_2 = []
    with jsonlines.open("../teacher_full/scitrek_full_qwen3_235b_a22b_thinking_0527_train_2.jsonl") as reader:
        for line in reader:
            subset_2.append(line)
    subset_3 = []
    with jsonlines.open("../teacher_full/scitrek_full_qwen3_235b_a22b_thinking_0527_train_3.jsonl") as reader:
        for line in reader:
            subset_3.append(line)
    for sample_0, sample_1, sample_2, sample_3 in zip(subset_0, subset_1, subset_2, subset_3):
        if "generations" in sample_0.keys():
            samples_train.append(sample_0)
            continue
        if "generations" in sample_1.keys():
            samples_train.append(sample_1)
            continue
        if "generations" in sample_2.keys():
            samples_train.append(sample_2)
            continue
        if "generations" in sample_3.keys():
            samples_train.append(sample_3)
            continue

    with jsonlines.open("../teacher_full/scitrek_full_qwen3_235b_a22b_thinking_0527_dev.jsonl") as reader:
        for line in reader:
            samples_dev.append(line)
    print("samples loaded", len(samples_train), len(samples_dev))


    def process_data(samples):
        message_data = []
        for sample in samples:
            # print(sample.keys())
            question = sample["question"]
            markdowns = get_full_texts(sample, articles_all)
            articles = "\n\n\n".join(markdowns)
            instruction = open("../instructions/instruction_full_scitrek.txt").read()
            instruction = instruction.replace("<question>", question)
            instruction = instruction.replace("<articles>", articles)

            reasonings = sample["reasonings"]
            generations = sample["generations"]
            answer = sample["answer"]
            target = ""
            generation_batches, references = pre_processing([generations], [answer])
            for reasoning, generation in zip(reasonings, generation_batches[0]):
                generation = normalize_text(generation)
                answer = normalize_text(references[0])
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

    data_folder = "scitrek_data"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    with jsonlines.open(f"{data_folder}/train.jsonl", "w") as writer:
        writer.write_all(train_message_data)
    with jsonlines.open(f"{data_folder}/dev.jsonl", "w") as writer:
        writer.write_all(dev_message_data)