import jsonlines
import os
import random
import sys
sys.path.append("../")
from preparation.scitrek.data_loading import load_articles, get_full_texts
from utils.answer_eval import normalize_text, pre_processing

if __name__ == "__main__":
    random.seed(42)
    dataset_dir = "../../SciTrek/benchmark"
    articles_all = load_articles(articles_folder=dataset_dir + "/article/")
    # read the data from teacher proxy outputs
    samples_train = []
    samples_dev = []
    with jsonlines.open("../reasoning/openrlhf/scitrek_proxy_qwen3_4b_instruct_0527_rl_proxy_train.jsonl") as reader:
        for line in reader:
            samples_train.append(line)
    with jsonlines.open("../reasoning/openrlhf/scitrek_proxy_qwen3_4b_instruct_0527_rl_proxy_dev.jsonl") as reader:
        for line in reader:
            samples_dev.append(line)
    print("samples loaded", len(samples_train), len(samples_dev))


    def process_data(samples):
        message_data = []
        for sample in samples:
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
    train_message_data = random.sample(train_message_data, int(len(train_message_data) * 0.2))
    dev_message_data = process_data(samples_dev)
    dev_message_data = random.sample(dev_message_data, 100)
    print("training/dev", len(train_message_data), len(dev_message_data))

    data_folder = "scitrek_data_reinforcement_qwen_1"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    with jsonlines.open(f"{data_folder}/train.jsonl", "w") as writer:
        writer.write_all(train_message_data)
    with jsonlines.open(f"{data_folder}/dev.jsonl", "w") as writer:
        writer.write_all(dev_message_data)