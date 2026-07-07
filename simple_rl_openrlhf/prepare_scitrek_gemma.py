import jsonlines
import os
import json
import random
import sys
from transformers import AutoTokenizer
sys.path.append("../")
from preparation.scitrek.data_loading import load_train, load_articles, get_full_texts


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    dataset_dir = "../../SciTrek/benchmark"
    articles_all = load_articles(articles_folder=dataset_dir + "/article/")
    samples_train, samples_dev = load_train(prefixes=["64k", "128k"], samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True)
    print("samples loaded", len(samples_train), len(samples_dev))

    def process_data(samples):
        message_data = []
        for sample in samples:
            question = sample["question"]
            answer = sample["answer"]
            markdowns = get_full_texts(sample, articles_all)
            articles = "\n\n\n".join(markdowns)
            instruction = open("../instructions/instruction_full_scitrek.txt").read()
            instruction = instruction.replace("<question>", question)
            instruction = instruction.replace("<articles>", articles)
            messages = [{"role": "user", "content": instruction}]
            text_tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True)
            print(len(text_tokenized))
            if len(text_tokenized) < 131072 - 2048:
                tmp = {"messages": messages, "answer": answer}
                message_data.append(tmp)
        return message_data

    train_message_data = process_data(samples_train)
    dev_message_data = process_data(samples_dev)
    dev_message_data = random.sample(dev_message_data, 100)
    print("training/dev", len(train_message_data), len(dev_message_data)) # 3200, 100

    data_folder = "scitrek_data_gemma"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    with jsonlines.open(f"{data_folder}/train.jsonl", "w") as writer:
        writer.write_all(train_message_data)
    with jsonlines.open(f"{data_folder}/val.jsonl", "w") as writer:
        writer.write_all(dev_message_data)
