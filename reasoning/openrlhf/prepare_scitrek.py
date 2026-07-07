import jsonlines
import os
import random
import sys
from transformers import AutoTokenizer
sys.path.append("../../")
from preparation.scitrek.data_loading import load_train, load_articles, get_proxy


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    dataset_dir = "../../../MDRL/bench"
    articles_all = load_articles(articles_folder=dataset_dir + "/article/")
    samples_train, samples_dev = load_train(prefixes=["64k", "128k"], samples_folder=dataset_dir + "/dataset/samples/final/",
                                  no_null=True, no_simple=True)

    def process_data(samples):
        max_length = 0
        message_data = []
        for sample in samples:
            question = sample["question"]
            answer = sample["answer"]
            articles = get_proxy(sample, articles_all)
            instruction = open("../../instructions/qwen_instruction_proxy_scitrek.txt").read()
            instruction = instruction.replace("<question>", question)
            instruction = instruction.replace("<articles>", articles)
            messages = [{"role": "user", "content": instruction}]
            text_tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True)
            if len(text_tokenized) > max_length:
                max_length = len(text_tokenized)
            tmp = {"messages": messages, "answer": answer}
            message_data.append(tmp)
        print("max length", max_length)
        return message_data

    train_message_data = process_data(samples_train)
    dev_message_data = process_data(samples_dev)
    dev_message_data = random.sample(dev_message_data, 100)
    print("training/dev", len(train_message_data), len(dev_message_data)) # 7290, 100

    if not os.path.exists("scitrek_data"):
        os.mkdir("scitrek_data")
    with jsonlines.open("scitrek_data/train.jsonl", "w") as writer:
        writer.write_all(train_message_data)
    with jsonlines.open("scitrek_data/val.jsonl", "w") as writer:
        writer.write_all(dev_message_data)
