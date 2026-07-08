import jsonlines
import os
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    data_folder = "scitrek_data_1t1"
    if os.path.exists(data_folder):
        samples_train = []
        with jsonlines.open(os.path.join(data_folder, "train.jsonl")) as reader:
            for line in reader:
                samples_train.append(line)
        samples_dev = []
        with jsonlines.open(os.path.join(data_folder, "val.jsonl")) as reader:
            for line in reader:
                samples_dev.append(line)
        contexts_length = []
        for sample in tqdm(samples_train + samples_dev):
            messages = sample["messages"]
            text_tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True)
            contexts_length.append(len(text_tokenized))
        print(data_folder, np.mean(contexts_length), np.std(contexts_length), np.max(contexts_length))


    data_folder = "scitrek_data_1t2"
    if os.path.exists(data_folder):
        samples_train = []
        with jsonlines.open(os.path.join(data_folder, "train.jsonl")) as reader:
            for line in reader:
                samples_train.append(line)
        samples_dev = []
        with jsonlines.open(os.path.join(data_folder, "val.jsonl")) as reader:
            for line in reader:
                samples_dev.append(line)
        contexts_length = []
        for sample in tqdm(samples_train + samples_dev):
            messages = sample["messages"]
            text_tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True)
            contexts_length.append(len(text_tokenized))
        print(data_folder, np.mean(contexts_length), np.std(contexts_length), np.max(contexts_length))


    data_folder = "scitrek_data_1t5"
    if os.path.exists(data_folder):
        samples_train = []
        with jsonlines.open(os.path.join(data_folder, "train.jsonl")) as reader:
            for line in reader:
                samples_train.append(line)
        samples_dev = []
        with jsonlines.open(os.path.join(data_folder, "val.jsonl")) as reader:
            for line in reader:
                samples_dev.append(line)
        contexts_length = []
        for sample in tqdm(samples_train + samples_dev):
            messages = sample["messages"]
            text_tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True)
            contexts_length.append(len(text_tokenized))
        print(data_folder, np.mean(contexts_length), np.std(contexts_length), np.max(contexts_length))


    data_folder = "hotpotqa_data_1t1"
    if os.path.exists(data_folder):
        samples_train = []
        with jsonlines.open(os.path.join(data_folder, "train.jsonl")) as reader:
            for line in reader:
                samples_train.append(line)
        samples_dev = []
        with jsonlines.open(os.path.join(data_folder, "val.jsonl")) as reader:
            for line in reader:
                samples_dev.append(line)
        contexts_length = []
        for sample in tqdm(samples_train + samples_dev):
            messages = sample["messages"]
            text_tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True)
            contexts_length.append(len(text_tokenized))
        print(data_folder, np.mean(contexts_length), np.std(contexts_length), np.max(contexts_length))


    data_folder = "hotpotqa_data_1t2"
    if os.path.exists(data_folder):
        samples_train = []
        with jsonlines.open(os.path.join(data_folder, "train.jsonl")) as reader:
            for line in reader:
                samples_train.append(line)
        samples_dev = []
        with jsonlines.open(os.path.join(data_folder, "val.jsonl")) as reader:
            for line in reader:
                samples_dev.append(line)
        contexts_length = []
        for sample in tqdm(samples_train + samples_dev):
            messages = sample["messages"]
            text_tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True)
            contexts_length.append(len(text_tokenized))
        print(data_folder, np.mean(contexts_length), np.std(contexts_length), np.max(contexts_length))


    data_folder = "hotpotqa_data_1t5"
    if os.path.exists(data_folder):
        samples_train = []
        with jsonlines.open(os.path.join(data_folder, "train.jsonl")) as reader:
            for line in reader:
                samples_train.append(line)
        samples_dev = []
        with jsonlines.open(os.path.join(data_folder, "val.jsonl")) as reader:
            for line in reader:
                samples_dev.append(line)
        contexts_length = []
        for sample in tqdm(samples_train + samples_dev):
            messages = sample["messages"]
            text_tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True)
            contexts_length.append(len(text_tokenized))
        print(data_folder, np.mean(contexts_length), np.std(contexts_length), np.max(contexts_length))